from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session, joinedload
from typing import List

from database import get_db
import models
import schemas
from services.rag_service import rag_engine
from utils import get_current_user_id

router = APIRouter()


def _resolve_chunk_info(db: Session, chroma_chunk_id: str):
    """
    Map a ChromaDB string chunk_id → (supabase_chunk_id, document_url).

    Single JOIN query: chunks → documents.
    Returns (int chunk_id, str source_uri) or (None, None) if not found.

    After running scripts/update_document_urls_from_catalog.py,
    source_uri will be a full https://mysu.sabanciuniv.edu/... URL.
    """
    if not chroma_chunk_id:
        return None, None

    row = (
        db.query(models.Chunk.chunk_id, models.Document.source_uri)
        .join(models.Document, models.Chunk.document_id == models.Document.document_id)
        .filter(models.Chunk.hash == chroma_chunk_id)
        .first()
    )
    if row:
        return row.chunk_id, row.source_uri
    return None, None


def _resolve_chunk_id(db: Session, chroma_chunk_id: str):
    """Thin wrapper kept for backward compatibility (used in retrieval_hits loop)."""
    chunk_id, _ = _resolve_chunk_info(db, chroma_chunk_id)
    return chunk_id


def _load_conversation_history(db: Session, conversation_id: int, limit: int = 10):
    """
    Load recent conversation messages for follow-up detection.
    Returns list of dicts with 'role' and 'content' keys,
    compatible with the RAG pipeline's history format.
    """
    messages = (
        db.query(models.ChatMessage)
        .filter(models.ChatMessage.conversation_id == conversation_id)
        .order_by(models.ChatMessage.created_at.desc())
        .limit(limit)
        .all()
    )
    # Reverse to chronological order
    messages = list(reversed(messages))
    return [
        {"role": msg.role, "content": msg.content}
        for msg in messages
    ]


@router.post("/chat", response_model=schemas.ChatResponse)
def chat_endpoint(
    req: schemas.ChatRequest,
    db: Session = Depends(get_db),
    current_user_id: str = Depends(get_current_user_id),  # JWT'den gelir
):
    """
    Tam Traceability RAG Akışı:
    1. Kullanıcı/Conversation Kontrolü
    2. Session ve Query Event Oluşturma
    3. Konuşma geçmişini yükleme (follow-up için)
    4. RAG Servisini Çağırma
    5. Retrieval Hit'lerini Kaydetme
    6. Answer Citations Kaydetme
    7. Cevabı (Answer) ve Mesajları (ChatMessage) Kaydetme
    """

    # ---------------------------------------------------------
    # A. HAZIRLIK: Kullanıcı ve Sohbeti Bul/Oluştur
    # ---------------------------------------------------------

    # user_id her zaman JWT token'dan gelir — request body'deki değer yok sayılır
    user_id = current_user_id

    # Conversation ID yoksa yeni sohbet başlat
    conversation_id = req.conversation_id
    if not conversation_id:
        new_conv = models.Conversation(
            user_id=user_id,
            title=req.query[:50] + ("..." if len(req.query) > 50 else "")
        )
        db.add(new_conv)
        db.commit()
        db.refresh(new_conv)
        conversation_id = new_conv.conversation_id
    else:
        # Güvenlik: Bu sohbet bu kullanıcıya mı ait?
        conv = db.query(models.Conversation).filter(
            models.Conversation.conversation_id == conversation_id,
            models.Conversation.user_id == user_id
        ).first()
        if not conv:
            raise HTTPException(status_code=403, detail="Bu sohbete erişim yetkiniz yok")

    # ---------------------------------------------------------
    # B. TRACEABILITY BAŞLANGICI: Session & Query Event
    # ---------------------------------------------------------

    query_session = models.QuerySession(
        user_id=user_id,
        conversation_id=conversation_id,
        client_meta={"source": "web-ui"}
    )
    db.add(query_session)
    db.commit()
    db.refresh(query_session)

    query_event = models.QueryEvent(
        session_id=query_session.session_id,
        query_text=req.query
    )
    db.add(query_event)
    db.commit()
    db.refresh(query_event)

    # Kullanıcı mesajını kaydet
    user_msg = models.ChatMessage(
        conversation_id=conversation_id,
        user_id=user_id,
        role="user",
        content=req.query,
        query_id=query_event.query_id
    )
    db.add(user_msg)
    db.commit()

    # ---------------------------------------------------------
    # C. KONUŞMA GEÇMİŞİ: Follow-up (parent-child) desteği
    # ---------------------------------------------------------

    history = _load_conversation_history(db, conversation_id, limit=10)

    # ---------------------------------------------------------
    # D. RAG MOTORU ÇALIŞIYOR 🧠
    # ---------------------------------------------------------

    answer_text, raw_sources, pipeline_result = rag_engine.query(req.query, history=history)

    # Extract pipeline data
    retrieved_chunks = pipeline_result.get("retrieved_chunks", [])
    context_chunks = pipeline_result.get("context_chunks", [])
    analysis = pipeline_result.get("analysis", {})

    # Update session language from analysis
    detected_lang = analysis.get("language", "tr")
    if detected_lang:
        query_session.lang = detected_lang
        db.commit()

    # ---------------------------------------------------------
    # E. RETRIEVAL HITS KAYDETME
    # ---------------------------------------------------------

    saved_hit_count = 0
    for rank_idx, chunk_data in enumerate(retrieved_chunks):
        chroma_cid = chunk_data.get("chunk_id", "")
        db_chunk_id = _resolve_chunk_id(db, chroma_cid)

        if db_chunk_id is None:
            # Chunk not yet synced to Supabase — skip this hit
            continue

        try:
            hit = models.RetrievalHit(
                query_id=query_event.query_id,
                chunk_id=db_chunk_id,
                rank=rank_idx + 1,
                score_dense=chunk_data.get("vec_score") or chunk_data.get("rrf_vec"),
                score_lex=chunk_data.get("bm25_score") or chunk_data.get("rrf_bm25"),
                score_graph=chunk_data.get("ce_score"),
            )
            db.add(hit)
            saved_hit_count += 1
        except Exception as e:
            print(f"[chat] Warning: Could not save retrieval_hit rank={rank_idx+1}: {e}")
            db.rollback()

    if saved_hit_count > 0:
        try:
            db.commit()
            print(f"[chat] Saved {saved_hit_count} retrieval_hits for query_id={query_event.query_id}")
        except Exception as e:
            print(f"[chat] Warning: retrieval_hits commit failed: {e}")
            db.rollback()

    # ---------------------------------------------------------
    # F. CEVAP (ANSWER) KAYDETME
    # ---------------------------------------------------------

    db_answer = models.Answer(
        query_id=query_event.query_id,
        text=answer_text,
        model_name="llama3"
    )
    db.add(db_answer)
    db.commit()
    db.refresh(db_answer)

    # ---------------------------------------------------------
    # G. ANSWER CITATIONS KAYDETME
    # ---------------------------------------------------------

    saved_citation_count = 0
    seen_chunk_ids = set()  # For unique constraint: uq_cite_once

    for order_idx, chunk_data in enumerate(context_chunks):
        chroma_cid = chunk_data.get("chunk_id", "")
        db_chunk_id = _resolve_chunk_id(db, chroma_cid)

        if db_chunk_id is None:
            # Chunk not yet synced to Supabase — skip this citation
            continue

        if db_chunk_id in seen_chunk_ids:
            # Skip duplicate chunk citations (unique constraint)
            continue
        seen_chunk_ids.add(db_chunk_id)

        try:
            citation = models.AnswerCitation(
                answer_id=db_answer.answer_id,
                chunk_id=db_chunk_id,
                order_idx=order_idx,
            )
            db.add(citation)
            saved_citation_count += 1
        except Exception as e:
            print(f"[chat] Warning: Could not save citation order_idx={order_idx}: {e}")
            db.rollback()

    if saved_citation_count > 0:
        try:
            db.commit()
            print(f"[chat] Saved {saved_citation_count} answer_citations for answer_id={db_answer.answer_id}")
        except Exception as e:
            print(f"[chat] Warning: answer_citations commit failed: {e}")
            db.rollback()

    # ---------------------------------------------------------
    # H. ASSISTANT MESAJI KAYDETME
    # ---------------------------------------------------------

    ai_msg = models.ChatMessage(
        conversation_id=conversation_id,
        role="assistant",
        content=answer_text,
        query_id=query_event.query_id,
        answer_id=db_answer.answer_id
    )
    db.add(ai_msg)
    db.commit()
    db.refresh(ai_msg)

    # ---------------------------------------------------------
    # I. FRONTEND'E CEVAP DÖNME
    # ---------------------------------------------------------

    formatted_sources = []
    for src in raw_sources[:3]:  # Limit to top 3 sources
        # Single query: ChromaDB hash → Supabase chunk_id + document URL
        chroma_cid = src.get("chunk_id", "")
        resolved_id, doc_url = _resolve_chunk_info(db, chroma_cid)
        resolved_id = resolved_id or 0

        formatted_sources.append(
            schemas.SourceReference(
                chunk_id=resolved_id,
                title=src.get("title", "Doc"),
                excerpt=src.get("excerpt", ""),
                score=None,  # Not shown to user
                url=doc_url,  # Full URL from documents.source_uri
            )
        )

    return schemas.ChatResponse(
        answer=answer_text,
        sources=formatted_sources,
        conversation_id=conversation_id,
        query_id=query_event.query_id,
        message_id=ai_msg.message_id,
        confidence=0.95
    )


def _get_message_sources(db: Session, answer_id: int) -> List[schemas.SourceReference]:
    """
    Fetch sources (citations) for a given answer_id.
    Loads citations with their chunks and documents in one joined query.
    Limits to top 3 sources. Score is not stored in DB so left as None.
    """
    if not answer_id:
        return []

    citations = (
        db.query(models.AnswerCitation)
        .options(
            joinedload(models.AnswerCitation.chunk).joinedload(models.Chunk.document)
        )
        .filter(models.AnswerCitation.answer_id == answer_id)
        .order_by(models.AnswerCitation.order_idx.asc())
        .limit(3)
        .all()
    )

    sources = []
    for citation in citations:
        chunk = citation.chunk
        if not chunk:
            continue

        doc = chunk.document
        doc_url = doc.source_uri if doc else None
        # Document title has priority over section header ("Introduction" etc.)
        title = (doc.title if doc else None) or chunk.section or "Document"

        sources.append(
            schemas.SourceReference(
                chunk_id=chunk.chunk_id,
                title=title,
                excerpt=(chunk.content[:150] + "...") if chunk.content else "",
                score=None,  # Not stored in DB; score badge hidden when None
                url=doc_url,
            )
        )

    return sources


@router.get("/chat/history", response_model=List[schemas.ConversationOut])
def get_chat_history(
    db: Session = Depends(get_db),
    current_user_id: str = Depends(get_current_user_id),  # JWT'den gelir
):
    """
    Giriş yapmış kullanıcının tüm sohbet geçmişini döner.
    Frontend ConversationData interface'i ile birebir uyumlu format.
    Includes sources (citations) and confidence scores for assistant messages.
    """
    conversations = (
        db.query(models.Conversation)
        .options(joinedload(models.Conversation.messages))
        .filter(
            models.Conversation.user_id == current_user_id,
            models.Conversation.archived == False,
        )
        .order_by(models.Conversation.created_at.desc())
        .all()
    )

    result = []
    for conv in conversations:
        # Mesajları zamana göre sırala
        sorted_messages = sorted(conv.messages, key=lambda m: m.created_at)

        # Son mesajı preview olarak kullan
        last_msg = sorted_messages[-1] if sorted_messages else None
        preview = (last_msg.content[:80] + "...") if last_msg and len(last_msg.content) > 80 else (last_msg.content if last_msg else "")

        # Konuşmanın timestamp'i = son mesajın zamanı veya oluşturulma zamanı
        conv_timestamp = (last_msg.created_at if last_msg else conv.created_at).isoformat()

        messages_out = []
        for msg in sorted_messages:
            # Fetch sources for assistant messages
            sources = None
            confidence = None
            if msg.role == "assistant" and msg.answer_id:
                sources = _get_message_sources(db, msg.answer_id)
                confidence = 0.95  # Default; could also be stored per answer if needed

            messages_out.append(
                schemas.MessageOut(
                    id=msg.message_id,
                    role=msg.role,
                    content=msg.content,
                    timestamp=msg.created_at.isoformat(),
                    sources=sources,
                    confidence=confidence,
                )
            )

        result.append(
            schemas.ConversationOut(
                id=conv.conversation_id,
                title=conv.title or "Yeni Sohbet",
                timestamp=conv_timestamp,
                preview=preview,
                messages=messages_out,
            )
        )

    return result
