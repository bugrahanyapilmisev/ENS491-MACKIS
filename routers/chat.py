from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session, joinedload
from typing import List

from database import get_db
import models
import schemas
from services.rag_service import rag_engine
from utils import get_current_user_id

router = APIRouter()


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
    3. RAG Servisini Çağırma
    4. Retrieval Hit'lerini Kaydetme
    5. Cevabı (Answer) ve Mesajları (ChatMessage) Kaydetme
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
    # C. RAG MOTORU ÇALIŞIYOR 🧠
    # ---------------------------------------------------------

    answer_text, raw_sources = rag_engine.query(req.query)

    # ---------------------------------------------------------
    # D. SONUÇLARI KAYDETME
    # ---------------------------------------------------------

    db_answer = models.Answer(
        query_id=query_event.query_id,
        text=answer_text,
        model_name="llama3"
    )
    db.add(db_answer)
    db.commit()
    db.refresh(db_answer)

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
    # E. FRONTEND'E CEVAP DÖNME
    # ---------------------------------------------------------

    formatted_sources = []
    for src in raw_sources:
        formatted_sources.append(
            schemas.SourceReference(
                chunk_id=0,
                title=src.get("title", "Doc"),
                excerpt=src.get("excerpt", ""),
                score=0.9
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


@router.get("/chat/history", response_model=List[schemas.ConversationOut])
def get_chat_history(
    db: Session = Depends(get_db),
    current_user_id: str = Depends(get_current_user_id),  # JWT'den gelir
):
    """
    Giriş yapmış kullanıcının tüm sohbet geçmişini döner.
    Frontend ConversationData interface'i ile birebir uyumlu format.
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

        messages_out = [
            schemas.MessageOut(
                id=msg.message_id,
                role=msg.role,
                content=msg.content,
                timestamp=msg.created_at.isoformat(),
            )
            for msg in sorted_messages
        ]

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

