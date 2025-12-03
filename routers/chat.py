from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from uuid import uuid4

# Kendi dosyalarımızdan importlar
from database import get_db
import models
import schemas
from services.rag_service import rag_engine # RAG motorunu buradan çağıracağız

router = APIRouter()

@router.post("/chat", response_model=schemas.ChatResponse)
def chat_endpoint(req: schemas.ChatRequest, db: Session = Depends(get_db)):
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
    
    # 1. Kullanıcı yoksa geçici bir 'Guest' kullanıcı bul veya oluştur
    # (Gerçek auth eklenince burası token'dan gelecek)
    user_id = req.user_id
    if not user_id:
        # Demo amaçlı: Veritabanındaki ilk user'ı al veya oluştur
        user = db.query(models.User).first()
        if not user:
            user = models.User(email="guest@demo.com", display_name="Misafir", role="guest")
            db.add(user)
            db.commit()
            db.refresh(user)
        user_id = user.user_id

    # 2. Conversation ID yoksa yeni sohbet başlat
    conversation_id = req.conversation_id
    if not conversation_id:
        new_conv = models.Conversation(
            user_id=user_id,
            title=req.query[:30] + "..." # İlk mesajdan başlık uydur
        )
        db.add(new_conv)
        db.commit()
        db.refresh(new_conv)
        conversation_id = new_conv.conversation_id

    # ---------------------------------------------------------
    # B. TRACEABILITY BAŞLANGICI: Session & Query Event
    # ---------------------------------------------------------

    # 3. Query Session (Varsa kullan, yoksa oluştur)
    # (Basitlik için her sohbete yeni bir session açıyoruz şimdilik)
    query_session = models.QuerySession(
        user_id=user_id,
        conversation_id=conversation_id,
        client_meta={"source": "web-ui"}
    )
    db.add(query_session)
    db.commit()
    db.refresh(query_session)

    # 4. Query Event (Kullanıcının sorusunu ham olarak kaydet)
    query_event = models.QueryEvent(
        session_id=query_session.session_id,
        query_text=req.query
    )
    db.add(query_event)
    db.commit()
    db.refresh(query_event)

    # 5. Kullanıcı Mesajını ChatMessage tablosuna da ekle (Sohbet geçmişi için)
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
    
    # rag_service.py içindeki query fonksiyonunu çağırıyoruz
    # Dönüş formatı: (cevap_metni, kaynak_listesi_dict)
    answer_text, raw_sources = rag_engine.query(req.query)

    # ---------------------------------------------------------
    # D. SONUÇLARI KAYDETME (Hits & Answers)
    # ---------------------------------------------------------

    # 6. Retrieval Hits (Hangi chunklar bulundu?)
    # Not: Gerçek chunk_id'leri rag_service'den dönmeli.
    # Şimdilik demo modunda chunk kaydı yapamıyoruz çünkü DB boş.
    # Veri yüklendiğinde buraya 'RetrievalHit' kayıt döngüsü eklenecek.

    # 7. Answer (Yapay zekanın cevabını kaydet)
    db_answer = models.Answer(
        query_id=query_event.query_id,
        text=answer_text,
        model_name="llama3" # Kullandığın model
    )
    db.add(db_answer)
    db.commit()
    db.refresh(db_answer)

    # 8. Asistan Mesajını ChatMessage tablosuna ekle
    ai_msg = models.ChatMessage(
        conversation_id=conversation_id,
        # user_id -> AI olduğu için NULL bırakıyoruz veya özel bir AI user olabilir
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
    
    # Kaynakları şemaya uygun hale getir
    formatted_sources = []
    for src in raw_sources:
        formatted_sources.append(
            schemas.SourceReference(
                chunk_id=0, # Şimdilik dummy
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