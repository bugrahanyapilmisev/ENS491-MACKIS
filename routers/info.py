# routers/info.py
"""
Info endpoints for the MACKIS frontend.

GET /api/suggestions  → 6 curated sample questions in Turkish (static)
GET /api/stats        → Real knowledge-base statistics from the database
"""

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from sqlalchemy import func

from database import get_db
import models

router = APIRouter(prefix="/api", tags=["info"])


# ---------------------------------------------------------------------------
# Static suggestion questions (Sabancı-specific, bilingual)
# ---------------------------------------------------------------------------
SUGGESTIONS = [
    {
        "question": "Erasmus değişim programı için minimum GNO kaç olmalı?",
        "hint": "Uluslararası değişim programı şartları",
        "emoji": "✈️",
    },
    {
        "question": "Kütüphaneden kaç kitap ödünç alabilirim ve iade süresi ne kadar?",
        "hint": "Kütüphane hizmetleri ve kuralları",
        "emoji": "📚",
    },
    {
        "question": "Uyarı cezası hangi durumlarda verilir?",
        "hint": "Öğrenci disiplin yönetmeliği",
        "emoji": "⚖️",
    },
    {
        "question": "Burs başvurusu için son başvuru tarihi ne zaman?",
        "hint": "Burs ve mali yardım bilgileri",
        "emoji": "🎓",
    },
    {
        "question": "Çift anadal programına nasıl başvurabilirim?",
        "hint": "Akademik program seçenekleri",
        "emoji": "📝",
    },
    {
        "question": "Mezuniyet için gereken minimum kredi sayısı nedir?",
        "hint": "Lisans mezuniyet gereksinimleri",
        "emoji": "🏛️",
    },
]


@router.get("/suggestions")
def get_suggestions():
    """Return 6 curated sample questions for the welcome screen."""
    return {"suggestions": SUGGESTIONS}


# ---------------------------------------------------------------------------
# Real knowledge-base statistics
# ---------------------------------------------------------------------------

@router.get("/stats")
def get_stats(db: Session = Depends(get_db)):
    """
    Return real knowledge-base counts from the database.

    Returns:
        chunk_count    – total indexed document chunks
        document_count – total unique source documents
        kg_topic_count – total KG topic nodes
        kg_edge_count  – total KG edges (triples)
    """
    try:
        chunk_count = db.query(func.count(models.Chunk.chunk_id)).scalar() or 0
        document_count = db.query(func.count(models.Document.document_id)).scalar() or 0
        kg_topic_count = (
            db.query(func.count(models.KGNode.node_id))
            .filter(models.KGNode.label == "topic")
            .scalar()
            or 0
        )
        # Fall back to total KG nodes when topic-label count is empty
        if kg_topic_count == 0:
            kg_topic_count = db.query(func.count(models.KGNode.node_id)).scalar() or 0

        kg_edge_count = db.query(func.count(models.KGEdge.edge_id)).scalar() or 0

        return {
            "document_count": document_count,
            "topic_count": kg_topic_count,
        }
    except Exception as exc:
        # Graceful degradation: return zeros on DB error rather than 500
        return {
            "document_count": 0,
            "topic_count": 0,
            "error": str(exc),
        }
