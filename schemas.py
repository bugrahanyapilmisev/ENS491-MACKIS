from pydantic import BaseModel, Field
from typing import List, Optional, Any
from datetime import datetime
from uuid import UUID

# -----------------------------------
# 1. Temel Parçalar
# -----------------------------------
class SourceReference(BaseModel):
    """Frontend'e gösterilecek kaynak bilgisi"""
    chunk_id: int
    title: str
    excerpt: str
    score: Optional[float] = None
    url: Optional[str] = None

# -----------------------------------
# 2. İstek Modelleri (Request)
# -----------------------------------
class ChatRequest(BaseModel):
    """Kullanıcıdan gelen mesaj formatı"""
    query: str
    conversation_id: Optional[int] = None # Eğer devam eden bir sohbetse ID gelir
    user_id: Optional[UUID] = None        # Login olmuş kullanıcı ise UUID gelir
    session_id: Optional[int] = None      # Opsiyonel traceability için

# -----------------------------------
# 3. Cevap Modelleri (Response)
# -----------------------------------
class ChatResponse(BaseModel):
    """Frontend'e dönecek nihai cevap"""
    answer: str
    sources: List[SourceReference] = []
    conversation_id: int          # Yeni başladıysa oluşan ID'yi döneriz
    query_id: int                 # Traceability için sorgu ID'si
    message_id: int               # Oluşan asistan mesajının ID'si
    confidence: float = 0.0