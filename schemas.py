from pydantic import BaseModel, Field
from typing import List, Optional, Any
from datetime import datetime
from uuid import UUID
from enum import Enum

# -----------------------------------
# 0. Enums
# -----------------------------------
class ChatModel(str, Enum):
    LLAMA31       = "llama3.1:latest"
    QWEN25_7B     = "qwen2.5:7b"
    GPT_OSS_120B  = "gpt-oss:120b-cloud"

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
    conversation_id: Optional[int] = None
    user_id: Optional[UUID] = None
    session_id: Optional[int] = None

# -----------------------------------
# 3. Cevap Modelleri (Response)
# -----------------------------------
class ChatResponse(BaseModel):
    """Frontend'e dönecek nihai cevap"""
    answer: str
    sources: List[SourceReference] = []
    conversation_id: int
    query_id: int
    message_id: int
    confidence: float = 0.0

# -----------------------------------
# 4. Chat Geçmişi Modelleri (History)
# -----------------------------------
class MessageOut(BaseModel):
    """Tek bir mesaj - frontend Message interface'i ile birebir uyumlu"""
    id: int              # message_id → id
    role: str            # "user" | "assistant"
    content: str
    timestamp: str       # ISO string, frontend bunu görüntüler
    sources: Optional[List[SourceReference]] = None  # Kaynaklar (assistant mesajları için)
    confidence: Optional[float] = None  # Güven skoru (assistant mesajları için)

    class Config:
        from_attributes = True

class ConversationOut(BaseModel):
    """Tek bir konuşma - frontend ConversationData interface'i ile birebir uyumlu"""
    id: int              # conversation_id → id
    title: str
    timestamp: str       # ISO string (konuşmanın son güncelleme zamanı)
    preview: str         # Son mesajın kısa hali
    messages: List[MessageOut]

    class Config:
        from_attributes = True

