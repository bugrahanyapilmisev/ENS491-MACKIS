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
    model: Optional[ChatModel] = ChatModel.LLAMA31  # null veya eksik olursa llama3.1 kullanılır
    conversation_id: Optional[int] = None            # Eğer devam eden bir sohbetse ID gelir

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

