# MACKIS RAG Pipeline - Mimari Dokümantasyonu

**Proje:** ENS491 Bitirme Projesi - MACKIS (Sabancı Üniversitesi Akıllı Asistan)
**Tarih:** Şubat 2026
**Versiyon:** 2.0

---

## 1. Genel Bakış

Bu dokümantasyon, MACKIS RAG (Retrieval-Augmented Generation) sisteminin yeniden yapılandırılmış mimarisini detaylı olarak açıklamaktadır.

### 1.1 Önceki Durum (v1)
- Tek dosyada 1502 satır kod (`services/rag_core.py`)
- Global state kullanımı (cache, index, model)
- Sıkı bağımlılıklar ve test zorluğu
- Karışık sorumluluklar

### 1.2 Yeni Mimari (v2)
- **5 Core Service**: Stateless utility sınıfları
- **4 Agent Class**: Yüksek seviye orkestratörler
- **1 Configuration Module**: Merkezi yapılandırma
- **1 Pipeline Orchestrator**: Geriye dönük uyumlu API

---

## 2. Dizin Yapısı

```
services/
├── __init__.py
├── config/
│   ├── __init__.py
│   └── settings.py              # RAGConfig dataclass
├── core/
│   ├── __init__.py
│   ├── embedding_service.py     # Ollama embedding + cache
│   ├── llm_service.py           # Ollama chat API
│   ├── vector_store.py          # ChromaDB işlemleri
│   ├── bm25_service.py          # BM25 index işlemleri
│   └── data_loader.py           # Parquet/pickle yükleme
├── agents/
│   ├── __init__.py
│   ├── query_analysis_agent.py  # Sorgu analizi
│   ├── retrieval_agent.py       # Hibrit arama
│   ├── ranking_agent.py         # Cross-encoder reranking
│   └── generation_agent.py      # Cevap üretimi
├── pipeline/
│   ├── __init__.py
│   └── rag_pipeline.py          # Ana orkestratör
├── rag_core.py                  # Geriye uyumluluk katmanı
├── rag_service.py               # FastAPI entegrasyonu
└── kg_service.py                # Knowledge Graph servisi
```

---

## 3. Configuration Module

### 3.1 settings.py Yapısı

```python
@dataclass
class OllamaConfig:
    host: str = "http://localhost:11434"
    embed_model: str = "bge-m3"
    embed_dim: int = 1024
    chat_model: str = "llama3.1:latest"
    timeout: int = 120
    chat_timeout: int = 300

@dataclass
class ChromaConfig:
    chroma_dir: str = ""
    collection_name: str = "mysu_v2_bge_m3"

@dataclass
class RetrievalConfig:
    top_k_chroma: int = 64
    top_k_bm25: int = 32
    top_k_final_base: int = 8
    top_k_final_max: int = 24
    bm25_weight: float = 0.3
    vector_weight: float = 0.7
    rrf_constant: int = 60

@dataclass
class RerankingConfig:
    model_name: str = "BAAI/bge-reranker-v2-m3"
    max_candidates: int = 24
    batch_size: int = 4
    max_text_length: int = 1500  # OOM önleme
    weight: float = 0.6
    score_threshold: float = 0.65

@dataclass
class KGConfig:
    kg_output_dir: str = ""
    kg_facts_llm_validated: str = ""
    kg_triples_llm_validated: str = ""
    active_source: str = "llm"

@dataclass
class RAGConfig:
    ollama: OllamaConfig
    chroma: ChromaConfig
    retrieval: RetrievalConfig
    reranking: RerankingConfig
    features: FeatureFlags
    paths: PathConfig
    kg: KGConfig

    @classmethod
    def from_env(cls) -> "RAGConfig":
        """Environment variables'dan yapılandırma yükle"""
```

### 3.2 Environment Variables

| Variable | Default | Açıklama |
|----------|---------|----------|
| `OLLAMA_HOST` | localhost:11434 | Ollama API adresi |
| `CHAT_MODEL` | qwen2.5:7b | LLM modeli |
| `EMBED_MODEL` | bge-m3 | Embedding modeli |
| `CROSS_MAX_CANDIDATES` | 24 | Reranking için max aday |
| `RERANKER_BATCH_SIZE` | 4 | Batch boyutu (OOM için) |
| `RERANKER_MAX_TEXT_LENGTH` | 1500 | Max metin uzunluğu |
| `KG_ACTIVE_SOURCE` | llm | KG kaynağı (llm/pattern/raw) |

---

## 4. Core Services

### 4.1 EmbeddingService
**Dosya:** `services/core/embedding_service.py`

```python
class EmbeddingService:
    def embed(self, text: str) -> np.ndarray:
        """Metin için embedding üret (SHA1 cache ile)"""

    def cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """İki vektör arası benzerlik"""

    def clear_cache(self):
        """Cache'i temizle"""
```

**Özellikler:**
- SHA1 tabanlı önbellekleme
- Exponential backoff ile retry
- L2 normalizasyon

### 4.2 LLMService
**Dosya:** `services/core/llm_service.py`

```python
class LLMService:
    def chat(self, prompt: str, system_prompt: str = None) -> str:
        """Ollama chat API çağrısı"""

    def chat_json(self, prompt: str, system_prompt: str = None) -> dict:
        """JSON formatında yanıt al"""
```

### 4.3 VectorStoreService
**Dosya:** `services/core/vector_store.py`

```python
class VectorStoreService:
    def get_collection(self) -> Collection:
        """ChromaDB collection'ı al"""

    def search(self, query_embedding: np.ndarray, top_k: int) -> List[Dict]:
        """Vektör araması"""

    def fetch_embeddings(self, ids: List[str]) -> Dict[str, np.ndarray]:
        """ID'lere göre embedding'leri getir"""
```

### 4.4 BM25Service
**Dosya:** `services/core/bm25_service.py`

```python
class BM25Service:
    def load_index(self) -> bool:
        """Pickle index yükle"""

    def tokenize(self, text: str) -> List[str]:
        """Türkçe diacritic desteği ile tokenize"""

    def search(self, query: str, top_k: int) -> List[Tuple[str, float]]:
        """BM25 araması"""
```

### 4.5 DataLoaderService
**Dosya:** `services/core/data_loader.py`

```python
class DataLoaderService:
    def load_chunk_df(self) -> pd.DataFrame:
        """Chunk parquet dosyasını yükle"""

    def load_doc_summaries(self) -> Dict[str, str]:
        """Doküman özetlerini yükle"""

    def get_all_chunks_for_doc(self, source_path: str) -> List[Dict]:
        """Bir dokümanın tüm chunk'larını getir"""
```

---

## 5. Agent Classes

### 5.1 QueryAnalysisAgent
**Dosya:** `services/agents/query_analysis_agent.py`

**Sorumluluk:** Sorgu analizi ve anlamlandırma

```python
class QueryAnalysisAgent:
    def analyze(self, query: str, history: List = None) -> Dict:
        """
        Returns:
            {
                "language": "tr" | "en" | None,
                "intent": "list_names" | "count_items" | "describe" | "other",
                "is_followup": bool,
                "anchor_query": Optional[str],
                "tags": List[str],
                "negated_terms": List[str]
            }
        """
```

**Alt metodlar:**
- `detect_language()`: langdetect + diacritic fallback
- `detect_intent()`: Keyword pattern matching
- `detect_followup()`: Semantic similarity kontrolü
- `infer_tags()`: LLM tabanlı tag çıkarımı
- `extract_negated_terms()`: Negasyon tespiti

### 5.2 RetrievalAgent
**Dosya:** `services/agents/retrieval_agent.py`

**Sorumluluk:** Hibrit arama ve sorgu genişletme

```python
class RetrievalAgent:
    def retrieve(self, query: str, language: str = None) -> List[Dict]:
        """Ana retrieval pipeline"""

    def hybrid_search(self, query: str, language: str) -> List[Dict]:
        """Vector + BM25 ile RRF fusion"""

    def expand_query(self, query: str, lang: str) -> List[str]:
        """LLM ile sorgu genişletme (2-3 alternatif)"""

    def generate_hypothetical_document(self, query: str) -> str:
        """HyDE: Hipotetik doküman üretimi"""

    def multi_query_retrieval(self, queries: List[str]) -> List[Dict]:
        """Çoklu sorgu sonuçlarını birleştir"""
```

**RRF (Reciprocal Rank Fusion) Formülü:**
```
score(d) = Σ 1/(k + rank_i(d))
```
Burada k = 60 (default)

### 5.3 RankingAgent
**Dosya:** `services/agents/ranking_agent.py`

**Sorumluluk:** Reranking, filtreleme ve seçim

```python
class RankingAgent:
    def rerank(self, query: str, candidates: List[Dict],
               query_tags: List[str] = None,
               negated_terms: List[str] = None) -> List[Dict]:
        """
        Pipeline:
        1. Cross-encoder reranking (batched)
        2. Tag-based boosting
        3. Negation penalty
        """

    def cross_encoder_rerank(self, query: str, candidates: List[Dict]) -> List[Dict]:
        """
        - Metin truncation (max 1500 char)
        - Batch processing (4 aday/batch)
        - Hybrid score combination
        """

    def mmr_select(self, candidates: List[Dict], query: str, k: int) -> List[Dict]:
        """MMR ile çeşitlilik seçimi"""
```

**Bellek Optimizasyonu:**
```python
# Metin kısaltma
def _truncate_text(self, text: str) -> str:
    max_chars = self.config.reranking.max_text_length  # 1500
    if len(text) <= max_chars:
        return text
    truncated = text[:max_chars]
    last_space = truncated.rfind(' ')
    return truncated[:last_space] + "..."

# Batch processing
for i in range(0, len(pairs), batch_size):  # batch_size = 4
    batch_pairs = pairs[i:i + batch_size]
    batch_scores = model.predict(batch_pairs)
```

### 5.4 GenerationAgent
**Dosya:** `services/agents/generation_agent.py`

**Sorumluluk:** Context oluşturma ve cevap üretimi

```python
class GenerationAgent:
    def generate(self, query: str, chunks: List[Dict],
                 language: str = None, kg_facts: List[str] = None) -> str:
        """
        1. Context oluştur
        2. System prompt seç (TR/EN)
        3. LLM çağrısı
        4. Sayı doğrulama
        """

    def build_context(self, chunks: List[Dict]) -> str:
        """Chunk'ları formatlı context'e dönüştür"""

    def verify_answer_numbers(self, answer: str, context: str) -> bool:
        """Hallucination tespiti (sayı kontrolü)"""
```

---

## 6. Pipeline Orchestrator

### 6.1 RAGPipeline
**Dosya:** `services/pipeline/rag_pipeline.py`

```python
class RAGPipeline:
    def __init__(self, config: RAGConfig = None):
        # Services
        self.embedding = EmbeddingService(config)
        self.llm = LLMService(config)
        self.vector_store = VectorStoreService(config)
        self.bm25 = BM25Service(config)
        self.data_loader = DataLoaderService(config)

        # Agents
        self.query_agent = QueryAnalysisAgent(self.embedding, self.llm, config)
        self.retrieval_agent = RetrievalAgent(...)
        self.ranking_agent = RankingAgent(...)
        self.generation_agent = GenerationAgent(...)

    def answer(self, query: str, history: List = None) -> str:
        """Ana pipeline akışı"""
```

### 6.2 Pipeline Akışı

```
┌─────────────────────────────────────────────────────────────┐
│                     USER QUERY                               │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│              1. QUERY ANALYSIS AGENT                         │
│  ├─ Dil tespiti (TR/EN)                                     │
│  ├─ Intent sınıflandırma                                    │
│  ├─ Followup kontrolü                                       │
│  ├─ Tag çıkarımı                                            │
│  └─ Negasyon tespiti                                        │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│              2. RETRIEVAL AGENT                              │
│  ├─ Query expansion (LLM)                                   │
│  ├─ HyDE generation                                         │
│  ├─ Multi-query hybrid search                               │
│  │   ├─ ChromaDB vector search (top_k=64)                   │
│  │   └─ BM25 lexical search (top_k=32)                      │
│  └─ RRF fusion                                              │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│              3. RANKING AGENT                                │
│  ├─ Cross-encoder reranking (batched, truncated)            │
│  ├─ Tag-based boosting                                      │
│  ├─ Negation penalty                                        │
│  └─ Threshold filtering                                     │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│           4. INTENT-AWARE SELECTION                          │
│  ├─ list/count intent → Get all chunks from top doc         │
│  └─ other intent → MMR diversity selection                  │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│           5. KG AUGMENTATION (Optional)                      │
│  └─ Query knowledge graph for related facts                 │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│              6. GENERATION AGENT                             │
│  ├─ Build context from chunks                               │
│  ├─ Generate answer (LLM)                                   │
│  └─ Verify numbers (hallucination check)                    │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│                      ANSWER                                  │
└─────────────────────────────────────────────────────────────┘
```

---

## 7. Supabase Entegrasyonu

### 7.1 Veritabanı Şeması

```sql
-- Documents
CREATE TABLE documents (
    document_id SERIAL PRIMARY KEY,
    source_type VARCHAR CHECK (source_type IN ('url','pdf','html','md','email','other')),
    source_uri VARCHAR,
    title VARCHAR,
    lang VARCHAR DEFAULT 'tr',
    department VARCHAR,
    hash VARCHAR UNIQUE,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Chunks
CREATE TABLE chunks (
    chunk_id SERIAL PRIMARY KEY,
    document_id INTEGER REFERENCES documents(document_id) ON DELETE CASCADE,
    ordinal INTEGER NOT NULL,
    section VARCHAR,
    page_num INTEGER,
    content TEXT NOT NULL,
    tokens INTEGER,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Chunk Embeddings (pgvector)
CREATE TABLE chunk_embeddings (
    chunk_id INTEGER PRIMARY KEY REFERENCES chunks(chunk_id) ON DELETE CASCADE,
    embedding VECTOR(1024) NOT NULL,
    model VARCHAR,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Knowledge Graph Nodes
CREATE TABLE kg_nodes (
    node_id SERIAL PRIMARY KEY,
    label VARCHAR NOT NULL,
    name VARCHAR NOT NULL,
    doc_id INTEGER REFERENCES documents(document_id),
    data JSONB
);

-- Knowledge Graph Edges
CREATE TABLE kg_edges (
    edge_id SERIAL PRIMARY KEY,
    src INTEGER REFERENCES kg_nodes(node_id) ON DELETE CASCADE,
    dst INTEGER REFERENCES kg_nodes(node_id) ON DELETE CASCADE,
    type VARCHAR NOT NULL,
    weight FLOAT DEFAULT 1.0,
    UNIQUE(src, dst, type)
);
```

### 7.2 Sync Script
**Dosya:** `scripts/sync/sync_to_supabase.py`

```bash
# Tüm verileri senkronize et
python scripts/sync/sync_to_supabase.py --all

# Sadece belirli tabloları
python scripts/sync/sync_to_supabase.py --documents
python scripts/sync/sync_to_supabase.py --chunks
python scripts/sync/sync_to_supabase.py --embeddings
python scripts/sync/sync_to_supabase.py --kg

# Dry-run (değişiklik yapmadan)
python scripts/sync/sync_to_supabase.py --dry-run
```

### 7.3 Senkronize Edilen Veriler

| Tablo | Kayıt Sayısı | Açıklama |
|-------|--------------|----------|
| documents | 1,873 | Benzersiz dokümanlar |
| chunks | 12,592 | Metin parçaları |
| chunk_embeddings | 12,592 | 1024-dim vektörler (pgvector) |
| kg_nodes | 3,973 | 1,709 topic + 2,264 entity |
| kg_edges | 1,883 | İlişki bağlantıları |

---

## 8. Bellek Optimizasyonları

### 8.1 Cross-Encoder OOM Sorunu

**Problem:** BAAI/bge-reranker-v2-m3 modeli uzun metinlerle çalışırken 9-12 GiB bellek talep ediyordu.

**Çözümler:**

1. **Metin Truncation:**
```python
max_text_length: int = 1500  # karaktere kısalt
```

2. **Batch Processing:**
```python
batch_size: int = 4  # 4'lü gruplar halinde işle
```

3. **Candidate Limiting:**
```python
max_candidates: int = 24  # max 24 aday rerank et
```

### 8.2 Yapılandırma ile Ayarlama

```bash
# .env dosyasında
RERANKER_BATCH_SIZE=2       # Daha az bellek
CROSS_MAX_CANDIDATES=16     # Daha az aday
RERANKER_MAX_TEXT_LENGTH=1000  # Daha kısa metin
```

---

## 9. Knowledge Graph Entegrasyonu

### 9.1 KG Dosya Yapısı

```
creating_kg/knowledge_graph/
├── llm_validated/           # LLM doğrulamalı (en yüksek kalite)
│   ├── kg_facts_llm_validated.json
│   ├── kg_triples_llm_validated.json
│   └── kg_index_llm_validated.pkl
├── pattern_validated/       # Pattern doğrulamalı
│   ├── kg_facts_validated.json
│   └── kg_triples_validated.json
└── raw/                     # Ham çıktılar
    ├── kg_facts.json
    └── kg_triples.json
```

### 9.2 KG Facts Formatı

```json
{
  "Erasmus": [
    {
      "relation": "requirement",
      "value": "minimum 2.50 GNO",
      "context_type": "academic",
      "source_chunk_id": "abc123...",
      "source_title": "Erasmus Yönergesi",
      "confidence": 0.95
    }
  ]
}
```

### 9.3 KG Triples Formatı

```json
{
  "triples": [
    {
      "head": "Erasmus Programı",
      "head_type": "program",
      "relation": "requires",
      "tail": "2.50 GNO",
      "tail_type": "requirement",
      "confidence": 0.9
    }
  ]
}
```

---

## 10. API Endpoints

### 10.1 Chat Endpoint
**Route:** `POST /chat`

```python
class ChatRequest(BaseModel):
    query: str
    conversation_id: Optional[int] = None
    user_id: Optional[UUID] = None
    session_id: Optional[int] = None

class ChatResponse(BaseModel):
    answer: str
    sources: List[SourceReference]
    conversation_id: int
    query_id: int
    message_id: int
    confidence: float
```

### 10.2 Auth Endpoint
**Route:** `POST /auth/login`

```python
class LoginRequest(BaseModel):
    email: str
    password: str

# Response
{
    "access_token": "eyJ...",
    "token_type": "bearer",
    "user_name": "Ahmet",
    "is_admin": false
}
```

---

## 11. Test Etme

### 11.1 RAG Pipeline Testi

```bash
cd testing
python test_rag_detailed.py
```

### 11.2 API Testi

```bash
# Server başlat
uvicorn main:app --reload

# Test isteği
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"query": "Erasmus için minimum GNO nedir?"}'
```

---

## 12. Sonuç

### 12.1 Kazanımlar

| Metrik | Önce | Sonra |
|--------|------|-------|
| Kod satırı (rag_core) | 1,502 | ~50 (wrapper) |
| Dosya başına satır | 1,502 | 50-200 |
| Test edilebilirlik | Düşük | Yüksek |
| Bağımlılık injection | Yok | Var |
| Yapılandırılabilirlik | Hardcoded | Environment |

### 12.2 Mimari Faydaları

1. **Modülerlik:** Her servis/agent bağımsız test edilebilir
2. **Bakım kolaylığı:** Net sorumluluk ayrımı
3. **Genişletilebilirlik:** Yeni agent/servis ekleme kolaylığı
4. **Yapılandırma:** Injectable configuration ile esneklik
5. **Debug kolaylığı:** Net sınırlar ile sorun tespiti

---


**Proje Sahibi:** Ahmet Çalışkan Murat Berke Türkan Buğrahan Yapılmışev
**Kurum:** Sabancı Üniversitesi
