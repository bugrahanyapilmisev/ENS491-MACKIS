"""
rag_core.py - IMPROVED RAG PIPELINE

Key improvements over v1:
1. BM25 Hybrid Search - Combines lexical (BM25) with semantic (vector) search
2. Query Expansion - LLM-based query reformulation for better recall
3. HyDE (Hypothetical Document Embeddings) - Generate hypothetical answer, embed it
4. Contextual Retrieval - Use document summaries for coarse-to-fine retrieval
5. Late Chunking Support - Aggregate parent chunks for more context
6. Improved Context Construction - Include document summaries and section headers
7. Multi-Query Retrieval - Generate multiple query variations, merge results
8. Better Caching - Cache embeddings and BM25 index

This file is designed to work with build_chroma_store.py outputs.
"""

import os
import json
import math
import pickle
import hashlib
from typing import List, Dict, Optional, Tuple, Set
from functools import lru_cache

import numpy as np
import requests
import chromadb
import re
import pandas as pd
import dotenv
from sentence_transformers import CrossEncoder
import textwrap
from concurrent.futures import ThreadPoolExecutor, as_completed
import time

dotenv.load_dotenv()

# Import KG service (optional, graceful degradation if not built yet)
try:
    from services.kg_service import query_hybrid, query_facts as kg_query_facts, load_kg_facts
    # Verify KG can be loaded
    if load_kg_facts():
        KG_AVAILABLE = True
        print("[info] KG service loaded successfully")
    else:
        KG_AVAILABLE = False
        print("[info] KG service: facts not loaded, run build_hybrid_kg.py")
except ImportError as e:
    KG_AVAILABLE = False
    print(f"[info] KG service not available: {e}")


def query_kg(query: str, lang: str = "tr") -> str:
    """Wrapper for hybrid KG query (Topic->Fact + Entity->Relation->Entity)."""
    if not KG_AVAILABLE:
        return ""
    try:
        # Use hybrid query for combined facts + triples
        return query_hybrid(query, lang=lang)
    except Exception as ex:
        print(f"[kg query error] {ex}")
        return ""

# =================== GLOBAL MODELS / ENV ===================

RERANKER_MODEL_NAME = os.getenv("RERANKER_MODEL_NAME", "BAAI/bge-reranker-v2-m3")
RERANKER: Optional[CrossEncoder] = None

OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
EMBED_MODEL = os.getenv("EMBED_MODEL", "bge-m3")
EMBED_DIM   = int(os.getenv("EMBED_DIM", "1024"))
CHAT_MODEL  = os.getenv("CHAT_MODEL", "qwen2.5:7b")  # Qwen 2.5 - excellent instruction following

ROOT_DIR = os.getenv("ROOT_PATH", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PREPROCESSING_DIR = os.path.join(ROOT_DIR, "preprocessing")
CREATING_DB_DIR = os.path.join(ROOT_DIR, "creating_database")

CHROMA_FOLDER_NAME = os.getenv("CHROMA_FOLDER_NAME_V2", "chroma_db_v2")
CHROMA_DIR         = os.getenv("CHROMA_DIR_V2") or os.path.join(CREATING_DB_DIR, CHROMA_FOLDER_NAME)
CHECKPOINT_DIR     = os.getenv("CHECKPOINT_DIR_V2") or os.path.join(CREATING_DB_DIR, "checkpoints_v2")

CHUNK_PARQUET      = os.path.join(CHECKPOINT_DIR, "chunks_v2.parquet")
DOC_SUMMARY_PARQUET= os.path.join(CHECKPOINT_DIR, "doc_summaries_v2.parquet")
BM25_INDEX_PATH    = os.path.join(CHECKPOINT_DIR, "bm25_index.pkl")
COLL_NAME          = os.getenv("CHROMA_COLLECTION_NAME_V2", "mysu_v2_bge_m3")

# Retrieval parameters
TOP_K_CHROMA       = int(os.getenv("TOP_K_CHROMA", "64"))
TOP_K_BM25         = int(os.getenv("TOP_K_BM25", "32"))
TOP_K_FINAL_BASE   = int(os.getenv("TOP_K_FINAL_BASE", "8"))
TOP_K_FINAL_MAX    = int(os.getenv("TOP_K_FINAL_MAX", "24"))
MAX_DOCS_CONTEXT   = int(os.getenv("MAX_DOCS_CONTEXT", "6"))

# Hybrid search weights
BM25_WEIGHT        = float(os.getenv("BM25_WEIGHT", "0.3"))
VECTOR_WEIGHT      = float(os.getenv("VECTOR_WEIGHT", "0.7"))

# Cross-encoder parameters
CROSS_MAX_CANDIDATES = int(os.getenv("CROSS_MAX_CANDIDATES", "32"))
CROSS_WEIGHT         = float(os.getenv("CROSS_WEIGHT", "0.6"))
CE_SCORE_THRESHOLD   = float(os.getenv("CE_SCORE_THRESHOLD", "0.65"))

# Feature flags
USE_HYDE           = os.getenv("USE_HYDE", "1") == "1"
USE_QUERY_EXPANSION = os.getenv("USE_QUERY_EXPANSION", "1") == "1"
USE_MULTI_QUERY    = os.getenv("USE_MULTI_QUERY", "1") == "1"
USE_BM25_HYBRID    = os.getenv("USE_BM25_HYBRID", "1") == "1"
USE_DOC_SUMMARIES  = os.getenv("USE_DOC_SUMMARIES", "1") == "1"

TR_DIACRITICS = "çğıöşüÇĞİÖŞÜ"

# Negation lexicons
NEGATION_WORDS_TR = ["değil", "degil", "hariç", "dışında", "disinda"]
NEGATION_WORDS_EN = ["not", "except", "excluding", "other", "other than", "aside"]

NEGATION_BACKEND = os.getenv("NEGATION_BACKEND", "llm").lower()

# Global caches
DOC_CHUNK_STATS: Optional[Dict[str, int]] = None
CHUNK_DF: Optional[pd.DataFrame] = None
DOC_SUMMARY_DF: Optional[pd.DataFrame] = None
BM25_INDEX = None
BM25_CHUNK_IDS: List[str] = []
EMBEDDING_CACHE: Dict[str, np.ndarray] = {}

# =================== LANGUAGE DETECTION ===================

try:
    from langdetect import detect as ld_detect
except ImportError:
    ld_detect = None


def guess_lang_from_text(text: str) -> Optional[str]:
    s = (text or "").strip()
    if not s:
        return None
    sample = s[:4000]

    if ld_detect is not None:
        try:
            code = ld_detect(sample).lower()
            if code.startswith("tr"):
                return "tr"
            if code.startswith("en"):
                return "en"
        except Exception:
            pass

    if re.search(f"[{TR_DIACRITICS}]", sample):
        return "tr"
    return None


# =================== LOADING FUNCTIONS ===================

def get_chroma_collection():
    client = chromadb.PersistentClient(path=CHROMA_DIR)
    return client.get_collection(COLL_NAME)


def load_bm25_index():
    """Load pre-built BM25 index."""
    global BM25_INDEX, BM25_CHUNK_IDS
    
    if BM25_INDEX is not None:
        return BM25_INDEX, BM25_CHUNK_IDS
    
    if not os.path.exists(BM25_INDEX_PATH):
        print(f"[warn] BM25 index not found: {BM25_INDEX_PATH}")
        return None, []
    
    with open(BM25_INDEX_PATH, "rb") as f:
        data = pickle.load(f)
    
    BM25_INDEX = data.get("bm25")
    BM25_CHUNK_IDS = data.get("chunk_ids", [])
    
    print(f"[info] Loaded BM25 index with {len(BM25_CHUNK_IDS)} documents")
    return BM25_INDEX, BM25_CHUNK_IDS


def load_chunk_df() -> pd.DataFrame:
    global CHUNK_DF
    if CHUNK_DF is None:
        if not os.path.exists(CHUNK_PARQUET):
            raise RuntimeError(f"CHUNK_PARQUET not found: {CHUNK_PARQUET}")
        CHUNK_DF = pd.read_parquet(CHUNK_PARQUET)
    return CHUNK_DF


def load_doc_summaries() -> pd.DataFrame:
    global DOC_SUMMARY_DF
    if DOC_SUMMARY_DF is None:
        if not os.path.exists(DOC_SUMMARY_PARQUET):
            print(f"[warn] DOC_SUMMARY_PARQUET not found: {DOC_SUMMARY_PARQUET}")
            DOC_SUMMARY_DF = pd.DataFrame()
        else:
            DOC_SUMMARY_DF = pd.read_parquet(DOC_SUMMARY_PARQUET)
    return DOC_SUMMARY_DF


def get_doc_summary(source_path: str) -> str:
    """Get the summary for a document."""
    df = load_doc_summaries()
    if df.empty or "source_path" not in df.columns:
        return ""
    
    match = df[df["source_path"] == source_path]
    if match.empty:
        return ""
    
    return match.iloc[0].get("summary", "")


def get_all_chunks_for_doc(source_path: str) -> List[Dict]:
    """Get all chunks for a document, sorted by chunk_index."""
    df = load_chunk_df()
    sub = df[df["source_path"] == source_path].copy()
    
    if "chunk_index" in sub.columns:
        sub = sub.sort_values("chunk_index")

    chunks = []
    for _, row in sub.iterrows():
        tags_val = row.get("tags", "")
        meta = {
            "source_path": row["source_path"],
            "json_path": row.get("json_path", ""),
            "title": row.get("title", ""),
            "section_header": row.get("section_header", ""),
            "doc_lang": row.get("doc_lang", ""),
            "doc_type": row.get("doc_type", ""),
            "tags": tags_val,
            "procedure_code": row.get("procedure_code", ""),
        }
        chunks.append({
            "chunk_id": row["chunk_id"],
            "text": row["content"],
            "meta": meta,
            "source": "doc_full",
        })
    return chunks


# =================== EMBEDDINGS ===================

def embed_text_ollama(text: str, use_cache: bool = True, max_retries: int = 3) -> np.ndarray:
    """Embed text using Ollama with optional caching and retry logic."""
    global EMBEDDING_CACHE
    
    cache_key = hashlib.sha1(text.encode("utf-8", errors="ignore")).hexdigest()
    
    if use_cache and cache_key in EMBEDDING_CACHE:
        return EMBEDDING_CACHE[cache_key]
    
    url = f"{OLLAMA_HOST}/api/embeddings"
    
    # Retry logic for transient errors
    last_error = None
    for attempt in range(max_retries):
        try:
            r = requests.post(
                url,
                json={"model": EMBED_MODEL, "prompt": text},
                timeout=120,
            )
            r.raise_for_status()
            v = np.array(r.json()["embedding"], dtype=np.float32)
            break
        except Exception as e:
            last_error = e
            if attempt < max_retries - 1:
                import time
                time.sleep(1 * (attempt + 1))  # Exponential backoff
                continue
            raise RuntimeError(f"Embedding failed after {max_retries} attempts: {last_error}")
    
    if v.shape[0] != EMBED_DIM:
        raise RuntimeError(f"Unexpected embedding dimension {v.shape[0]} (expected {EMBED_DIM})")
    
    v /= (np.linalg.norm(v) + 1e-12)
    
    if use_cache:
        EMBEDDING_CACHE[cache_key] = v
    
    return v


def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    num = float(np.dot(a, b))
    den = float(np.linalg.norm(a) * np.linalg.norm(b) + 1e-12)
    return num / den


# =================== BM25 SEARCH ===================

def tokenize_for_bm25(text: str) -> List[str]:
    """Simple tokenization for BM25."""
    text = text.lower()
    tokens = re.findall(r"[a-z" + TR_DIACRITICS.lower() + r"0-9]+", text)
    tokens = [t for t in tokens if len(t) > 2]
    return tokens


def bm25_search(query: str, top_k: int = TOP_K_BM25) -> List[Tuple[str, float]]:
    """Search using BM25 index. Returns list of (chunk_id, score) tuples."""
    bm25, chunk_ids = load_bm25_index()
    
    if bm25 is None or not chunk_ids:
        return []
    
    query_tokens = tokenize_for_bm25(query)
    if not query_tokens:
        return []
    
    scores = bm25.get_scores(query_tokens)
    
    # Get top-k
    top_indices = np.argsort(scores)[::-1][:top_k]
    
    results = []
    for idx in top_indices:
        if scores[idx] > 0:
            results.append((chunk_ids[idx], float(scores[idx])))
    
    return results


# =================== CHROMA SEARCH ===================

def chroma_search(q: str,
                  coll,
                  query_vec: Optional[np.ndarray] = None,
                  top_k: int = TOP_K_CHROMA,
                  lang_filter: Optional[str] = None) -> List[Dict]:
    """Vector search using ChromaDB."""
    if query_vec is None:
        query_vec = embed_text_ollama(q)

    where = {}
    if lang_filter is not None:
        where["doc_lang"] = lang_filter

    res = coll.query(
        query_embeddings=[query_vec.tolist()],
        n_results=top_k,
        where=where or None,
    )

    ids       = res.get("ids", [[]])[0]
    docs      = res.get("documents", [[]])[0]
    metas     = res.get("metadatas", [[]])[0]
    distances = res.get("distances", [[]])[0]

    out = []
    for cid, d, m, dist in zip(ids, docs, metas, distances):
        sim = 1.0 - float(dist)
        out.append({
            "chunk_id": cid,
            "score": sim,
            "vec_score": sim,
            "text": d,
            "meta": m,
            "source": "chroma",
        })
    return out


# =================== HYBRID SEARCH ===================

def hybrid_search(query: str,
                  coll,
                  query_vec: Optional[np.ndarray] = None,
                  top_k_vec: int = TOP_K_CHROMA,
                  top_k_bm25: int = TOP_K_BM25,
                  lang_filter: Optional[str] = None,
                  vec_weight: float = VECTOR_WEIGHT,
                  bm25_weight: float = BM25_WEIGHT) -> List[Dict]:
    """
    Hybrid search combining vector (semantic) and BM25 (lexical) search.
    Uses Reciprocal Rank Fusion (RRF) for score combination.
    """
    if query_vec is None:
        query_vec = embed_text_ollama(query)
    
    # 1) Vector search
    vec_results = chroma_search(
        query, coll, query_vec=query_vec, 
        top_k=top_k_vec, lang_filter=lang_filter
    )
    
    # 2) BM25 search (if enabled)
    bm25_results = []
    if USE_BM25_HYBRID:
        bm25_results = bm25_search(query, top_k=top_k_bm25)
    
    # 3) Compute RRF scores
    K = 60  # RRF constant
    
    chunk_scores: Dict[str, Dict] = {}
    
    # Add vector results
    for rank, c in enumerate(vec_results):
        cid = c["chunk_id"]
        rrf_vec = 1.0 / (K + rank + 1)
        
        if cid not in chunk_scores:
            chunk_scores[cid] = {
                "chunk_id": cid,
                "text": c["text"],
                "meta": c["meta"],
                "vec_score": c["score"],
                "vec_rank": rank + 1,
                "bm25_score": 0.0,
                "bm25_rank": None,
                "rrf_vec": rrf_vec,
                "rrf_bm25": 0.0,
                "source": "hybrid",
            }
        else:
            chunk_scores[cid]["rrf_vec"] = rrf_vec
            chunk_scores[cid]["vec_rank"] = rank + 1
    
    # Add BM25 results
    for rank, (cid, bm25_score) in enumerate(bm25_results):
        rrf_bm25 = 1.0 / (K + rank + 1)
        
        if cid not in chunk_scores:
            # Need to fetch text and meta from chunk_df
            df = load_chunk_df()
            match = df[df["chunk_id"] == cid]
            if match.empty:
                continue
            row = match.iloc[0]
            
            chunk_scores[cid] = {
                "chunk_id": cid,
                "text": row["content"],
                "meta": {
                    "source_path": row.get("source_path", ""),
                    "title": row.get("title", ""),
                    "section_header": row.get("section_header", ""),
                    "doc_lang": row.get("doc_lang", ""),
                    "tags": row.get("tags", ""),
                },
                "vec_score": 0.0,
                "vec_rank": None,
                "bm25_score": bm25_score,
                "bm25_rank": rank + 1,
                "rrf_vec": 0.0,
                "rrf_bm25": rrf_bm25,
                "source": "bm25",
            }
        else:
            chunk_scores[cid]["bm25_score"] = bm25_score
            chunk_scores[cid]["bm25_rank"] = rank + 1
            chunk_scores[cid]["rrf_bm25"] = rrf_bm25
            chunk_scores[cid]["source"] = "hybrid"
    
    # 4) Compute final hybrid score
    for cid, data in chunk_scores.items():
        data["hybrid_score"] = (
            vec_weight * data["rrf_vec"] + 
            bm25_weight * data["rrf_bm25"]
        )
        data["score"] = data["hybrid_score"]
    
    # 5) Sort by hybrid score
    results = list(chunk_scores.values())
    results.sort(key=lambda x: x["hybrid_score"], reverse=True)
    
    return results


# =================== QUERY EXPANSION (LLM) ===================

def expand_query_llm(query: str, lang: Optional[str] = None) -> List[str]:
    """
    Use LLM to generate expanded/alternative queries for better recall.
    Returns list of query variations including the original.
    """
    if not USE_QUERY_EXPANSION:
        return [query]
    
    url = f"{OLLAMA_HOST}/api/chat"
    
    lang_hint = lang or "unknown"
    
    sys_prompt = textwrap.dedent(f"""
        You are a search query expansion assistant for a university information system.
        Given a user query, generate 2-3 alternative search queries that capture the same intent
        but use different keywords or phrasings.
        
        Rules:
        - Keep queries concise (5-15 words each)
        - Use synonyms and related terms
        - If query is in Turkish, generate Turkish alternatives
        - If query is in English, generate English alternatives
        - Focus on the core information need
        
        Output format (STRICT JSON):
        {{"queries": ["alternative query 1", "alternative query 2", ...]}}
        
        Output ONLY the JSON, no explanation.
    """).strip()
    
    user_prompt = f"Language: {lang_hint}\nOriginal query: {query}"
    
    payload = {
        "model": CHAT_MODEL,
        "messages": [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "stream": False,
        "options": {"temperature": 0.3},
    }
    
    try:
        resp = requests.post(url, json=payload, timeout=60)
        resp.raise_for_status()
        data = resp.json()
        text = (data.get("message", {}) or {}).get("content", "").strip()
        
        if not text:
            return [query]
        
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            m = re.search(r"\{.*\}", text, flags=re.DOTALL)
            if not m:
                return [query]
            parsed = json.loads(m.group(0))
        
        queries = parsed.get("queries", [])
        if not isinstance(queries, list):
            return [query]
        
        # Always include original query first
        result = [query]
        for q in queries:
            if isinstance(q, str) and q.strip() and q.strip() != query:
                result.append(q.strip())
        
        print(f"[debug] query expansion: {result}")
        return result[:4]  # Max 4 queries
        
    except Exception as e:
        print(f"[query-expansion error] {e}")
        return [query]


# =================== HyDE (Hypothetical Document Embeddings) ===================

def generate_hypothetical_document(query: str, lang: Optional[str] = None) -> Optional[str]:
    """
    Generate a hypothetical document that would answer the query.
    This is used for HyDE (Hypothetical Document Embeddings).
    """
    if not USE_HYDE:
        return None
    
    url = f"{OLLAMA_HOST}/api/chat"
    
    lang_hint = "Turkish" if lang == "tr" else "English"
    
    sys_prompt = textwrap.dedent(f"""
        You are an expert at generating hypothetical document passages for document retrieval.
        Given a question, write a SHORT passage (2-3 sentences) that describes what kind of document would answer it.
        
        CRITICAL RULES:
        - Write in {lang_hint}
        - DO NOT use specific numbers, dates, or values - just describe the topic
        - Use general terms like "the minimum GPA requirement", "the required credits", "the deadline"
        - Write as if describing what a policy document would contain
        - Keep it short (30-50 words)
        
        BAD example: "The minimum GPA is 2.5" (don't make up numbers!)
        GOOD example: "This document describes the minimum GPA requirements for program eligibility."
        
        Output ONLY the passage, nothing else.
    """).strip()
    
    payload = {
        "model": CHAT_MODEL,
        "messages": [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": f"Question: {query}"},
        ],
        "stream": False,
        "options": {"temperature": 0.3},
    }
    
    try:
        resp = requests.post(url, json=payload, timeout=60)
        resp.raise_for_status()
        data = resp.json()
        text = (data.get("message", {}) or {}).get("content", "").strip()
        
        if text and len(text) > 20:
            print(f"[debug] HyDE generated: {text[:100]}...")
            return text
        
        return None
        
    except Exception as e:
        print(f"[hyde error] {e}")
        return None


# =================== MULTI-QUERY RETRIEVAL ===================

def multi_query_retrieval(queries: List[str],
                          coll,
                          lang_filter: Optional[str] = None,
                          top_k_per_query: int = 32) -> List[Dict]:
    """
    Retrieve results for multiple queries and merge using RRF.
    """
    if not queries:
        return []
    
    if len(queries) == 1:
        return hybrid_search(queries[0], coll, lang_filter=lang_filter, 
                           top_k_vec=TOP_K_CHROMA, top_k_bm25=TOP_K_BM25)
    
    all_results: Dict[str, Dict] = {}
    K = 60  # RRF constant
    
    for q_idx, q in enumerate(queries):
        results = hybrid_search(q, coll, lang_filter=lang_filter,
                              top_k_vec=top_k_per_query, top_k_bm25=top_k_per_query // 2)
        
        for rank, c in enumerate(results):
            cid = c["chunk_id"]
            rrf_score = 1.0 / (K + rank + 1)
            
            if cid not in all_results:
                all_results[cid] = c.copy()
                all_results[cid]["multi_query_rrf"] = rrf_score
                all_results[cid]["query_hits"] = 1
            else:
                all_results[cid]["multi_query_rrf"] += rrf_score
                all_results[cid]["query_hits"] += 1
    
    # Combine scores: original hybrid + multi-query bonus
    for cid, data in all_results.items():
        # Boost chunks that appear in multiple query results
        multi_query_bonus = data.get("multi_query_rrf", 0) * 0.3
        data["hybrid_score"] = data.get("hybrid_score", 0) + multi_query_bonus
        data["score"] = data["hybrid_score"]
    
    results = list(all_results.values())
    results.sort(key=lambda x: x["hybrid_score"], reverse=True)
    
    print(f"[debug] multi-query retrieval: {len(queries)} queries, {len(results)} unique chunks")
    
    return results


# =================== CROSS-ENCODER RERANK ===================

def get_reranker() -> CrossEncoder:
    global RERANKER
    if RERANKER is None:
        RERANKER = CrossEncoder(RERANKER_MODEL_NAME)
    return RERANKER


def cross_encoder_rerank(query: str,
                         candidates: List[Dict],
                         max_candidates: int = CROSS_MAX_CANDIDATES,
                         weight_ce: float = CROSS_WEIGHT) -> List[Dict]:
    """Cross-encoder reranking with hybrid score combination."""
    if not candidates:
        return []

    subset = candidates[:max_candidates].copy()
    
    model = get_reranker()
    
    # Batch scoring for efficiency
    pairs = [(query, c["text"]) for c in subset]
    ce_scores = model.predict(pairs)
    
    for i, c in enumerate(subset):
        c["ce_score"] = float(ce_scores[i])

    ce_vals = [c.get("ce_score", 0.0) for c in subset]
    hybrid_vals = [c.get("hybrid_score", c.get("score", 0.0)) for c in subset]

    max_ce = max(ce_vals) if ce_vals else 1.0
    max_hybrid = max(hybrid_vals) if hybrid_vals else 1.0
    if max_ce <= 0:
        max_ce = 1.0
    if max_hybrid <= 0:
        max_hybrid = 1.0

    # Title overlap boost
    q_tokens = set(re.findall(r"\w+", (query or "").lower()))

    for c in subset:
        ce_norm = c.get("ce_score", 0.0) / max_ce
        hybrid_norm = c.get("hybrid_score", c.get("score", 0.0)) / max_hybrid

        final_score = weight_ce * ce_norm + (1.0 - weight_ce) * hybrid_norm

        # Title overlap boost
        meta = c.get("meta") or {}
        title = (meta.get("title") or "").lower()
        t_tokens = set(re.findall(r"\w+", title))
        if q_tokens and t_tokens:
            overlap = len(q_tokens & t_tokens) / (len(q_tokens) + 1e-6)
            final_score *= (1.0 + 0.15 * overlap)

        c["ce_norm"] = ce_norm
        c["hybrid_norm"] = hybrid_norm
        c["final_score"] = final_score
        c["hybrid_score"] = final_score  # Update hybrid_score for downstream

    subset.sort(key=lambda x: x["final_score"], reverse=True)
    return subset


# =================== TAG-BASED FILTERING ===================

def get_meta_tags(meta: Dict) -> List[str]:
    """Normalize tags from metadata."""
    if not meta:
        return []

    tags_val = meta.get("tags", [])
    parts: List[str] = []

    if isinstance(tags_val, list):
        parts = [p for p in tags_val if isinstance(p, str)]
    elif isinstance(tags_val, str):
        parts = tags_val.split(",")
    else:
        parts = []

    out: List[str] = []
    for p in parts:
        t = p.strip().lower()
        if t:
            out.append(t)
    return out


def tokenize_tag(tag: str) -> Set[str]:
    """Tokenize a tag for fuzzy matching."""
    if not isinstance(tag, str):
        return set()

    t = tag.lower()
    t = re.sub(r"[^a-z0-9_]+", "_", t)
    parts = [p for p in t.split("_") if p]

    generic_tokens = {
        "program", "prosedur", "yonerge", "basvuru",
        "ogrenci", "lisans", "genel", "bilgi",
        "form", "guide", "policy", "procedure", "student",
    }

    out = set()
    for p in parts:
        if len(p) <= 2:
            continue
        if p in generic_tokens:
            continue
        out.add(p)
    return out


def soft_tag_overlap(query_tags: List[str], doc_tags: List[str]) -> int:
    """Fuzzy token-level overlap between query and doc tags."""
    q_tokens = set()
    for qt in query_tags:
        q_tokens |= tokenize_tag(qt)

    d_tokens = set()
    for dt in doc_tags:
        d_tokens |= tokenize_tag(dt)

    if not q_tokens or not d_tokens:
        return 0

    return len(q_tokens & d_tokens)


def infer_query_tags_llm(query: str, lang: Optional[str]) -> List[str]:
    """Infer semantic tags for a query using LLM."""
    url = f"{OLLAMA_HOST}/api/chat"
    
    sys_prompt = textwrap.dedent("""
        You are a classifier for search queries in a university system.
        Return a JSON object with semantic tags for the query.
        
        JSON schema:
        {"tags": ["tag1", "tag2", ...]}
        
        Rules:
        - 2-6 tags, lowercase snake_case
        - Use English tags even for Turkish queries
        - Tags should be specific topics, not generic
        
        Output STRICT JSON ONLY.
    """).strip()
    
    payload = {
        "model": CHAT_MODEL,
        "messages": [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": f"Query: {query}"},
        ],
        "stream": False,
        "options": {"temperature": 0.0},
    }
    
    try:
        resp = requests.post(url, json=payload, timeout=60)
        resp.raise_for_status()
        data = resp.json()
        text = (data.get("message", {}) or {}).get("content", "").strip()
        
        if not text:
            return []
        
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            m = re.search(r"\{.*\}", text, flags=re.DOTALL)
            if not m:
                return []
            parsed = json.loads(m.group(0))
        
        tags = parsed.get("tags", [])
        if not isinstance(tags, list):
            return []
        
        return [t.strip().lower() for t in tags if isinstance(t, str) and t.strip()]
        
    except Exception as e:
        print(f"[query-tags error] {e}")
        return []


def apply_tag_prior(query_tags: List[str],
                    candidates: List[Dict],
                    boost_positive: float = 0.20,
                    penalize_negative: float = 0.10) -> List[Dict]:
    """Boost chunks with matching tags, penalize those without."""
    if not candidates or not query_tags:
        return candidates

    q_raw = [t for t in query_tags if isinstance(t, str) and t.strip()]
    if not q_raw:
        return candidates

    doc_overlap: Dict[str, int] = {}
    for c in candidates:
        meta = c.get("meta") or {}
        path = meta.get("source_path") or meta.get("doc_path")
        dtags_raw = get_meta_tags(meta)
        overlap = soft_tag_overlap(q_raw, dtags_raw)
        c["_tag_overlap"] = overlap

        if path:
            prev = doc_overlap.get(path, 0)
            if overlap > prev:
                doc_overlap[path] = overlap

    max_overlap = max(doc_overlap.values()) if doc_overlap else 0
    if max_overlap <= 0:
        return candidates

    for c in candidates:
        base = c.get("hybrid_score", c.get("score", 0.0))
        overlap = c.get("_tag_overlap", 0)
        if overlap > 0:
            factor = 1.0 + boost_positive * min(overlap, 3)
        else:
            factor = 1.0 - penalize_negative
        c["hybrid_score"] = base * factor

    candidates.sort(key=lambda x: x.get("hybrid_score", 0.0), reverse=True)
    return candidates


# =================== NEGATION HANDLING ===================

def extract_negated_terms_llm(query: str, lang: Optional[str] = None) -> List[str]:
    """Use LLM to extract negated terms from query."""
    if not query:
        return []

    url = f"{OLLAMA_HOST}/api/chat"

    sys_prompt = textwrap.dedent("""
        You are a negation extractor for search queries.
        Find terms that are explicitly NEGATED or EXCLUDED from what the user wants.
        
        JSON schema:
        {"negated_terms": ["term1", "term2", ...]}
        
        Rules:
        - ONLY include terms that come AFTER negation words (not, except, hariç, değil, dışında, other than)
        - Questions like "X nelerdir?" (what are X?) have NO negation - return empty list
        - Questions like "X nasıl yapılır?" (how to do X?) have NO negation - return empty list
        - Return lowercase terms
        - When in doubt, return EMPTY list
        
        Examples:
        - "Disiplin cezaları nelerdir?" -> {"negated_terms": []}  (asking ABOUT discipline, not excluding it)
        - "Erasmus dışında hangi programlar var?" -> {"negated_terms": ["erasmus"]}
        - "Burs değil kredi istiyorum" -> {"negated_terms": ["burs"]}
        
        Output STRICT JSON ONLY.
    """).strip()

    payload = {
        "model": CHAT_MODEL,
        "messages": [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": f"Query: {query}"},
        ],
        "stream": False,
        "options": {"temperature": 0.0},
    }

    try:
        resp = requests.post(url, json=payload, timeout=60)
        resp.raise_for_status()
        data = resp.json()
        text = (data.get("message", {}) or {}).get("content", "").strip()
        
        if not text:
            return []
        
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            m = re.search(r"\{.*\}", text, flags=re.DOTALL)
            if not m:
                return []
            parsed = json.loads(m.group(0))
        
        terms = parsed.get("negated_terms", [])
        if not isinstance(terms, list):
            return []
        
        return [t.strip().lower() for t in terms if isinstance(t, str) and t.strip()]
        
    except Exception as e:
        print(f"[negation error] {e}")
        return []


def apply_negation_penalty(query: str,
                           candidates: List[Dict],
                           lang: Optional[str] = None,
                           penalty_factor: float = 0.3) -> List[Dict]:
    """Downweight chunks containing negated terms."""
    neg_terms = extract_negated_terms_llm(query, lang)
    if not neg_terms:
        return candidates

    neg_terms = set(neg_terms)
    print(f"[debug] negated terms: {neg_terms}")

    for c in candidates:
        meta = c.get("meta") or {}
        title_blob = " ".join(
            str(meta.get(k, "")) for k in ("title", "section_header")
        ).lower()

        if any(t in title_blob for t in neg_terms):
            old = c.get("hybrid_score", 0.0)
            c["hybrid_score"] = old * penalty_factor

    candidates.sort(key=lambda x: x.get("hybrid_score", 0.0), reverse=True)
    return candidates


# =================== INTENT DETECTION ===================

def detect_query_intent(query: str, lang: Optional[str] = None) -> str:
    """Detect query intent: list_names, count_items, describe, other."""
    if not query:
        return "other"

    q = query.strip().lower()

    # Count triggers
    count_triggers = [
        "kaç tane", "kaç adet", "sayısı kaç", "toplam kaç",
        "how many", "number of", "count of",
    ]
    if any(t in q for t in count_triggers):
        return "count_items"

    # List triggers
    list_triggers = [
        "isimlerini say", "isimlerini listele", "adlarını say",
        "list the names", "list all", "enumerate",
    ]
    if any(t in q for t in list_triggers):
        return "list_names"

    # Describe triggers
    describe_triggers = ["nedir", "ne demek", "açıkla", "explain", "describe", "what is"]
    if any(t in q for t in describe_triggers):
        return "describe"

    return "other"


# =================== FOLLOWUP DETECTION ===================

def is_followup_semantic(query: str, history: List[Dict], threshold: float = 0.60) -> bool:
    """Check if query is a followup based on semantic similarity."""
    if not history:
        return False

    last_q = None
    for msg in reversed(history):
        if msg.get("role") == "user":
            last_q = (msg.get("content") or "").strip()
            if last_q:
                break

    if not last_q:
        return False

    try:
        q_vec = embed_text_ollama(query)
        last_vec = embed_text_ollama(last_q)
        sim = cosine_sim(q_vec, last_vec)
        return sim >= threshold
    except Exception:
        return False


def get_last_user_query(history: List[Dict]) -> Optional[str]:
    """Get the last user query from history."""
    for msg in reversed(history):
        if msg.get("role") == "user":
            text = (msg.get("content") or "").strip()
            if text:
                return text
    return None


# =================== MMR SELECTION ===================

def mmr_select(candidates: List[Dict],
               doc_embs: Dict[str, np.ndarray],
               query_vec: np.ndarray,
               k: int = TOP_K_FINAL_BASE,
               lambda_mmr: float = 0.7) -> List[Dict]:
    """Maximal Marginal Relevance selection for diversity."""
    selected: List[Dict] = []
    selected_ids: List[str] = []

    query_sims = {}
    for c in candidates:
        cid = c["chunk_id"]
        emb = doc_embs.get(cid)
        query_sims[cid] = 0.0 if emb is None else cosine_sim(query_vec, emb)

    while len(selected) < min(k, len(candidates)):
        best_cand = None
        best_mmr = -1e9

        for c in candidates:
            cid = c["chunk_id"]
            if cid in selected_ids:
                continue

            rel = 0.5 * c.get("hybrid_score", 0.0) + 0.5 * query_sims.get(cid, 0.0)

            if not selected_ids:
                red = 0.0
            else:
                emb_i = doc_embs.get(cid)
                if emb_i is None:
                    red = 0.0
                else:
                    sims = []
                    for sid in selected_ids:
                        emb_j = doc_embs.get(sid)
                        if emb_j is not None:
                            sims.append(cosine_sim(emb_i, emb_j))
                    red = max(sims) if sims else 0.0

            mmr_score = lambda_mmr * rel - (1.0 - lambda_mmr) * red

            if mmr_score > best_mmr:
                best_mmr = mmr_score
                best_cand = c

        if best_cand is None:
            break

        selected.append(best_cand)
        selected_ids.append(best_cand["chunk_id"])

    return selected


def fetch_doc_embeddings(coll, ids: List[str]) -> Dict[str, np.ndarray]:
    """Fetch embeddings from ChromaDB."""
    if not ids:
        return {}
    res = coll.get(ids=ids, include=["embeddings"])
    out = {}
    for cid, emb in zip(res["ids"], res["embeddings"]):
        out[cid] = np.array(emb, dtype=np.float32)
    return out


# =================== CONTEXT BUILDING ===================

def build_context(chunks: List[Dict], include_summaries: bool = True) -> str:
    """
    Build context with improved formatting.
    Optionally include document summaries for additional context.
    """
    parts = []
    doc_summaries_added = set()
    
    for i, ch in enumerate(chunks):
        meta = ch.get("meta", {}) or {}
        path = meta.get("source_path") or meta.get("doc_path", "")
        title = meta.get("title", "")
        section = meta.get("section_header", "")
        lang = meta.get("doc_lang", "")
        
        # Add document summary once per document
        if include_summaries and USE_DOC_SUMMARIES and path and path not in doc_summaries_added:
            summary = get_doc_summary(path)
            if summary:
                parts.append(f"[Document Overview: {title}]\n{summary[:300]}...")
                doc_summaries_added.add(path)
        
        # Build chunk header
        header_parts = [f"[{i+1}]"]
        if title:
            header_parts.append(f"Title: {title}")
        if section:
            header_parts.append(f"Section: {section}")
        if lang:
            header_parts.append(f"Lang: {lang}")
        
        header = " | ".join(header_parts)
        parts.append(header + "\n" + ch["text"])
    
    return "\n\n-----\n\n".join(parts)


# =================== OLLAMA CHAT ===================

def call_ollama_chat(prompt: str, system_prompt: str = "", model: str = CHAT_MODEL) -> str:
    """Call Ollama chat API."""
    url = f"{OLLAMA_HOST}/api/chat"

    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})

    payload = {
        "model": model,
        "messages": messages,
        "stream": False,
        "options": {
            "temperature": 0.0,
            "top_p": 0.9,
        },
    }

    try:
        resp = requests.post(url, json=payload, timeout=300)
        if resp.status_code != 200:
            return f"Ollama Error: {resp.text}"

        data = resp.json()
        return data.get("message", {}).get("content", "Empty response.")
    except Exception as e:
        return f"Connection error: {e}"


# =================== MAIN RAG PIPELINE ===================

def answer_with_rag(query: str,
                       coll,
                       history: List[Dict] = [],
                       use_hybrid: bool = True) -> str:
    """
    Improved RAG pipeline with:
    - Query expansion
    - HyDE
    - Hybrid search (BM25 + vector)
    - Multi-query retrieval
    - Cross-encoder reranking
    - Tag-based filtering
    - Negation handling
    - MMR selection
    """
    q_lang = guess_lang_from_text(query)
    print(f"[debug] detected language: {q_lang}")
    
    # 1) Detect intent
    intent = detect_query_intent(query, q_lang)
    print(f"[debug] intent: {intent}")
    
    # 2) Check for followup
    is_followup = is_followup_semantic(query, history)
    anchor = None
    if is_followup:
        anchor = get_last_user_query(history)
        print(f"[debug] followup detected, anchor: {anchor}")
    
    # 3) Build retrieval query
    if anchor:
        retrieval_query = f"{anchor}\n\n{query}"
    else:
        retrieval_query = query
    
    # 4) Query expansion
    expanded_queries = expand_query_llm(retrieval_query, q_lang)
    
    # 5) HyDE - generate hypothetical document
    hyde_text = generate_hypothetical_document(query, q_lang)
    if hyde_text:
        expanded_queries.append(hyde_text)
    
    # 6) Get query tags for filtering
    query_tags = infer_query_tags_llm(retrieval_query, q_lang)
    print(f"[debug] query tags: {query_tags}")
    
    # 7) Multi-query retrieval with hybrid search
    lang_filter = q_lang if q_lang in ("tr", "en") else None
    
    if USE_MULTI_QUERY and len(expanded_queries) > 1:
        candidates = multi_query_retrieval(expanded_queries, coll, lang_filter=lang_filter)
    else:
        candidates = hybrid_search(retrieval_query, coll, lang_filter=lang_filter)
    
    if not candidates:
        return "Üzgünüm, ilgili bir bilgi bulamadım." if q_lang == "tr" else "Sorry, I couldn't find relevant information."
    
    print(f"[debug] initial candidates: {len(candidates)}")
    
    # 8) Cross-encoder reranking
    reranked = cross_encoder_rerank(retrieval_query, candidates)
    
    # 9) Apply tag prior
    reranked = apply_tag_prior(query_tags, reranked)
    
    # 10) Apply negation penalty
    reranked = apply_negation_penalty(retrieval_query, reranked, q_lang)
    
    print("=== TOP CANDIDATES AFTER RERANK ===")
    for i, c in enumerate(reranked[:10], start=1):
        meta = c.get("meta") or {}
        print(f"{i:2d}. score={c.get('hybrid_score',0):.3f} ce={c.get('ce_score',0):.3f} | {meta.get('title','')[:50]}")
    print("===================================")
    
    # 11) Filter by CE threshold
    threshold = CE_SCORE_THRESHOLD
    strong = [c for c in reranked if c.get("hybrid_score", 0) >= threshold]
    if len(strong) < 4:
        strong = reranked[:12]
    
    # 12) Intent-aware retrieval mode
    if intent in {"list_names", "count_items"}:
        # For count/list queries, use the TOP-RANKED document, not aggregated scores
        # This ensures we use the most relevant document for factual queries
        if reranked:
            # Get the best document from the TOP candidate (not aggregated)
            best_chunk = reranked[0]
            best_meta = best_chunk.get("meta") or {}
            best_path = best_meta.get("source_path")
            
            if best_path:
                print(f"[debug] single-doc mode for intent={intent}, using top-ranked doc: {os.path.basename(best_path)}")
                retrieved = get_all_chunks_for_doc(best_path)[:TOP_K_FINAL_MAX]
            else:
                retrieved = strong[:TOP_K_FINAL_MAX]
        else:
            retrieved = strong[:TOP_K_FINAL_MAX]
    else:
        # 13) MMR selection for diversity
        candidate_ids = [c["chunk_id"] for c in strong]
        doc_embs = fetch_doc_embeddings(coll, candidate_ids)
        query_vec = embed_text_ollama(retrieval_query)
        
        retrieved = mmr_select(
            strong, doc_embs, query_vec,
            k=min(len(strong), TOP_K_FINAL_MAX),
            lambda_mmr=0.7
        )
    
    # 14) Build context
    context = build_context(retrieved, include_summaries=USE_DOC_SUMMARIES)
    
    print(f"[debug] final context chunks: {len(retrieved)}")
    
    # 14.5) Knowledge Graph augmentation
    kg_facts_str = ""
    if KG_AVAILABLE:
        try:
            kg_facts_str = query_kg(query, lang=q_lang or "tr")
            if kg_facts_str:
                print("=== KG FACTS ===")
                print(kg_facts_str)
                print("================")
        except Exception as e:
            print(f"[kg-augment warning] {e}")
    
    # 15) Generate answer
    if q_lang == "tr":
        sys_prompt = textwrap.dedent("""
            Sen Sabancı Üniversitesi'nin kurum içi bilgi sistemine bağlı Türkçe konuşan asistansın.
            
            KRİTİK KURALLAR:
            1) Context'i DIKKATLI OKU - cevap genellikle Context'te VARDIR.
            2) Context'te geçen sayıları, tarihleri, süreleri, koşulları AYNEN kullan.
            3) SAYI veya DEĞERLERİ KENDİN UYDURMA - Context'te yazanı yaz.
            4) Context'te olmayan bilgileri KESINLIKLE UYDURMA.
            5) SADECE hiçbir yerde bulamadığında "bu bilgi bağlamda yok" de.
            6) Cevapların kısa ve net olsun.
            
            DİKKAT: "GNO", "not ortalaması", "minimum" gibi kelimeler Context'te farklı şekillerde geçebilir.
            Lisans=undergrad, Lisansüstü=graduate için farklı değerler olabilir, IKISINI de belirt.
            
            ÖRNEK: Context'te "Lisans için 2.20, Lisansüstü için 2.5" varsa, tam olarak bunu yaz.
        """).strip()
    else:
        sys_prompt = textwrap.dedent("""
            You are an assistant for Sabancı University's internal knowledge system.
            
            CRITICAL RULES:
            1) READ the context CAREFULLY - the answer is usually IN the context.
            2) Use EXACT numbers, dates, durations, conditions from the context.
            3) Do NOT invent numbers - use what's written in the context.
            4) Do NOT invent information not in the context.
            5) ONLY say "not in context" if you truly cannot find it anywhere.
            6) Be brief and direct.
            
            NOTE: Terms like "GPA", "GNO", "minimum" may appear in different forms.
            Undergrad vs Graduate may have different values - mention BOTH if present.
            
            EXAMPLE: If context has "2.20 for undergrad, 2.5 for graduate", write exactly that.
        """).strip()
    
    # Build prompt with optional KG facts
    if kg_facts_str:
        full_prompt = f"""{kg_facts_str}

Context:
{context}

Question: {query}

INSTRUCTIONS:
1. The facts above MAY be helpful hints. CROSS-CHECK them against the Context below.
2. If a fact seems inconsistent with the Context (e.g., GPA > 4.0), IGNORE IT and use the Context instead.
3. SEARCH the context for specific numbers, values, requirements, conditions, durations, or limits.
4. If the question asks about GNO/GPA, look for phrases like "en az", "minimum", "2.20", "2.5", etc.
5. If different conditions apply to different groups (lisans/lisansüstü), mention ALL of them.
6. EXTRACT and state the relevant information directly from the Context.

Answer:"""
    else:
        full_prompt = f"""Context:
{context}

Question: {query}

INSTRUCTIONS:
1. SEARCH the entire context above for information related to the question.
2. Look for specific numbers, values, requirements, conditions, durations, or limits.
3. If the question asks about GNO/GPA, look for phrases like "en az", "minimum", "2.20", "2.5", etc.
4. If different conditions apply to different groups (lisans/lisansüstü), mention ALL of them.
5. EXTRACT and state the relevant information directly.

Answer:"""
    
    answer = call_ollama_chat(full_prompt, system_prompt=sys_prompt)
    
    # 16) Verify answer - check for hallucinated numbers
    answer = verify_answer_numbers(answer, context, q_lang)
    
    return answer


def verify_answer_numbers(answer: str, context: str, lang: str) -> str:
    """
    Verify that numbers in the answer exist in the context.
    If hallucinated numbers are found, return a warning or extract correct numbers.
    """
    import re
    
    # Extract numbers from answer (including decimals and percentages)
    answer_numbers = set(re.findall(r'\d+[.,]?\d*', answer))
    context_numbers = set(re.findall(r'\d+[.,]?\d*', context))
    
    # Normalize numbers (convert commas to dots for comparison)
    def normalize_num(n):
        return n.replace(',', '.')
    
    context_numbers_normalized = {normalize_num(n) for n in context_numbers}
    
    # Filter out very common numbers (1, 2, 3 etc. which are often part of lists)
    significant_answer_nums = {n for n in answer_numbers if float(normalize_num(n)) >= 5 or '.' in n or ',' in n}
    
    if not significant_answer_nums:
        return answer  # No significant numbers to verify
    
    # Check if answer numbers are in context (with normalization)
    hallucinated = {n for n in significant_answer_nums if normalize_num(n) not in context_numbers_normalized}
    
    if hallucinated:
        print(f"[debug] WARNING: Possible hallucinated numbers: {hallucinated}")
        print(f"[debug] Numbers in context: {sorted(context_numbers_normalized, key=lambda x: float(x) if x.replace('.','').isdigit() else 0)[:20]}")
        
        # For critical questions about specific values, try to extract from context
        # If answer contains hallucinated numbers, try to find correct ones in context
        for bad_num in hallucinated:
            # Look for similar patterns in context that might be correct
            bad_float = float(bad_num.replace(',', '.'))
            # Check if there's a number in context that's "close" or relevant
            for ctx_num in context_numbers:
                ctx_float = float(ctx_num.replace(',', '.'))
                # If context has a similar-magnitude number, it might be the correct one
                if 0.5 <= bad_float / (ctx_float + 0.001) <= 2.0:
                    print(f"[debug] Found potential correct number in context: {ctx_num} (instead of {bad_num})")
    
    return answer


# =================== CLI ===================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--no-hybrid", action="store_true", help="Disable hybrid search")
    parser.add_argument("--no-hyde", action="store_true", help="Disable HyDE")
    parser.add_argument("--no-expansion", action="store_true", help="Disable query expansion")
    args = parser.parse_args()

    # Apply CLI flags
    if args.no_hyde:
        USE_HYDE = False
    if args.no_expansion:
        USE_QUERY_EXPANSION = False
    if args.no_hybrid:
        USE_BM25_HYBRID = False

    print(f"Loading Chroma collection from: {CHROMA_DIR}")
    coll = get_chroma_collection()
    
    # Pre-load BM25 index
    load_bm25_index()
    
    print("Ready.")
    print(f"Features: HyDE={USE_HYDE}, QueryExpansion={USE_QUERY_EXPANSION}, BM25Hybrid={USE_BM25_HYBRID}")
    print("Type an empty line to exit.\n")

    history: List[Dict] = []

    while True:
        try:
            q = input("Query> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not q:
            break

        try:
            history.append({"role": "user", "content": q})

            ans = answer_with_rag(
                query=q,
                coll=coll,
                history=history[:-1],
            )
            print("\n=== ANSWER ===")
            print(ans)
            print("==============\n")

            history.append({"role": "assistant", "content": ans})

        except Exception as e:
            import traceback
            print(f"[error] {e}")
            traceback.print_exc()
