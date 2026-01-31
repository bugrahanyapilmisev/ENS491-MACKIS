"""
rag_core.py - Backward Compatibility Layer

This module provides backward compatibility with the original monolithic rag_core.py
by wrapping the new modular architecture.

All original exports are maintained for existing code that imports from this module.

The original implementation has been refactored into:
- services/config/settings.py - Configuration management
- services/core/ - Core services (embedding, llm, vector_store, bm25, data_loader)
- services/agents/ - Agent classes (query_analysis, retrieval, ranking, generation)
- services/pipeline/rag_pipeline.py - Main orchestrator

For new code, prefer importing from the modular structure directly.
"""

import os
from typing import Dict, List, Optional

from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Import from new modular architecture
from services.config.settings import RAGConfig
from services.pipeline.rag_pipeline import (
    RAGPipeline,
    create_pipeline,
    answer_with_rag,
    get_pipeline,
    KG_AVAILABLE,
)

# Load configuration from environment for backward compatibility constants
_config = RAGConfig.from_env()

# =================== BACKWARD COMPATIBLE CONSTANTS ===================

# Ollama configuration
OLLAMA_HOST = _config.ollama.host
EMBED_MODEL = _config.ollama.embed_model
EMBED_DIM = _config.ollama.embed_dim
CHAT_MODEL = _config.ollama.chat_model

# Path configuration
ROOT_DIR = _config.paths.root_dir
PREPROCESSING_DIR = _config.paths.preprocessing_dir
CREATING_DB_DIR = _config.paths.creating_db_dir
CHECKPOINT_DIR = _config.paths.checkpoint_dir
CHROMA_DIR = _config.chroma.chroma_dir
COLL_NAME = _config.chroma.collection_name

CHUNK_PARQUET = _config.paths.chunk_parquet
DOC_SUMMARY_PARQUET = _config.paths.doc_summary_parquet
BM25_INDEX_PATH = _config.paths.bm25_index_path

# Retrieval parameters
TOP_K_CHROMA = _config.retrieval.top_k_chroma
TOP_K_BM25 = _config.retrieval.top_k_bm25
TOP_K_FINAL_BASE = _config.retrieval.top_k_final_base
TOP_K_FINAL_MAX = _config.retrieval.top_k_final_max
MAX_DOCS_CONTEXT = _config.retrieval.max_docs_context

# Hybrid search weights
BM25_WEIGHT = _config.retrieval.bm25_weight
VECTOR_WEIGHT = _config.retrieval.vector_weight

# Cross-encoder parameters
RERANKER_MODEL_NAME = _config.reranking.model_name
CROSS_MAX_CANDIDATES = _config.reranking.max_candidates
CROSS_WEIGHT = _config.reranking.weight
CE_SCORE_THRESHOLD = _config.reranking.score_threshold

# Feature flags
USE_HYDE = _config.features.use_hyde
USE_QUERY_EXPANSION = _config.features.use_query_expansion
USE_MULTI_QUERY = _config.features.use_multi_query
USE_BM25_HYBRID = _config.features.use_bm25_hybrid
USE_DOC_SUMMARIES = _config.features.use_doc_summaries
NEGATION_BACKEND = _config.features.negation_backend

# =================== BACKWARD COMPATIBLE FUNCTIONS ===================

# Lazy-initialized pipeline
_pipeline: Optional[RAGPipeline] = None


def _get_pipeline() -> RAGPipeline:
    """Get or create pipeline instance."""
    global _pipeline
    if _pipeline is None:
        _pipeline = RAGPipeline(_config)
    return _pipeline


def get_chroma_collection():
    """Get ChromaDB collection - backward compatible."""
    return _get_pipeline().get_chroma_collection()


def load_bm25_index():
    """Load BM25 index - backward compatible."""
    pipeline = _get_pipeline()
    pipeline.bm25_service.load_index()
    return pipeline.bm25_service._bm25, pipeline.bm25_service._chunk_ids


def load_chunk_df():
    """Load chunk dataframe - backward compatible."""
    return _get_pipeline().data_loader.load_chunk_df()


def load_doc_summaries():
    """Load document summaries - backward compatible."""
    return _get_pipeline().data_loader.load_doc_summaries()


def get_doc_summary(source_path: str) -> str:
    """Get document summary - backward compatible."""
    return _get_pipeline().data_loader.get_doc_summary(source_path)


def get_all_chunks_for_doc(source_path: str) -> List[Dict]:
    """Get all chunks for document - backward compatible."""
    return _get_pipeline().data_loader.get_all_chunks_for_doc(source_path)


def embed_text_ollama(text: str, use_cache: bool = True, max_retries: int = 3):
    """Embed text using Ollama - backward compatible."""
    return _get_pipeline().embedding_service.embed(text, use_cache)


def cosine_sim(a, b) -> float:
    """Cosine similarity - backward compatible."""
    from services.core.embedding_service import EmbeddingService
    return EmbeddingService.cosine_similarity(a, b)


def call_ollama_chat(prompt: str, system_prompt: str = "", model: str = None) -> str:
    """Call Ollama chat - backward compatible."""
    pipeline = _get_pipeline()
    return pipeline.llm_service.chat(
        prompt,
        system_prompt,
        temperature=0.0,
        model=model or CHAT_MODEL
    )


def guess_lang_from_text(text: str) -> Optional[str]:
    """Detect language - backward compatible."""
    return _get_pipeline().query_agent.detect_language(text)


def detect_query_intent(query: str, lang: Optional[str] = None) -> str:
    """Detect query intent - backward compatible."""
    return _get_pipeline().query_agent.detect_intent(query)


def hybrid_search(
    query: str,
    coll=None,  # Ignored
    query_vec=None,
    top_k_vec: int = None,
    top_k_bm25: int = None,
    lang_filter: Optional[str] = None,
    vec_weight: float = None,
    bm25_weight: float = None,
) -> List[Dict]:
    """Hybrid search - backward compatible."""
    pipeline = _get_pipeline()
    return pipeline.retrieval_agent.hybrid_search(
        query,
        language=lang_filter,
        query_vec=query_vec,
        top_k_vec=top_k_vec or TOP_K_CHROMA,
        top_k_bm25=top_k_bm25 or TOP_K_BM25,
    )


def cross_encoder_rerank(
    query: str,
    candidates: List[Dict],
    max_candidates: int = None,
    weight_ce: float = None,
) -> List[Dict]:
    """Cross-encoder rerank - backward compatible."""
    pipeline = _get_pipeline()
    return pipeline.ranking_agent.cross_encoder_rerank(query, candidates)


def build_context(chunks: List[Dict], include_summaries: bool = True) -> str:
    """Build context - backward compatible."""
    return _get_pipeline().generation_agent.build_context(chunks, include_summaries)


# =================== KG INTEGRATION ===================

def query_kg(query: str, lang: str = "tr") -> str:
    """Query knowledge graph - backward compatible wrapper."""
    if not KG_AVAILABLE:
        return ""
    try:
        from services.kg_service import query_hybrid
        return query_hybrid(query, lang=lang)
    except Exception as e:
        print(f"[KG Query Error] {e}")
        return ""


# =================== CLI ===================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="RAG Core CLI")
    parser.add_argument("--no-hybrid", action="store_true", help="Disable hybrid search")
    parser.add_argument("--no-hyde", action="store_true", help="Disable HyDE")
    parser.add_argument("--no-expansion", action="store_true", help="Disable query expansion")
    args = parser.parse_args()

    # Create pipeline with potentially modified config
    config = RAGConfig.from_env()
    if args.no_hyde:
        config.features.use_hyde = False
    if args.no_expansion:
        config.features.use_query_expansion = False
    if args.no_hybrid:
        config.features.use_bm25_hybrid = False

    pipeline = RAGPipeline(config)

    print("Ready.")
    print(f"Features: HyDE={config.features.use_hyde}, "
          f"QueryExpansion={config.features.use_query_expansion}, "
          f"BM25Hybrid={config.features.use_bm25_hybrid}")
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

            ans = pipeline.answer(query=q, history=history[:-1])

            print("\n=== ANSWER ===")
            print(ans)
            print("==============\n")

            history.append({"role": "assistant", "content": ans})

        except Exception as e:
            import traceback
            print(f"[error] {e}")
            traceback.print_exc()


# =================== EXPORTS ===================

__all__ = [
    # Main function
    "answer_with_rag",
    # Configuration constants
    "OLLAMA_HOST",
    "EMBED_MODEL",
    "EMBED_DIM",
    "CHAT_MODEL",
    "CHROMA_DIR",
    "COLL_NAME",
    "TOP_K_CHROMA",
    "TOP_K_BM25",
    "TOP_K_FINAL_BASE",
    "TOP_K_FINAL_MAX",
    "BM25_WEIGHT",
    "VECTOR_WEIGHT",
    "CROSS_WEIGHT",
    "CE_SCORE_THRESHOLD",
    "USE_HYDE",
    "USE_QUERY_EXPANSION",
    "USE_MULTI_QUERY",
    "USE_BM25_HYBRID",
    "USE_DOC_SUMMARIES",
    # Backward compatible functions
    "get_chroma_collection",
    "load_bm25_index",
    "load_chunk_df",
    "load_doc_summaries",
    "get_doc_summary",
    "get_all_chunks_for_doc",
    "embed_text_ollama",
    "cosine_sim",
    "call_ollama_chat",
    "guess_lang_from_text",
    "detect_query_intent",
    "hybrid_search",
    "cross_encoder_rerank",
    "build_context",
    "query_kg",
    "KG_AVAILABLE",
    # New modular exports
    "RAGPipeline",
    "create_pipeline",
    "get_pipeline",
    "RAGConfig",
]
