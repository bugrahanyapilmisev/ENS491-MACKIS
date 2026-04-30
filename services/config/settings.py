"""
Centralized configuration with dependency injection support.

This module provides dataclass-based configuration for the RAG pipeline,
allowing for easy testing, customization, and environment-based configuration.
"""

from dataclasses import dataclass, field
from typing import Optional
import os

from dotenv import load_dotenv


@dataclass
class OllamaConfig:
    """Ollama API configuration."""
    host: str = "http://localhost:11434"
    embed_model: str = "bge-m3"
    embed_dim: int = 4096  # Qwen3-Embedding-8B native dim; use 1024 for BGE-M3/Ollama
    chat_model: str = "qwen3.5:latest"
    timeout: int = 120
    chat_timeout: int = 900
    max_retries: int = 1
    openrouter_api_key: str = ""
    # Embedding provider: "ollama" or "openrouter"
    embed_provider: str = "ollama"
    # OpenRouter embedding model (Qwen3-Embedding-8B is top multilingual open-source)
    embed_model_openrouter: str = "qwen/qwen3-embedding-8b"
    # Instruction prefixes for instruction-aware embedding (Qwen3-Embedding style)
    # These give ~2-5% nDCG improvement on retrieval tasks
    embed_instruction_query: str = (
        "Instruct: Given a user question about university policies, "
        "retrieve the most relevant university policy document chunk\nQuery: "
    )
    embed_instruction_doc: str = (
        "Instruct: Represent this university policy document chunk for retrieval\nDocument: "
    )


@dataclass
class ChromaConfig:
    """ChromaDB configuration."""
    chroma_dir: str = ""
    collection_name: str = "mysu_v2_bge_m3"


@dataclass
class RetrievalConfig:
    """Retrieval parameters."""
    top_k_chroma: int = 80
    top_k_bm25: int = 48
    top_k_final_base: int = 8
    top_k_final_max: int = 40
    max_docs_context: int = 15
    bm25_weight: float = 0.3
    vector_weight: float = 0.7
    rrf_constant: int = 60


@dataclass
class RerankingConfig:
    """Cross-encoder reranking parameters."""
    model_name: str = "BAAI/bge-reranker-v2-m3"
    max_candidates: int = 40  # Increased for Qwen 3.5
    batch_size: int = 4       # Reduced from 8
    max_text_length: int = 1500  # Truncate text for cross-encoder
    weight: float = 0.6
    score_threshold: float = 0.65
    tag_boost_positive: float = 0.20
    tag_penalize_negative: float = 0.10
    negation_penalty_factor: float = 0.3
    mmr_lambda: float = 0.7


@dataclass
class FeatureFlags:
    """Feature toggles for pipeline components."""
    use_hyde: bool = True
    use_query_expansion: bool = True
    use_multi_query: bool = True
    use_bm25_hybrid: bool = True
    use_doc_summaries: bool = True
    negation_backend: str = "llm"


@dataclass
class KGConfig:
    """Knowledge Graph configuration."""
    kg_output_dir: str = ""
    # LLM validated files (highest quality)
    kg_facts_llm_validated: str = ""
    kg_index_llm_validated: str = ""
    kg_triples_llm_validated: str = ""
    # Pattern validated files
    kg_facts_pattern_validated: str = ""
    kg_index_pattern_validated: str = ""
    kg_triples_pattern_validated: str = ""
    # Raw/base files
    kg_facts: str = ""
    kg_triples: str = ""
    kg_entities: str = ""
    kg_raw_extractions: str = ""
    # Active source (which validation to use: "llm", "pattern", "raw")
    active_source: str = "llm"


@dataclass
class PathConfig:
    """File system paths."""
    root_dir: str = ""
    preprocessing_dir: str = ""
    creating_db_dir: str = ""
    creating_kg_dir: str = ""
    checkpoint_dir: str = ""
    chunk_parquet: str = ""
    doc_summary_parquet: str = ""
    bm25_index_path: str = ""


@dataclass
class RAGConfig:
    """Master configuration container."""
    ollama: OllamaConfig = field(default_factory=OllamaConfig)
    chroma: ChromaConfig = field(default_factory=ChromaConfig)
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)
    reranking: RerankingConfig = field(default_factory=RerankingConfig)
    features: FeatureFlags = field(default_factory=FeatureFlags)
    paths: PathConfig = field(default_factory=PathConfig)
    kg: KGConfig = field(default_factory=KGConfig)

    @classmethod
    def from_env(cls, env_path: Optional[str] = None) -> "RAGConfig":
        """
        Load configuration from environment variables.

        Args:
            env_path: Optional path to .env file. If None, uses default dotenv behavior.

        Returns:
            RAGConfig instance with values from environment.
        """
        if env_path:
            load_dotenv(env_path)
        else:
            load_dotenv()

        # Determine root directory
        root_dir = os.getenv(
            "ROOT_PATH",
            os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        )
        preprocessing_dir = os.path.join(root_dir, "preprocessing")
        creating_db_dir = os.path.join(root_dir, "creating_database")
        creating_kg_dir = os.path.join(root_dir, "creating_kg")

        # Checkpoint directory with fallback
        checkpoint_dir = os.getenv("CHECKPOINT_DIR_V2") or os.path.join(
            creating_db_dir, "checkpoints_v2"
        )

        # Chroma directory with fallback
        chroma_dir = os.getenv("CHROMA_DIR_V2") or os.path.join(
            creating_db_dir, "chroma_db_v2"
        )

        # KG output directory
        kg_output_dir = os.path.join(creating_kg_dir, "knowledge_graph")
        kg_llm_validated_dir = os.path.join(kg_output_dir, "llm_validated")
        kg_pattern_validated_dir = os.path.join(kg_output_dir, "pattern_validated")

        return cls(
            ollama=OllamaConfig(
                host=os.getenv("OLLAMA_HOST", "http://localhost:11434"),
                embed_model=os.getenv("EMBED_MODEL", "bge-m3"),
                embed_dim=int(os.getenv("EMBED_DIM", "1024")),
                chat_model=os.getenv("CHAT_MODEL", "qwen3.5:latest"),
                timeout=int(os.getenv("OLLAMA_TIMEOUT", "120")),
                chat_timeout=int(os.getenv("OLLAMA_CHAT_TIMEOUT", "900")),
                max_retries=int(os.getenv("OLLAMA_MAX_RETRIES", "3")),
                openrouter_api_key=os.getenv("OPENROUTER_API_KEY", ""),
                embed_provider=os.getenv("EMBED_PROVIDER", "ollama"),
                embed_model_openrouter=os.getenv(
                    "EMBED_MODEL_OPENROUTER", "qwen/qwen3-embedding-8b"
                ),
                embed_instruction_query=os.getenv(
                    "EMBED_INSTRUCTION_QUERY",
                    "Instruct: Given a user question about university policies, "
                    "retrieve the most relevant university policy document chunk\nQuery: "
                ),
                embed_instruction_doc=os.getenv(
                    "EMBED_INSTRUCTION_DOC",
                    "Instruct: Represent this university policy document chunk for retrieval\nDocument: "
                ),
            ),
            chroma=ChromaConfig(
                chroma_dir=chroma_dir,
                # Prefer V3 collection (Qwen3-Embedding) if available; fall back to V2 (BGE-M3)
                collection_name=os.getenv(
                    "COLL_NAME_V3",
                    os.getenv("CHROMA_COLLECTION_NAME_V2", "mysu_v2_bge_m3")
                ),
            ),
            retrieval=RetrievalConfig(
                top_k_chroma=int(os.getenv("TOP_K_CHROMA", "80")),
                top_k_bm25=int(os.getenv("TOP_K_BM25", "48")),
                top_k_final_base=int(os.getenv("TOP_K_FINAL_BASE", "8")),
                top_k_final_max=int(os.getenv("TOP_K_FINAL_MAX", "40")),
                max_docs_context=int(os.getenv("MAX_DOCS_CONTEXT", "15")),
                bm25_weight=float(os.getenv("BM25_WEIGHT", "0.3")),
                vector_weight=float(os.getenv("VECTOR_WEIGHT", "0.7")),
                rrf_constant=int(os.getenv("RRF_CONSTANT", "60")),
            ),
            reranking=RerankingConfig(
                model_name=os.getenv("RERANKER_MODEL_NAME", "BAAI/bge-reranker-v2-m3"),
                max_candidates=int(os.getenv("CROSS_MAX_CANDIDATES", "40")),
                batch_size=int(os.getenv("RERANKER_BATCH_SIZE", "4")),
                max_text_length=int(os.getenv("RERANKER_MAX_TEXT_LENGTH", "1500")),
                weight=float(os.getenv("CROSS_WEIGHT", "0.6")),
                score_threshold=float(os.getenv("CE_SCORE_THRESHOLD", "0.65")),
                tag_boost_positive=float(os.getenv("TAG_BOOST_POSITIVE", "0.20")),
                tag_penalize_negative=float(os.getenv("TAG_PENALIZE_NEGATIVE", "0.10")),
                negation_penalty_factor=float(os.getenv("NEGATION_PENALTY_FACTOR", "0.3")),
                mmr_lambda=float(os.getenv("MMR_LAMBDA", "0.7")),
            ),
            features=FeatureFlags(
                use_hyde=os.getenv("USE_HYDE", "1") == "1",
                use_query_expansion=os.getenv("USE_QUERY_EXPANSION", "1") == "1",
                use_multi_query=os.getenv("USE_MULTI_QUERY", "1") == "1",
                use_bm25_hybrid=os.getenv("USE_BM25_HYBRID", "1") == "1",
                use_doc_summaries=os.getenv("USE_DOC_SUMMARIES", "1") == "1",
                negation_backend=os.getenv("NEGATION_BACKEND", "llm").lower(),
            ),
            paths=PathConfig(
                root_dir=root_dir,
                preprocessing_dir=preprocessing_dir,
                creating_db_dir=creating_db_dir,
                creating_kg_dir=creating_kg_dir,
                checkpoint_dir=checkpoint_dir,
                chunk_parquet=os.getenv("CHUNK_PARQUET_PATH", os.path.join(checkpoint_dir, "chunks_v3.parquet")),
                doc_summary_parquet=os.getenv("DOC_SUMMARY_PARQUET_PATH", os.path.join(checkpoint_dir, "doc_summaries_v3.parquet")),
                bm25_index_path=os.getenv("BM25_INDEX_PATH", os.path.join(checkpoint_dir, "bm25_index_v3.pkl")),
            ),
            kg=KGConfig(
                kg_output_dir=kg_output_dir,
                # LLM validated files (highest quality)
                kg_facts_llm_validated=os.path.join(kg_llm_validated_dir, "kg_facts_llm_validated.json"),
                kg_index_llm_validated=os.path.join(kg_llm_validated_dir, "kg_index_llm_validated.pkl"),
                kg_triples_llm_validated=os.path.join(kg_llm_validated_dir, "kg_triples_llm_validated.json"),
                # Pattern validated files
                kg_facts_pattern_validated=os.path.join(kg_pattern_validated_dir, "kg_facts_validated.json"),
                kg_index_pattern_validated=os.path.join(kg_pattern_validated_dir, "kg_index_validated.pkl"),
                kg_triples_pattern_validated=os.path.join(kg_pattern_validated_dir, "kg_triples_validated.json"),
                # Raw/base files
                kg_facts=os.path.join(kg_output_dir, "kg_facts.json"),
                kg_triples=os.path.join(kg_output_dir, "kg_triples.json"),
                kg_entities=os.path.join(kg_output_dir, "kg_entities.json"),
                kg_raw_extractions=os.path.join(kg_output_dir, "kg_raw_extractions.json"),
                # Active source (which validation to use)
                active_source=os.getenv("KG_ACTIVE_SOURCE", "llm"),
            ),
        )

    def __repr__(self) -> str:
        """Return a readable string representation."""
        return (
            f"RAGConfig(\n"
            f"  ollama={self.ollama},\n"
            f"  chroma={self.chroma},\n"
            f"  retrieval={self.retrieval},\n"
            f"  reranking={self.reranking},\n"
            f"  features={self.features},\n"
            f"  kg={self.kg}\n"
            f")"
        )
