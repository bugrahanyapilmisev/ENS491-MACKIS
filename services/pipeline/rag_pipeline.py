"""
RAG Pipeline Orchestrator - Main entry point.

This module coordinates all agents to process queries through:
1. Query Analysis
2. Retrieval
3. Ranking
4. Generation

Maintains backward compatibility with the original answer_with_rag() function.
"""

from typing import Dict, List, Optional

from services.config.settings import RAGConfig
from services.core.embedding_service import EmbeddingService
from services.core.llm_service import LLMService
from services.core.vector_store import VectorStoreService
from services.core.bm25_service import BM25Service
from services.core.data_loader import DataLoaderService
from services.agents.query_analysis_agent import QueryAnalysisAgent
from services.agents.retrieval_agent import RetrievalAgent
from services.agents.ranking_agent import RankingAgent
from services.agents.generation_agent import GenerationAgent


# Optional KG integration
try:
    from services.kg_service import query_hybrid as kg_query_hybrid, load_kg_facts
    # Verify KG can be loaded
    if load_kg_facts():
        KG_AVAILABLE = True
        print("[RAGPipeline] KG service loaded successfully")
    else:
        KG_AVAILABLE = False
        print("[RAGPipeline] KG service: facts not loaded")
except ImportError:
    KG_AVAILABLE = False
    kg_query_hybrid = None
    print("[RAGPipeline] KG service not available")


class RAGPipeline:
    """
    Main RAG Pipeline Orchestrator.

    Coordinates all agents to process queries through:
    1. Query Analysis - Intent, language, followup, tags, negation
    2. Retrieval - Hybrid search, query expansion, HyDE
    3. Ranking - Cross-encoder, tag boosting, negation penalty
    4. Generation - Context building, answer generation
    """

    def __init__(self, config: Optional[RAGConfig] = None):
        """
        Initialize pipeline with configuration.

        Args:
            config: RAG configuration. If None, loads from environment.
        """
        self.config = config or RAGConfig.from_env()

        # Initialize services with shared embedding cache
        self._embedding_cache: Dict = {}

        self.embedding_service = EmbeddingService(
            self.config.ollama,
            cache=self._embedding_cache
        )
        self.llm_service = LLMService(self.config.ollama)
        self.vector_store = VectorStoreService(self.config.chroma)
        self.bm25_service = BM25Service(self.config.paths)
        self.data_loader = DataLoaderService(self.config.paths)

        # Initialize agents
        self.query_agent = QueryAnalysisAgent(
            self.embedding_service,
            self.llm_service,
            self.config
        )
        self.retrieval_agent = RetrievalAgent(
            self.embedding_service,
            self.llm_service,
            self.vector_store,
            self.bm25_service,
            self.data_loader,
            self.config
        )
        self.ranking_agent = RankingAgent(
            self.embedding_service,
            self.vector_store,
            self.config
        )
        self.generation_agent = GenerationAgent(
            self.llm_service,
            self.data_loader,
            self.config
        )

        # Pre-load indexes
        self.bm25_service.load_index()

        print(f"[RAGPipeline] Initialized with {self.vector_store.count} chunks")

    def answer(
        self,
        query: str,
        history: Optional[List[Dict]] = None,
        use_hybrid: bool = True
    ) -> Dict:
        """
        Main entry point - process query and return structured result.

        Args:
            query: User question.
            history: Optional conversation history.
            use_hybrid: Whether to use hybrid search (ignored, always True).

        Returns:
            Dict with:
            - answer: Generated answer string
            - retrieved_chunks: All reranked chunks with scores
            - context_chunks: Final chunks used for generation
            - analysis: Query analysis result (language, intent, followup, etc.)
        """
        history = history or []

        # 1. Query Analysis
        analysis = self.query_agent.analyze(query, history)
        language = analysis["language"]
        intent = analysis["intent"]
        is_followup = analysis["is_followup"]
        anchor = analysis["anchor_query"]
        query_tags = analysis["tags"]
        negated_terms = analysis["negated_terms"]
        expanded_queries = analysis.get("expanded_queries", [query])

        print(f"[RAGPipeline] language={language}, intent={intent}, followup={is_followup}")
        print(f"[RAGPipeline] tags={query_tags}")
        if negated_terms:
            print(f"[RAGPipeline] negated_terms={negated_terms}")

        # 2. Build retrieval query
        retrieval_query = self.query_agent.build_retrieval_query(query, analysis)

        # 3. Retrieval — pass pre-expanded queries + intent-based HyDE skip
        skip_hyde = intent in {"count_items", "list_names"}
        candidates = self.retrieval_agent.retrieve(
            retrieval_query,
            language=language,
            use_expansion=self.config.features.use_query_expansion,
            use_hyde=self.config.features.use_hyde and not skip_hyde,
            pre_expanded=expanded_queries if self.config.features.use_query_expansion else None
        )

        if not candidates:
            return {
                "answer": self._no_results_message(language),
                "retrieved_chunks": [],
                "context_chunks": [],
                "analysis": analysis,
            }

        print(f"[RAGPipeline] Initial candidates: {len(candidates)}")

        # 4. Ranking
        reranked = self.ranking_agent.rerank(
            retrieval_query,
            candidates,
            query_tags=query_tags,
            negated_terms=negated_terms
        )

        # Debug: Print top candidates
        print("=== TOP CANDIDATES AFTER RERANK ===")
        for i, c in enumerate(reranked[:10], start=1):
            meta = c.get("meta") or {}
            print(f"{i:2d}. score={c.get('hybrid_score', 0):.3f} "
                  f"ce={c.get('ce_score', 0):.3f} | {meta.get('title', '')[:50]}")
        print("===================================")

        # 5. Filter by threshold
        strong = self.ranking_agent.filter_by_threshold(reranked)

        # 6. Document-level diversity + chunk cap
        max_ctx = self.config.retrieval.max_docs_context
        max_per_doc = max(3, max_ctx // 3)

        if intent in {"list_names", "count_items"} and reranked:
            # Use all chunks from top document for factual queries
            best_chunk = reranked[0]
            best_meta = best_chunk.get("meta") or {}
            best_path = best_meta.get("source_path")

            if best_path:
                print(f"[RAGPipeline] Single-doc mode for intent={intent}")
                retrieved = self.data_loader.get_all_chunks_for_doc(best_path)
                retrieved = retrieved[:max_ctx]
            else:
                retrieved = self.ranking_agent.document_level_select(
                    strong, max_chunks=max_ctx, max_per_doc=max_per_doc
                )
        else:
            # Document-level diversity selection
            retrieved = self.ranking_agent.document_level_select(
                strong, max_chunks=max_ctx, max_per_doc=max_per_doc
            )

        print(f"[RAGPipeline] Final context chunks: {len(retrieved)}")

        # 7. Knowledge Graph augmentation (optional)
        kg_facts = ""
        if KG_AVAILABLE and kg_query_hybrid:
            try:
                kg_facts = kg_query_hybrid(query, lang=language or "tr")
                if kg_facts:
                    print("[RAGPipeline] KG facts retrieved")
                    print("=== KG FACTS ===")
                    print(kg_facts[:500] + "..." if len(kg_facts) > 500 else kg_facts)
                    print("================")
            except Exception as e:
                print(f"[RAGPipeline] KG augmentation warning: {e}")

        # 8. Generation
        answer = self.generation_agent.generate(
            query,
            retrieved,
            language=language,
            kg_facts=kg_facts
        )

        return {
            "answer": answer,
            "retrieved_chunks": reranked,       # All reranked candidates with scores
            "context_chunks": retrieved,         # Final chunks used for generation (citations)
            "analysis": analysis,
        }

    def _no_results_message(self, language: Optional[str]) -> str:
        """Return appropriate no-results message."""
        if language == "tr":
            return "Üzgünüm, ilgili bir bilgi bulamadım."
        return "Sorry, I couldn't find relevant information."

    def search_only(
        self,
        query: str,
        top_k: int = 10,
        language: Optional[str] = None
    ) -> List[Dict]:
        """
        Search without generation (for debugging).

        Args:
            query: Search query.
            top_k: Number of results.
            language: Optional language filter.

        Returns:
            List of search results.
        """
        return self.retrieval_agent.search_only(query, top_k, language)

    def get_chroma_collection(self):
        """Return underlying ChromaDB collection for backward compatibility."""
        return self.vector_store.get_collection()

    def get_stats(self) -> Dict:
        """Get pipeline statistics."""
        return {
            "chunk_count": self.vector_store.count,
            "bm25_loaded": self.bm25_service.is_loaded,
            "bm25_doc_count": self.bm25_service.document_count,
            "embedding_cache_size": self.embedding_service.cache_size,
            "kg_available": KG_AVAILABLE,
            "features": {
                "use_hyde": self.config.features.use_hyde,
                "use_query_expansion": self.config.features.use_query_expansion,
                "use_multi_query": self.config.features.use_multi_query,
                "use_bm25_hybrid": self.config.features.use_bm25_hybrid,
                "use_doc_summaries": self.config.features.use_doc_summaries,
            }
        }


# Global pipeline instance for backward compatibility
_pipeline_instance: Optional[RAGPipeline] = None


def create_pipeline(config: Optional[RAGConfig] = None) -> RAGPipeline:
    """
    Create a new RAGPipeline instance.

    Args:
        config: Optional configuration.

    Returns:
        New RAGPipeline instance.
    """
    return RAGPipeline(config)


def answer_with_rag(
    query: str,
    coll=None,  # Ignored - kept for compatibility
    history: Optional[List[Dict]] = None,
    use_hybrid: bool = True
) -> str:
    """
    Backward-compatible wrapper for the original answer_with_rag function.

    Note: The 'coll' parameter is ignored as the pipeline manages its own
    ChromaDB connection.

    Args:
        query: User question.
        coll: Ignored (ChromaDB collection - pipeline manages this).
        history: Optional conversation history.
        use_hybrid: Ignored (always uses hybrid search).

    Returns:
        Generated answer string.
    """
    global _pipeline_instance

    if _pipeline_instance is None:
        _pipeline_instance = RAGPipeline()

    result = _pipeline_instance.answer(query, history, use_hybrid)
    # Backward compat: return only the answer string
    return result["answer"] if isinstance(result, dict) else result


def get_pipeline() -> RAGPipeline:
    """
    Get or create the global pipeline instance.

    Returns:
        Global RAGPipeline instance.
    """
    global _pipeline_instance

    if _pipeline_instance is None:
        _pipeline_instance = RAGPipeline()

    return _pipeline_instance
