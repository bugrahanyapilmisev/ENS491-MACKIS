# backend/services/rag_service.py
"""
RAG Service - Uses modular RAG pipeline with hybrid search

Features:
- BM25 + Vector hybrid search
- Query expansion and HyDE
- Cross-encoder reranking
- Multi-query retrieval
- Knowledge graph augmentation

This service wraps the modular RAGPipeline for use in FastAPI endpoints.
"""

from typing import List, Dict, Tuple, Optional

from services.pipeline.rag_pipeline import RAGPipeline, get_pipeline
from services.config.settings import RAGConfig


class RAGService:
    """
    RAG Service wrapper for FastAPI integration.

    Provides a simple interface to the modular RAG pipeline.
    """

    def __init__(self, config: Optional[RAGConfig] = None):
        """
        Initialize RAG Service.

        Args:
            config: Optional configuration. If None, loads from environment.
        """
        print("🚀 RAG Engine (Modular Architecture) Loading...")

        try:
            # Initialize the modular pipeline
            self.pipeline = RAGPipeline(config)

            # Get collection reference for backward compatibility
            self.coll = self.pipeline.get_chroma_collection()

            print(f"✅ ChromaDB Connected: {self.pipeline.vector_store.count} chunks indexed")

            if self.pipeline.bm25_service.is_loaded:
                print(f"✅ BM25 Index Loaded: {self.pipeline.bm25_service.document_count} documents")
            else:
                print("⚠️ BM25 index not available (will use vector-only search)")

            # Report feature status
            stats = self.pipeline.get_stats()
            print(f"✅ Features: HyDE={stats['features']['use_hyde']}, "
                  f"QueryExpansion={stats['features']['use_query_expansion']}, "
                  f"MultiQuery={stats['features']['use_multi_query']}")

            if stats.get('kg_available'):
                print("✅ Knowledge Graph: Available")

            print("✅ RAG Engine Ready!")

        except Exception as e:
            print(f"❌ RAG Engine Initialization Error: {e}")
            import traceback
            traceback.print_exc()
            self.pipeline = None
            self.coll = None

    def query(
        self,
        user_query: str,
        history: Optional[List[Dict]] = None
    ) -> Tuple[str, List[Dict]]:
        """
        Process user query through the RAG pipeline.

        Args:
            user_query: The user's question
            history: Previous conversation messages

        Returns:
            Tuple of (answer_text, sources)
        """
        history = history or []
        print(f"🔍 Processing Query: {user_query[:100]}...")

        if not self.pipeline:
            return "Database connection unavailable.", []

        try:
            # Call the modular RAG pipeline
            answer_text = self.pipeline.answer(
                query=user_query,
                history=history,
            )

            # TODO: Extract sources from the pipeline if needed
            # For now, return empty sources list
            sources = []

            return answer_text, sources

        except Exception as e:
            print(f"❌ RAG Pipeline Error: {e}")
            import traceback
            traceback.print_exc()
            return "Sorry, a technical error occurred while processing your question.", []

    def search_only(
        self,
        user_query: str,
        top_k: int = 5,
        language: Optional[str] = None
    ) -> List[Dict]:
        """
        Perform search without generating an answer.
        Useful for debugging and testing.

        Args:
            user_query: Search query
            top_k: Number of results to return
            language: Optional language filter

        Returns:
            List of search results with metadata
        """
        if not self.pipeline:
            return []

        try:
            # Use pipeline's search_only method
            results = self.pipeline.search_only(user_query, top_k, language)

            # Format results
            formatted = []
            for c in results:
                meta = c.get("meta", {})
                formatted.append({
                    "chunk_id": c.get("chunk_id", ""),
                    "title": meta.get("title", ""),
                    "section": meta.get("section_header", ""),
                    "source_path": meta.get("source_path", ""),
                    "text_preview": c.get("text", "")[:200],
                    "score": c.get("hybrid_score", c.get("score", 0)),
                })

            return formatted

        except Exception as e:
            print(f"Search error: {e}")
            return []

    def get_stats(self) -> Dict:
        """
        Get service statistics.

        Returns:
            Dict with pipeline statistics
        """
        if not self.pipeline:
            return {"status": "unavailable"}

        return self.pipeline.get_stats()

    @property
    def is_available(self) -> bool:
        """Check if service is available."""
        return self.pipeline is not None


# Global Instance - Can be imported by FastAPI app
rag_engine = RAGService()
