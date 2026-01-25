# backend/services/rag_service.py
"""
RAG Service - Uses improved rag_core with hybrid search

Features:
- BM25 + Vector hybrid search
- Query expansion and HyDE
- Cross-encoder reranking
- Multi-query retrieval
"""

import os
import sys
from typing import List, Dict, Tuple

# Add services directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from rag_core import (
    get_chroma_collection,
    load_bm25_index,
    answer_with_rag,
    CHROMA_DIR,
    BM25_INDEX_PATH,
)


class RAGService:
    def __init__(self):
        print("🚀 RAG Engine (Hybrid Search + Reranking) Loading...")
        
        # 1. Check directories
        if not os.path.exists(CHROMA_DIR):
            print(f"⚠️ WARNING: ChromaDB folder not found: {CHROMA_DIR}")
            print("Please run build_chroma_store.py first.")
        
        # 2. ChromaDB Connection
        try:
            self.coll = get_chroma_collection()
            count = self.coll.count()
            print(f"✅ ChromaDB Connected: {count} chunks indexed")
        except Exception as e:
            print(f"❌ ChromaDB Error: {e}")
            self.coll = None

        # 3. Load BM25 Index
        try:
            bm25, chunk_ids = load_bm25_index()
            if bm25:
                print(f"✅ BM25 Index Loaded: {len(chunk_ids)} documents")
            else:
                print("⚠️ BM25 index not available (will use vector-only search)")
        except Exception as e:
            print(f"⚠️ BM25 loading warning: {e}")
        
        print("✅ RAG Engine Ready!")

    def query(self, user_query: str, history: List[Dict] = []) -> Tuple[str, List[Dict]]:
        """
        Process user query through the RAG pipeline.
        
        Args:
            user_query: The user's question
            history: Previous conversation messages
            
        Returns:
            Tuple of (answer_text, sources)
        """
        print(f"🔍 Processing Query: {user_query[:100]}...")
        
        if not self.coll:
            return "Database connection unavailable.", []

        try:
            # Call the  RAG pipeline
            answer_text = answer_with_rag(
                query=user_query,
                coll=self.coll,
                history=history,
                use_hybrid=True  # Enable hybrid search
            )
            
            # TODO: Extract sources from the answer if needed
            # For now, return empty sources list
            sources = []
            
            return answer_text, sources

        except Exception as e:
            print(f"❌ RAG Pipeline Error: {e}")
            import traceback
            traceback.print_exc()
            return "Sorry, a technical error occurred while processing your question.", []

    def search_only(self, user_query: str, top_k: int = 5) -> List[Dict]:
        """
        Perform search without generating an answer.
        Useful for debugging and testing.
        """
        from rag_core import hybrid_search, cross_encoder_rerank
        
        if not self.coll:
            return []
        
        try:
            # Get candidates via hybrid search
            candidates = hybrid_search(user_query, self.coll)
            
            # Rerank
            reranked = cross_encoder_rerank(user_query, candidates)
            
            # Return top results
            results = []
            for c in reranked[:top_k]:
                meta = c.get("meta", {})
                results.append({
                    "title": meta.get("title", ""),
                    "source_path": meta.get("source_path", ""),
                    "text_preview": c.get("text", "")[:200],
                    "score": c.get("final_score", 0),
                })
            
            return results
        except Exception as e:
            print(f"Search error: {e}")
            return []


# Global Instance - Can be imported by FastAPI app
rag_engine = RAGService()


# For backward compatibility, also export as rag_engine
rag_engine = rag_engine
