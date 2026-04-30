"""
Retrieval agent for hybrid search and query expansion.

This agent handles:
- Hybrid search (Vector + BM25 with RRF fusion)
- LLM-based query expansion
- HyDE (Hypothetical Document Embeddings)
- Multi-query retrieval with result merging
"""

from typing import Dict, List, Optional
import textwrap

import numpy as np

from services.core.embedding_service import EmbeddingService
from services.core.llm_service import LLMService
from services.core.vector_store import VectorStoreService
from services.core.bm25_service import BM25Service
from services.core.data_loader import DataLoaderService
from services.config.settings import RAGConfig


class RetrievalAgent:
    """Handles retrieval operations with hybrid search."""

    def __init__(
        self,
        embedding_service: EmbeddingService,
        llm_service: LLMService,
        vector_store: VectorStoreService,
        bm25_service: BM25Service,
        data_loader: DataLoaderService,
        config: RAGConfig
    ):
        """
        Initialize retrieval agent.

        Args:
            embedding_service: Service for text embeddings.
            llm_service: Service for LLM operations.
            vector_store: Service for vector search.
            bm25_service: Service for BM25 search.
            data_loader: Service for chunk data loading.
            config: RAG configuration.
        """
        self.embedding = embedding_service
        self.llm = llm_service
        self.vector_store = vector_store
        self.bm25 = bm25_service
        self.data_loader = data_loader
        self.config = config

    def retrieve(
        self,
        query: str,
        language: Optional[str] = None,
        use_expansion: bool = True,
        use_hyde: bool = True,
        pre_expanded: Optional[List[str]] = None
    ) -> List[Dict]:
        """
        Main retrieval method combining all strategies.

        Args:
            query: User query.
            language: Detected language for filtering.
            use_expansion: Whether to use query expansion.
            use_hyde: Whether to use HyDE.
            pre_expanded: Pre-expanded queries from query analysis.
                          When provided, skips the separate expansion LLM call.

        Returns:
            List of candidate chunks with scores.
        """
        # If pre-expanded queries are provided, use them directly
        if pre_expanded and len(pre_expanded) > 0:
            expanded_queries = list(pre_expanded)
            print(f"[Retrieval] Using {len(expanded_queries)} pre-expanded queries (no extra LLM call)")
        else:
            expanded_queries = [query]
            # Query expansion (separate LLM call — only if not pre-expanded)
            if use_expansion and self.config.features.use_query_expansion:
                expanded_queries = self.expand_query(query, language)

        # HyDE (Hypothetical Document Embeddings)
        if use_hyde and self.config.features.use_hyde:
            hyde_text = self.generate_hypothetical_document(query, language)
            if hyde_text:
                expanded_queries.append(hyde_text)

        # Multi-query or single query retrieval
        if self.config.features.use_multi_query and len(expanded_queries) > 1:
            return self.multi_query_retrieval(expanded_queries, language)
        else:
            return self.hybrid_search(query, language)

    def hybrid_search(
        self,
        query: str,
        language: Optional[str] = None,
        query_vec: Optional[np.ndarray] = None,
        top_k_vec: Optional[int] = None,
        top_k_bm25: Optional[int] = None,
    ) -> List[Dict]:
        """
        Hybrid search combining vector and BM25.

        Args:
            query: Search query.
            language: Optional language filter.
            query_vec: Optional pre-computed query embedding.
            top_k_vec: Number of vector search results.
            top_k_bm25: Number of BM25 results.

        Returns:
            List of candidate dicts with hybrid scores.
        """
        top_k_vec = top_k_vec or self.config.retrieval.top_k_chroma
        top_k_bm25 = top_k_bm25 or self.config.retrieval.top_k_bm25

        # Embed query if not provided
        # is_query=True applies the instruction prefix for instruction-aware models
        # (Qwen3-Embedding-8B): "Instruct: ... retrieve the relevant document chunk\nQuery: <text>"
        if query_vec is None:
            query_vec = self.embedding.embed(query, is_query=True)

        # Vector search
        lang_filter = {"doc_lang": language} if language in ("tr", "en") else None
        vec_results = self.vector_store.search(
            query_vec,
            top_k=top_k_vec,
            where_filter=lang_filter
        )

        # BM25 search
        bm25_results = []
        if self.config.features.use_bm25_hybrid:
            bm25_results = self.bm25.search(query, top_k=top_k_bm25)

        # RRF fusion
        return self._rrf_fusion(vec_results, bm25_results)

    def _rrf_fusion(
        self,
        vec_results: List[Dict],
        bm25_results: List[tuple]
    ) -> List[Dict]:
        """
        Reciprocal Rank Fusion to combine vector and BM25 results.

        Args:
            vec_results: Results from vector search.
            bm25_results: Results from BM25 search as (chunk_id, score) tuples.

        Returns:
            Fused and sorted results.
        """
        K = self.config.retrieval.rrf_constant
        chunk_scores: Dict[str, Dict] = {}

        # Add vector results
        for rank, c in enumerate(vec_results):
            cid = c["chunk_id"]
            rrf_vec = 1.0 / (K + rank + 1)

            chunk_scores[cid] = {
                **c,
                "vec_rank": rank + 1,
                "bm25_score": 0.0,
                "bm25_rank": None,
                "rrf_vec": rrf_vec,
                "rrf_bm25": 0.0,
                "source": "hybrid",
            }

        # Add BM25 results
        for rank, (cid, bm25_score) in enumerate(bm25_results):
            rrf_bm25 = 1.0 / (K + rank + 1)

            if cid not in chunk_scores:
                # Fetch from data loader
                chunk_data = self.data_loader.get_chunk_by_id(cid)
                if not chunk_data:
                    continue

                chunk_scores[cid] = {
                    **chunk_data,
                    "score": 0.0,
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

        # Compute final hybrid score
        vec_weight = self.config.retrieval.vector_weight
        bm25_weight = self.config.retrieval.bm25_weight

        for data in chunk_scores.values():
            data["hybrid_score"] = (
                vec_weight * data["rrf_vec"] +
                bm25_weight * data["rrf_bm25"]
            )
            data["score"] = data["hybrid_score"]

        # Sort by hybrid score
        results = list(chunk_scores.values())
        results.sort(key=lambda x: x["hybrid_score"], reverse=True)

        return results

    def expand_query(
        self,
        query: str,
        lang: Optional[str] = None
    ) -> List[str]:
        """
        Generate expanded/alternative queries using LLM with synonym awareness.

        The prompt instructs the LLM to include common abbreviations and their
        expansions, formal/informal variations, and related university terms.

        Args:
            query: Original query.
            lang: Detected language.

        Returns:
            List of queries including original + alternatives (max 4).
        """
        sys_prompt = textwrap.dedent("""
            You are a search query expansion assistant for a university information system.
            Given a user query, generate 2-3 alternative search queries that capture the same intent.

            Rules:
            - Keep queries concise (5-15 words each)
            - Use synonyms and related terms
            - IMPORTANT: Include common abbreviations and their expansions.
              Examples: GNO <-> GPA, ÇAP <-> çift anadal <-> double major,
              AKTS <-> ECTS, yandal <-> minor, kayıt dondurma <-> dönem izni,
              not yükseltme <-> ders tekrarı
            - If the query uses an abbreviation, expand it in an alternative.
              If it uses the full form, include the abbreviation.
            - If query is in Turkish, generate Turkish alternatives
            - If query is in English, generate English alternatives
            - Include both formal and informal variations of key terms
            - Focus on the core information need

            Output format (STRICT JSON):
            {"queries": ["alternative query 1", "alternative query 2", ...]}

            Output ONLY the JSON, no explanation.
        """).strip()

        lang_hint = lang or "unknown"
        result = self.llm.chat_json(
            f"Language: {lang_hint}\nOriginal query: {query}",
            sys_prompt,
            temperature=0.3
        )

        if not result:
            return [query]

        queries = result.get("queries", [])
        if not isinstance(queries, list):
            return [query]

        # Always include original query first
        expanded = [query]
        for q in queries:
            if isinstance(q, str) and q.strip() and q.strip() != query:
                expanded.append(q.strip())

        print(f"[Retrieval] Query expansion: {expanded}")
        return expanded[:4]

    def generate_hypothetical_document(
        self,
        query: str,
        lang: Optional[str] = None
    ) -> Optional[str]:
        """
        Generate hypothetical document for HyDE.

        The generated document describes what a relevant document would contain,
        without making up specific facts. This helps semantic matching.

        Args:
            query: User query.
            lang: Detected language.

        Returns:
            Hypothetical document text, or None on failure.
        """
        lang_hint = "Turkish" if lang == "tr" else "English"

        sys_prompt = textwrap.dedent(f"""
            You are an expert at generating hypothetical document passages for retrieval.
            Given a question, write a SHORT passage (2-3 sentences) describing what kind of
            document would answer it.

            CRITICAL RULES:
            - Write in {lang_hint}
            - DO NOT use specific numbers, dates, or values - just describe the topic
            - Use general terms like "the minimum GPA requirement", "the required credits"
            - Write as if describing what a policy document would contain
            - Keep it short (30-50 words)

            BAD example: "The minimum GPA is 2.5" (don't make up numbers!)
            GOOD example: "This document describes the minimum GPA requirements for program eligibility."

            Output ONLY the passage, nothing else.
        """).strip()

        text = self.llm.chat(
            f"Question: {query}",
            sys_prompt,
            temperature=0.3
        )

        # Validate response
        if not text:
            return None
        if text.startswith(("Ollama Error", "Connection", "LLM error")):
            return None
        if len(text) < 20:
            return None

        print(f"[Retrieval] HyDE generated: {text[:100]}...")
        return text

    def multi_query_retrieval(
        self,
        queries: List[str],
        language: Optional[str]
    ) -> List[Dict]:
        """
        Retrieve for multiple queries and merge using RRF.

        Args:
            queries: List of query variations.
            language: Language filter.

        Returns:
            Merged and sorted candidates.
        """
        if not queries:
            return []

        if len(queries) == 1:
            return self.hybrid_search(queries[0], language)

        all_results: Dict[str, Dict] = {}
        K = self.config.retrieval.rrf_constant

        # Retrieve for each query
        for q in queries:
            results = self.hybrid_search(
                q,
                language,
                top_k_vec=self.config.retrieval.top_k_chroma // 2,
                top_k_bm25=self.config.retrieval.top_k_bm25 // 2,
            )

            # Apply RRF across queries
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

        # Combine scores: boost chunks appearing in multiple queries
        for data in all_results.values():
            multi_query_bonus = data.get("multi_query_rrf", 0) * 0.3
            data["hybrid_score"] = data.get("hybrid_score", 0) + multi_query_bonus
            data["score"] = data["hybrid_score"]

        # Sort by final score
        results = list(all_results.values())
        results.sort(key=lambda x: x["hybrid_score"], reverse=True)

        print(f"[Retrieval] Multi-query: {len(queries)} queries, {len(results)} unique chunks")
        return results

    def search_only(
        self,
        query: str,
        top_k: int = 10,
        language: Optional[str] = None
    ) -> List[Dict]:
        """
        Simple search without expansion or HyDE (for debugging).

        Args:
            query: Search query.
            top_k: Number of results.
            language: Optional language filter.

        Returns:
            List of search results.
        """
        return self.hybrid_search(
            query,
            language,
            top_k_vec=top_k,
            top_k_bm25=top_k // 2
        )[:top_k]
