"""
Ranking agent for reranking and selection.

This agent handles:
- Cross-encoder reranking with hybrid score combination
- Tag-based filtering and boosting
- Negation penalty application
- MMR (Maximal Marginal Relevance) selection for diversity
"""

from typing import Dict, List, Set, Optional
import re

import numpy as np
from sentence_transformers import CrossEncoder

from services.core.embedding_service import EmbeddingService
from services.core.vector_store import VectorStoreService
from services.config.settings import RAGConfig


class RankingAgent:
    """Handles reranking, tag filtering, and selection."""

    def __init__(
        self,
        embedding_service: EmbeddingService,
        vector_store: VectorStoreService,
        config: RAGConfig
    ):
        """
        Initialize ranking agent.

        Args:
            embedding_service: Service for embeddings.
            vector_store: Service for fetching embeddings.
            config: RAG configuration.
        """
        self.embedding = embedding_service
        self.vector_store = vector_store
        self.config = config
        self._reranker: Optional[CrossEncoder] = None

    def _truncate_text(self, text: str) -> str:
        """Truncate text to max characters to avoid OOM in cross-encoder."""
        max_chars = self.config.reranking.max_text_length
        if len(text) <= max_chars:
            return text
        # Truncate at word boundary
        truncated = text[:max_chars]
        last_space = truncated.rfind(' ')
        if last_space > max_chars * 0.8:
            truncated = truncated[:last_space]
        return truncated + "..."

    def _get_reranker(self) -> CrossEncoder:
        """Lazy-load cross-encoder model."""
        if self._reranker is None:
            print(f"[Ranking] Loading reranker: {self.config.reranking.model_name}")
            self._reranker = CrossEncoder(self.config.reranking.model_name)
        return self._reranker

    def rerank(
        self,
        query: str,
        candidates: List[Dict],
        query_tags: Optional[List[str]] = None,
        negated_terms: Optional[List[str]] = None
    ) -> List[Dict]:
        """
        Full reranking pipeline.

        Steps:
        1. Cross-encoder reranking
        2. Tag-based filtering (if tags provided)
        3. Negation penalty (if negated terms provided)

        Args:
            query: User query.
            candidates: Retrieval candidates.
            query_tags: Semantic tags inferred from query.
            negated_terms: Terms to exclude/penalize.

        Returns:
            Reranked and filtered candidates.
        """
        if not candidates:
            return []

        # 1. Cross-encoder rerank
        reranked = self.cross_encoder_rerank(query, candidates)

        # 2. Tag-based filtering
        if query_tags:
            reranked = self.apply_tag_prior(query_tags, reranked)

        # 3. Negation penalty
        if negated_terms:
            reranked = self.apply_negation_penalty(negated_terms, reranked)

        return reranked

    def cross_encoder_rerank(
        self,
        query: str,
        candidates: List[Dict]
    ) -> List[Dict]:
        """
        Cross-encoder reranking with hybrid score combination.

        Args:
            query: User query.
            candidates: Candidates to rerank.

        Returns:
            Reranked candidates with updated scores.
        """
        max_candidates = self.config.reranking.max_candidates
        batch_size = self.config.reranking.batch_size
        weight_ce = self.config.reranking.weight

        # Take top candidates for reranking
        subset = candidates[:max_candidates].copy()

        if not subset:
            return []

        # Get cross-encoder scores in batches to avoid OOM
        model = self._get_reranker()

        # Truncate text to avoid memory issues (cross-encoder has ~512 token limit)
        pairs = [(query, self._truncate_text(c["text"])) for c in subset]

        # Batch prediction to avoid memory issues
        ce_scores_list = []
        for i in range(0, len(pairs), batch_size):
            batch_pairs = pairs[i:i + batch_size]
            batch_scores = model.predict(batch_pairs)
            ce_scores_list.extend(batch_scores)

        ce_scores = np.array(ce_scores_list)

        # Add scores to candidates
        for i, c in enumerate(subset):
            c["ce_score"] = float(ce_scores[i])

        # Normalize scores
        ce_vals = [c.get("ce_score", 0.0) for c in subset]
        hybrid_vals = [c.get("hybrid_score", c.get("score", 0.0)) for c in subset]

        max_ce = max(ce_vals) if ce_vals else 1.0
        max_hybrid = max(hybrid_vals) if hybrid_vals else 1.0

        if max_ce <= 0:
            max_ce = 1.0
        if max_hybrid <= 0:
            max_hybrid = 1.0

        # Query tokens for title overlap
        q_tokens = set(re.findall(r"\w+", (query or "").lower()))

        for c in subset:
            ce_norm = c.get("ce_score", 0.0) / max_ce
            hybrid_norm = c.get("hybrid_score", c.get("score", 0.0)) / max_hybrid

            # Weighted combination
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
            c["hybrid_score"] = final_score

        # Sort by final score
        subset.sort(key=lambda x: x["final_score"], reverse=True)
        return subset

    def apply_tag_prior(
        self,
        query_tags: List[str],
        candidates: List[Dict],
        boost_positive: Optional[float] = None,
        penalize_negative: Optional[float] = None
    ) -> List[Dict]:
        """
        Boost chunks with matching tags, penalize those without.

        Args:
            query_tags: Semantic tags from query.
            candidates: Candidates to filter.
            boost_positive: Boost factor for matches.
            penalize_negative: Penalty factor for non-matches.

        Returns:
            Re-scored candidates.
        """
        if not candidates or not query_tags:
            return candidates

        boost_positive = boost_positive or self.config.reranking.tag_boost_positive
        penalize_negative = penalize_negative or self.config.reranking.tag_penalize_negative

        # Calculate tag overlap for each candidate
        for c in candidates:
            meta = c.get("meta") or {}
            doc_tags = self._get_meta_tags(meta)
            overlap = self._soft_tag_overlap(query_tags, doc_tags)
            c["_tag_overlap"] = overlap

        max_overlap = max(c.get("_tag_overlap", 0) for c in candidates)
        if max_overlap <= 0:
            return candidates

        # Apply boost/penalty
        for c in candidates:
            base = c.get("hybrid_score", c.get("score", 0.0))
            overlap = c.get("_tag_overlap", 0)

            if overlap > 0:
                factor = 1.0 + boost_positive * min(overlap, 3)
            else:
                factor = 1.0 - penalize_negative

            c["hybrid_score"] = base * factor

        # Re-sort
        candidates.sort(key=lambda x: x.get("hybrid_score", 0.0), reverse=True)
        return candidates

    def apply_negation_penalty(
        self,
        negated_terms: List[str],
        candidates: List[Dict],
        penalty_factor: Optional[float] = None
    ) -> List[Dict]:
        """
        Downweight chunks containing negated terms.

        Args:
            negated_terms: Terms to penalize.
            candidates: Candidates to process.
            penalty_factor: Score multiplier for matches.

        Returns:
            Re-scored candidates.
        """
        if not negated_terms:
            return candidates

        penalty_factor = penalty_factor or self.config.reranking.negation_penalty_factor
        neg_terms = set(t.lower() for t in negated_terms)

        print(f"[Ranking] Applying negation penalty for: {neg_terms}")

        for c in candidates:
            meta = c.get("meta") or {}
            title_blob = " ".join(
                str(meta.get(k, "")) for k in ("title", "section_header")
            ).lower()

            if any(t in title_blob for t in neg_terms):
                old = c.get("hybrid_score", 0.0)
                c["hybrid_score"] = old * penalty_factor
                c["negation_penalized"] = True

        # Re-sort
        candidates.sort(key=lambda x: x.get("hybrid_score", 0.0), reverse=True)
        return candidates

    def mmr_select(
        self,
        candidates: List[Dict],
        query: str,
        k: Optional[int] = None,
        lambda_mmr: Optional[float] = None
    ) -> List[Dict]:
        """
        Maximal Marginal Relevance selection for diversity.

        Balances relevance with diversity by penalizing candidates
        that are too similar to already selected ones.

        Args:
            candidates: Candidates to select from.
            query: Original query for relevance.
            k: Number of candidates to select.
            lambda_mmr: Balance parameter (higher = more relevance).

        Returns:
            Selected diverse candidates.
        """
        if not candidates:
            return []

        k = k or self.config.retrieval.top_k_final_base
        lambda_mmr = lambda_mmr or self.config.reranking.mmr_lambda

        # Get query embedding
        query_vec = self.embedding.embed(query)

        # Fetch embeddings for candidates
        candidate_ids = [c["chunk_id"] for c in candidates]
        doc_embs = self.vector_store.fetch_embeddings(candidate_ids)

        selected: List[Dict] = []
        selected_ids: List[str] = []

        # Pre-compute query similarities
        query_sims = {}
        for c in candidates:
            cid = c["chunk_id"]
            emb = doc_embs.get(cid)
            query_sims[cid] = 0.0 if emb is None else self.embedding.cosine_similarity(query_vec, emb)

        # Greedy selection
        while len(selected) < min(k, len(candidates)):
            best_cand = None
            best_mmr = -1e9

            for c in candidates:
                cid = c["chunk_id"]
                if cid in selected_ids:
                    continue

                # Relevance: combination of hybrid score and query similarity
                rel = 0.5 * c.get("hybrid_score", 0.0) + 0.5 * query_sims.get(cid, 0.0)

                # Redundancy: max similarity to already selected
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
                                sims.append(self.embedding.cosine_similarity(emb_i, emb_j))
                        red = max(sims) if sims else 0.0

                # MMR score
                mmr_score = lambda_mmr * rel - (1.0 - lambda_mmr) * red

                if mmr_score > best_mmr:
                    best_mmr = mmr_score
                    best_cand = c

            if best_cand is None:
                break

            selected.append(best_cand)
            selected_ids.append(best_cand["chunk_id"])

        return selected

    def filter_by_threshold(
        self,
        candidates: List[Dict],
        threshold: Optional[float] = None,
        min_count: int = 4,
        fallback_count: int = 12
    ) -> List[Dict]:
        """
        Filter candidates by score threshold with fallback.

        Args:
            candidates: Candidates to filter.
            threshold: Score threshold.
            min_count: Minimum candidates to keep.
            fallback_count: Number to return if threshold yields too few.

        Returns:
            Filtered candidates.
        """
        threshold = threshold or self.config.reranking.score_threshold

        strong = [c for c in candidates if c.get("hybrid_score", 0) >= threshold]

        if len(strong) < min_count:
            return candidates[:fallback_count]

        return strong

    def document_level_select(
        self,
        candidates: List[Dict],
        max_chunks: Optional[int] = None,
        max_per_doc: int = 4
    ) -> List[Dict]:
        """
        Select chunks with document-level diversity.

        Groups chunks by source document, scores each document by its best
        chunk, and greedily selects top chunks while capping per-document
        representation. This prevents a single document from dominating
        the context and ensures diversity across sources.

        Args:
            candidates: Ranked candidates to select from.
            max_chunks: Total chunks to return (defaults to max_docs_context).
            max_per_doc: Maximum chunks from any single document.

        Returns:
            Selected candidates with document-level diversity.
        """
        if not candidates:
            return []

        max_chunks = max_chunks or self.config.retrieval.max_docs_context

        # Group by source document
        doc_groups: Dict[str, List[Dict]] = {}
        for c in candidates:
            meta = c.get("meta") or {}
            path = meta.get("source_path") or meta.get("doc_path", "unknown")
            if path not in doc_groups:
                doc_groups[path] = []
            doc_groups[path].append(c)

        # Score each document by its best chunk score
        doc_best_scores = {}
        for path, chunks in doc_groups.items():
            doc_best_scores[path] = max(
                c.get("hybrid_score", 0) for c in chunks
            )

        # Sort documents by best score (descending)
        sorted_docs = sorted(
            doc_best_scores.items(), key=lambda x: x[1], reverse=True
        )

        # Greedily select chunks with per-document cap
        selected: List[Dict] = []
        for path, _ in sorted_docs:
            if len(selected) >= max_chunks:
                break
            doc_chunks = doc_groups[path][:max_per_doc]
            remaining_slots = max_chunks - len(selected)
            selected.extend(doc_chunks[:remaining_slots])

        print(f"[Ranking] Document-level select: {len(candidates)} candidates -> "
              f"{len(selected)} chunks from {min(len(sorted_docs), max_chunks)} docs")

        return selected

    def _get_meta_tags(self, meta: Dict) -> List[str]:
        """Extract and normalize tags from metadata."""
        tags_val = meta.get("tags", [])
        parts = []

        if isinstance(tags_val, list):
            parts = [p for p in tags_val if isinstance(p, str)]
        elif isinstance(tags_val, str):
            parts = tags_val.split(",")

        return [p.strip().lower() for p in parts if p.strip()]

    def _soft_tag_overlap(
        self,
        query_tags: List[str],
        doc_tags: List[str]
    ) -> int:
        """Fuzzy token-level overlap between query and doc tags."""
        q_tokens = set()
        for qt in query_tags:
            q_tokens |= self._tokenize_tag(qt)

        d_tokens = set()
        for dt in doc_tags:
            d_tokens |= self._tokenize_tag(dt)

        if not q_tokens or not d_tokens:
            return 0

        return len(q_tokens & d_tokens)

    def _tokenize_tag(self, tag: str) -> Set[str]:
        """Tokenize a tag for fuzzy matching."""
        if not isinstance(tag, str):
            return set()

        t = re.sub(r"[^a-z0-9_]+", "_", tag.lower())
        parts = [p for p in t.split("_") if p]

        # Generic tokens to filter out
        generic_tokens = {
            "program", "prosedur", "yonerge", "basvuru",
            "ogrenci", "lisans", "genel", "bilgi",
            "form", "guide", "policy", "procedure", "student",
        }

        return {p for p in parts if len(p) > 2 and p not in generic_tokens}
