"""
BM25 service for lexical search.

This service handles BM25 operations with:
- Lazy index loading from pickle
- Turkish diacritics support in tokenization
- Keyword-based search with scoring
"""

from typing import List, Tuple, Optional
import os
import pickle
import re

import numpy as np

from services.config.settings import PathConfig


# Turkish diacritics for tokenization
TR_DIACRITICS = "çğıöşüÇĞİÖŞÜ"


class BM25Service:
    """Handles BM25 lexical search operations."""

    def __init__(self, config: PathConfig):
        """
        Initialize BM25 service.

        Args:
            config: Path configuration with bm25_index_path.
        """
        self.config = config
        self._bm25 = None
        self._chunk_ids: List[str] = []
        self._loaded = False

    def load_index(self) -> bool:
        """
        Load pre-built BM25 index from pickle file.

        Returns:
            True if index loaded successfully, False otherwise.
        """
        if self._loaded:
            return True

        if not os.path.exists(self.config.bm25_index_path):
            print(f"[BM25Service] Index not found: {self.config.bm25_index_path}")
            return False

        try:
            with open(self.config.bm25_index_path, "rb") as f:
                data = pickle.load(f)

            self._bm25 = data.get("bm25")
            self._chunk_ids = data.get("chunk_ids", [])
            self._loaded = True

            print(f"[BM25Service] Loaded index with {len(self._chunk_ids)} documents")
            return True

        except Exception as e:
            print(f"[BM25Service] Error loading index: {e}")
            return False

    def tokenize(self, text: str) -> List[str]:
        """
        Tokenize text for BM25 search.

        Handles Turkish diacritics and filters short tokens.

        Args:
            text: Text to tokenize.

        Returns:
            List of tokens.
        """
        text = text.lower()
        # Match alphanumeric + Turkish diacritics
        pattern = r"[a-z" + TR_DIACRITICS.lower() + r"0-9]+"
        tokens = re.findall(pattern, text)
        # Filter tokens with length > 2
        return [t for t in tokens if len(t) > 2]

    def search(
        self,
        query: str,
        top_k: int = 32
    ) -> List[Tuple[str, float]]:
        """
        Search using BM25 index.

        Args:
            query: Search query string.
            top_k: Number of results to return.

        Returns:
            List of (chunk_id, score) tuples, sorted by score descending.
        """
        if not self.load_index():
            return []

        if self._bm25 is None or not self._chunk_ids:
            return []

        query_tokens = self.tokenize(query)
        if not query_tokens:
            return []

        # Get BM25 scores
        scores = self._bm25.get_scores(query_tokens)

        # Get top-k indices
        top_indices = np.argsort(scores)[::-1][:top_k]

        # Build results with positive scores only
        results = []
        for idx in top_indices:
            if scores[idx] > 0:
                results.append((self._chunk_ids[idx], float(scores[idx])))

        return results

    def search_with_details(
        self,
        query: str,
        top_k: int = 32
    ) -> List[dict]:
        """
        Search with detailed results including tokens.

        Args:
            query: Search query string.
            top_k: Number of results to return.

        Returns:
            List of result dicts with chunk_id, score, and matched tokens.
        """
        if not self.load_index():
            return []

        if self._bm25 is None or not self._chunk_ids:
            return []

        query_tokens = self.tokenize(query)
        if not query_tokens:
            return []

        scores = self._bm25.get_scores(query_tokens)
        top_indices = np.argsort(scores)[::-1][:top_k]

        results = []
        for idx in top_indices:
            if scores[idx] > 0:
                results.append({
                    "chunk_id": self._chunk_ids[idx],
                    "bm25_score": float(scores[idx]),
                    "query_tokens": query_tokens,
                })

        return results

    @property
    def is_loaded(self) -> bool:
        """Check if index is loaded."""
        return self._loaded

    @property
    def document_count(self) -> int:
        """Return indexed document count."""
        return len(self._chunk_ids)

    def get_stats(self) -> dict:
        """
        Get BM25 index statistics.

        Returns:
            Dict with index statistics.
        """
        return {
            "loaded": self._loaded,
            "document_count": len(self._chunk_ids),
            "index_path": self.config.bm25_index_path,
            "index_exists": os.path.exists(self.config.bm25_index_path),
        }
