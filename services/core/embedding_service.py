"""
Embedding service using Ollama with caching and retry logic.

This service handles text embedding with:
- SHA1-keyed caching for efficiency
- Exponential backoff retry for reliability
- L2 normalization for consistent similarity computation
"""

from typing import Dict, Optional
import hashlib
import time

import numpy as np
import requests

from services.config.settings import OllamaConfig


class EmbeddingService:
    """Handles text embedding with caching and retry logic."""

    def __init__(
        self,
        config: OllamaConfig,
        cache: Optional[Dict[str, np.ndarray]] = None
    ):
        """
        Initialize embedding service.

        Args:
            config: Ollama configuration with host, model, dimensions.
            cache: Optional shared cache dictionary. If None, creates internal cache.
        """
        self.config = config
        self._cache = cache if cache is not None else {}
        self._url = f"{config.host}/api/embeddings"

    def embed(self, text: str, use_cache: bool = True) -> np.ndarray:
        """
        Embed text using Ollama with optional caching.

        Args:
            text: Text to embed.
            use_cache: Whether to use/update cache.

        Returns:
            Normalized embedding vector of shape (embed_dim,).

        Raises:
            RuntimeError: If embedding fails after max retries.
        """
        cache_key = hashlib.sha1(text.encode("utf-8", errors="ignore")).hexdigest()

        if use_cache and cache_key in self._cache:
            return self._cache[cache_key]

        vector = self._embed_with_retry(text)

        if use_cache:
            self._cache[cache_key] = vector

        return vector

    def embed_batch(
        self,
        texts: list,
        use_cache: bool = True
    ) -> list:
        """
        Embed multiple texts.

        Args:
            texts: List of texts to embed.
            use_cache: Whether to use/update cache.

        Returns:
            List of embedding vectors.
        """
        return [self.embed(text, use_cache) for text in texts]

    def _embed_with_retry(self, text: str) -> np.ndarray:
        """
        Embed with exponential backoff retry.

        Args:
            text: Text to embed.

        Returns:
            Normalized embedding vector.

        Raises:
            RuntimeError: If embedding fails after max retries.
        """
        last_error = None

        for attempt in range(self.config.max_retries):
            try:
                response = requests.post(
                    self._url,
                    json={"model": self.config.embed_model, "prompt": text},
                    timeout=self.config.timeout,
                )
                response.raise_for_status()

                embedding = response.json().get("embedding")
                if embedding is None:
                    raise RuntimeError("No embedding in response")

                vector = np.array(embedding, dtype=np.float32)

                if vector.shape[0] != self.config.embed_dim:
                    raise RuntimeError(
                        f"Unexpected embedding dimension {vector.shape[0]}, "
                        f"expected {self.config.embed_dim}"
                    )

                # L2 normalize
                norm = np.linalg.norm(vector)
                if norm > 0:
                    vector = vector / norm

                return vector

            except Exception as e:
                last_error = e
                if attempt < self.config.max_retries - 1:
                    # Exponential backoff
                    sleep_time = (attempt + 1) * 1.0
                    time.sleep(sleep_time)

        raise RuntimeError(
            f"Embedding failed after {self.config.max_retries} attempts: {last_error}"
        )

    @staticmethod
    def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
        """
        Compute cosine similarity between two vectors.

        Args:
            a: First vector.
            b: Second vector.

        Returns:
            Cosine similarity score in [-1, 1].
        """
        numerator = float(np.dot(a, b))
        denominator = float(np.linalg.norm(a) * np.linalg.norm(b) + 1e-12)
        return numerator / denominator

    def clear_cache(self) -> None:
        """Clear the embedding cache."""
        self._cache.clear()

    @property
    def cache_size(self) -> int:
        """Return current cache size."""
        return len(self._cache)

    def get_cache_stats(self) -> Dict:
        """Return cache statistics."""
        return {
            "size": len(self._cache),
            "memory_mb": sum(
                v.nbytes for v in self._cache.values()
            ) / (1024 * 1024) if self._cache else 0
        }
