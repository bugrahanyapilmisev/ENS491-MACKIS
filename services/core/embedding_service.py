"""
Embedding service supporting both Ollama (local) and OpenRouter (cloud) backends.

Features:
- SHA1-keyed caching for efficiency
- Exponential backoff retry for reliability
- L2 normalization for consistent similarity computation
- OpenRouter support: Qwen3-Embedding-8B (best multilingual open-source)
- Instruction-aware embedding: separate prefixes for queries vs documents
- Batch embedding support for OpenRouter (multiple texts per API call)
"""

from typing import Dict, List, Optional
import hashlib
import time

import numpy as np
import requests

from services.config.settings import OllamaConfig


class EmbeddingService:
    """Handles text embedding with caching and retry logic.

    Supports two backends:
      - "ollama"     : local Ollama server (e.g. bge-m3)
      - "openrouter" : OpenRouter API (e.g. qwen/qwen3-embedding-8b)

    When OpenRouter is used, instruction-aware prefixes are applied:
      - Queries get  ``config.embed_instruction_query`` prefix
      - Documents get ``config.embed_instruction_doc`` prefix
    """

    OPENROUTER_EMBED_URL = "https://openrouter.ai/api/v1/embeddings"

    def __init__(
        self,
        config: OllamaConfig,
        cache: Optional[Dict[str, np.ndarray]] = None
    ):
        """
        Initialize embedding service.

        Args:
            config: Ollama/OpenRouter configuration.
            cache: Optional shared cache dictionary. If None, creates internal cache.
        """
        self.config = config
        self._cache = cache if cache is not None else {}

        # Determine backend
        self._use_openrouter = (
            config.embed_provider == "openrouter"
            and bool(config.openrouter_api_key)
        )

        if self._use_openrouter:
            self._url = self.OPENROUTER_EMBED_URL
            self._headers = {
                "Authorization": f"Bearer {config.openrouter_api_key}",
                "HTTP-Referer": "http://localhost:3000",
                "X-Title": "MACKIS RAG",
                "Content-Type": "application/json",
            }
            print(
                f"[EmbeddingService] Backend: OpenRouter "
                f"(model={config.embed_model_openrouter})"
            )
        else:
            self._url = f"{config.host}/api/embeddings"
            self._headers = {}
            print(
                f"[EmbeddingService] Backend: Ollama "
                f"(model={config.embed_model})"
            )

    # ─────────────────────────────────────────────────────────────────────────
    # Public API
    # ─────────────────────────────────────────────────────────────────────────

    def embed(
        self,
        text: str,
        use_cache: bool = True,
        is_query: bool = False,
    ) -> np.ndarray:
        """
        Embed a single text with optional caching.

        Args:
            text: Text to embed.
            use_cache: Whether to use/update cache.
            is_query: If True, applies query instruction prefix (OpenRouter only).
                      If False, applies document instruction prefix.

        Returns:
            Normalized embedding vector of shape (embed_dim,).
        """
        prefixed = self._apply_instruction(text, is_query)
        cache_key = hashlib.sha1(prefixed.encode("utf-8", errors="ignore")).hexdigest()

        if use_cache and cache_key in self._cache:
            return self._cache[cache_key]

        vector = self._embed_one_with_retry(prefixed)

        if use_cache:
            self._cache[cache_key] = vector

        return vector

    def embed_batch(
        self,
        texts: List[str],
        use_cache: bool = True,
        is_query: bool = False,
        batch_size: int = 64,
    ) -> List[np.ndarray]:
        """
        Embed multiple texts efficiently.

        For OpenRouter: sends batches of up to ``batch_size`` texts per API call.
        For Ollama: falls back to sequential embedding.

        Args:
            texts: List of texts to embed.
            use_cache: Whether to use/update cache.
            is_query: Query vs document mode (affects instruction prefix).
            batch_size: Max texts per OpenRouter API call (max 2048 per OR docs).

        Returns:
            List of normalized embedding vectors.
        """
        if self._use_openrouter:
            return self._embed_batch_openrouter(texts, use_cache, is_query, batch_size)
        else:
            return [self.embed(t, use_cache, is_query) for t in texts]

    # ─────────────────────────────────────────────────────────────────────────
    # Instruction-aware prefix
    # ─────────────────────────────────────────────────────────────────────────

    def _apply_instruction(self, text: str, is_query: bool) -> str:
        """Prepend instruction prefix for instruction-aware models (Qwen3-Embedding)."""
        if not self._use_openrouter:
            return text  # Ollama models (BGE-M3) don't use instruction prefixes
        if is_query:
            return self.config.embed_instruction_query + text
        else:
            return self.config.embed_instruction_doc + text

    # ─────────────────────────────────────────────────────────────────────────
    # Single-text embedding with retry
    # ─────────────────────────────────────────────────────────────────────────

    def _embed_one_with_retry(self, text: str) -> np.ndarray:
        """Embed a single (already-prefixed) text with exponential backoff."""
        last_error = None

        for attempt in range(max(self.config.max_retries, 3)):
            try:
                if self._use_openrouter:
                    return self._embed_openrouter_single(text)
                else:
                    return self._embed_ollama(text)
            except Exception as e:
                last_error = e
                if attempt < self.config.max_retries - 1:
                    sleep_time = (attempt + 1) * 2.0
                    time.sleep(sleep_time)

        raise RuntimeError(
            f"Embedding failed after {self.config.max_retries} attempts: {last_error}"
        )

    # ─────────────────────────────────────────────────────────────────────────
    # Ollama backend
    # ─────────────────────────────────────────────────────────────────────────

    def _embed_ollama(self, text: str) -> np.ndarray:
        """Call local Ollama embedding API."""
        response = requests.post(
            self._url,
            json={"model": self.config.embed_model, "prompt": text},
            timeout=self.config.timeout,
        )
        response.raise_for_status()

        embedding = response.json().get("embedding")
        if embedding is None:
            raise RuntimeError("No embedding in Ollama response")

        vector = np.array(embedding, dtype=np.float32)

        if vector.shape[0] != self.config.embed_dim:
            raise RuntimeError(
                f"Unexpected embedding dim {vector.shape[0]}, expected {self.config.embed_dim}"
            )

        return self._normalize(vector)

    # ─────────────────────────────────────────────────────────────────────────
    # OpenRouter backend
    # ─────────────────────────────────────────────────────────────────────────

    def _embed_openrouter_single(self, text: str) -> np.ndarray:
        """Call OpenRouter embedding API for a single text."""
        payload = {
            "model": self.config.embed_model_openrouter,
            "input": text,
            "encoding_format": "float",
        }
        # Qwen3-Embedding supports Matryoshka — optionally request smaller dim
        # We keep the default (full dim from the model, usually 1024 or 4096)

        response = requests.post(
            self._url,
            json=payload,
            headers=self._headers,
            timeout=self.config.timeout,
        )
        response.raise_for_status()

        data = response.json()
        embedding = data.get("data", [{}])[0].get("embedding")
        if embedding is None:
            raise RuntimeError(f"No embedding in OpenRouter response: {data}")

        vector = np.array(embedding, dtype=np.float32)
        return self._normalize(vector)

    def _embed_batch_openrouter(
        self,
        texts: List[str],
        use_cache: bool,
        is_query: bool,
        batch_size: int,
    ) -> List[np.ndarray]:
        """
        Batch embed via OpenRouter.  Splits into chunks of ``batch_size``,
        uses cache for already-seen texts, sends only uncached texts.
        """
        prefixed = [self._apply_instruction(t, is_query) for t in texts]
        results: List[Optional[np.ndarray]] = [None] * len(prefixed)
        uncached_idx: List[int] = []
        uncached_texts: List[str] = []

        # Check cache first
        for i, p in enumerate(prefixed):
            key = hashlib.sha1(p.encode("utf-8", errors="ignore")).hexdigest()
            if use_cache and key in self._cache:
                results[i] = self._cache[key]
            else:
                uncached_idx.append(i)
                uncached_texts.append(p)

        # Batch embed uncached
        for chunk_start in range(0, len(uncached_texts), batch_size):
            chunk = uncached_texts[chunk_start: chunk_start + batch_size]
            chunk_indices = uncached_idx[chunk_start: chunk_start + batch_size]

            vecs = self._call_openrouter_batch(chunk)

            for local_i, (global_i, vec) in enumerate(zip(chunk_indices, vecs)):
                results[global_i] = vec
                if use_cache:
                    key = hashlib.sha1(
                        chunk[local_i].encode("utf-8", errors="ignore")
                    ).hexdigest()
                    self._cache[key] = vec

        return results  # type: ignore[return-value]

    def _call_openrouter_batch(self, texts: List[str]) -> List[np.ndarray]:
        """Send a single batch request to OpenRouter and return normalized vectors."""
        last_error = None
        for attempt in range(max(self.config.max_retries, 3)):
            try:
                payload = {
                    "model": self.config.embed_model_openrouter,
                    "input": texts,
                    "encoding_format": "float",
                }
                response = requests.post(
                    self._url,
                    json=payload,
                    headers=self._headers,
                    timeout=max(self.config.timeout, 60),
                )
                response.raise_for_status()
                data = response.json()
                embeddings = data.get("data", [])
                # Sort by index to preserve order
                embeddings.sort(key=lambda x: x.get("index", 0))
                return [
                    self._normalize(np.array(e["embedding"], dtype=np.float32))
                    for e in embeddings
                ]
            except Exception as e:
                last_error = e
                sleep_time = (attempt + 1) * 2.0
                print(f"[EmbeddingService] Batch retry {attempt+1}: {e}")
                time.sleep(sleep_time)

        raise RuntimeError(
            f"OpenRouter batch embedding failed after retries: {last_error}"
        )

    # ─────────────────────────────────────────────────────────────────────────
    # Utilities
    # ─────────────────────────────────────────────────────────────────────────

    @staticmethod
    def _normalize(vector: np.ndarray) -> np.ndarray:
        """L2-normalize a vector."""
        norm = np.linalg.norm(vector)
        if norm > 0:
            vector = vector / norm
        return vector

    @staticmethod
    def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
        """Compute cosine similarity between two vectors."""
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
            "backend": "openrouter" if self._use_openrouter else "ollama",
            "model": (
                self.config.embed_model_openrouter
                if self._use_openrouter
                else self.config.embed_model
            ),
            "memory_mb": (
                sum(v.nbytes for v in self._cache.values()) / (1024 * 1024)
                if self._cache
                else 0
            ),
        }
