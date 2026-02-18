"""
Vector store service for ChromaDB operations.

This service handles ChromaDB operations with:
- Lazy collection loading
- Vector similarity search with optional filtering
- Embedding retrieval by ID
"""

from typing import Dict, List, Optional, Any

import numpy as np
import chromadb

from services.config.settings import ChromaConfig


class VectorStoreService:
    """Handles ChromaDB vector operations."""

    def __init__(self, config: ChromaConfig):
        """
        Initialize vector store service.

        Args:
            config: ChromaDB configuration with directory and collection name.
        """
        self.config = config
        self._client: Optional[chromadb.PersistentClient] = None
        self._collection = None

    def get_collection(self):
        """
        Get or create ChromaDB collection.

        Returns:
            ChromaDB collection instance.

        Raises:
            Exception: If collection cannot be loaded.
        """
        if self._collection is None:
            self._client = chromadb.PersistentClient(path=self.config.chroma_dir)
            self._collection = self._client.get_collection(self.config.collection_name)
            print(f"[VectorStore] Loaded collection '{self.config.collection_name}' "
                  f"with {self._collection.count()} documents")
        return self._collection

    def search(
        self,
        query_embedding: np.ndarray,
        top_k: int = 64,
        where_filter: Optional[Dict] = None,
        include_embeddings: bool = False,
    ) -> List[Dict]:
        """
        Perform vector similarity search.

        Args:
            query_embedding: Query vector for similarity search.
            top_k: Number of results to return.
            where_filter: Optional metadata filter (e.g., {"doc_lang": "tr"}).
            include_embeddings: Whether to include embeddings in results.

        Returns:
            List of result dicts with chunk_id, score, text, meta, source.
        """
        collection = self.get_collection()

        include = ["documents", "metadatas", "distances"]
        if include_embeddings:
            include.append("embeddings")

        result = collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=top_k,
            where=where_filter if where_filter else None,
            include=include,
        )

        ids = result.get("ids", [[]])[0]
        docs = result.get("documents", [[]])[0]
        metas = result.get("metadatas", [[]])[0]
        distances = result.get("distances", [[]])[0]
        embeddings = result.get("embeddings", [[]])[0] if include_embeddings else [None] * len(ids)

        results = []
        for i, (cid, doc, meta, dist) in enumerate(zip(ids, docs, metas, distances)):
            # Convert distance to similarity (ChromaDB uses L2 by default)
            similarity = 1.0 - float(dist)

            result_dict = {
                "chunk_id": cid,
                "score": similarity,
                "vec_score": similarity,
                "text": doc,
                "meta": meta or {},
                "source": "chroma",
            }

            if include_embeddings and embeddings[i] is not None:
                result_dict["embedding"] = np.array(embeddings[i], dtype=np.float32)

            results.append(result_dict)

        return results

    def fetch_embeddings(self, ids: List[str]) -> Dict[str, np.ndarray]:
        """
        Fetch embeddings for given chunk IDs.

        Args:
            ids: List of chunk IDs to fetch.

        Returns:
            Dict mapping chunk_id to embedding vector.
        """
        if not ids:
            return {}

        collection = self.get_collection()

        try:
            result = collection.get(ids=ids, include=["embeddings"])

            output = {}
            fetched_ids = result.get("ids", [])
            fetched_embeddings = result.get("embeddings", [])

            for cid, emb in zip(fetched_ids, fetched_embeddings):
                if emb is not None:
                    output[cid] = np.array(emb, dtype=np.float32)

            return output

        except Exception as e:
            print(f"[VectorStore] Error fetching embeddings: {e}")
            return {}

    def get_by_ids(
        self,
        ids: List[str],
        include_embeddings: bool = False
    ) -> List[Dict]:
        """
        Get documents by their IDs.

        Args:
            ids: List of chunk IDs.
            include_embeddings: Whether to include embeddings.

        Returns:
            List of document dicts.
        """
        if not ids:
            return []

        collection = self.get_collection()

        include = ["documents", "metadatas"]
        if include_embeddings:
            include.append("embeddings")

        result = collection.get(ids=ids, include=include)

        fetched_ids = result.get("ids", [])
        docs = result.get("documents", [])
        metas = result.get("metadatas", [])
        embeddings = result.get("embeddings", []) if include_embeddings else [None] * len(fetched_ids)

        results = []
        for cid, doc, meta, emb in zip(fetched_ids, docs, metas, embeddings):
            result_dict = {
                "chunk_id": cid,
                "text": doc,
                "meta": meta or {},
            }
            if include_embeddings and emb is not None:
                result_dict["embedding"] = np.array(emb, dtype=np.float32)
            results.append(result_dict)

        return results

    @property
    def count(self) -> int:
        """Return total document count in collection."""
        try:
            return self.get_collection().count()
        except Exception:
            return 0

    @property
    def is_loaded(self) -> bool:
        """Check if collection is loaded."""
        return self._collection is not None

    def peek(self, n: int = 5) -> List[Dict]:
        """
        Peek at first n documents in collection.

        Args:
            n: Number of documents to return.

        Returns:
            List of document dicts.
        """
        collection = self.get_collection()
        result = collection.peek(limit=n)

        ids = result.get("ids", [])
        docs = result.get("documents", [])
        metas = result.get("metadatas", [])

        return [
            {"chunk_id": cid, "text": doc, "meta": meta}
            for cid, doc, meta in zip(ids, docs, metas)
        ]
