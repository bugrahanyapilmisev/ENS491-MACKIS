"""
Data loading service for chunks and summaries.

This service handles loading from parquet files with:
- Lazy loading with caching
- Chunk retrieval by ID or document path
- Document summary retrieval
"""

from typing import Dict, List, Optional
import os

import pandas as pd

from services.config.settings import PathConfig


class DataLoaderService:
    """Handles loading of chunk data and document summaries."""

    def __init__(self, config: PathConfig):
        """
        Initialize data loader service.

        Args:
            config: Path configuration with parquet file paths.
        """
        self.config = config
        self._chunk_df: Optional[pd.DataFrame] = None
        self._doc_summary_df: Optional[pd.DataFrame] = None

    def load_chunk_df(self) -> pd.DataFrame:
        """
        Load chunk dataframe from parquet.

        Returns:
            DataFrame with all chunks.

        Raises:
            RuntimeError: If chunk parquet file not found.
        """
        if self._chunk_df is None:
            if not os.path.exists(self.config.chunk_parquet):
                raise RuntimeError(
                    f"Chunk parquet not found: {self.config.chunk_parquet}"
                )
            self._chunk_df = pd.read_parquet(self.config.chunk_parquet)
            print(f"[DataLoader] Loaded {len(self._chunk_df)} chunks")

        return self._chunk_df

    def load_doc_summaries(self) -> pd.DataFrame:
        """
        Load document summaries from parquet.

        Returns:
            DataFrame with document summaries, or empty DataFrame if not found.
        """
        if self._doc_summary_df is None:
            if not os.path.exists(self.config.doc_summary_parquet):
                print(f"[DataLoader] Doc summary parquet not found: "
                      f"{self.config.doc_summary_parquet}")
                self._doc_summary_df = pd.DataFrame()
            else:
                self._doc_summary_df = pd.read_parquet(self.config.doc_summary_parquet)
                print(f"[DataLoader] Loaded {len(self._doc_summary_df)} document summaries")

        return self._doc_summary_df

    def get_doc_summary(self, source_path: str) -> str:
        """
        Get summary for a document by its source path.

        Args:
            source_path: Path to the source document.

        Returns:
            Summary string, or empty string if not found.
        """
        df = self.load_doc_summaries()

        if df.empty or "source_path" not in df.columns:
            return ""

        match = df[df["source_path"] == source_path]
        if match.empty:
            return ""

        return str(match.iloc[0].get("summary", ""))

    def get_all_chunks_for_doc(self, source_path: str) -> List[Dict]:
        """
        Get all chunks for a document, sorted by chunk_index.

        Args:
            source_path: Path to the source document.

        Returns:
            List of chunk dicts with chunk_id, text, meta, source.
        """
        df = self.load_chunk_df()
        sub = df[df["source_path"] == source_path].copy()

        if sub.empty:
            return []

        # Sort by chunk_index if available
        if "chunk_index" in sub.columns:
            sub = sub.sort_values("chunk_index")

        chunks = []
        for _, row in sub.iterrows():
            meta = {
                "source_path": row.get("source_path", ""),
                "json_path": row.get("json_path", ""),
                "title": row.get("title", ""),
                "section_header": row.get("section_header", ""),
                "doc_lang": row.get("doc_lang", ""),
                "doc_type": row.get("doc_type", ""),
                "tags": row.get("tags", ""),
                "procedure_code": row.get("procedure_code", ""),
            }
            chunks.append({
                "chunk_id": row["chunk_id"],
                "text": row.get("content", ""),
                "meta": meta,
                "source": "doc_full",
            })

        return chunks

    def get_chunk_by_id(self, chunk_id: str) -> Optional[Dict]:
        """
        Get a single chunk by its ID.

        Args:
            chunk_id: Unique chunk identifier.

        Returns:
            Chunk dict, or None if not found.
        """
        df = self.load_chunk_df()
        match = df[df["chunk_id"] == chunk_id]

        if match.empty:
            return None

        row = match.iloc[0]
        return {
            "chunk_id": row["chunk_id"],
            "text": row.get("content", ""),
            "meta": {
                "source_path": row.get("source_path", ""),
                "title": row.get("title", ""),
                "section_header": row.get("section_header", ""),
                "doc_lang": row.get("doc_lang", ""),
                "tags": row.get("tags", ""),
                "procedure_code": row.get("procedure_code", ""),
            }
        }

    def get_chunks_by_ids(self, chunk_ids: List[str]) -> List[Dict]:
        """
        Get multiple chunks by their IDs.

        Args:
            chunk_ids: List of chunk IDs.

        Returns:
            List of chunk dicts (preserves order).
        """
        if not chunk_ids:
            return []

        df = self.load_chunk_df()
        id_set = set(chunk_ids)

        # Filter and preserve order
        results = []
        id_to_row = {}

        for _, row in df[df["chunk_id"].isin(id_set)].iterrows():
            id_to_row[row["chunk_id"]] = row

        for cid in chunk_ids:
            if cid in id_to_row:
                row = id_to_row[cid]
                results.append({
                    "chunk_id": row["chunk_id"],
                    "text": row.get("content", ""),
                    "meta": {
                        "source_path": row.get("source_path", ""),
                        "title": row.get("title", ""),
                        "section_header": row.get("section_header", ""),
                        "doc_lang": row.get("doc_lang", ""),
                        "tags": row.get("tags", ""),
                    }
                })

        return results

    def get_unique_documents(self) -> List[str]:
        """
        Get list of unique document source paths.

        Returns:
            List of unique source_path values.
        """
        df = self.load_chunk_df()
        if "source_path" not in df.columns:
            return []
        return df["source_path"].unique().tolist()

    def get_stats(self) -> Dict:
        """
        Get data loader statistics.

        Returns:
            Dict with statistics about loaded data.
        """
        chunk_df = self.load_chunk_df() if self._chunk_df is not None else None
        summary_df = self.load_doc_summaries() if self._doc_summary_df is not None else None

        return {
            "chunks_loaded": self._chunk_df is not None,
            "summaries_loaded": self._doc_summary_df is not None,
            "chunk_count": len(chunk_df) if chunk_df is not None else 0,
            "summary_count": len(summary_df) if summary_df is not None else 0,
            "unique_documents": len(chunk_df["source_path"].unique()) if chunk_df is not None and "source_path" in chunk_df.columns else 0,
        }

    def reload(self) -> None:
        """Force reload of all data."""
        self._chunk_df = None
        self._doc_summary_df = None
        self.load_chunk_df()
        self.load_doc_summaries()
