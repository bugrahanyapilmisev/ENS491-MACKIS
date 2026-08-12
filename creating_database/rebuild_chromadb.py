"""
rebuild_chromadb.py — Rebuild ChromaDB from parquet checkpoints.

Recreates the HNSW index from pre-computed vectors (no API calls needed).
Takes ~2-5 minutes for 17,788 documents.

Usage:
    python rebuild_chromadb.py
"""

import os
import sys
import shutil
import time
import numpy as np
import pandas as pd
import chromadb

# ── Config ──────────────────────────────────────────────────────────────────
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
CHROMA_DIR = os.path.join(CURRENT_DIR, "chroma_db_v2")
COLLECTION_NAME = "mysu_v3_qwen3"

# Try local checkpoints first, then fallback to external backup
LOCAL_CHECKPOINTS = os.path.join(CURRENT_DIR, "checkpoints_v2")
BACKUP_CHECKPOINTS = r"C:\bitirme3\aaa\checkpoints_v2"

def _find_file(filename):
    """Find a checkpoint file, checking local dir first then backup."""
    local = os.path.join(LOCAL_CHECKPOINTS, filename)
    if os.path.exists(local):
        return local
    backup = os.path.join(BACKUP_CHECKPOINTS, filename)
    if os.path.exists(backup):
        return backup
    return local  # Will fail with a clear error message

CHUNKS_PATH = _find_file("chunks_v3.parquet")
VECTORS_PATH = _find_file("vectors_v3.parquet")

BATCH_SIZE = 500  # ChromaDB batch insert limit


def main():
    print("=" * 60)
    print("[REBUILD] ChromaDB from parquet checkpoints")
    print("=" * 60)

    # 1. Verify source files exist
    for path, name in [(CHUNKS_PATH, "chunks"), (VECTORS_PATH, "vectors")]:
        if not os.path.exists(path):
            print(f"[ERROR] {name} file not found: {path}")
            sys.exit(1)

    # 2. Load chunks
    print(f"\n[1/5] Loading chunks...")
    chunks_df = pd.read_parquet(CHUNKS_PATH)
    print(f"  Loaded {len(chunks_df)} chunks")

    # 3. Load vectors
    print(f"[2/5] Loading pre-computed vectors...")
    vectors_df = pd.read_parquet(VECTORS_PATH)
    print(f"  Loaded {len(vectors_df)} vectors")
    print(f"  Columns: {list(vectors_df.columns)}")

    # Check vector dimensions
    sample_vec = vectors_df.iloc[0]["vector"]
    if isinstance(sample_vec, (list, np.ndarray)):
        dim = len(sample_vec)
    else:
        print(f"[ERROR] Unexpected vector type: {type(sample_vec)}")
        sys.exit(1)
    print(f"  Vector dimension: {dim}")

    # 4. Delete corrupted ChromaDB
    print(f"\n[3/5] Removing corrupted ChromaDB at {CHROMA_DIR}...")
    if os.path.exists(CHROMA_DIR):
        shutil.rmtree(CHROMA_DIR)
        print(f"  Deleted old database")
    os.makedirs(CHROMA_DIR, exist_ok=True)

    # 5. Create fresh ChromaDB
    print(f"[4/5] Creating fresh ChromaDB collection '{COLLECTION_NAME}'...")
    client = chromadb.PersistentClient(path=CHROMA_DIR)
    collection = client.create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"}  # Use cosine similarity
    )
    print(f"  Collection created")

    # 6. Prepare data from vectors_df (it has id, vector, document, metadata)
    print(f"\n[5/5] Inserting {len(vectors_df)} documents in batches of {BATCH_SIZE}...")

    all_ids = []
    all_documents = []
    all_metadatas = []
    all_embeddings = []
    skipped = 0

    for _, row in vectors_df.iterrows():
        chunk_id = str(row.get("id", ""))
        content = str(row.get("document", ""))
        vec = row.get("vector")
        meta_raw = row.get("metadata")

        if not chunk_id or vec is None:
            skipped += 1
            continue

        # Parse metadata
        meta = {}
        if isinstance(meta_raw, dict):
            meta = meta_raw
        elif isinstance(meta_raw, str):
            try:
                import json
                meta = json.loads(meta_raw)
            except Exception:
                meta = {}

        # ChromaDB only accepts str, int, float, bool in metadata
        clean_meta = {}
        for k, v in meta.items():
            if v is None:
                continue
            if isinstance(v, (str, int, float, bool)):
                clean_meta[k] = v
            elif isinstance(v, list):
                clean_meta[k] = str(v)
            else:
                clean_meta[k] = str(v)

        # ChromaDB 1.3.4+ rejects np.float32 — convert to plain Python float
        if isinstance(vec, np.ndarray):
            embedding = vec.astype(float).tolist()
        elif isinstance(vec, list):
            embedding = [float(x) for x in vec]
        else:
            embedding = vec

        all_ids.append(chunk_id)
        all_documents.append(content)
        all_metadatas.append(clean_meta)
        all_embeddings.append(embedding)

    if skipped:
        print(f"  Skipped {skipped} chunks (no matching vector)")

    # Batch insert
    total = len(all_ids)
    start_time = time.time()

    for i in range(0, total, BATCH_SIZE):
        end = min(i + BATCH_SIZE, total)
        collection.add(
            ids=all_ids[i:end],
            documents=all_documents[i:end],
            metadatas=all_metadatas[i:end],
            embeddings=all_embeddings[i:end],
        )
        elapsed = time.time() - start_time
        rate = (i + BATCH_SIZE) / elapsed if elapsed > 0 else 0
        eta = (total - end) / rate if rate > 0 else 0
        print(f"  Inserted {end}/{total} ({100*end/total:.0f}%) | ETA: {eta:.0f}s")

    elapsed = time.time() - start_time

    # 7. Verify
    final_count = collection.count()
    print(f"\n{'=' * 60}")
    print(f"[DONE] ChromaDB rebuilt successfully!")
    print(f"  Collection: {COLLECTION_NAME}")
    print(f"  Documents:  {final_count}")
    print(f"  Dimension:  {dim}")
    print(f"  Time:       {elapsed:.1f}s")
    print(f"  Location:   {CHROMA_DIR}")
    print(f"{'=' * 60}")

    # Quick sanity check
    print(f"\n[VERIFY] Testing peek and query...")
    try:
        peek = collection.peek(limit=2)
        print(f"  Peek OK: {peek['ids']}")
    except Exception as e:
        print(f"  Peek FAILED: {e}")

    try:
        dummy = [0.0] * dim
        result = collection.query(query_embeddings=[dummy], n_results=3)
        print(f"  Query OK: returned {len(result['ids'][0])} results")
    except Exception as e:
        print(f"  Query FAILED: {e}")


if __name__ == "__main__":
    main()
