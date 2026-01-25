
import os
import json
import numpy as np
import pyarrow.parquet as pq
import chromadb
from dotenv import load_dotenv

load_dotenv()

# ================= CONFIG =================

# Directory setup relative to this script
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
CHROMA_DIR_V2 =  os.path.join(CURRENT_DIR, "chroma_db_v3")
CHECKPOINT_DIR_V2 = os.getenv("CHECKPOINT_DIR_V2") or os.path.join(CURRENT_DIR, "checkpoints_v2")

VECTORS_PARQUET = os.path.join(CHECKPOINT_DIR_V2, "vectors_v2.parquet")

COLL_NAME = os.getenv("COLL_NAME_V2", "mysu_v2_bge_m3")
EMBED_DIM = int(os.getenv("EMBED_DIM", "1024"))
BATCH_UPSERT = int(os.getenv("BATCH_UPSERT", "64"))

def main():
    print("=" * 60)
    print("RESTORE CHROMA DB V2")
    print("=" * 60)
    print(f"Parquet Source: {VECTORS_PARQUET}")
    print(f"Chroma Target:  {CHROMA_DIR_V2}")
    print(f"Collection:     {COLL_NAME}")
    print("=" * 60)

    if not os.path.exists(VECTORS_PARQUET):
        print(f"[error] Vectors parquet file not found: {VECTORS_PARQUET}")
        return

    # 1) Load vectors checkpoint
    print("[1/5] Loading vectors parquet...")
    try:
        table = pq.read_table(VECTORS_PARQUET)
    except Exception as e:
        print(f"[error] Failed to read parquet: {e}")
        return

    ids         = table["id"].to_pylist()
    docs        = table["document"].to_pylist()
    metas_json  = table["metadata"].to_pylist()
    vecs_list   = table["vector"].to_pylist()

    print(f"[info] Loaded {len(ids)} vectors from parquet")

    if not ids:
        print("[warn] No vectors found. Exiting.")
        return

    vectors = np.asarray(vecs_list, dtype=np.float32)
    if vectors.shape[1] != EMBED_DIM:
        print(f"[warn] Dimension mismatch! Parquet has {vectors.shape[1]}, config expects {EMBED_DIM}.")
        # Proceeding anyway usually causes Chroma error, but we'll let it try or user can adjust config.

    metadatas = []
    for m in metas_json:
        try:
            metadatas.append(json.loads(m))
        except:
            metadatas.append({})

    # 2) Connect to Chroma
    print("[2/5] Connecting to ChromaDB...")
    client = chromadb.PersistentClient(path=CHROMA_DIR_V2)

    # 3) Drop old collection (very important!)
    print(f"[3/5] Resetting collection '{COLL_NAME}'...")
    try:
        client.delete_collection(COLL_NAME)
        print(f"  - Deleted old collection")
    except Exception as e:
        print(f"  - No existing collection to delete or delete failed ({e})")

    # 4) Recreate collection
    print("[4/5] Creating new collection...")
    coll = client.get_or_create_collection(
        name=COLL_NAME,
        metadata={"hnsw:space": "cosine"},
    )

    # 5) Upsert in batches
    print(f"[5/5] Upserting {len(ids)} items...")
    n = len(ids)
    for i in range(0, n, BATCH_UPSERT):
        batch_ids   = ids[i : i + BATCH_UPSERT]
        batch_docs  = docs[i : i + BATCH_UPSERT]
        batch_meta  = metadatas[i : i + BATCH_UPSERT]
        batch_vecs  = vectors[i : i + BATCH_UPSERT]

        coll.upsert(
            ids=batch_ids,
            embeddings=batch_vecs.tolist(),
            metadatas=batch_meta,
            documents=batch_docs,
        )
        if (i // BATCH_UPSERT) % 5 == 0:
            print(f"  - restored {min(i+BATCH_UPSERT, n)}/{n}")

    print("\n✅ Restore finished!")
    print(f"Final Collection Count: {coll.count()}")

if __name__ == "__main__":
    main()
