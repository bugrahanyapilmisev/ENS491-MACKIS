import os
import json
import numpy as np
import pyarrow.parquet as pq
import chromadb

CHROMA_DIR = r"C:\\Users\\kosot\\Documents\\bitirme\\preprocessing\\chroma_db_mysu_RESTORE"
CHECKPOINT_DIR = r"C:\\Users\\kosot\\Documents\\bitirme\\preprocessing\\checkpoints_plus2"
VECTORS_PARQUET = os.path.join(CHECKPOINT_DIR, "vectors_plus2.parquet")

COLL_NAME = "mysu_surecharitasi_bge_m3_v1"
EMBED_DIM = 1024
BATCH_UPSERT = 64

def main():
    # 1) Load vectors checkpoint
    table = pq.read_table(VECTORS_PARQUET)

    ids         = table["id"].to_pylist()
    docs        = table["document"].to_pylist()
    metas_json  = table["metadata"].to_pylist()
    vecs_list   = table["vector"].to_pylist()

    vectors = np.asarray(vecs_list, dtype=np.float32)
    if vectors.shape[1] != EMBED_DIM:
        raise RuntimeError(f"Bad dim: {vectors.shape[1]} vs {EMBED_DIM}")

    metadatas = [json.loads(m) for m in metas_json]

    # 2) Connect to Chroma
    client = chromadb.PersistentClient(path=CHROMA_DIR)

    # 3) Drop old collection (very important!)
    try:
        client.delete_collection(COLL_NAME)
        print(f"[info] deleted old collection {COLL_NAME}")
    except Exception as e:
        print(f"[info] no existing collection {COLL_NAME} or delete failed: {e}")

    # 4) Recreate collection
    coll = client.get_or_create_collection(
        name=COLL_NAME,
        metadata={"hnsw:space": "cosine"},
    )

    # 5) Upsert in batches
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
        print(f"[restore] upserted {i+len(batch_ids)}/{n}")

    print("✅ Restore finished.")
    print("Collection count:", coll.count())

if __name__ == "__main__":
    main()
