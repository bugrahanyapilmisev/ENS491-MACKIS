# export_chroma_to_parquet_plus2.py
import os
import json
import chromadb
import pyarrow as pa
import pyarrow.parquet as pq

# ========= EDIT THESE =========
CHROMA_DIR  = r"C:\\Users\\kosot\\Documents\\bitirme\\preprocessing\\chroma_db_mysu"
COLL_NAME   = "mysu_surecharitasi_bge_m3_v1"

OUT_DIR     = r"C:\\Users\\kosot\\Documents\\bitirme\\preprocessing\\vector_export"
OUT_FILE    = os.path.join(OUT_DIR, "mysu_vectors.parquet")

BATCH       = 1000   # how many records to read from Chroma per call
# ==============================


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    client = chromadb.PersistentClient(path=CHROMA_DIR)
    coll   = client.get_collection(name=COLL_NAME)

    total = coll.count()
    print(f"Collection {COLL_NAME} has {total} vectors")

    all_ids   = []
    all_vecs  = []
    all_docs  = []
    all_metas = []

    offset = 0
    while offset < total:
        n = min(BATCH, total - offset)

        # ⚠️ don't include "ids" here – Chroma returns them by default
        res = coll.get(
            limit=n,
            offset=offset,
            include=["embeddings", "documents", "metadatas"],
        )

        ids   = res["ids"]
        vecs  = res["embeddings"]
        docs  = res["documents"]
        metas = res["metadatas"]

        all_ids.extend(ids)
        all_vecs.extend(vecs)
        all_docs.extend(docs)
        all_metas.extend([json.dumps(m, ensure_ascii=False) for m in metas])

        offset += n
        print(f"Exported {offset}/{total}")

    # Build Arrow table
    table = pa.table({
        "id":       pa.array(all_ids,   type=pa.string()),
        "vector":   pa.array(all_vecs,  type=pa.list_(pa.float32())),
        "document": pa.array(all_docs,  type=pa.string()),
        "metadata": pa.array(all_metas, type=pa.string()),
    })

    pq.write_table(table, OUT_FILE)
    print(f"✅ Exported {total} vectors to {OUT_FILE}")


if __name__ == "__main__":
    main()
