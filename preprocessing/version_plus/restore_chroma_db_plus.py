# restore_chroma_db_plus.py
# Restore single-collection Chroma from version_plus Parquet shards.

import os, json
import pyarrow.dataset as ds
import chromadb
from tqdm import tqdm  # pip install tqdm

# --- paths for version_plus ---
PARQUET_DIR = r"C:\\Users\\kosot\\Documents\\bitirme\\preprocessing\\version_plus\\checkpoints_plus\\parquet_bge"
CHROMA_DIR  = r"C:\\Users\\kosot\\Documents\\bitirme\\preprocessing\\version_plus\\chroma_db_clean_plus"

COLL_NAME   = "mysu_surecharitasi_bge"  # same as in build_chroma_store_plus.py

def iter_rows(parquet_dir, batch_size=2000):
    d = ds.dataset(parquet_dir, format="parquet")
    tbl = d.to_table()  # portable across pyarrow versions
    n = tbl.num_rows
    for i in range(0, n, batch_size):
        chunk = tbl.slice(i, min(batch_size, n - i)).to_pydict()
        ids   = chunk["id"]
        docs  = chunk["document"]
        metas = [json.loads(s) for s in chunk["metadata"]]
        vecs  = [list(v) for v in chunk["vector"]]
        yield ids, vecs, metas, docs

def main():
    os.makedirs(CHROMA_DIR, exist_ok=True)
    client = chromadb.PersistentClient(path=CHROMA_DIR)
    coll = client.get_or_create_collection(
        name=COLL_NAME,
        metadata={"hnsw:space": "cosine"}
    )

    total = 0
    for ids, vecs, metas, docs in tqdm(iter_rows(PARQUET_DIR), unit="batch"):
        coll.upsert(ids=ids, embeddings=vecs, metadatas=metas, documents=docs)
        total += len(ids)

    print(f"Restored {total} vectors into {COLL_NAME}. count() -> {coll.count()}")

if __name__ == "__main__":
    main()
