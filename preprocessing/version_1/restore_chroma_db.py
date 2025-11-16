# restore_chroma_from_parquet.py  (version using to_table + slicing)
import os, json
import pyarrow.dataset as ds
import chromadb
from tqdm import tqdm  # pip install tqdm

PARQUET_BGE   = r"C:\\Users\\kosot\\Documents\\bitirme\\preprocessing\\version_1\\checkpoints\\parquet_bge"
PARQUET_NOMIC = r"C:\\Users\\kosot\\Documents\\bitirme\\preprocessing\\version_1\\checkpoints\\parquet_nomic"
CHROMA_DIR    = r"C:\\Users\\kosot\\Documents\\bitirme\\preprocessing\\version_1\\chroma_db_clean"  # new, non-synced local dir

TARGET = [
    ("mysu_surecharitasi_bge",   PARQUET_BGE),
    ("mysu_surecharitasi_nomic", PARQUET_NOMIC),
]

def iter_rows(parquet_dir, batch_size=2000):
    d = ds.dataset(parquet_dir, format="parquet")
    # Load as a single table, then slice into batches (portable across pyarrow versions)
    tbl = d.to_table()  # uses multithreading internally
    n = tbl.num_rows
    for i in range(0, n, batch_size):
        chunk = tbl.slice(i, min(batch_size, n - i)).to_pydict()
        ids   = chunk["id"]
        docs  = chunk["document"]
        metas = [json.loads(s) for s in chunk["metadata"]]
        vecs  = chunk["vector"]
        vecs  = [list(v) for v in vecs]  # ensure plain lists
        yield ids, vecs, metas, docs

def main():
    os.makedirs(CHROMA_DIR, exist_ok=True)
    client = chromadb.PersistentClient(path=CHROMA_DIR)

    for coll_name, pq_dir in TARGET:
        print(f"\nRestoring {coll_name} from {pq_dir} ...")
        coll = client.get_or_create_collection(
            name=coll_name,
            metadata={"hnsw:space": "cosine"}
        )
        total = 0
        for ids, vecs, metas, docs in tqdm(iter_rows(pq_dir), unit="batch"):
            coll.upsert(ids=ids, embeddings=vecs, metadatas=metas, documents=docs)
            total += len(ids)
        print(f"Restored {total} vectors into {coll_name}. count() -> {coll.count()}")

if __name__ == "__main__":
    main()
