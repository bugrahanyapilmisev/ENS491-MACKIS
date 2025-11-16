# peek_chroma.py  — fixed
import chromadb, numpy as np

CHROMA_DIR = r"C:\\Users\\kosot\\Documents\\bitirme\\preprocessing\\chroma_db_clean"
COLL_NAMES = {
    "bge-m3": "mysu_surecharitasi_bge",
    "nomic-embed-text": "mysu_surecharitasi_nomic",
}

client = chromadb.PersistentClient(path=CHROMA_DIR)

for model, name in COLL_NAMES.items():
    coll = client.get_or_create_collection(name=name, metadata={"hnsw:space":"cosine"})
    print(f"\n=== {name} ({model}) ===")
    print("count:", coll.count())
    # Do NOT include "ids" here — it's returned automatically
    rows = coll.get(limit=3, include=["documents","metadatas","embeddings"])
    for i in range(len(rows["ids"])):
        eid = rows["ids"][i]
        doc = rows["documents"][i]
        meta = rows["metadatas"][i]
        emb = rows["embeddings"][i]
        preview = (doc[:160] + ("..." if len(doc) > 160 else "")).replace("\n", " ")
        print("- id:", eid)
        print("  preview:", preview)
        print("  path:", meta.get("doc_path"))
        print("  idx:", meta.get("chunk_index"))
        print("  emb dim:", len(emb))
