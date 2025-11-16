# peek_chroma_plus.py
# Quick sanity inspector for version_plus collection.

import chromadb

CHROMA_DIR = r"C:\\Users\\kosot\\Documents\\bitirme\\preprocessing\\version_plus\\chroma_db_clean_plus"
COLL_NAME  = "mysu_surecharitasi_bge"

client = chromadb.PersistentClient(path=CHROMA_DIR)
coll   = client.get_or_create_collection(name=COLL_NAME, metadata={"hnsw:space":"cosine"})

print(f"\n=== {COLL_NAME} (version_plus) ===")
print("count:", coll.count())

rows = coll.get(include=["documents", "metadatas"], limit=3)
for i in range(len(rows["documents"])):
    meta = rows["metadatas"][i]
    doc  = rows["documents"][i]
    preview = (doc[:180] + ("..." if len(doc) > 180 else "")).replace("\n"," ")
    print(f"- id: {meta.get('doc_path','')[-50:]} ::#{meta.get('chunk_index')}")
    print(f"  h1/h2/h3: {meta.get('h1','')} > {meta.get('h2','')} > {meta.get('h3','')}")
    print(f"  lang: {meta.get('lang')}, doctype: {meta.get('doctype')}")
    print(f"  preview: {preview}")
