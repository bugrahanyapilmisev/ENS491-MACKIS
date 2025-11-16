# rag_query_chroma.py
import re, requests, numpy as np, chromadb

OLLAMA = "http://localhost:11434"
CHROMA_DIR = r"C:\\Users\\kosot\\Documents\\bitirme\\preprocessing\\chroma_db_clean"   # your restored path

COLL_BGE   = "mysu_surecharitasi_bge"      # TR/multilingual
COLL_NOMIC = "mysu_surecharitasi_nomic"    # EN

def is_turkish(s: str) -> bool:
    return bool(re.search(r"[çğıöşüÇĞİÖŞÜ]", s))

def embed(text: str, model: str):
    r = requests.post(f"{OLLAMA}/api/embeddings", json={"model": model, "prompt": text}, timeout=60)
    r.raise_for_status()
    v = np.array(r.json()["embedding"], dtype=np.float32)
    v /= (np.linalg.norm(v) + 1e-12)
    return v.tolist()

def ask_llm(question: str, ctx: list, model="llama3.1"):
    system = "You are a helpful assistant. Answer ONLY using the provided context. If unsure, say you don't know."
    prompt = f"{system}\n\nContext:\n" + "\n\n---\n\n".join(ctx) + f"\n\nQuestion:\n{question}\n\nAnswer:"
    r = requests.post(f"{OLLAMA}/api/generate", json={"model": model, "prompt": prompt, "stream": False}, timeout=120)
    r.raise_for_status()
    return r.json()["response"]

def main():
    q = input("Soru / Question: ").strip()

    client = chromadb.PersistentClient(path=CHROMA_DIR)
    coll_tr = client.get_or_create_collection(COLL_BGE,   metadata={"hnsw:space":"cosine"})
    coll_en = client.get_or_create_collection(COLL_NOMIC, metadata={"hnsw:space":"cosine"})

    # ✅ Embed per collection/model
    qvec_tr = embed(q, "bge-m3")            # 1024-D
    qvec_en = embed(q, "nomic-embed-text")  # 768-D

    def query(coll, qvec, k=5):
        res = coll.query(query_embeddings=[qvec], n_results=k,
                         include=["documents","metadatas","distances"])
        out = []
        for doc, meta, dist in zip(res["documents"][0], res["metadatas"][0], res["distances"][0]):
            sim = 1.0 - float(dist)
            out.append((sim, doc, meta))
        return out

    hits = query(coll_tr, qvec_tr, 5) + query(coll_en, qvec_en, 5)
    hits.sort(key=lambda x: x[0], reverse=True)
    ctx = [h[1] for h in hits[:6]]

    print("\nTop contexts:")
    for i, (sim, doc, meta) in enumerate(hits[:6], 1):
        prev = (doc[:140] + ("..." if len(doc) > 140 else "")).replace("\n"," ")
        print(f"{i:02d}. sim={sim:.3f} idx={meta.get('chunk_index')} path={meta.get('doc_path')}\n    {prev}")

    print("\nAnswer:\n")
    print(ask_llm(q, ctx))

if __name__ == "__main__":
    main()
