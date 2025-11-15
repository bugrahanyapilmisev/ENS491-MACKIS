# rag_pdf_ollama.py
import os, json, math, time, re
import numpy as np
import requests
from pypdf import PdfReader
import faiss, json
OLLAMA_HOST = "http://localhost:11434"
EMBED_MODEL = "nomic-embed-text"      # English-friendly
GEN_MODEL   = "llama3.1"    # the LLM that will answer


def save_index(index, chunks, path_prefix="akd_index"):
    faiss.write_index(index, f"{path_prefix}.faiss")
    with open(f"{path_prefix}.chunks.json","w",encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False)

def load_index(path_prefix="akd_index"):
    index = faiss.read_index(f"{path_prefix}.faiss")
    with open(f"{path_prefix}.chunks.json","r",encoding="utf-8") as f:
        chunks = json.load(f)
    return index, chunks
# === A) Load & clean PDF ===
def load_pdf_text(pdf_path: str) -> str:
    reader = PdfReader(pdf_path)
    pages = []
    for p in reader.pages:
        t = p.extract_text() or ""
        pages.append(t)
    text = "\n".join(pages)
    # light cleanup
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n\s*\n\s*\n+", "\n\n", text)
    return text.strip()

# === B) Chunking (simple, char-based) ===
def chunk_text(text: str, chunk_size=1200, overlap=200):
    chunks = []
    i = 0
    n = len(text)
    while i < n:
        j = min(i + chunk_size, n)
        chunk = text[i:j]
        # try not to cut inside a word/sentence too badly
        if j < n:
            k = chunk.rfind(". ")
            if k > 200:  # avoid too small tail
                chunk = chunk[:k+1]
                j = i + len(chunk)
        chunks.append(chunk.strip())
        i = max(j - overlap, j)  # move forward with overlap
    # filter empties
    return [c for c in chunks if c]

# === C) Embedding via Ollama REST ===
def embed(text: str, model=EMBED_MODEL):
    url = f"{OLLAMA_HOST}/api/embeddings"
    resp = requests.post(url, json={"model": model, "prompt": text}, timeout=120)
    resp.raise_for_status()
    return np.array(resp.json()["embedding"], dtype=np.float32)

def embed_many(texts, model=EMBED_MODEL, sleep_s=0.0):
    vecs = []
    for t in texts:
        v = embed(t, model=model)
        vecs.append(v)
        if sleep_s: time.sleep(sleep_s)
    arr = np.vstack(vecs)
    # cosine similarity with FAISS = normalize to unit length + inner product
    norms = np.linalg.norm(arr, axis=1, keepdims=True) + 1e-12
    arr = arr / norms
    return arr

# === D) Build FAISS index ===
def build_faiss_index(embeddings: np.ndarray):
    import faiss  # lazy import
    d = embeddings.shape[1]
    index = faiss.IndexFlatIP(d)   # inner product (cosine after normalization)
    index.add(embeddings)
    return index

# === E) Simple retriever ===
def retrieve(query: str, index, chunks, k=4, model=EMBED_MODEL):
    q = embed(query, model=model).astype(np.float32)
    q = q / (np.linalg.norm(q) + 1e-12)
    import faiss
    D, I = index.search(q.reshape(1, -1), k)
    ctx = [chunks[i] for i in I[0]]
    return ctx, D[0].tolist(), I[0].tolist()

# === F) Ask LLM with retrieved context ===
def ask_llm(question: str, context_chunks, model=GEN_MODEL):
    # You can tune the prompt to your style/policy
    system = (
        "You are a helpful assistant. Answer in the language of the question. "
        "Use ONLY the provided context. If unsure, say you don't know."
    )
    context = "\n\n---\n\n".join(context_chunks)
    prompt = f"{system}\n\nContext:\n{context}\n\nQuestion:\n{question}\n\nAnswer:"
    url = f"{OLLAMA_HOST}/api/generate"
    resp = requests.post(url, json={"model": model, "prompt": prompt, "stream": False}, timeout=300)
    resp.raise_for_status()
    return resp.json()["response"]

# === G) Glue everything ===
def main():
    # 1) point to your PDF (update the path!)
    PDF_PATH = "C:\\Users\\kosot\\OneDrive\\Masaüstü\\CS\\bitirme\\crawler_for_srdoc\\mysu_dump2\\surecharitasi\\sites\\mysu.sabanciuniv.edu.surecharitasi\\files\\akademikdurustlugedair_mektup_0.pdf"
    if not os.path.isfile(PDF_PATH):
        print("Update PDF_PATH to your local file.")
        return

    print("Loading PDF...")
    text = load_pdf_text(PDF_PATH)
    print(f"Total characters: {len(text):,}")

    print("Chunking...")
    chunks = chunk_text(text, chunk_size=1200, overlap=200)
    print(f"Chunks: {len(chunks)}")

    print("Embedding chunks with bge-m3...")
    embs = embed_many(chunks, model=EMBED_MODEL)
    print(f"Emb shape: {embs.shape}")

    print("Building FAISS index...")
    index = build_faiss_index(embs)
    save_index(index, chunks, "akd_index")
    index, chunks = load_index("akd_index")
    # Example Q&A
    print("\n=== Ask something about the PDF ===")
    while True:
        q = input("\nSoru (quit to exit): ").strip()
        if q.lower() in {"q","quit","exit"}:
            break
        ctx, scores, ids = retrieve(q, index, chunks, k=4, model=EMBED_MODEL)
        print("Top scores:", [round(s,3) for s in scores])
        print("Answer:\n", ask_llm(q, ctx, model=GEN_MODEL))

if __name__ == "__main__":
    main()
