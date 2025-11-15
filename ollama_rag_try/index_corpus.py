# index_corpus.py
import os, re, json, math, pathlib, hashlib
from pathlib import Path
from bs4 import BeautifulSoup
from langdetect import detect
from pypdf import PdfReader
import numpy as np
from tqdm import tqdm
import requests

DATA_ROOT = "C:\\Users\\kosot\\OneDrive\\Masaüstü\\CS\\bitirme\\crawler_for_srdoc\\mysu_dump2"   # <- change if needed
OUT_DIR   = "store"
OUT_VEC   = Path(OUT_DIR) / "vectors.npz"   # matrix
OUT_META  = Path(OUT_DIR) / "meta.jsonl"    # per-chunk metadata

# --- Ollama endpoints ---
OLLAMA_HOST = "http://localhost:11434"
EMBED_URL   = f"{OLLAMA_HOST}/api/embeddings"

# Choose models
TR_EMBED = "bge-m3"
EN_EMBED = "nomic-embed-text"

# --- chunking params ---
CHUNK_TOKENS = 300      # rough heuristic (characters used instead of tokens)
CHUNK_OVERLAP = 60

# ----------------- helpers -----------------
def read_text_from_file(p: Path) -> str:
    ext = p.suffix.lower()
    try:
        if ext in [".txt", ".md", ".csv", ".log"]:
            return p.read_text("utf-8", errors="ignore")
        if ext in [".html", ".htm"]:
            html = p.read_text("utf-8", errors="ignore")
            soup = BeautifulSoup(html, "lxml")
            # remove script/style
            for tag in soup(["script", "style", "noscript"]):
                tag.extract()
            return soup.get_text("\n", strip=True)
        if ext == ".pdf":
            txt = []
            with open(p, "rb") as f:
                pdf = PdfReader(f)
                for page in pdf.pages:
                    t = page.extract_text() or ""
                    txt.append(t)
            return "\n".join(txt)
        # unsupported binaries -> skip
        return ""
    except Exception:
        return ""

def chunk_text(s: str, size=CHUNK_TOKENS, overlap=CHUNK_OVERLAP):
    s = re.sub(r"[ \t]+", " ", s)
    s = re.sub(r"\n{2,}", "\n", s)
    s = s.strip()
    if not s:
        return []
    # simple char-based chunking
    chunks = []
    i = 0
    while i < len(s):
        chunk = s[i:i+size]
        if chunk.strip():
            chunks.append(chunk)
        i += size - overlap
        if i <= 0:
            break
    return chunks

def choose_embed_model(text: str) -> str:
    # very short text can confuse detector; default EN
    try:
        lang = detect(text[:1000]) if len(text) >= 20 else "en"
    except Exception:
        lang = "en"
    if lang.startswith("tr"):
        return TR_EMBED
    return EN_EMBED

def embed_text(model: str, text: str) -> np.ndarray:
    resp = requests.post(
        EMBED_URL,
        json={"model": model, "prompt": text},
        timeout=60,
    )
    resp.raise_for_status()
    vec = resp.json()["embedding"]
    return np.array(vec, dtype=np.float32)

def sha1(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8", errors="ignore")).hexdigest()

# ----------------- main -----------------
def main():
    Path(OUT_DIR).mkdir(parents=True, exist_ok=True)
    vectors = []
    metas = []

    all_files = []
    for root, _, files in os.walk(DATA_ROOT):
        for fn in files:
            ext = Path(fn).suffix.lower()
            if ext in {".txt",".md",".csv",".log",".html",".htm",".pdf"}:
                all_files.append(Path(root) / fn)

    for p in tqdm(all_files, desc="Indexing"):
        text = read_text_from_file(p)
        if not text:
            continue

        chunks = chunk_text(text)
        if not chunks:
            continue

        for idx, ch in enumerate(chunks):
            # pick embed model based on detected language of the chunk
            model = choose_embed_model(ch)
            vec = embed_text(model, ch)

            vectors.append(vec)
            metas.append({
                "id": sha1(f"{p}::{idx}::{len(ch)}"),
                "source": str(p).replace("\\", "/"),
                "chunk_index": idx,
                "model": model,
                "chars": len(ch),
                "preview": ch[:200].replace("\n"," "),
            })

    if not vectors:
        print("No text found — check DATA_ROOT.")
        return

    M = np.vstack(vectors)  # [N, D]
    # save
    np.savez_compressed(OUT_VEC, vectors=M)
    with open(OUT_META, "w", encoding="utf-8") as f:
        for m in metas:
            f.write(json.dumps(m, ensure_ascii=False) + "\n")

    print(f"Saved: {OUT_VEC} and {OUT_META}")
    print(f"Chunks: {len(metas)}, dim: {M.shape[1]}")

if __name__ == "__main__":
    main()
