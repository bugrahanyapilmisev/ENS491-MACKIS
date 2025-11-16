# build_chroma_store_robust.py
# Fast, resumable Chroma builder with strong preprocessing + Parquet checkpoints (for easy Qdrant migration later).

import os, re, json, time, hashlib, pathlib, math, sqlite3, logging, unicodedata
from typing import List, Dict, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import requests
from pypdf import PdfReader
from bs4 import BeautifulSoup
import pyarrow as pa
import pyarrow.parquet as pq
import chromadb

# -------------------- CONFIG --------------------
OLLAMA_HOST = "http://localhost:11434"

# Known Ollama embedding dims (keep in sync with your pulled models)
EMBED_MODELS = {
    "bge-m3": 1024,              # Turkish/multilingual
    "nomic-embed-text": 768,     # English
}

# Paths
ROOT_DIR   = r":\\Users\\kosot\\Documents\\bitirme\\crawler_for_srdoc\\mysu_dump2"
CHROMA_DIR = r"C:\\Users\\kosot\\Documents\\bitirme\\preprocessing\\chroma_db_clean"

# One Chroma collection per model
COLL_NAMES = {
    "bge-m3": "mysu_surecharitasi_bge",
    "nomic-embed-text": "mysu_surecharitasi_nomic",
}

# Checkpointing & logs
CHECKPOINT_DIR = r"C:\\Users\\kosot\\Documents\\bitirme\\preprocessing\\chechkpoints"
PARQUET_OUT_TR = os.path.join(CHECKPOINT_DIR, "parquet_bge")       # bge-m3 shards
PARQUET_OUT_EN = os.path.join(CHECKPOINT_DIR, "parquet_nomic")     # nomic shards
MANIFEST_DB    = os.path.join(CHECKPOINT_DIR, "manifest.sqlite3")
LOG_FILE       = os.path.join(CHECKPOINT_DIR, "build.log")

# Performance knobs
MAX_WORKERS_EMBED = 6          # concurrent requests to Ollama embeddings
BATCH_UPSERT      = 64         # chunks per upsert to Chroma
CHUNK_SIZE        = 900        # ~900 chars window
CHUNK_OVERLAP     = 200        # ~200 chars worth of sentence overlap
MIN_TEXT_LEN      = 50         # skip very small docs
REQUEST_TIMEOUT_S = 120        # per-embedding HTTP timeout
RETRY_MAX         = 4          # embedding retries
BACKOFF_BASE      = 0.7        # backoff (exp + jitter)
PARQUET_SHARD_EVERY = 2000     # write parquet shard after every N new vectors per model

TR_DIACRITICS = "çğıöşüÇĞİÖŞÜ"

# -------------------- LOGGING --------------------
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
for d in (PARQUET_OUT_TR, PARQUET_OUT_EN, CHROMA_DIR):
    os.makedirs(d, exist_ok=True)

logging.basicConfig(
    filename=LOG_FILE,
    filemode="a",
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
print(f"Logs: {LOG_FILE}")

# -------------------- UTILS --------------------
def sha1_bytes(b: bytes) -> str:
    return hashlib.sha1(b).hexdigest()

def sha1_text(s: str) -> str:
    return sha1_bytes(s.encode("utf-8", errors="ignore"))

def normalize_ws_keep_diacritics(text: str) -> str:
    # Preserve Turkish diacritics (NFC), normalize spacing/blanklines
    text = unicodedata.normalize("NFC", text)
    text = text.replace("\xa0", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\s*\n\s*\n\s*\n+", "\n\n", text)
    return text.strip()

def fix_pdf_wrapping(text: str) -> str:
    # Remove hyphenation at EOL: "algo-\n rithm" -> "algorithm"
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)
    # Join lines likely same paragraph (linebreak not followed by capital/digit)
    text = re.sub(r"(?<![\.!?:])\n(?!\s*[A-Z" + TR_DIACRITICS + r"0-9])", " ", text)
    return normalize_ws_keep_diacritics(text)

def load_pdf_text(path: str) -> Tuple[str, str]:
    reader = PdfReader(path)
    pages = [(p.extract_text() or "") for p in reader.pages]
    text = "\n".join(pages)
    text = fix_pdf_wrapping(text)
    return text, ""  # titles unreliable in PDFs

def load_html_text(path: str) -> Tuple[str, str]:
    html = pathlib.Path(path).read_text(encoding="utf-8", errors="ignore")
    soup = BeautifulSoup(html, "lxml")
    # Remove scripts, styles, noscript early
    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()
    # Remove common boilerplate
    for sel in [
        "nav", "footer", "header", ".menu", ".navbar", ".breadcrumb", ".sidebar",
        "#cookie-banner", ".cookie", ".cookies", ".site-footer", ".site-header",
        ".social-share", ".pagination", ".advert", ".ad", ".banner"
    ]:
        for node in soup.select(sel):
            node.decompose()

    # Prefer main/article if present
    main = soup.select_one("main") or soup.select_one("article")
    body = main if main else soup

    title = (soup.title.string.strip() if soup.title and soup.title.string else "")
    text = body.get_text("\n")
    text = normalize_ws_keep_diacritics(text)

    # Remove very short “menu-like” lines
    lines = [ln.strip() for ln in text.split("\n")]
    cleaned = []
    for ln in lines:
        if len(ln) >= 3 and not re.fullmatch(r"[•·\-\u2022]+", ln):
            cleaned.append(ln)
    text = "\n".join(cleaned)
    return text, title

def guess_lang_model(path: str, text_hint: str) -> str:
    p = path.lower()
    if "\\tr\\" in p or "/tr/" in p:
        return "bge-m3"
    if "\\en\\" in p or "/en/" in p:
        return "nomic-embed-text"
    if re.search(f"[{TR_DIACRITICS}]", text_hint[:2000]):
        return "bge-m3"
    return "nomic-embed-text"

# sentence-aware chunking
SENT_SPLIT = re.compile(r"(?<=[\.!?…])\s+(?=[A-Z" + TR_DIACRITICS + r"0-9])")

def sentence_split(text: str) -> List[str]:
    sents = SENT_SPLIT.split(text)
    merged = []
    buf = ""
    for s in sents:
        s = s.strip()
        if not s:
            continue
        if len(buf) + len(s) < 150:
            buf = (buf + " " + s).strip()
        else:
            if buf:
                merged.append(buf)
            buf = s
    if buf:
        merged.append(buf)
    return merged

def chunk_text(text: str, chunk_size: int, overlap: int) -> List[Dict]:
    sents = sentence_split(text)
    out, idx = [], 0
    i = 0
    # Sentence-level sliding window
    while i < len(sents):
        buf = []
        total = 0
        j = i
        while j < len(sents) and total + len(sents[j]) + 1 <= chunk_size:
            buf.append(sents[j])
            total += len(sents[j]) + 1
            j += 1
        if not buf:
            buf = [sents[i]]
            j = i + 1
        chunk = " ".join(buf).strip()
        if chunk:
            # Approx offsets just for debug
            start = max(0, text.find(buf[0]))
            end = start + len(chunk)
            out.append({"text": chunk, "start": start, "end": end, "index": idx})
            idx += 1
        # overlap ~ by sentences (roughly map overlap chars to ~sentences)
        step = max(1, len(buf) - max(1, math.ceil(overlap / 100)))
        i += step
    return out

def dedupe_chunks(chunks: List[Dict]) -> List[Dict]:
    """Deduplicate inside a document."""
    seen = set()
    deduped = []
    for ch in chunks:
        key = sha1_text(ch["text"])
        if key in seen:
            continue
        seen.add(key)
        deduped.append(ch)
    return deduped

# -------------------- MANIFEST (resume + global dedup) --------------------
def init_manifest(db_path: str = MANIFEST_DB):
    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    c.execute("""CREATE TABLE IF NOT EXISTS files(
        path TEXT PRIMARY KEY,
        mtime REAL,
        size INTEGER,
        content_sha1 TEXT,
        processed INTEGER DEFAULT 0,
        model_hint TEXT,
        total_chunks INTEGER DEFAULT 0
    )""")
    c.execute("""CREATE TABLE IF NOT EXISTS errors(
        path TEXT,
        stage TEXT,
        message TEXT,
        ts REAL
    )""")
    c.execute("""CREATE TABLE IF NOT EXISTS chunk_hashes(
        chunk_sha1 TEXT PRIMARY KEY
    )""")
    conn.commit()
    return conn

def record_file(conn, path: str, mtime: float, size: int, content_sha1: str, model_hint: str, total_chunks: int):
    c = conn.cursor()
    c.execute("""INSERT OR REPLACE INTO files(path, mtime, size, content_sha1, processed, model_hint, total_chunks)
                 VALUES (?,?,?,?,COALESCE((SELECT processed FROM files WHERE path=?),0),?,?)""",
              (path, mtime, size, content_sha1, path, model_hint, total_chunks))
    conn.commit()

def mark_processed(conn, path: str):
    c = conn.cursor()
    c.execute("UPDATE files SET processed=1 WHERE path=?", (path,))
    conn.commit()

def already_processed(conn, path: str, mtime: float, size: int, content_sha1: str) -> bool:
    c = conn.cursor()
    c.execute("SELECT processed, mtime, size, content_sha1 FROM files WHERE path=?", (path,))
    row = c.fetchone()
    if not row:
        return False
    processed, old_mtime, old_size, old_sha = row
    return processed == 1 and old_mtime == mtime and old_size == size and old_sha == content_sha1

def log_error(conn, path: str, stage: str, message: str):
    c = conn.cursor()
    c.execute("INSERT INTO errors(path, stage, message, ts) VALUES (?,?,?,?)",
              (path, stage, message[:4000], time.time()))
    conn.commit()
    logging.error(f"[{stage}] {path} :: {message}")

def is_new_chunk(conn, chunk_sha: str) -> bool:
    c = conn.cursor()
    c.execute("SELECT 1 FROM chunk_hashes WHERE chunk_sha1=?", (chunk_sha,))
    return c.fetchone() is None

def remember_chunk(conn, chunk_sha: str):
    c = conn.cursor()
    c.execute("INSERT OR IGNORE INTO chunk_hashes(chunk_sha1) VALUES (?)", (chunk_sha,))
    conn.commit()

# -------------------- EMBEDDING (parallel + retry) --------------------
def embed_one(text: str, model: str) -> np.ndarray:
    url = f"{OLLAMA_HOST}/api/embeddings"
    for attempt in range(RETRY_MAX):
        try:
            r = requests.post(url, json={"model": model, "prompt": text}, timeout=REQUEST_TIMEOUT_S)
            r.raise_for_status()
            v = np.array(r.json()["embedding"], dtype=np.float32)
            # L2 normalize for cosine
            v /= (np.linalg.norm(v) + 1e-12)
            exp_dim = EMBED_MODELS[model]
            if v.shape[0] != exp_dim:
                raise RuntimeError(f"Unexpected dim for {model}: got {v.shape[0]} vs {exp_dim}")
            return v
        except Exception as e:
            sleep = (BACKOFF_BASE * (2 ** attempt)) * (0.5 + np.random.rand())
            time.sleep(sleep)
            if attempt == RETRY_MAX - 1:
                raise
    raise RuntimeError("Unreachable retry loop")

def embed_many_parallel(texts: List[str], model: str, max_workers=MAX_WORKERS_EMBED) -> np.ndarray:
    vecs = [None] * len(texts)
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = {ex.submit(embed_one, texts[i], model): i for i in range(len(texts))}
        for fut in as_completed(futures):
            i = futures[fut]
            vecs[i] = fut.result()
    arr = np.vstack(vecs)
    return arr

# -------------------- CHROMA --------------------
def get_or_create_collection(client, name: str):
    # Ensure cosine space for HNSW
    return client.get_or_create_collection(name=name, metadata={"hnsw:space": "cosine"})

# -------------------- PARQUET CHECKPOINT --------------------
def write_parquet_shard(out_dir: str, shard_ix: int, ids, vectors, docs, metas):
    table = pa.table({
        "id": pa.array(ids),
        "vector": pa.array(vectors, type=pa.list_(pa.float32())),
        "document": pa.array(docs, type=pa.string()),
        "metadata": pa.array([json.dumps(m, ensure_ascii=False) for m in metas], type=pa.string())
    })
    fn = os.path.join(out_dir, f"vectors_{shard_ix:08d}.parquet")
    pq.write_table(table, fn)
    logging.info(f"[parquet] wrote {fn}")

# -------------------- MAIN --------------------
def main():
    conn = init_manifest(MANIFEST_DB)
    client = chromadb.PersistentClient(path=CHROMA_DIR)
    colls = {m: get_or_create_collection(client, COLL_NAMES[m]) for m in EMBED_MODELS}

    # Gather files
    exts = {".pdf", ".html", ".htm"}
    files = []
    for root, _, fnames in os.walk(ROOT_DIR):
        for fn in fnames:
            if os.path.splitext(fn)[1].lower() in exts:
                files.append(os.path.join(root, fn))
    files.sort()

    # Parquet shard counters/accumulators
    shard_ix = {"bge-m3": 0, "nomic-embed-text": 0}
    parquet_out = {"bge-m3": PARQUET_OUT_TR, "nomic-embed-text": PARQUET_OUT_EN}
    os.makedirs(parquet_out["bge-m3"], exist_ok=True)
    os.makedirs(parquet_out["nomic-embed-text"], exist_ok=True)

    total_upserts = {"bge-m3": 0, "nomic-embed-text": 0}
    # small accumulators for last shard write
    shard_buf = {
        "bge-m3": {"ids": [], "vecs": [], "docs": [], "metas": []},
        "nomic-embed-text": {"ids": [], "vecs": [], "docs": [], "metas": []},
    }

    processed_files = 0
    started_at = time.time()

    for path in files:
        try:
            stat = os.stat(path)
            mtime, size = stat.st_mtime, stat.st_size

            with open(path, "rb") as f:
                content_bytes = f.read()
            content_sha = sha1_bytes(content_bytes)

            if already_processed(conn, path, mtime, size, content_sha):
                continue

            # Read
            ext = os.path.splitext(path)[1].lower()
            if ext == ".pdf":
                text, title = load_pdf_text(path)
            else:
                text, title = load_html_text(path)

            if len(text) < MIN_TEXT_LEN:
                record_file(conn, path, mtime, size, content_sha, "", 0)
                mark_processed(conn, path)
                continue

            # Chunk + dedupe (per document)
            chunks = chunk_text(text, CHUNK_SIZE, CHUNK_OVERLAP)
            chunks = dedupe_chunks(chunks)
            model = guess_lang_model(path, text)

            record_file(conn, path, mtime, size, content_sha, model, len(chunks))

            # Process in mini-batches for this file (bounded memory)
            for i in range(0, len(chunks), BATCH_UPSERT):
                batch = chunks[i:i+BATCH_UPSERT]

                # Global dedup across files
                filtered = []
                for b in batch:
                    ch_sha = sha1_text(b["text"])
                    if is_new_chunk(conn, ch_sha):
                        remember_chunk(conn, ch_sha)
                        filtered.append((b, ch_sha))
                if not filtered:
                    continue

                texts = [b["text"] for b, _ in filtered]
                ids   = [sha1_text(f"{path}::{b['index']}") for b, _ in filtered]
                metas = [{
                    "doc_path": path,
                    "title": title,
                    "model": model,
                    "start": b["start"],
                    "end": b["end"],
                    "chunk_index": b["index"],
                    "char_count": len(b["text"]),
                    "lang": "tr" if model == "bge-m3" else "en",
                    "preview": b["text"][:220],
                } for b, _ in filtered]

                # Embed (parallel)
                vectors = embed_many_parallel(texts, model=model, max_workers=MAX_WORKERS_EMBED)

                # Upsert to Chroma
                coll = colls[model]
                coll.upsert(ids=ids, embeddings=vectors.tolist(), metadatas=metas, documents=texts)

                # Update counters + parquet shard buffers
                total_upserts[model] += len(ids)
                sb = shard_buf[model]
                sb["ids"].extend(ids)
                sb["vecs"].extend(vectors.tolist())
                sb["docs"].extend(texts)
                sb["metas"].extend(metas)

                # Periodically write a shard for migration safety
                if len(sb["ids"]) >= PARQUET_SHARD_EVERY:
                    shard_ix[model] += 1
                    write_parquet_shard(parquet_out[model], shard_ix[model],
                                        sb["ids"], sb["vecs"], sb["docs"], sb["metas"])
                    # reset buffer
                    sb["ids"].clear(); sb["vecs"].clear(); sb["docs"].clear(); sb["metas"].clear()

            mark_processed(conn, path)
            processed_files += 1

            if processed_files % 25 == 0:
                elapsed = time.time() - started_at
                print(f"[progress] files={processed_files} upserts(TR)={total_upserts['bge-m3']} upserts(EN)={total_upserts['nomic-embed-text']} elapsed={elapsed:.1f}s")

        except Exception as e:
            log_error(conn, path, "process", str(e))
            continue

    # Final shards (flush tails)
    for model in ["bge-m3", "nomic-embed-text"]:
        sb = shard_buf[model]
        if sb["ids"]:
            shard_ix[model] += 1
            write_parquet_shard(parquet_out[model], shard_ix[model],
                                sb["ids"], sb["vecs"], sb["docs"], sb["metas"])

    # Summary
    print("✅ Done.")
    for m, name in COLL_NAMES.items():
        try:
            c = colls[m]
            print(f" - {name} ({m}): {c.count()} vectors")
        except Exception as e:
            print(f" - {name} ({m}): count unavailable ({e})")
    print(f"Parquet out: {PARQUET_OUT_TR} / {PARQUET_OUT_EN}")
    print(f"Manifest DB: {MANIFEST_DB}")
    print(f"Logs:        {LOG_FILE}")

if __name__ == "__main__":
    main()
