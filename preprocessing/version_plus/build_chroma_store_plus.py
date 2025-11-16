# build_chroma_store_plus.py
# Single-model (bge-m3) Chroma builder + heading-aware, section-aware chunking + Parquet checkpoints.

import os, re, json, time, hashlib, pathlib, math, sqlite3, logging, unicodedata
from typing import List, Dict, Tuple, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import requests
from pypdf import PdfReader
from bs4 import BeautifulSoup
import pyarrow as pa
import pyarrow.parquet as pq
import chromadb

# ============ CONFIG ============
OLLAMA_HOST = "http://localhost:11434"
EMBED_MODEL = "bge-m3"
EMBED_DIM   = 1024

# --- FIX YOUR PATHS HERE ---
ROOT_DIR   = r"C:\\Users\\kosot\\Documents\\bitirme\\crawler_for_srdoc\\mysu_dump2"   # <- was broken in your snippet
CHROMA_DIR = r"C:\\Users\\kosot\\Documents\\bitirme\\preprocessing\\version_plus\\chroma_db_clean_plus"

COLL_NAME  = "mysu_surecharitasi_bge"  # single collection (TR+EN)

CHECKPOINT_DIR   = r"C:\\Users\\kosot\\Documents\\bitirme\\preprocessing\\version_plus\\checkpoints_plus"
PARQUET_OUT_DIR  = os.path.join(CHECKPOINT_DIR, "parquet_bge")
MANIFEST_DB      = os.path.join(CHECKPOINT_DIR, "manifest.sqlite3")
LOG_FILE         = os.path.join(CHECKPOINT_DIR, "build.log")

# Performance
MAX_WORKERS_EMBED   = 6
BATCH_UPSERT        = 64
CHUNK_SIZE          = 900
CHUNK_OVERLAP       = 180
MIN_TEXT_LEN        = 50
REQUEST_TIMEOUT_S   = 120
RETRY_MAX           = 4
BACKOFF_BASE        = 0.7
PARQUET_SHARD_EVERY = 2000

TR_DIACRITICS = "çğıöşüÇĞİÖŞÜ"

# ============ LOGGING ============
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(PARQUET_OUT_DIR, exist_ok=True)
os.makedirs(CHROMA_DIR, exist_ok=True)

logging.basicConfig(
    filename=LOG_FILE,
    filemode="a",
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
print(f"Logs: {LOG_FILE}")

# ============ UTILS ============
def sha1_bytes(b: bytes) -> str:
    return hashlib.sha1(b).hexdigest()

def sha1_text(s: str) -> str:
    return sha1_bytes(s.encode("utf-8", errors="ignore"))

def normalize_ws_keep_diacritics(text: str) -> str:
    text = unicodedata.normalize("NFC", text)
    text = text.replace("\xa0", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\s*\n\s*\n\s*\n+", "\n\n", text)
    return text.strip()

def glue_codes(text: str) -> str:
    # Keep codes like FIRO-C42003-03, PIRO-C420-0301 intact (remove EOL hyphenation around them)
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)
    return text

def fix_pdf_wrapping(text: str) -> str:
    text = glue_codes(text)
    text = re.sub(r"(?<![\.!?:])\n(?!\s*[A-Z" + TR_DIACRITICS + r"0-9])", " ", text)
    return normalize_ws_keep_diacritics(text)

def load_pdf_text(path: str) -> Tuple[str, str]:
    reader = PdfReader(path)
    pages = [(p.extract_text() or "") for p in reader.pages]
    text = "\n".join(pages)
    text = fix_pdf_wrapping(text)
    return text, ""

def load_html_sections(path: str) -> Tuple[str, List[Dict], str]:
    """Return full plain text, plus a list of sections with heading hierarchy for section-aware chunking."""
    html = pathlib.Path(path).read_text(encoding="utf-8", errors="ignore")
    soup = BeautifulSoup(html, "lxml")
    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()
    for sel in ["nav","footer","header",".menu",".navbar",".breadcrumb",".sidebar","#cookie-banner",".cookie",".cookies",".site-footer",".site-header",".social-share",".pagination",".advert",".ad",".banner"]:
        for node in soup.select(sel):
            node.decompose()

    title = (soup.title.string.strip() if soup.title and soup.title.string else "")
    body  = soup.select_one("main") or soup.select_one("article") or soup

    # Build flattened text (for fallback offsets) + sectionize by headings
    # Sections: [{h1,h2,h3,text}]
    sections = []
    cur = {"h1":"","h2":"","h3":"","text":[]}

    def close_section():
        if cur["text"]:
            sections.append({
                "h1": cur["h1"], "h2": cur["h2"], "h3": cur["h3"],
                "text": normalize_ws_keep_diacritics("\n".join(cur["text"]))
            })
            cur["text"] = []

    for node in body.descendants:
        if not getattr(node, "name", None):
            continue
        name = node.name.lower()
        if name in ["h1","h2","h3"]:
            close_section()
            heading_text = normalize_ws_keep_diacritics(node.get_text(" ").strip())
            if name == "h1": cur["h1"], cur["h2"], cur["h3"] = heading_text, "", ""
            elif name == "h2": cur["h2"], cur["h3"] = heading_text, ""
            else: cur["h3"] = heading_text
        elif name in ["p","li","td","th","blockquote","pre"]:
            t = normalize_ws_keep_diacritics(node.get_text(" ").strip())
            if len(t) >= 1:
                cur["text"].append(t)
    close_section()

    full_text = normalize_ws_keep_diacritics(body.get_text("\n"))
    full_text = glue_codes(full_text)

    return full_text, sections, title

# sentence split (kept)
SENT_SPLIT = re.compile(r"(?<=[\.!?…])\s+(?=[A-Z" + TR_DIACRITICS + r"0-9])")
def sentence_split(text: str) -> List[str]:
    sents = SENT_SPLIT.split(text)
    merged, buf = [], ""
    for s in sents:
        s = s.strip()
        if not s: continue
        if len(buf) + len(s) < 150:
            buf = (buf + " " + s).strip()
        else:
            if buf: merged.append(buf)
            buf = s
    if buf: merged.append(buf)
    return merged

def chunk_text(text: str, chunk_size: int, overlap: int) -> List[Dict]:
    sents = sentence_split(text)
    out, idx, i = [], 0, 0
    while i < len(sents):
        buf, total, j = [], 0, i
        while j < len(sents) and total + len(sents[j]) + 1 <= chunk_size:
            buf.append(sents[j]); total += len(sents[j]) + 1; j += 1
        if not buf: buf, j = [sents[i]], i+1
        chunk = " ".join(buf).strip()
        if chunk:
            start = max(0, text.find(buf[0]))
            end   = start + len(chunk)
            out.append({"text": chunk, "start": start, "end": end, "index": idx})
            idx += 1
        step = max(1, len(buf) - max(1, math.ceil(overlap / 100)))
        i += step
    return out

def chunk_sections(sections: List[Dict], chunk_size: int, overlap: int, doc_text_for_offsets: str) -> List[Dict]:
    """Chunk each section separately; attach section metadata."""
    out, idx = [], 0
    for sec in sections:
        stext = sec["text"]
        if not stext: continue
        chunks = chunk_text(stext, chunk_size, overlap)
        for ch in chunks:
            ch["index"] = idx
            # map approx offsets to the doc for debugging
            start = doc_text_for_offsets.find(ch["text"][:30])
            ch["start"] = max(0, start)
            ch["end"] = start + len(ch["text"]) if start >= 0 else ch["start"] + len(ch["text"])
            ch["h1"], ch["h2"], ch["h3"] = sec["h1"], sec["h2"], sec["h3"]
            out.append(ch)
            idx += 1
    return out

def dedupe_chunks(chunks: List[Dict]) -> List[Dict]:
    seen, ded = set(), []
    for ch in chunks:
        key = sha1_text(ch["text"])
        if key in seen: continue
        seen.add(key); ded.append(ch)
    return ded

# ============ MANIFEST ============
def init_manifest(db_path: str = MANIFEST_DB):
    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    c.execute("""CREATE TABLE IF NOT EXISTS files(
        path TEXT PRIMARY KEY,
        mtime REAL,
        size INTEGER,
        content_sha1 TEXT,
        processed INTEGER DEFAULT 0,
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

def record_file(conn, path: str, mtime: float, size: int, content_sha1: str, total_chunks: int):
    c = conn.cursor()
    c.execute("""INSERT OR REPLACE INTO files(path, mtime, size, content_sha1, processed, total_chunks)
                 VALUES (?,?,?,?,COALESCE((SELECT processed FROM files WHERE path=?),0),?)""",
              (path, mtime, size, content_sha1, path, total_chunks))
    conn.commit()

def mark_processed(conn, path: str):
    c = conn.cursor()
    c.execute("UPDATE files SET processed=1 WHERE path=?", (path,))
    conn.commit()

def already_processed(conn, path: str, mtime: float, size: int, content_sha1: str) -> bool:
    c = conn.cursor()
    c.execute("SELECT processed, mtime, size, content_sha1 FROM files WHERE path=?", (path,))
    row = c.fetchone()
    if not row: return False
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

# ============ EMBEDDING ============
def embed_one(text: str, model: str = EMBED_MODEL) -> np.ndarray:
    url = f"{OLLAMA_HOST}/api/embeddings"
    for attempt in range(RETRY_MAX):
        try:
            r = requests.post(url, json={"model": model, "prompt": text}, timeout=REQUEST_TIMEOUT_S)
            r.raise_for_status()
            v = np.array(r.json()["embedding"], dtype=np.float32)
            v /= (np.linalg.norm(v) + 1e-12)   # L2 norm
            if v.shape[0] != EMBED_DIM:
                raise RuntimeError(f"Unexpected dim: got {v.shape[0]} vs {EMBED_DIM}")
            return v
        except Exception as e:
            sleep = (BACKOFF_BASE * (2 ** attempt)) * (0.5 + np.random.rand())
            time.sleep(sleep)
            if attempt == RETRY_MAX - 1:
                raise

def embed_many_parallel(texts: List[str], max_workers=MAX_WORKERS_EMBED) -> np.ndarray:
    vecs = [None] * len(texts)
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = {ex.submit(embed_one, texts[i]): i for i in range(len(texts))}
        for fut in as_completed(futures):
            i = futures[fut]
            vecs[i] = fut.result()
    return np.vstack(vecs)

# ============ CHROMA & PARQUET ============
def get_or_create_collection(client, name: str):
    # If your Chroma accepts HNSW params, add them here:
    return client.get_or_create_collection(
        name=name,
        metadata={
            "hnsw:space": "cosine",
            # "hnsw:M": "64",
            # "hnsw:ef_construction": "200",
            # "hnsw:ef_search": "128",
        }
    )

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

# ============ MAIN ============
def main():
    conn = init_manifest(MANIFEST_DB)
    client = chromadb.PersistentClient(path=CHROMA_DIR)
    coll  = get_or_create_collection(client, COLL_NAME)

    exts = {".pdf", ".html", ".htm"}
    files = []
    for root, _, fnames in os.walk(ROOT_DIR):
        for fn in fnames:
            if os.path.splitext(fn)[1].lower() in exts:
                files.append(os.path.join(root, fn))
    files.sort()

    total_upserts = 0
    shard_ix = 0
    sb = {"ids": [], "vecs": [], "docs": [], "metas": []}
    processed_files, started_at = 0, time.time()

    for path in files:
        try:
            stat = os.stat(path)
            mtime, size = stat.st_mtime, stat.st_size
            with open(path, "rb") as f:
                content_bytes = f.read()
            content_sha = sha1_bytes(content_bytes)

            if already_processed(conn, path, mtime, size, content_sha):
                continue

            ext = os.path.splitext(path)[1].lower()
            if ext == ".pdf":
                doc_text, title = load_pdf_text(path)
                sections = [{"h1":"","h2":"","h3":"","text":doc_text}]  # treat whole PDF as one section
                doc_for_offsets = doc_text
            else:
                doc_for_offsets, sections, title = load_html_sections(path)

            if len(doc_for_offsets) < MIN_TEXT_LEN:
                record_file(conn, path, mtime, size, content_sha, 0)
                mark_processed(conn, path)
                continue

            chunks = chunk_sections(sections, CHUNK_SIZE, CHUNK_OVERLAP, doc_for_offsets)
            chunks = dedupe_chunks(chunks)
            record_file(conn, path, mtime, size, content_sha, len(chunks))

            # in bounded mini-batches
            for i in range(0, len(chunks), BATCH_UPSERT):
                batch = chunks[i:i+BATCH_UPSERT]
                filtered = []
                for b in batch:
                    ch_sha = sha1_text(b["text"])
                    if is_new_chunk(conn, ch_sha):
                        remember_chunk(conn, ch_sha)
                        filtered.append((b, ch_sha))
                if not filtered:
                    continue

                texts = [b["text"] for b,_ in filtered]
                ids   = [sha1_text(f"{path}::{b['index']}") for b,_ in filtered]
                metas = [{
                    "doc_path": path,
                    "title": title,
                    "start": b["start"],
                    "end": b["end"],
                    "chunk_index": b["index"],
                    "char_count": len(b["text"]),
                    "lang": "tr" if re.search(f"[{TR_DIACRITICS}]", b["text"][:400]) else "en",
                    "h1": b.get("h1",""), "h2": b.get("h2",""), "h3": b.get("h3",""),
                    "preview": b["text"][:220],
                    "doctype": "pdf" if ext==".pdf" else "html"
                } for b,_ in filtered]

                vecs = embed_many_parallel(texts)
                coll.upsert(ids=ids, embeddings=vecs.tolist(), metadatas=metas, documents=texts)

                total_upserts += len(ids)
                sb["ids"].extend(ids); sb["vecs"].extend(vecs.tolist())
                sb["docs"].extend(texts); sb["metas"].extend(metas)

                if len(sb["ids"]) >= PARQUET_SHARD_EVERY:
                    shard_ix += 1
                    write_parquet_shard(PARQUET_OUT_DIR, shard_ix, sb["ids"], sb["vecs"], sb["docs"], sb["metas"])
                    sb = {"ids": [], "vecs": [], "docs": [], "metas": []}

            mark_processed(conn, path)
            processed_files += 1
            if processed_files % 25 == 0:
                elapsed = time.time() - started_at
                print(f"[progress] files={processed_files} upserts={total_upserts} elapsed={elapsed:.1f}s")

        except Exception as e:
            log_error(conn, path, "process", str(e))
            continue

    if sb["ids"]:
        shard_ix += 1
        write_parquet_shard(PARQUET_OUT_DIR, shard_ix, sb["ids"], sb["vecs"], sb["docs"], sb["metas"])

    print("✅ Done.")
    try:
        print(f" - {COLL_NAME}: {coll.count()} vectors")
    except Exception as e:
        print(f" - {COLL_NAME}: count unavailable ({e})")
    print(f"Parquet out: {PARQUET_OUT_DIR}")
    print(f"Manifest DB: {MANIFEST_DB}")
    print(f"Logs:        {LOG_FILE}")

if __name__ == "__main__":
    main()
