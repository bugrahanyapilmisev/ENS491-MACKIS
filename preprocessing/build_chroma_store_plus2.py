# build_chroma_store_plus2.py
import os
import re
import json
import time
import math
import hashlib
import pathlib
from typing import List, Dict, Tuple

import numpy as np
import pandas as pd
import requests
from pypdf import PdfReader
from bs4 import BeautifulSoup

try:
    import docx  # for .docx
except ImportError:
    docx = None

try:
    import textract  # optional, for legacy .doc
except ImportError:
    textract = None

print(textract)
import chromadb
from concurrent.futures import ThreadPoolExecutor, as_completed

import pyarrow as pa
import pyarrow.parquet as pq

# ================= CONFIG =================
OLLAMA_HOST = "http://localhost:11434"
EMBED_MODEL = "bge-m3"
EMBED_DIM   = 1024

ROOT_DIR    = os.getenv("ROOT_DIR")
CHROMA_DIR  = r"C:\\Users\\kosot\\Documents\\bitirme\\preprocessing\\chroma_db_mysu"
CHECKPOINT_DIR = r"C:\\Users\\kosot\\Documents\\bitirme\\preprocessing\\checkpoints_plus2"

CHUNK_PARQUET   = os.path.join(CHECKPOINT_DIR, "chunks_plus2.parquet")
VECTORS_PARQUET = os.path.join(CHECKPOINT_DIR, "vectors_plus2.parquet")

COLL_NAME   = "mysu_surecharitasi_bge_m3_v1"

# performance
MAX_WORKERS_EMBED = 6
BATCH_UPSERT      = 64
REQUEST_TIMEOUT_S = 120
RETRY_MAX         = 4
BACKOFF_BASE      = 0.7

CHUNK_SIZE    = 900
CHUNK_OVERLAP = 180
MIN_TEXT_LEN  = 50

TR_DIACRITICS = "çğıöşüÇĞİÖŞÜ"

os.makedirs(CHROMA_DIR, exist_ok=True)
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

# ================ UTILS ====================
def sha1_bytes(b: bytes) -> str:
    return hashlib.sha1(b).hexdigest()

def sha1_text(s: str) -> str:
    return sha1_bytes(s.encode("utf-8", errors="ignore"))

def normalize_ws_keep_diacritics(text: str) -> str:
    text = text.replace("\xa0", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\s*\n\s*\n\s*\n+", "\n\n", text)
    return text.strip()

def glue_codes(text: str) -> str:
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)
    return text

def fix_pdf_wrapping(text: str) -> str:
    text = glue_codes(text)
    text = re.sub(
        r"(?<![\.!?:])\n(?!\s*[A-Z" + TR_DIACRITICS + r"0-9])",
        " ",
        text,
    )
    return normalize_ws_keep_diacritics(text)

# ============ LANGUAGE DETECTION ============
try:
    from langdetect import detect as ld_detect
except ImportError:
    ld_detect = None

def guess_lang_from_text(text: str) -> str:
    s = (text or "").strip()
    if not s:
        return "en"
    sample = s[:4000]
    if ld_detect is not None:
        try:
            code = ld_detect(sample)
            code = (code or "").lower()
            if code.startswith("tr"):
                return "tr"
            if code.startswith("en"):
                return "en"
        except Exception:
            pass
    if re.search(f"[{TR_DIACRITICS}]", sample):
        return "tr"
    return "en"

def detect_html_lang_from_soup(soup: BeautifulSoup) -> str:
    html_lang = ""
    html_tag = soup.find("html")
    if html_tag is not None:
        lang_attr = html_tag.get("lang") or html_tag.get("xml:lang")
        if lang_attr:
            code = lang_attr.split("-")[0].lower()
            if code in ("tr", "en"):
                html_lang = code
    return html_lang

def decide_doc_lang(html_lang: str, text_lang: str) -> str:
    """
    IMPORTANT: we DO NOT trust path /en or /tr here.
    Only HTML lang + text lang.
    """
    for c in (html_lang, text_lang):
        if not c:
            continue
        c = c.lower()
        if c.startswith("tr"):
            return "tr"
        if c.startswith("en"):
            return "en"
    return "en"

# ================ LOADERS ===================
def load_pdf_text(path: str) -> Tuple[str, str]:
    reader = PdfReader(path)
    pages = [(p.extract_text() or "") for p in reader.pages]
    text = "\n".join(pages)
    text = fix_pdf_wrapping(text)
    title = ""
    return text, title

def load_html_sections(path: str):
    """
    Return:
      full_text, sections, title, html_lang
    sections: list of {h1,h2,h3,text}
    """
    html = pathlib.Path(path).read_text(encoding="utf-8", errors="ignore")
    soup = BeautifulSoup(html, "lxml")

    html_lang = detect_html_lang_from_soup(soup)

    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()
    for sel in [
        "nav", "footer", "header", ".menu", ".navbar", ".breadcrumb", ".sidebar",
        "#cookie-banner", ".cookie", ".cookies", ".site-footer", ".site-header",
        ".social-share", ".pagination", ".advert", ".ad", ".banner"
    ]:
        for node in soup.select(sel):
            node.decompose()

    title = (soup.title.string.strip() if soup.title and soup.title.string else "")
    body = soup.select_one("main") or soup.select_one("article") or soup

    sections = []
    cur = {"h1": "", "h2": "", "h3": "", "text": []}

    def close_section():
        if cur["text"]:
            sections.append({
                "h1": cur["h1"],
                "h2": cur["h2"],
                "h3": cur["h3"],
                "text": normalize_ws_keep_diacritics("\n".join(cur["text"])),
            })
            cur["text"] = []

    for node in body.descendants:
        if not getattr(node, "name", None):
            continue
        name = node.name.lower()
        if name in ["h1", "h2", "h3"]:
            close_section()
            heading_text = normalize_ws_keep_diacritics(node.get_text(" ").strip())
            if name == "h1":
                cur["h1"], cur["h2"], cur["h3"] = heading_text, "", ""
            elif name == "h2":
                cur["h2"], cur["h3"] = heading_text, ""
            else:
                cur["h3"] = heading_text
        elif name in ["p", "li", "td", "th", "blockquote", "pre"]:
            t = normalize_ws_keep_diacritics(node.get_text(" ").strip())
            if len(t) >= 1:
                cur["text"].append(t)
    close_section()

    full_text = normalize_ws_keep_diacritics(body.get_text("\n"))
    full_text = glue_codes(full_text)

    return full_text, sections, title, html_lang

def load_docx_text(path: str) -> Tuple[str, str]:
    if docx is None:
        raise RuntimeError("python-docx not installed; cannot read .docx")
    d = docx.Document(path)
    parts = []
    for p in d.paragraphs:
        if p.text:
            parts.append(p.text)
    text = normalize_ws_keep_diacritics("\n".join(parts))
    title = os.path.splitext(os.path.basename(path))[0]
    return text, title

def load_legacy_doc_text(path: str) -> Tuple[str, str]:
    if textract is None:
        raise RuntimeError("textract not installed; cannot read .doc")
    raw = textract.process(path)
    text = normalize_ws_keep_diacritics(raw.decode("utf-8", errors="ignore"))
    title = os.path.splitext(os.path.basename(path))[0]
    return text, title

def load_excel_text(path: str) -> Tuple[str, str]:
    import pandas as pd
    xls = pd.ExcelFile(path)
    parts = []
    for sheet in xls.sheet_names:
        df = xls.parse(sheet)
        parts.append(f"Sheet: {sheet}")
        parts.append(df.to_string(index=False))
    text = normalize_ws_keep_diacritics("\n".join(parts))
    title = os.path.splitext(os.path.basename(path))[0]
    return text, title

def load_csv_text(path: str) -> Tuple[str, str]:
    import pandas as pd
    df = pd.read_csv(path)
    text = normalize_ws_keep_diacritics(df.to_string(index=False))
    title = os.path.splitext(os.path.basename(path))[0]
    return text, title

# =============== CHUNKING ===================
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
    out = []
    idx = 0
    i = 0
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
            start = max(0, text.find(buf[0]))
            out.append({"text": chunk, "start": start, "end": start + len(chunk), "index": idx})
            idx += 1
        step = max(1, len(buf) - 1)
        i += step
    return out

def chunk_sections(sections: List[Dict], chunk_size: int, overlap: int, doc_text: str) -> List[Dict]:
    out = []
    idx = 0
    for sec in sections:
        stext = sec["text"]
        if not stext:
            continue
        chunks = chunk_text(stext, chunk_size, overlap)
        for ch in chunks:
            ch["index"] = idx
            # best-effort mapping to original text
            start = doc_text.find(ch["text"][:30])
            if start < 0:
                start = 0
            ch["start"] = start
            ch["end"] = start + len(ch["text"])
            ch["h1"] = sec.get("h1", "")
            ch["h2"] = sec.get("h2", "")
            ch["h3"] = sec.get("h3", "")
            out.append(ch)
            idx += 1
    return out

def dedupe_chunks(chunks: List[Dict]) -> List[Dict]:
    seen = set()
    ded = []
    for ch in chunks:
        key = sha1_text(ch["text"])
        if key in seen:
            continue
        seen.add(key)
        ded.append(ch)
    return ded

# ============ EMBEDDINGS (OLLAMA) ============
def embed_one(text: str) -> np.ndarray:
    url = f"{OLLAMA_HOST}/api/embeddings"
    for attempt in range(RETRY_MAX):
        try:
            r = requests.post(
                url,
                json={"model": EMBED_MODEL, "prompt": text},
                timeout=REQUEST_TIMEOUT_S,
            )
            r.raise_for_status()
            v = np.array(r.json()["embedding"], dtype=np.float32)
            v /= (np.linalg.norm(v) + 1e-12)
            if v.shape[0] != EMBED_DIM:
                raise RuntimeError(f"Unexpected dim: {v.shape[0]} vs {EMBED_DIM}")
            return v
        except Exception:
            time.sleep(BACKOFF_BASE * (2 ** attempt))
            if attempt == RETRY_MAX - 1:
                raise
    raise RuntimeError("unreachable")

def embed_many_parallel(texts: List[str]) -> np.ndarray:
    vecs = [None] * len(texts)
    with ThreadPoolExecutor(max_workers=MAX_WORKERS_EMBED) as ex:
        futs = {ex.submit(embed_one, texts[i]): i for i in range(len(texts))}
        for fut in as_completed(futs):
            i = futs[fut]
            vecs[i] = fut.result()
    return np.vstack(vecs)

# ================ MAIN ======================
def main():
    client = chromadb.PersistentClient(path=CHROMA_DIR)
    coll = client.get_or_create_collection(
        name=COLL_NAME,
        metadata={"hnsw:space": "cosine"},
    )

    exts = {".pdf", ".html", ".htm", ".docx", ".doc", ".xlsx", ".xls", ".csv"}
    files = []
    for root, _, fnames in os.walk(ROOT_DIR):
        for fn in fnames:
            ext = os.path.splitext(fn)[1].lower()
            if ext in exts:
                files.append(os.path.join(root, fn))
    files.sort()

    global_seen_chunks = set()  # SHA1(text) for global dedupe
    chunk_rows = []             # for chunks_plus2.parquet
    vector_rows = []            # for vectors_plus2.parquet

    total_upserts = 0
    started = time.time()

    for fi, path in enumerate(files, 1):
        try:
            ext = os.path.splitext(path)[1].lower()
            size_bytes = os.path.getsize(path)

            # ---------- LOAD ----------
            if ext in [".html", ".htm"]:
                doc_text, sections, title, html_lang = load_html_sections(path)
            elif ext == ".pdf":
                doc_text, title = load_pdf_text(path)
                sections = [{"h1": "", "h2": "", "h3": "", "text": doc_text}]
                html_lang = ""
            elif ext == ".docx":
                doc_text, title = load_docx_text(path)
                sections = [{"h1": "", "h2": "", "h3": "", "text": doc_text}]
                html_lang = ""
            elif ext == ".doc":
                doc_text, title = load_legacy_doc_text(path)
                sections = [{"h1": "", "h2": "", "h3": "", "text": doc_text}]
                html_lang = ""
            elif ext in [".xlsx", ".xls"]:
                doc_text, title = load_excel_text(path)
                sections = [{"h1": "", "h2": "", "h3": "", "text": doc_text}]
                html_lang = ""
            elif ext == ".csv":
                doc_text, title = load_csv_text(path)
                sections = [{"h1": "", "h2": "", "h3": "", "text": doc_text}]
                html_lang = ""
            else:
                continue

            if len(doc_text) < MIN_TEXT_LEN:
                print(f"[skip] too short: {path}")
                continue

            # ---------- LANG ----------
            text_lang = guess_lang_from_text(doc_text)
            doc_lang = decide_doc_lang(html_lang, text_lang)

            # ---------- CHUNKING ----------
            chunks = chunk_sections(sections, CHUNK_SIZE, CHUNK_OVERLAP, doc_text)
            chunks = dedupe_chunks(chunks)
            if not chunks:
                continue

            # ---------- EMBED + UPSERT ----------
            for i in range(0, len(chunks), BATCH_UPSERT):
                batch = chunks[i : i + BATCH_UPSERT]

                texts = []
                ids = []
                metas = []

                for ch in batch:
                    ch_text = ch["text"]
                    ch_sha = sha1_text(ch_text)
                    if ch_sha in global_seen_chunks:
                        continue
                    global_seen_chunks.add(ch_sha)

                    chunk_index = ch["index"]
                    chunk_id = sha1_text(f"{path}::{chunk_index}")

                    meta = {
                        "doc_path": path,
                        "title": title,
                        "h1": ch.get("h1", ""),
                        "h2": ch.get("h2", ""),
                        "h3": ch.get("h3", ""),
                        "chunk_index": int(chunk_index),
                        "start": int(ch.get("start", 0)),
                        "end": int(ch.get("end", 0)),
                        "content_lang": doc_lang,   # we keep content_lang, but it is doc-level here
                        "html_lang": html_lang,
                        "doc_lang": doc_lang,
                        "doctype": ext.lstrip("."),
                        "bytes": int(size_bytes),
                    }

                    chunk_rows.append({
                        "chunk_id": chunk_id,
                        "content": ch_text,
                        "doc_path": path,
                        "title": title,
                        "h1": meta["h1"],
                        "h2": meta["h2"],
                        "h3": meta["h3"],
                        "chunk_index": int(chunk_index),
                        "start": meta["start"],
                        "end": meta["end"],
                        "content_lang": doc_lang,
                        "html_lang": html_lang,
                        "doc_lang": doc_lang,
                        "doctype": meta["doctype"],
                        "bytes": int(size_bytes),
                    })

                    texts.append(ch_text)
                    ids.append(chunk_id)
                    metas.append(meta)

                if not texts:
                    continue

                vecs = embed_many_parallel(texts)
                coll.upsert(
                    ids=ids,
                    embeddings=vecs.tolist(),
                    metadatas=metas,
                    documents=texts,
                )
                total_upserts += len(ids)

                # also record vector checkpoint
                for j in range(len(ids)):
                    vector_rows.append({
                        "id": ids[j],
                        "vector": vecs[j].astype(np.float32).tolist(),
                        "document": texts[j],
                        "metadata": json.dumps(metas[j], ensure_ascii=False),
                    })

            if fi % 20 == 0:
                elapsed = time.time() - started
                print(f"[progress] files={fi}/{len(files)} upserts={total_upserts} elapsed={elapsed:.1f}s")

        except Exception as e:
            print(f"[error] {path}: {e}")
            continue

    # ---------- WRITE CHUNKS PARQUET ----------
    if chunk_rows:
        df = pd.DataFrame(chunk_rows)
        df.to_parquet(CHUNK_PARQUET, index=False)
        print(f"✅ chunks parquet written: {CHUNK_PARQUET} (rows={len(df)})")

    # ---------- WRITE VECTORS PARQUET ----------
    if vector_rows:
        ids = [r["id"] for r in vector_rows]
        docs = [r["document"] for r in vector_rows]
        metas_json = [r["metadata"] for r in vector_rows]
        vecs = [r["vector"] for r in vector_rows]

        table = pa.table({
            "id":       pa.array(ids, type=pa.string()),
            "vector":   pa.array(vecs, type=pa.list_(pa.float32())),
            "document": pa.array(docs, type=pa.string()),
            "metadata": pa.array(metas_json, type=pa.string()),
        })
        pq.write_table(table, VECTORS_PARQUET)
        print(f"✅ vectors parquet written: {VECTORS_PARQUET} (rows={len(ids)})")

    print("✅ Build finished.")
    try:
        print(f"Collection {COLL_NAME} count() -> {coll.count()}")
    except Exception as e:
        print(f"count() failed: {e}")

if __name__ == "__main__":
    main()
