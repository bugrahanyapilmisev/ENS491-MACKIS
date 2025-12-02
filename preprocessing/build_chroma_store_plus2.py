# build_chroma_store_plus2.py
#
# - Reads preprocessed JSON docs from PREPROCESSED_ROOT (output of preprocess_docs.py)
# - Chunks text, embeds with OLLAMA bge-m3
# - Upserts into Chroma collection
# - Writes two checkpoints:
#     * chunks_plus2.parquet  (chunk text + metadata)
#     * vectors_plus2.parquet (embeddings + metadata)
#
# Now with:
# - LLM-based OPEN-ENDED semantic tags per document (metadata["tags"])

import os
import re
import json
import time
import math
import hashlib
import pathlib
import textwrap
from typing import List, Dict, Tuple

import numpy as np
import pandas as pd
import requests
import chromadb
from concurrent.futures import ThreadPoolExecutor, as_completed

import pyarrow as pa
import pyarrow.parquet as pq
from dotenv import load_dotenv

# ================= CONFIG =================
load_dotenv()

OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
EMBED_MODEL = os.getenv("EMBED_MODEL", "bge-m3")
EMBED_DIM   = int(os.getenv("EMBED_DIM", "1024"))
CHAT_MODEL  = os.getenv("CHAT_MODEL", "llama3.2")  # used for tag LLM

# Paths / dirs
BASE_DIR = os.getenv("PROJECT_ROOT") or os.getcwd()
PREPROCESSING_DIR = os.getenv("PREPROCESSING_PATH") or os.path.join(BASE_DIR, "preprocessing")
PRE_ROOT = os.getenv("PREPROCESSED_ROOT") or os.path.join(PREPROCESSING_DIR, "preprocessed_docs")

CHROMA_DIR = os.getenv("CHROMA_DIR") or os.path.join(PREPROCESSING_DIR, "chroma_db_mysu")
CHECKPOINT_DIR = os.getenv("CHECKPOINT_DIR") or os.path.join(PREPROCESSING_DIR, "checkpoints_plus2")

CHUNK_PARQUET   = os.path.join(CHECKPOINT_DIR, "chunks_plus2.parquet")
VECTORS_PARQUET = os.path.join(CHECKPOINT_DIR, "vectors_plus2.parquet")

COLL_NAME   = os.getenv("COLL_NAME", "mysu_surecharitasi_bge_m3_v1")

# performance
MAX_WORKERS_EMBED = int(os.getenv("MAX_WORKERS_EMBED", "6"))
BATCH_UPSERT      = int(os.getenv("BATCH_UPSERT", "64"))
REQUEST_TIMEOUT_S = int(os.getenv("REQUEST_TIMEOUT_S", "120"))
RETRY_MAX         = int(os.getenv("RETRY_MAX", "4"))
BACKOFF_BASE      = float(os.getenv("BACKOFF_BASE", "0.7"))

CHUNK_SIZE    = int(os.getenv("CHUNK_SIZE", "900"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "180"))
MIN_TEXT_LEN  = int(os.getenv("MIN_TEXT_LEN", "50"))

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

def decide_doc_lang(lang_field: str, text_lang: str) -> str:
    """
    lang_field: 'lang' from preprocess_docs JSON (may be 'tr'/'en')
    text_lang: language guessed from text content
    Priority: lang_field if valid, else text_lang, else 'en'
    """
    for c in (lang_field, text_lang):
        if not c:
            continue
        c = c.lower()
        if c.startswith("tr"):
            return "tr"
        if c.startswith("en"):
            return "en"
    return "en"

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

# ============ LLM TAGGING (OPEN-ENDED TAGS) ============

def call_ollama_json(prompt: str, system_prompt: str = "") -> dict:
    """
    Call Ollama /api/chat and expect STRICT JSON in the response.
    If anything goes wrong, return {}.
    """
    url = f"{OLLAMA_HOST}/api/chat"

    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})

    payload = {
        "model": CHAT_MODEL,
        "messages": messages,
        "stream": False,
        "options": {
            "temperature": 0.0,
        },
    }

    try:
        resp = requests.post(url, json=payload, timeout=300)
        resp.raise_for_status()
        data = resp.json()
        text = (data.get("message", {}) or {}).get("content", "").strip()
        if not text:
            return {}

        # Try direct JSON first
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            m = re.search(r"\{.*\}", text, flags=re.DOTALL)
            if not m:
                return {}
            return json.loads(m.group(0))
    except Exception as e:
        print(f"[tagger-llm error] {e}")
        return {}

def infer_doc_tags_llm(title: str, doc_text: str, doc_lang: str) -> Dict:
    """
    Use an LLM to infer OPEN-ENDED semantic tags for a university document.

    Output schema:
    {
      "tags": ["short_snake_case_tag1", "tag2", ...]
    }

    - tags is an *unbounded* vocabulary: the model can invent any tag string.
    - We only suggest some canonical labels (erasmus_study, erasmus_internship, ...)
      in the prompt, but we DO NOT restrict the set of possible tags.
    """

    body_sample = (doc_text or "")[:2500]
    lang_hint = doc_lang or "unknown"

    sys_prompt = textwrap.dedent("""
        You are an expert classifier for internal university documents.
        Your ONLY job is to read the document title and a text excerpt and
        return a JSON object describing what the document is about via TAGS.

        JSON schema (STRICT):
        {
          "tags": ["short_snake_case_tag1", "tag2", ...]
        }

        Rules for tags:
        - 3 to 12 tags is ideal, but NOT a hard limit.
        - Tags must be SHORT, lowercase, snake_case strings.
        - Tags should be semantically informative:
            * topics (e.g. "erasmus_plus", "student_mobility", "scholarships")
            * document role (e.g. "procedure", "directive", "regulation", "form")
            * target audience (e.g. "undergraduate_students", "graduate_students")
            * important programs or units (e.g. "faculty_of_engineering", "international_office")
        - Use English tags even if the document is in Turkish.
        - Do NOT include extremely generic tags like "document", "general", "other".

        IMPORTANT CONVENTION (NOT a hard restriction):
        - If the document is clearly about Erasmus *study* / exchange mobility,
          please include a tag "erasmus_study".
        - If the document is clearly about Erasmus *internship* / traineeship,
          please include a tag "erasmus_internship".
        - If the document is clearly about student admission, add something like
          "undergraduate_admission" or "graduate_admission", as appropriate.

        However, these are ONLY conventions. You are free to invent any additional tags
        that make sense. The overall tag vocabulary is OPEN-ENDED and unbounded.

        Output requirements:
        - Output STRICT JSON ONLY, no explanation, no markdown.
        - Make sure the JSON is valid and matches the schema.
    """).strip()

    user_prompt = textwrap.dedent(f"""
        Language: {lang_hint}

        Title:
        {title or "(no title)"}

        Excerpt:
        {body_sample}
    """).strip()

    raw = call_ollama_json(user_prompt, system_prompt=sys_prompt)

    tags = raw.get("tags", [])
    if not isinstance(tags, list):
        tags = []

    clean_tags = []
    seen = set()
    for t in tags:
        if not isinstance(t, str):
            continue
        tt = t.strip().lower()
        if not tt:
            continue
        if tt in seen:
            continue
        seen.add(tt)
        clean_tags.append(tt)

    return {
        "tags": clean_tags,
    }

# ================ MAIN ======================

def main():
    # 1) Collect all preprocessed JSON files
    json_files: List[str] = []
    for root, _, fnames in os.walk(PRE_ROOT):
        for fn in fnames:
            if not fn.lower().endswith(".json"):
                continue
            json_files.append(os.path.join(root, fn))
    json_files.sort()

    print(f"[info] Found {len(json_files)} preprocessed JSON docs under {PRE_ROOT}")

    # 2) Create Chroma collection
    client = chromadb.PersistentClient(path=CHROMA_DIR)
    coll = client.get_or_create_collection(
        name=COLL_NAME,
        metadata={"hnsw:space": "cosine"},
    )

    global_seen_chunks = set()  # SHA1(text) for global dedupe
    chunk_rows = []             # for chunks_plus2.parquet
    vector_rows = []            # for vectors_plus2.parquet

    total_upserts = 0
    started = time.time()

    for fi, path in enumerate(json_files, 1):
        try:
            # ---------- LOAD JSON ----------
            with open(path, "r", encoding="utf-8") as f:
                doc = json.load(f)

            doc_text = normalize_ws_keep_diacritics(doc.get("text", "") or "")
            if len(doc_text) < MIN_TEXT_LEN:
                print(f"[skip] too short text in {path}")
                continue

            title       = (doc.get("title") or "").strip()
            html_lang   = (doc.get("html_lang") or "").strip()
            lang_field  = (doc.get("lang") or "").strip()
            source_path = (doc.get("source_path") or "").strip()

            # Fallbacks
            if not source_path:
                # relative path of this JSON inside PRE_ROOT
                source_path = os.path.relpath(path, PRE_ROOT).replace("\\", "/")

            text_lang = guess_lang_from_text(doc_text)
            doc_lang  = decide_doc_lang(lang_field, text_lang)

            # ---------- LLM-BASED SEMANTIC TAGGING (OPEN-ENDED) ----------
            tag_info = infer_doc_tags_llm(
                title=title,
                doc_text=doc_text,
                doc_lang=doc_lang,
            )
            tags = tag_info.get("tags", [])
            tags_str = ",".join(tags)

            # Treat the whole document as one section; h1 = title
            sections = [{
                "h1": title,
                "h2": "",
                "h3": "",
                "text": doc_text,
            }]

            # ---------- CHUNKING ----------
            chunks = chunk_sections(sections, CHUNK_SIZE, CHUNK_OVERLAP, doc_text)
            chunks = dedupe_chunks(chunks)
            if not chunks:
                continue

            # ---------- EMBED + UPSERT ----------
            size_bytes = len(doc_text.encode("utf-8", errors="ignore"))
            json_rel   = os.path.relpath(path, PRE_ROOT).replace("\\", "/")

            for i in range(0, len(chunks), BATCH_UPSERT):
                batch = chunks[i: i + BATCH_UPSERT]

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
                    chunk_id = sha1_text(f"{source_path}::{chunk_index}")

                    meta = {
                        "chunk_id": chunk_id,
                        "source_path": source_path,          # original relative path from preprocess
                        "json_path": json_rel,               # path of this JSON under PRE_ROOT
                        "title": title,
                        "h1": ch.get("h1", ""),
                        "h2": ch.get("h2", ""),
                        "h3": ch.get("h3", ""),
                        "chunk_index": int(chunk_index),
                        "start": int(ch.get("start", 0)),
                        "end": int(ch.get("end", 0)),
                        "content_lang": doc_lang,
                        "html_lang": html_lang,
                        "doc_lang": doc_lang,
                        "doctype": pathlib.Path(source_path).suffix.lstrip(".") or "txt",
                        "bytes": int(size_bytes),

                        # OPEN-ENDED TAGS LIST
                        # OPEN-ENDED TAGS, BUT AS STRINGS (Chroma doesn't allow list in metadata)
                        "tags": tags_str,  # e.g. "erasmus_study,faculty_of_engineering"
                        "tags_json": json.dumps(tags, ensure_ascii=False),
                    }

                    chunk_rows.append({
                        "chunk_id": chunk_id,
                        "content": ch_text,
                        "source_path": source_path,
                        "json_path": json_rel,
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

                        # tags stored as CSV string for quick inspection
                        "tags": tags_str,
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
                print(f"[progress] json files={fi}/{len(json_files)} "
                      f"upserts={total_upserts} elapsed={elapsed:.1f}s")

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
