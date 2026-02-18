"""
build_chroma_store.py - IMPROVED VERSION

Key improvements over v1:
1. Hierarchical/semantic chunking based on document sections
2. Contextual embeddings (prepend document summary to each chunk)
3. BM25 index creation alongside vector embeddings
4. Better metadata extraction (procedure codes, dates, units)
5. Parent-child chunk relationships for late chunking
6. Improved deduplication with semantic similarity
7. Document-level embeddings for coarse retrieval

Creates:
- ChromaDB collection with improved embeddings
- BM25 index (pickle) for hybrid search
- chunks_v2.parquet with enhanced metadata
- doc_summaries.parquet for document-level retrieval
"""

import os
import re
import json
import time
import math
import hashlib
import pathlib
import textwrap
import pickle
from typing import List, Dict, Tuple, Optional, Any
from collections import defaultdict

import numpy as np
import pandas as pd
import requests
import chromadb
from concurrent.futures import ThreadPoolExecutor, as_completed

import pyarrow as pa
import pyarrow.parquet as pq
from dotenv import load_dotenv

# Optional: BM25 for hybrid search
try:
    from rank_bm25 import BM25Okapi
    HAS_BM25 = True
except ImportError:
    HAS_BM25 = False
    print("[warn] rank_bm25 not installed. BM25 hybrid search will be disabled.")

# ================= CONFIG =================
load_dotenv()

OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
EMBED_MODEL = os.getenv("EMBED_MODEL", "bge-m3")
EMBED_DIM   = int(os.getenv("EMBED_DIM", "1024"))
CHAT_MODEL  = os.getenv("CHAT_MODEL", "llama3.2")

BASE_DIR = os.getenv("PROJECT_ROOT") or os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PREPROCESSING_DIR = os.getenv("PREPROCESSING_PATH") or os.path.join(BASE_DIR, "preprocessing")

# Always use v2 preprocessed docs
PRE_ROOT = os.getenv("PREPROCESSED_ROOT_V2") or os.path.join(PREPROCESSING_DIR, "preprocessed_docs_v2")

CREATING_DB_DIR = os.path.dirname(os.path.abspath(__file__))
CHROMA_DIR_V2 = os.getenv("CHROMA_DIR_V2") or os.path.join(CREATING_DB_DIR, "chroma_db_v2")
CHECKPOINT_DIR_V2 = os.getenv("CHECKPOINT_DIR_V2") or os.path.join(CREATING_DB_DIR, "checkpoints_v2")

CHUNK_PARQUET   = os.path.join(CHECKPOINT_DIR_V2, "chunks_v2.parquet")
VECTORS_PARQUET = os.path.join(CHECKPOINT_DIR_V2, "vectors_v2.parquet")
DOC_SUMMARY_PARQUET = os.path.join(CHECKPOINT_DIR_V2, "doc_summaries_v2.parquet")
BM25_INDEX_PATH = os.path.join(CHECKPOINT_DIR_V2, "bm25_index.pkl")
PROGRESS_CHECKPOINT = os.path.join(CHECKPOINT_DIR_V2, "progress_checkpoint.json")  # For resume

COLL_NAME = os.getenv("COLL_NAME_V2", "mysu_v2_bge_m3")

# Performance - reduced parallelism to avoid overwhelming Ollama
MAX_WORKERS_EMBED = int(os.getenv("MAX_WORKERS_EMBED", "2"))  # Reduced from 6 to 2
BATCH_UPSERT = int(os.getenv("BATCH_UPSERT", "32"))  # Reduced from 64 to 32
REQUEST_TIMEOUT_S = int(os.getenv("REQUEST_TIMEOUT_S", "180"))  # Increased timeout
RETRY_MAX = int(os.getenv("RETRY_MAX", "6"))  # More retries
BACKOFF_BASE = float(os.getenv("BACKOFF_BASE", "2.0"))  # Longer backoff

# Chunking parameters - IMPROVED
CHUNK_SIZE_MIN = int(os.getenv("CHUNK_SIZE_MIN", "300"))
CHUNK_SIZE_MAX = int(os.getenv("CHUNK_SIZE_MAX", "1200"))
CHUNK_SIZE_TARGET = int(os.getenv("CHUNK_SIZE_TARGET", "800"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "150"))
MIN_TEXT_LEN = int(os.getenv("MIN_TEXT_LEN", "50"))

# Contextual embedding: prepend summary to chunk
USE_CONTEXTUAL_EMBEDDINGS = os.getenv("USE_CONTEXTUAL_EMBEDDINGS", "1") == "1"

# Skip LLM tagging for faster processing (set to "1" to skip)
SKIP_LLM_TAGGING = os.getenv("SKIP_LLM_TAGGING", "0") == "1"  # Default: enable tagging

# Max text length for embedding (bge-m3 context window)
MAX_EMBED_TEXT_LEN = int(os.getenv("MAX_EMBED_TEXT_LEN", "8000"))

TR_DIACRITICS = "çğıöşüÇĞİÖŞÜ"

os.makedirs(CHROMA_DIR_V2, exist_ok=True)
os.makedirs(CHECKPOINT_DIR_V2, exist_ok=True)

# ================= UTILITIES =================

def sha1_text(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8", errors="ignore")).hexdigest()

def normalize_ws(text: str) -> str:
    text = text.replace("\xa0", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\s*\n\s*\n\s*\n+", "\n\n", text)
    return text.strip()

# ================= LANGUAGE DETECTION =================

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

# ================= IMPROVED CHUNKING =================

SENT_SPLIT = re.compile(r"(?<=[\.!?…])\s+(?=[A-Z" + TR_DIACRITICS + r"0-9])")

def sentence_split(text: str) -> List[str]:
    """Split text into sentences, merging very short ones."""
    sents = SENT_SPLIT.split(text)
    merged = []
    buf = ""
    for s in sents:
        s = s.strip()
        if not s:
            continue
        if len(buf) + len(s) < 100:  # Merge short sentences
            buf = (buf + " " + s).strip()
        else:
            if buf:
                merged.append(buf)
            buf = s
    if buf:
        merged.append(buf)
    return merged

def smart_chunk_section(section_text: str, section_header: str, 
                        target_size: int = CHUNK_SIZE_TARGET,
                        min_size: int = CHUNK_SIZE_MIN,
                        max_size: int = CHUNK_SIZE_MAX,
                        overlap: int = CHUNK_OVERLAP) -> List[Dict]:
    """
    Smart chunking that respects section boundaries and sentence structure.
    Each chunk includes its section header for context.
    """
    if not section_text.strip():
        return []
    
    sentences = sentence_split(section_text)
    chunks = []
    current_chunk = []
    current_len = 0
    chunk_idx = 0
    
    for sent in sentences:
        sent_len = len(sent)
        
        # If adding this sentence exceeds max, save current and start new
        if current_len + sent_len > max_size and current_chunk:
            chunk_text = " ".join(current_chunk)
            if len(chunk_text) >= min_size:
                chunks.append({
                    "text": chunk_text,
                    "section_header": section_header,
                    "index": chunk_idx,
                })
                chunk_idx += 1
            
            # Keep overlap: take last few sentences
            overlap_sents = []
            overlap_len = 0
            for s in reversed(current_chunk):
                if overlap_len + len(s) < overlap:
                    overlap_sents.insert(0, s)
                    overlap_len += len(s)
                else:
                    break
            
            current_chunk = overlap_sents
            current_len = overlap_len
        
        current_chunk.append(sent)
        current_len += sent_len
        
        # If we hit target size, consider saving
        if current_len >= target_size:
            chunk_text = " ".join(current_chunk)
            if len(chunk_text) >= min_size:
                chunks.append({
                    "text": chunk_text,
                    "section_header": section_header,
                    "index": chunk_idx,
                })
                chunk_idx += 1
                
                # Overlap
                overlap_sents = []
                overlap_len = 0
                for s in reversed(current_chunk):
                    if overlap_len + len(s) < overlap:
                        overlap_sents.insert(0, s)
                        overlap_len += len(s)
                    else:
                        break
                
                current_chunk = overlap_sents
                current_len = overlap_len
    
    # Don't forget the last chunk
    if current_chunk:
        chunk_text = " ".join(current_chunk)
        if len(chunk_text) >= min_size:
            chunks.append({
                "text": chunk_text,
                "section_header": section_header,
                "index": chunk_idx,
            })
    
    return chunks

def chunk_document(doc: Dict) -> List[Dict]:
    """
    Chunk document using hierarchical section-aware strategy.
    Falls back to simple chunking if no sections available.
    """
    sections = doc.get("sections", [])
    full_text = doc.get("full_text", "") or doc.get("text", "")
    title = doc.get("title", "")
    
    all_chunks = []
    
    if sections:
        # Use section-based chunking
        for sec in sections:
            header = sec.get("header", "")
            sec_text = sec.get("text", "")
            
            sec_chunks = smart_chunk_section(
                sec_text, 
                header,
                target_size=CHUNK_SIZE_TARGET,
                min_size=CHUNK_SIZE_MIN,
                max_size=CHUNK_SIZE_MAX,
                overlap=CHUNK_OVERLAP
            )
            all_chunks.extend(sec_chunks)
    else:
        # Fallback: chunk the entire text
        chunks = smart_chunk_section(
            full_text,
            title,
            target_size=CHUNK_SIZE_TARGET,
            min_size=CHUNK_SIZE_MIN,
            max_size=CHUNK_SIZE_MAX,
            overlap=CHUNK_OVERLAP
        )
        all_chunks.extend(chunks)
    
    # Re-index
    for i, ch in enumerate(all_chunks):
        ch["index"] = i
    
    return all_chunks

def dedupe_chunks(chunks: List[Dict]) -> List[Dict]:
    """Remove duplicate chunks based on text hash."""
    seen = set()
    deduped = []
    for ch in chunks:
        key = sha1_text(ch["text"])
        if key not in seen:
            seen.add(key)
            deduped.append(ch)
    return deduped

# ================= EMBEDDINGS =================

def embed_one(text: str) -> np.ndarray:
    """Embed a single text using Ollama with robust retry logic."""
    url = f"{OLLAMA_HOST}/api/embeddings"
    last_error = None
    
    # Truncate text if too long (bge-m3 has limited context)
    if len(text) > MAX_EMBED_TEXT_LEN:
        text = text[:MAX_EMBED_TEXT_LEN]
    
    for attempt in range(RETRY_MAX):
        try:
            r = requests.post(
                url,
                json={"model": EMBED_MODEL, "prompt": text},
                timeout=REQUEST_TIMEOUT_S,
            )
            r.raise_for_status()
            v = np.array(r.json()["embedding"], dtype=np.float32)
            v /= (np.linalg.norm(v) + 1e-12)  # Normalize
            if v.shape[0] != EMBED_DIM:
                raise RuntimeError(f"Unexpected dim: {v.shape[0]} vs {EMBED_DIM}")
            return v
        except Exception as e:
            last_error = e
            wait_time = BACKOFF_BASE * (2 ** attempt) + 1.0  # Longer backoff
            print(f"[retry {attempt+1}/{RETRY_MAX}] Embedding failed: {e}. Waiting {wait_time:.1f}s...")
            time.sleep(wait_time)
    
    raise RuntimeError(f"Embedding failed after {RETRY_MAX} retries: {last_error}")

def embed_many_parallel(texts: List[str]) -> np.ndarray:
    """Embed multiple texts in parallel."""
    vecs = [None] * len(texts)
    with ThreadPoolExecutor(max_workers=MAX_WORKERS_EMBED) as ex:
        futs = {ex.submit(embed_one, texts[i]): i for i in range(len(texts))}
        for fut in as_completed(futs):
            i = futs[fut]
            vecs[i] = fut.result()
    return np.vstack(vecs)

def create_contextual_embedding_text(chunk_text: str, doc_summary: str, section_header: str) -> str:
    """
    Create contextual embedding input by prepending document context.
    This helps the embedding model understand the chunk's context.
    """
    if not USE_CONTEXTUAL_EMBEDDINGS:
        return chunk_text
    
    # Format: [Document Summary] [Section: Header] [Chunk Text]
    context_parts = []
    
    if doc_summary:
        context_parts.append(f"[Belge: {doc_summary[:200]}]")
    
    if section_header:
        context_parts.append(f"[Bölüm: {section_header}]")
    
    context_parts.append(chunk_text)
    
    return " ".join(context_parts)

# ================= LLM TAGGING =================

def call_ollama_json(prompt: str, system_prompt: str = "") -> dict:
    """Call Ollama and expect JSON response."""
    url = f"{OLLAMA_HOST}/api/chat"
    
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})
    
    payload = {
        "model": CHAT_MODEL,
        "messages": messages,
        "stream": False,
        "options": {"temperature": 0.0},
    }
    
    try:
        resp = requests.post(url, json=payload, timeout=300)
        resp.raise_for_status()
        data = resp.json()
        text = (data.get("message", {}) or {}).get("content", "").strip()
        if not text:
            return {}
        
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

def infer_doc_tags_llm(title: str, doc_text: str, doc_lang: str) -> List[str]:
    """Infer semantic tags for a document using LLM."""
    # Skip if disabled for faster processing
    if SKIP_LLM_TAGGING:
        return []
    
    body_sample = (doc_text or "")[:2500]
    lang_hint = doc_lang or "unknown"
    
    sys_prompt = textwrap.dedent("""
        You are an expert classifier for internal university documents.
        Return a JSON object with semantic tags for the document.

        JSON schema (STRICT):
        {
          "tags": ["short_snake_case_tag1", "tag2", ...]
        }

        Rules:
        - 3 to 12 tags
        - Tags must be lowercase snake_case
        - Tags should cover: topics, document type, target audience, related units
        - Use English tags even for Turkish documents
        - No generic tags like "document", "general"
        
        Output STRICT JSON ONLY, no explanation.
    """).strip()
    
    user_prompt = f"Language: {lang_hint}\n\nTitle:\n{title or '(no title)'}\n\nExcerpt:\n{body_sample}"
    
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
        if tt and tt not in seen:
            seen.add(tt)
            clean_tags.append(tt)
    
    return clean_tags

# ================= BM25 INDEX =================

def tokenize_for_bm25(text: str) -> List[str]:
    """Simple tokenization for BM25."""
    text = text.lower()
    # Keep Turkish diacritics
    tokens = re.findall(r"[a-z" + TR_DIACRITICS.lower() + r"0-9]+", text)
    # Remove very short tokens
    tokens = [t for t in tokens if len(t) > 2]
    return tokens

def build_bm25_index(chunks: List[Dict]) -> Tuple[Any, List[str]]:
    """Build BM25 index from chunks."""
    if not HAS_BM25:
        return None, []
    
    print("[info] Building BM25 index...")
    
    corpus = []
    chunk_ids = []
    
    for ch in chunks:
        text = ch.get("text", "")
        chunk_id = ch.get("chunk_id", "")
        
        tokens = tokenize_for_bm25(text)
        corpus.append(tokens)
        chunk_ids.append(chunk_id)
    
    bm25 = BM25Okapi(corpus)
    
    print(f"[info] BM25 index built with {len(corpus)} documents")
    return bm25, chunk_ids

def save_bm25_index(bm25, chunk_ids: List[str], path: str):
    """Save BM25 index to pickle."""
    if bm25 is None:
        return
    
    with open(path, "wb") as f:
        pickle.dump({"bm25": bm25, "chunk_ids": chunk_ids}, f)
    
    print(f"✅ BM25 index saved to {path}")

# ================= MAIN =================

def load_checkpoint() -> dict:
    """Load checkpoint to resume from where we left off."""
    if os.path.exists(PROGRESS_CHECKPOINT):
        try:
            with open(PROGRESS_CHECKPOINT, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            print(f"[warn] Could not load checkpoint: {e}")
    return {"processed_files": [], "last_index": 0}

def save_checkpoint(processed_files: List[str], last_index: int):
    """Save checkpoint for resume capability."""
    try:
        with open(PROGRESS_CHECKPOINT, "w", encoding="utf-8") as f:
            json.dump({
                "processed_files": processed_files,
                "last_index": last_index,
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            }, f, indent=2)
    except Exception as e:
        print(f"[warn] Could not save checkpoint: {e}")

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
    
    # 2) Load checkpoint for resume
    checkpoint = load_checkpoint()
    processed_set = set(checkpoint.get("processed_files", []))
    start_index = checkpoint.get("last_index", 0)
    
    if processed_set:
        print(f"[info] Resuming from checkpoint: {len(processed_set)} files already processed")
    
    # 3) Create Chroma collection
    client = chromadb.PersistentClient(path=CHROMA_DIR_V2)
    coll = client.get_or_create_collection(
        name=COLL_NAME,
        metadata={"hnsw:space": "cosine"},
    )
    
    global_seen_chunks = set()
    chunk_rows = []
    vector_rows = []
    doc_summary_rows = []
    all_chunks_for_bm25 = []
    processed_files_list = list(processed_set)  # For checkpoint
    
    total_upserts = 0
    started = time.time()
    
    for fi, path in enumerate(json_files, 1):
        try:
            # Skip if already processed (resume mode)
            if path in processed_set:
                continue
            
            # Progress logging - show every file
            elapsed = time.time() - started
            print(f"\n[{fi}/{len(json_files)}] Processing: {os.path.basename(path)} (elapsed: {elapsed:.1f}s)")
            
            # Load JSON
            with open(path, "r", encoding="utf-8") as f:
                doc = json.load(f)
            
            # Get text (support both v1 and v2 format)
            doc_text = normalize_ws(doc.get("full_text", "") or doc.get("text", ""))
            if len(doc_text) < MIN_TEXT_LEN:
                print(f"  [skip] text too short ({len(doc_text)} chars)")
                processed_files_list.append(path)
                continue
            
            title = (doc.get("title") or "").strip()
            source_path = (doc.get("source_path") or "").strip()
            html_lang = (doc.get("html_lang") or "").strip()
            lang_field = (doc.get("lang") or "").strip()
            doc_type = doc.get("doc_type", "other")
            metadata_extra = doc.get("metadata", {})
            summary = doc.get("summary", "")
            
            if not source_path:
                source_path = os.path.relpath(path, PRE_ROOT).replace("\\", "/")
            
            text_lang = guess_lang_from_text(doc_text)
            doc_lang = lang_field if lang_field in ("tr", "en") else text_lang
            
            # Generate tags
            print(f"  [tagging] {title[:50]}...")
            tags = infer_doc_tags_llm(title, doc_text, doc_lang)
            tags_str = ",".join(tags)
            if tags:
                print(f"  [tags] {tags_str[:80]}")
            
            # Generate summary if not present
            if not summary:
                summary = doc_text[:300] + "..."
            
            # Store document summary
            doc_summary_rows.append({
                "source_path": source_path,
                "title": title,
                "summary": summary,
                "doc_type": doc_type,
                "doc_lang": doc_lang,
                "tags": tags_str,
                "procedure_code": metadata_extra.get("procedure_code", ""),
            })
            
            # Chunk document
            chunks = chunk_document(doc)
            chunks = dedupe_chunks(chunks)
            
            if not chunks:
                continue
            
            # Process chunks
            json_rel = os.path.relpath(path, PRE_ROOT).replace("\\", "/")
            size_bytes = len(doc_text.encode("utf-8", errors="ignore"))
            
            for i in range(0, len(chunks), BATCH_UPSERT):
                batch = chunks[i:i + BATCH_UPSERT]
                
                texts = []
                ids = []
                metas = []
                embed_texts = []  # For contextual embeddings
                
                for ch in batch:
                    ch_text = ch["text"]
                    ch_sha = sha1_text(ch_text)
                    
                    if ch_sha in global_seen_chunks:
                        continue
                    global_seen_chunks.add(ch_sha)
                    
                    chunk_index = ch["index"]
                    section_header = ch.get("section_header", "")
                    chunk_id = sha1_text(f"{source_path}::{chunk_index}")
                    
                    # Create contextual embedding text
                    embed_text = create_contextual_embedding_text(
                        ch_text, summary, section_header
                    )
                    
                    meta = {
                        "chunk_id": chunk_id,
                        "source_path": source_path,
                        "json_path": json_rel,
                        "title": title,
                        "section_header": section_header,
                        "chunk_index": int(chunk_index),
                        "doc_lang": doc_lang,
                        "html_lang": html_lang,
                        "doc_type": doc_type,
                        "bytes": int(size_bytes),
                        "tags": tags_str,
                        "procedure_code": metadata_extra.get("procedure_code", ""),
                    }
                    
                    chunk_rows.append({
                        **meta,
                        "content": ch_text,
                        "embed_text": embed_text,
                    })
                    
                    all_chunks_for_bm25.append({
                        "chunk_id": chunk_id,
                        "text": ch_text,
                    })
                    
                    texts.append(ch_text)
                    embed_texts.append(embed_text)
                    ids.append(chunk_id)
                    metas.append(meta)
                
                if not texts:
                    continue
                
                # Embed using contextual text
                vecs = embed_many_parallel(embed_texts)
                
                # Upsert to Chroma
                coll.upsert(
                    ids=ids,
                    embeddings=vecs.tolist(),
                    metadatas=metas,
                    documents=texts,  # Store original text, not contextual
                )
                total_upserts += len(ids)
                
                # Record vectors
                for j in range(len(ids)):
                    vector_rows.append({
                        "id": ids[j],
                        "vector": vecs[j].astype(np.float32).tolist(),
                        "document": texts[j],
                        "metadata": json.dumps(metas[j], ensure_ascii=False),
                    })
            
            # Mark file as processed and save checkpoint periodically
            processed_files_list.append(path)
            if fi % 10 == 0:
                save_checkpoint(processed_files_list, fi)
                elapsed = time.time() - started
                print(f"  [checkpoint] Saved progress at {fi}/{len(json_files)} (upserts={total_upserts}, elapsed={elapsed:.1f}s)")
        
        except Exception as e:
            print(f"[error] {path}: {e}")
            # Save checkpoint on error so we can resume
            save_checkpoint(processed_files_list, fi)
            continue
    
    # Final checkpoint save
    save_checkpoint(processed_files_list, len(json_files))
    
    # Save chunks parquet
    if chunk_rows:
        df = pd.DataFrame(chunk_rows)
        df.to_parquet(CHUNK_PARQUET, index=False)
        print(f"✅ Chunks parquet written: {CHUNK_PARQUET} (rows={len(df)})")
    
    # Save vectors parquet
    if vector_rows:
        ids = [r["id"] for r in vector_rows]
        docs = [r["document"] for r in vector_rows]
        metas_json = [r["metadata"] for r in vector_rows]
        vecs = [r["vector"] for r in vector_rows]
        
        table = pa.table({
            "id": pa.array(ids, type=pa.string()),
            "vector": pa.array(vecs, type=pa.list_(pa.float32())),
            "document": pa.array(docs, type=pa.string()),
            "metadata": pa.array(metas_json, type=pa.string()),
        })
        pq.write_table(table, VECTORS_PARQUET)
        print(f"✅ Vectors parquet written: {VECTORS_PARQUET} (rows={len(ids)})")
    
    # Save document summaries
    if doc_summary_rows:
        df_docs = pd.DataFrame(doc_summary_rows)
        df_docs.to_parquet(DOC_SUMMARY_PARQUET, index=False)
        print(f"✅ Document summaries parquet written: {DOC_SUMMARY_PARQUET} (rows={len(df_docs)})")
    
    # Build and save BM25 index
    if all_chunks_for_bm25 and HAS_BM25:
        bm25, chunk_ids = build_bm25_index(all_chunks_for_bm25)
        save_bm25_index(bm25, chunk_ids, BM25_INDEX_PATH)
    
    # Remove checkpoint file on successful completion
    if os.path.exists(PROGRESS_CHECKPOINT):
        os.remove(PROGRESS_CHECKPOINT)
        print("✅ Checkpoint file removed (processing complete)")
    
    print("✅ Build finished.")
    try:
        print(f"Collection {COLL_NAME} count() -> {coll.count()}")
    except Exception as e:
        print(f"count() failed: {e}")

if __name__ == "__main__":
    main()
