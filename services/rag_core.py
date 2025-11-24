# rag_try_ollama_plus2.py  (updated with chroma-mmr)

import os
import json
import pickle
import math
from typing import List, Dict, Optional

import numpy as np
import requests
import chromadb
import re
import pandas as pd  # for reading chunks_plus2.parquet
import dotenv

# =============== CONFIG ====================

dotenv.load_dotenv()

OLLAMA_HOST = "http://localhost:11434"
EMBED_MODEL = "bge-m3"    # must match builder
EMBED_DIM   = 1024
CHAT_MODEL  = "llama3.2"  # change if you use something else

PREPROCESSING_DIR = os.getenv("PREPROCESSING_PATH")

# 2. Chroma Klasör Adını Al (YENİ KISIM)



CHROMA_DIR     = os.getenv("CHROMA_DIR") or os.path.join(PREPROCESSING_DIR, "chroma_db_mysu_plus2")
CHECKPOINT_DIR = os.getenv("CHECKPOINT_DIR") 
BM25DIR        = PREPROCESSING_DIR

BM25_PKL       = os.path.join(BM25DIR, "bm25_plus2.pkl")
CHUNK_PARQUET  = os.path.join(CHECKPOINT_DIR, "chunks_plus2.parquet")


COLL_NAME      = "mysu_surecharitasi_bge_m3_v1"

# retrieval sizes
TOP_K_CHROMA     = 24   # candidate pool from vectors
TOP_K_BM25       = 24   # candidate pool from BM25
TOP_K_FINAL_BASE = 8    # minimum final context size
TOP_K_FINAL_MAX  = 32   # maximum final context size (for big docs)

TR_DIACRITICS = "çğıöşüÇĞİÖŞÜ"
TOKEN_RE      = re.compile(r"\w+", flags=re.UNICODE)

# caches
DOC_CHUNK_STATS: Optional[Dict[str, int]] = None
CHUNK_DF: Optional[pd.DataFrame] = None

# =============== LANGUAGE DETECTION ====================

try:
    from langdetect import detect as ld_detect
except ImportError:
    ld_detect = None


def guess_lang_from_text(text: str) -> Optional[str]:
    s = (text or "").strip()
    if not s:
        return None
    sample = s[:4000]

    # langdetect if available
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

    # fallback: check Turkish chars
    if re.search(f"[{TR_DIACRITICS}]", sample):
        return "tr"

    # default: unknown
    return None


# =============== TOKENIZER FOR BM25 ============

def tokenize(text: str) -> List[str]:
    text = (text or "").lower()
    return TOKEN_RE.findall(text)


# =============== CHROMA ====================

def get_chroma_collection():
    client = chromadb.PersistentClient(path=CHROMA_DIR)
    coll = client.get_collection(COLL_NAME)
    return coll


# =============== BM25 LOAD + SEARCH ==========

def load_bm25_index():
    with open(BM25_PKL, "rb") as f:
        pack = pickle.load(f)
    bm25       = pack["bm25"]
    chunk_ids  = pack["chunk_ids"]
    texts      = pack["texts"]
    doc_paths  = pack["doc_paths"]
    titles     = pack["titles"]
    doc_langs  = pack["doc_langs"]

    id2idx = {cid: i for i, cid in enumerate(chunk_ids)}

    return {
        "bm25": bm25,
        "chunk_ids": chunk_ids,
        "texts": texts,
        "doc_paths": doc_paths,
        "titles": titles,
        "doc_langs": doc_langs,
        "id2idx": id2idx,
    }


def bm25_search(q: str,
                bm25_pack: Dict,
                top_k: int = TOP_K_BM25,
                lang_filter: Optional[str] = None) -> List[Dict]:
    bm25      = bm25_pack["bm25"]
    texts     = bm25_pack["texts"]
    chunk_ids = bm25_pack["chunk_ids"]
    doc_paths = bm25_pack["doc_paths"]
    titles    = bm25_pack["titles"]
    doc_langs = bm25_pack["doc_langs"]

    query_tokens = tokenize(q)
    scores = bm25.get_scores(query_tokens)  # np.array

    idxs = np.argsort(scores)[::-1]

    results = []
    for idx in idxs:
        score = float(scores[idx])
        if score <= 0:
            break

        lang = doc_langs[idx]
        if lang_filter is not None and lang != lang_filter:
            continue

        results.append({
            "chunk_id": chunk_ids[idx],
            "score": score,
            "text": texts[idx],
            "meta": {
                "doc_path": doc_paths[idx],
                "title": titles[idx],
                "doc_lang": lang,
            },
            "source": "bm25",
        })
        if len(results) >= top_k:
            break
    return results


# =============== DOC CHUNK STATS & FULL-DOC CHUNKS =====

def load_doc_chunk_stats() -> Dict[str, int]:
    """
    Reads chunks_plus2.parquet once and builds:
        { doc_path: total_number_of_chunks_in_that_doc }
    """
    global DOC_CHUNK_STATS
    if DOC_CHUNK_STATS is not None:
        return DOC_CHUNK_STATS

    if not os.path.exists(CHUNK_PARQUET):
        print(f"[warn] CHUNK_PARQUET not found: {CHUNK_PARQUET}")
        DOC_CHUNK_STATS = {}
        return DOC_CHUNK_STATS

    df = pd.read_parquet(CHUNK_PARQUET)

    if "doc_path" not in df.columns:
        raise RuntimeError("chunks_plus2.parquet has no 'doc_path' column.")
    if "chunk_id" not in df.columns:
        raise RuntimeError("chunks_plus2.parquet has no 'chunk_id' column.")

    stats = (
        df.groupby("doc_path")["chunk_id"]
          .count()
          .to_dict()
    )

    DOC_CHUNK_STATS = stats
    print(f"[info] Loaded doc chunk stats for {len(DOC_CHUNK_STATS)} documents.")
    return DOC_CHUNK_STATS


def load_chunk_df() -> pd.DataFrame:
    """
    Load the full chunks_plus2.parquet as a DataFrame (cached).
    Used to pull all chunks for a given doc_path when we want full-doc context.
    """
    global CHUNK_DF
    if CHUNK_DF is None:
        if not os.path.exists(CHUNK_PARQUET):
            raise RuntimeError(f"CHUNK_PARQUET not found: {CHUNK_PARQUET}")
        CHUNK_DF = pd.read_parquet(CHUNK_PARQUET)
    return CHUNK_DF


def get_all_chunks_for_doc(doc_path: str) -> List[Dict]:
    """
    Return all chunks for a given doc_path, sorted by chunk_index,
    in the same structure as retrieval results (chunk_id, text, meta, ...).
    """
    df = load_chunk_df()
    sub = df[df["doc_path"] == doc_path].copy()
    if "chunk_index" in sub.columns:
        sub = sub.sort_values("chunk_index")

    chunks = []
    for _, row in sub.iterrows():
        meta = {
            "doc_path": row["doc_path"],
            "title": row.get("title", ""),
            "h1": row.get("h1", ""),
            "h2": row.get("h2", ""),
            "h3": row.get("h3", ""),
            "doc_lang": row.get("doc_lang", row.get("content_lang", "")),
        }
        chunks.append({
            "chunk_id": row["chunk_id"],
            "text": row["content"],
            "meta": meta,
            "source": "doc_full",
        })
    return chunks


def decide_top_k_for_candidates(candidates: List[Dict]) -> int:
    """
    Decide TOP_K_FINAL dynamically based only on document sizes.

    - Look at doc_paths present in candidates.
    - Use the largest document size (in #chunks) among them.
    - Use K = min(largest_doc_size, TOP_K_FINAL_MAX), but at least TOP_K_FINAL_BASE.
    - Also cannot exceed len(candidates).
    """
    if not candidates:
        return TOP_K_FINAL_BASE

    stats = load_doc_chunk_stats()
    if not stats:
        return TOP_K_FINAL_BASE

    doc_sizes = []
    for c in candidates:
        meta = c.get("meta") or {}
        path = meta.get("doc_path")
        if not path:
            continue
        if path in stats:
            doc_sizes.append(stats[path])

    if not doc_sizes:
        return TOP_K_FINAL_BASE

    largest_doc_size = max(doc_sizes)

    # K ~ size of the biggest doc among candidates
    k = largest_doc_size

    # clamp
    k = max(TOP_K_FINAL_BASE, k)
    k = min(k, TOP_K_FINAL_MAX)
    k = min(k, len(candidates))

    return k


# =============== OLLAMA EMBEDDINGS ============

def embed_text_ollama(text: str) -> np.ndarray:
    url = f"{OLLAMA_HOST}/api/embeddings"
    r = requests.post(
        url,
        json={"model": EMBED_MODEL, "prompt": text},
        timeout=120,
    )
    r.raise_for_status()
    v = np.array(r.json()["embedding"], dtype=np.float32)
    if v.shape[0] != EMBED_DIM:
        raise RuntimeError(f"Unexpected embedding dimension {v.shape[0]} (expected {EMBED_DIM})")
    # normalize (same as in builder)
    v /= (np.linalg.norm(v) + 1e-12)
    return v


# =============== CHROMA SEARCH ===============

def chroma_search(q: str,
                  coll,
                  query_vec: Optional[np.ndarray] = None,
                  top_k: int = TOP_K_CHROMA,
                  lang_filter: Optional[str] = None) -> List[Dict]:

    if query_vec is None:
        query_vec = embed_text_ollama(q)

    where = {}
    if lang_filter is not None:
        where["doc_lang"] = lang_filter

    res = coll.query(
        query_embeddings=[query_vec.tolist()],
        n_results=top_k,
        where=where or None,
    )

    ids        = res.get("ids", [[]])[0]
    docs       = res.get("documents", [[]])[0]
    metas      = res.get("metadatas", [[]])[0]
    distances  = res.get("distances", [[]])[0]  # cosine distance: 0 = identical

    out = []
    for cid, d, m, dist in zip(ids, docs, metas, distances):
        sim = 1.0 - float(dist)  # similarity = 1 - distance
        out.append({
            "chunk_id": cid,
            "score": sim,
            "text": d,
            "meta": m,
            "source": "chroma",
        })
    return out


# =============== HYBRID MERGE ================

def hybrid_merge(bm_results: List[Dict],
                 vec_results: List[Dict],
                 w_bm25: float = 0.6,
                 w_vec: float = 0.4) -> List[Dict]:
    combined: Dict[str, Dict] = {}

    bm_scores = [r["score"] for r in bm_results]
    max_bm = max(bm_scores) if bm_scores else 1.0

    for r in bm_results:
        cid = r["chunk_id"]
        norm_bm = r["score"] / max_bm if max_bm > 0 else 0.0
        combined.setdefault(cid, {
            "chunk_id": cid,
            "text": r["text"],
            "meta": r["meta"],
            "bm25_score": 0.0,
            "vec_score": 0.0,
        })
        combined[cid]["bm25_score"] = max(combined[cid]["bm25_score"], norm_bm)

    vec_scores = [r["score"] for r in vec_results]
    max_vec = max(vec_scores) if vec_scores else 1.0

    for r in vec_results:
        cid = r["chunk_id"]
        norm_vec = r["score"] / max_vec if max_vec > 0 else 0.0
        meta = r["meta"] or {}
        combined.setdefault(cid, {
            "chunk_id": cid,
            "text": r["text"],
            "meta": meta,
            "bm25_score": 0.0,
            "vec_score": 0.0,
        })
        combined[cid]["vec_score"] = max(combined[cid]["vec_score"], norm_vec)
        if not combined[cid]["meta"]:
            combined[cid]["meta"] = meta

    out = []
    for cid, entry in combined.items():
        hybrid_score = w_bm25 * entry["bm25_score"] + w_vec * entry["vec_score"]
        entry["hybrid_score"] = hybrid_score
        out.append(entry)

    out.sort(key=lambda x: x["hybrid_score"], reverse=True)
    return out


# =============== FETCH EMBEDDINGS ============

def fetch_doc_embeddings(coll, ids: List[str]) -> Dict[str, np.ndarray]:
    """Get stored embeddings from Chroma for given chunk_ids."""
    if not ids:
        return {}
    res = coll.get(ids=ids, include=["embeddings"])
    out = {}
    for cid, emb in zip(res["ids"], res["embeddings"]):
        out[cid] = np.array(emb, dtype=np.float32)
    return out


def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    num = float(np.dot(a, b))
    den = float(np.linalg.norm(a) * np.linalg.norm(b) + 1e-12)
    return num / den


# =============== MMR =========================

def mmr_select(candidates: List[Dict],
               doc_embs: Dict[str, np.ndarray],
               query_vec: np.ndarray,
               k: int = TOP_K_FINAL_BASE,
               lambda_mmr: float = 0.7) -> List[Dict]:
    """
    candidates: list of {chunk_id, text, meta, hybrid_score, bm25_score, vec_score}
    doc_embs: dict chunk_id -> np.array embedding
    query_vec: query embedding
    """
    selected: List[Dict] = []
    selected_ids: List[str] = []

    # precompute similarity to query (based on embeddings)
    query_sims = {}
    for c in candidates:
        cid = c["chunk_id"]
        emb = doc_embs.get(cid)
        if emb is None:
            query_sims[cid] = 0.0
        else:
            query_sims[cid] = cosine_sim(query_vec, emb)

    while len(selected) < min(k, len(candidates)):
        best_cand = None
        best_mmr = -1e9

        for c in candidates:
            cid = c["chunk_id"]
            if cid in selected_ids:
                continue

            # relevance: combine hybrid_score & query_sims
            rel = 0.5 * c.get("hybrid_score", 0.0) + 0.5 * query_sims.get(cid, 0.0)

            # redundancy: max similarity to already selected docs
            if not selected_ids:
                red = 0.0
            else:
                emb_i = doc_embs.get(cid)
                if emb_i is None:
                    red = 0.0
                else:
                    sims = []
                    for sid in selected_ids:
                        emb_j = doc_embs.get(sid)
                        if emb_j is not None:
                            sims.append(cosine_sim(emb_i, emb_j))
                    red = max(sims) if sims else 0.0

            mmr_score = lambda_mmr * rel - (1.0 - lambda_mmr) * red

            if mmr_score > best_mmr:
                best_mmr = mmr_score
                best_cand = c

        if best_cand is None:
            break

        selected.append(best_cand)
        selected_ids.append(best_cand["chunk_id"])

    return selected


# =============== OLLAMA CHAT ==================

def call_ollama_chat(prompt: str, system_prompt: str = "", model: str = CHAT_MODEL) -> str:
    url_chat = f"{OLLAMA_HOST}/api/chat"
    
    # Mesaj yapısı
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})

    # Payload
    payload = {
        "model": model,
        "messages": messages,
        "stream": False
    }

    try:
        print(f"📤 Ollama'ya İstek Gönderiliyor... Model: {model}")
        
        # İsteği at
        resp = requests.post(url_chat, json=payload, timeout=300)
        
        # Hata varsa detayını al
        if resp.status_code != 200:
            print(f"❌ OLLAMA HATASI (Status: {resp.status_code}): {resp.text}")
            return f"Ollama Hatası: {resp.text}"

        # Cevabı al
        data = resp.json()
        return data.get("message", {}).get("content", "Boş cevap döndü.")

    except Exception as e:
        print(f"❌ Bağlantı Hatası: {e}")
        return "Üzgünüm, yapay zeka servisine bağlanılamadı."


# =============== CONTEXT CONSTRUCTION =========

def build_context(chunks: List[Dict]) -> str:
    parts = []
    for i, ch in enumerate(chunks):
        meta = ch.get("meta", {}) or {}
        path = meta.get("doc_path", "")
        title = meta.get("title", "")
        lang = meta.get("doc_lang", "")
        header = f"[{i+1}] lang={lang} | title={title} | file={os.path.basename(path)}"
        parts.append(header + "\n" + ch["text"])
    return "\n\n-----\n\n".join(parts)


def answer_with_rag(query: str,
                    mode: str,
                    bm25_pack: Dict,
                    coll,
                    history: List[Dict] = []) -> str:
    # auto language detection from query
    q_lang = guess_lang_from_text(query)
    print(f"(auto-detected query language: {q_lang})")

    # determine language filter for docs
    lang_filter = q_lang if q_lang in ("tr", "en") else None

    # embed query once for Chroma + MMR
    query_vec = embed_text_ollama(query)

    # ---------- MODES ----------
    if mode == "chroma":
        # Single-main-doc mode: pick best doc, then stream its chunks
        vec_candidates = chroma_search(
            query,
            coll,
            query_vec=query_vec,
            top_k=TOP_K_CHROMA,
            lang_filter=lang_filter,
        )

        # Aggregate scores per document to find the "main" doc
        doc_scores = {}
        for c in vec_candidates:
            meta = c.get("meta") or {}
            path = meta.get("doc_path")
            if not path:
                continue
            doc_scores[path] = doc_scores.get(path, 0.0) + c["score"]

        if doc_scores:
            # Pick the best document
            best_doc_path = max(doc_scores.items(), key=lambda x: x[1])[0]

            # Load ALL chunks for that document and then limit by doc size / global max
            doc_chunks = get_all_chunks_for_doc(best_doc_path)
            top_k_final = min(len(doc_chunks), TOP_K_FINAL_MAX)
            retrieved = doc_chunks[:top_k_final]
        else:
            # Fallback: if we somehow have no doc_paths, just use candidates with dynamic K
            top_k_final = decide_top_k_for_candidates(vec_candidates)
            retrieved = vec_candidates[:top_k_final]

    elif mode == "chroma-mmr":
        # Pure Chroma multi-document mode with MMR (no BM25)
        vec_results = chroma_search(
            query,
            coll,
            query_vec=query_vec,
            top_k=TOP_K_CHROMA,
            lang_filter=lang_filter,
        )

        # Prepare scores
        for r in vec_results:
            r["bm25_score"]   = 0.0
            r["vec_score"]    = r["score"]
            r["hybrid_score"] = r["score"]  # we only care about vector similarity here

        candidate_ids = [c["chunk_id"] for c in vec_results]
        doc_embs = fetch_doc_embeddings(coll, candidate_ids)

        k = decide_top_k_for_candidates(vec_results)
        selected = mmr_select(vec_results, doc_embs, query_vec,
                              k=k, lambda_mmr=0.7)
        retrieved = selected

    elif mode == "bm25":
        bm_results = bm25_search(
            query,
            bm25_pack,
            top_k=TOP_K_FINAL_BASE,
            lang_filter=lang_filter,
        )
        # adapt to same structure
        for r in bm_results:
            r["bm25_score"] = r["score"]
            r["vec_score"] = 0.0
            r["hybrid_score"] = r["score"]
        retrieved = bm_results

    elif mode == "hybrid":
        bm_results  = bm25_search(
            query,
            bm25_pack,
            top_k=TOP_K_BM25,
            lang_filter=lang_filter,
        )
        vec_results = chroma_search(
            query,
            coll,
            query_vec=query_vec,
            top_k=TOP_K_CHROMA,
            lang_filter=lang_filter,
        )
        merged = hybrid_merge(bm_results, vec_results)
        k = decide_top_k_for_candidates(merged)
        retrieved = merged[:k]

    elif mode == "hybrid-mmr":
        bm_results  = bm25_search(
            query,
            bm25_pack,
            top_k=TOP_K_BM25,
            lang_filter=lang_filter,
        )
        vec_results = chroma_search(
            query,
            coll,
            query_vec=query_vec,
            top_k=TOP_K_CHROMA,
            lang_filter=lang_filter,
        )
        merged = hybrid_merge(bm_results, vec_results)

        # get embeddings for all candidate ids
        candidate_ids = [c["chunk_id"] for c in merged]
        doc_embs = fetch_doc_embeddings(coll, candidate_ids)

        # apply MMR on merged candidates
        k = decide_top_k_for_candidates(merged)
        selected = mmr_select(merged, doc_embs, query_vec,
                              k=k, lambda_mmr=0.7)
        retrieved = selected

    else:
        raise ValueError("mode must be one of: chroma, chroma-mmr, bm25, hybrid, hybrid-mmr")

    context = build_context(retrieved)

    print("=== DEBUG CONTEXT START ===")
    print(context)
    print("=== DEBUG CONTEXT END ===")

    # system prompt: answer in the same language as query
    if q_lang == "tr":
        sys_prompt = (
        "Sen Sabancı Üniversitesi öğrencileri için geliştirilmiş, sadece TÜRKÇE konuşan profesyonel bir asistansın.\n\n"
        
        "GÖREVİN:\n"
        "Sana verilen 'Context' (Bağlam) içindeki bilgileri kullanarak kullanıcının sorusunu cevaplamaktır.\n\n"
        
        "KESİN KURALLAR (ASLA İHLAL ETME):\n"
        "1. DİL: Cevabın %100 AKICI VE DÜZGÜN İSTANBUL TÜRKÇESİ olmalıdır.\n"
        "2. YASAK: Cümle içinde ASLA İngilizce kelime (determine, prior, date, begin vb.) KULLANMA. Hepsini Türkçeye çevir.\n"
        "3. YASAK: 'Beginir', 'withdrawa' gibi uydurma ekler ve kelimeler kullanma.\n"
        "4. SADAKAT: Sadece Context içindeki bilgiyi kullan. Bilgi yoksa uydurma, 'Bilgi bulunamadı' de.\n"
        "5. ÜSLUP: Resmi, net ve anlaşılır ol.\n"
        "6. TERİMLER: 'Withdraw' kavramını 'Dersten Çekilme' olarak, 'Add-Drop' kavramını 'Ders Ekleme-Bırakma' olarak kullan.\n"
    )
    elif q_lang == "en":
        sys_prompt = (
            "You are a helpful assistant answering questions about Sabancı University "
            "regulations, procedures, and documents. Use ONLY the provided context. "
            "If something is not in the context, say you don't know. Answer in English."
        )
    else:
        # default: bilingual-ish
        sys_prompt = (
            "You are a helpful assistant answering questions about Sabancı University "
            "regulations, procedures, and documents based ONLY on the provided context."
        )

    history_text = ""
    if history:
        history_text = "Chat History:\n" + "\n".join(
            [f"{msg['role'].upper()}: {msg['content']}" for msg in history[-5:]]
        ) + "\n\n"

    full_prompt = (
        f"Context:\n{context}\n\n"
        f"{history_text}"  # <-- Geçmişi buraya gömdük
        f"Question:\n{query}\n\n"
        f"Answer based only on the context above."
    )

    return call_ollama_chat(full_prompt, system_prompt=sys_prompt)

# =============== CLI MAIN =====================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        choices=["chroma", "chroma-mmr", "bm25", "hybrid", "hybrid-mmr"],
        default="chroma-mmr",
        help="retrieval mode"
    )
    args = parser.parse_args()

    print(f"Loading Chroma collection from: {CHROMA_DIR}")
    coll = get_chroma_collection()

    print(f"Loading BM25 index from: {BM25_PKL}")
    bm25_pack = load_bm25_index()

    # preload doc stats so you see the info line at startup
    load_doc_chunk_stats()

    print("Ready.")
    print(f"Mode = {args.mode}")
    print("Type an empty line to exit.\n")

    while True:
        try:
            q = input("Query> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not q:
            break

        try:
            ans = answer_with_rag(
                query=q,
                mode=args.mode,
                bm25_pack=bm25_pack,
                coll=coll,
            )
            print("\n=== ANSWER ===")
            print(ans)
            print("==============\n")
        except Exception as e:
            print(f"[error] {e}")
