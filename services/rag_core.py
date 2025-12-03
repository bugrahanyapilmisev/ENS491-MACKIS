# rag_core.py
# Generic RAG core:
# - Chroma + CE rerank + MMR
# - CE threshold filtering
# - Intent-aware single-doc mode (list_names / count_items) GENERIC
# - History-aware followups WITHOUT any domain-specific keywords
# - Tag-aware retrieval using open-ended LLM tags from build_chroma_store_plus2.py

import os
import json
import math
from typing import List, Dict, Optional, Tuple

import numpy as np
import requests
import chromadb
import re
import pandas as pd
import dotenv
from sentence_transformers import CrossEncoder
import textwrap

dotenv.load_dotenv()

# ---------- GLOBAL MODELS / ENV ----------

RERANKER_MODEL_NAME = os.getenv("RERANKER_MODEL_NAME", "BAAI/bge-reranker-v2-m3")
RERANKER: Optional[CrossEncoder] = None

OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
EMBED_MODEL = os.getenv("EMBED_MODEL", "bge-m3")
EMBED_DIM   = int(os.getenv("EMBED_DIM", "1024"))
CHAT_MODEL  = os.getenv("CHAT_MODEL", "llama3.2")

PREPROCESSING_DIR  = os.getenv("PREPROCESSING_PATH")
CHROMA_FOLDER_NAME = os.getenv("CHROMA_FOLDER_NAME", "chroma_db")
CHROMA_DIR         = os.getenv("CHROMA_DIR") or os.path.join(PREPROCESSING_DIR, CHROMA_FOLDER_NAME)
CHECKPOINT_DIR     = os.path.join(PREPROCESSING_DIR, "checkpoints_plus2")
CHUNK_PARQUET      = os.path.join(CHECKPOINT_DIR, "chunks_plus2.parquet")
COLL_NAME          = os.getenv("CHROMA_COLLECTION_NAME", "mysu_surecharitasi_bge_m3_v1")

TOP_K_CHROMA       = int(os.getenv("TOP_K_CHROMA", "64"))
TOP_K_FINAL_BASE   = int(os.getenv("TOP_K_FINAL_BASE", "8"))
TOP_K_FINAL_MAX    = int(os.getenv("TOP_K_FINAL_MAX", "24"))
MAX_DOCS_CONTEXT   = int(os.getenv("MAX_DOCS_CONTEXT", "6"))

# Stricter, but still overridable via .env
CROSS_MAX_CANDIDATES = int(os.getenv("CROSS_MAX_CANDIDATES", "24"))
CROSS_WEIGHT         = float(os.getenv("CROSS_WEIGHT", "0.7"))
CE_SCORE_THRESHOLD   = float(os.getenv("CE_SCORE_THRESHOLD", "0.70"))

USE_ANCHOR_DOCS = os.getenv("USE_ANCHOR_DOCS", "0") == "1"  # generic, default off

TR_DIACRITICS = "çğıöşüÇĞİÖŞÜ"

# Generic negation lexicons (for all domains)
NEGATION_WORDS_TR = ["değil", "degil", "hariç", "dışında", "disinda"]
NEGATION_WORDS_EN = ["not", "except", "excluding", "other", "other than", "aside"]

# Backend for negation extraction:
# - "llm": use an LLM with a negation-specialized prompt (default)
# - "heuristic": simple window-based rules (no model call)
NEGATION_BACKEND = os.getenv("NEGATION_BACKEND", "llm").lower()

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

    if ld_detect is not None:
        try:
            code = ld_detect(sample).lower()
            if code.startswith("tr"):
                return "tr"
            if code.startswith("en"):
                return "en"
        except Exception:
            pass

    if re.search(f"[{TR_DIACRITICS}]", sample):
        return "tr"
    return None


def _title_overlap_boost(query: str, title: str) -> float:
    """Generic token overlap boost between query and doc title."""
    q_tokens = set(re.findall(r"\w+", (query or "").lower()))
    t_tokens = set(re.findall(r"\w+", (title or "").lower()))
    if not q_tokens or not t_tokens:
        return 0.0
    inter = q_tokens & t_tokens
    return len(inter) / (len(q_tokens) + 1e-6)


def filter_top_docs_by_score(
    candidates: List[Dict],
    max_docs: int = MAX_DOCS_CONTEXT,
    query: str = ""
) -> List[Dict]:
    """Group by doc_path and keep top docs based on score + generic title overlap."""
    if not candidates or max_docs <= 0:
        return candidates

    doc_scores = {}
    doc_title_overlap = {}

    q = query or ""

    for c in candidates:
        meta = c.get("meta") or {}
        path = meta.get("source_path") or meta.get("doc_path")
        if not path:
            continue

        base_score = c.get("hybrid_score")
        if base_score is None:
            base_score = c.get("score", 0.0)

        title = meta.get("title", "")
        overlap = _title_overlap_boost(q, title)

        score = float(base_score) + 0.3 * overlap

        prev = doc_scores.get(path)
        if prev is None or score > prev:
            doc_scores[path] = score

        prev_ov = doc_title_overlap.get(path, 0.0)
        if overlap > prev_ov:
            doc_title_overlap[path] = overlap

    if not doc_scores:
        return candidates

    overlapping_paths = {p for p, ov in doc_title_overlap.items() if ov > 0}

    if overlapping_paths:
        candidate_paths = overlapping_paths
    else:
        candidate_paths = set(doc_scores.keys())

    sorted_docs = sorted(
        [(p, s) for p, s in doc_scores.items() if p in candidate_paths],
        key=lambda x: x[1],
        reverse=True
    )

    top_paths = {p for p, _ in sorted_docs[:max_docs]}

    filtered = []
    for c in candidates:
        meta = c.get("meta") or {}
        path = meta.get("source_path") or meta.get("doc_path")
        if path in top_paths:
            filtered.append(c)

    return filtered or candidates


# =============== CHROMA ====================

def get_chroma_collection():
    client = chromadb.PersistentClient(path=CHROMA_DIR)
    return client.get_collection(COLL_NAME)


# =============== DOC STATS & FULL DOC CHUNKS ============

def load_doc_chunk_stats() -> Dict[str, int]:
    global DOC_CHUNK_STATS
    if DOC_CHUNK_STATS is not None:
        return DOC_CHUNK_STATS

    if not os.path.exists(CHUNK_PARQUET):
        print(f"[warn] CHUNK_PARQUET not found: {CHUNK_PARQUET}")
        DOC_CHUNK_STATS = {}
        return DOC_CHUNK_STATS

    df = pd.read_parquet(CHUNK_PARQUET)
    if "source_path" not in df.columns or "chunk_id" not in df.columns:
        raise RuntimeError("chunks_plus2.parquet must contain 'source_path' and 'chunk_id'.")

    stats = df.groupby("source_path")["chunk_id"].count().to_dict()
    DOC_CHUNK_STATS = stats
    print(f"[info] Loaded doc chunk stats for {len(DOC_CHUNK_STATS)} documents.")
    return DOC_CHUNK_STATS


def load_chunk_df() -> pd.DataFrame:
    global CHUNK_DF
    if CHUNK_DF is None:
        if not os.path.exists(CHUNK_PARQUET):
            raise RuntimeError(f"CHUNK_PARQUET not found: {CHUNK_PARQUET}")
        CHUNK_DF = pd.read_parquet(CHUNK_PARQUET)
    return CHUNK_DF


def get_all_chunks_for_doc(source_path: str) -> List[Dict]:
    df = load_chunk_df()
    sub = df[df["source_path"] == source_path].copy()
    if "chunk_index" in sub.columns:
        sub = sub.sort_values("chunk_index")

    chunks = []
    for _, row in sub.iterrows():
        # Row may or may not have 'tags' column, depending on version of builder
        tags_val = ""
        try:
            if "tags" in row.index:
                tags_val = row["tags"]
        except Exception:
            tags_val = ""

        meta = {
            "source_path": row["source_path"],
            "json_path": row.get("json_path", ""),
            "title": row.get("title", ""),
            "h1": row.get("h1", ""),
            "h2": row.get("h2", ""),
            "h3": row.get("h3", ""),
            "doc_lang": row.get("doc_lang", row.get("content_lang", "")),
            # keep as-is; get_meta_tags() will normalize later
            "tags": tags_val,
        }
        chunks.append({
            "chunk_id": row["chunk_id"],
            "text": row["content"],
            "meta": meta,
            "source": "doc_full",
        })
    return chunks


def decide_top_k_for_candidates(candidates: List[Dict]) -> int:
    if not candidates:
        return TOP_K_FINAL_BASE

    stats = load_doc_chunk_stats()
    if not stats:
        return TOP_K_FINAL_BASE

    doc_sizes = []
    for c in candidates:
        meta = c.get("meta") or {}
        path = meta.get("source_path") or meta.get("doc_path")
        if path and path in stats:
            doc_sizes.append(stats[path])

    if not doc_sizes:
        return TOP_K_FINAL_BASE

    largest_doc_size = max(doc_sizes)
    k = largest_doc_size
    k = max(TOP_K_FINAL_BASE, k)
    k = min(k, TOP_K_FINAL_MAX)
    k = min(k, len(candidates))
    return k


# =============== EMBEDDINGS ====================

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
    v /= (np.linalg.norm(v) + 1e-12)
    return v


def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    num = float(np.dot(a, b))
    den = float(np.linalg.norm(a) * np.linalg.norm(b) + 1e-12)
    return num / den


# =============== CHROMA SEARCH ====================

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
    distances  = res.get("distances", [[]])[0]

    out = []
    for cid, d, m, dist in zip(ids, docs, metas, distances):
        sim = 1.0 - float(dist)
        out.append({
            "chunk_id": cid,
            "score": sim,
            "text": d,
            "meta": m,
            "source": "chroma",
        })
    return out


def fetch_doc_embeddings(coll, ids: List[str]) -> Dict[str, np.ndarray]:
    if not ids:
        return {}
    res = coll.get(ids=ids, include=["embeddings"])
    out = {}
    for cid, emb in zip(res["ids"], res["embeddings"]):
        out[cid] = np.array(emb, dtype=np.float32)
    return out


# =============== SMALL JSON CHAT HELPER (USED FOR TAGGING) ============

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


# =============== QUERY TAGGING (OPEN-ENDED, MATCHING DOC TAGS) =========

def infer_query_tags_llm(query: str, lang: Optional[str]) -> List[str]:
    """
    Infer open-ended semantic tags for the *user query*.
    Tag style is intentionally aligned with doc tags from build_chroma_store_plus2.py.

    Output:
      ["erasmus_study", "student_mobility", ...]
    """

    lang_hint = lang or "unknown"

    sys_prompt = textwrap.dedent("""
        You are an expert classifier for search queries in a university information system.
        Your ONLY job is to read the user query and return a JSON object describing the
        main topics of the query via TAGS.

        JSON schema (STRICT):
        {
          "tags": ["short_snake_case_tag1", "tag2", ...]
        }

        Rules for tags:
        - 2 to 8 tags is ideal, but NOT a hard limit.
        - Tags must be SHORT, lowercase, snake_case strings.
        - Tags should be semantically informative:
            * topics        (e.g. "erasmus_plus", "student_mobility", "scholarships")
            * document role (e.g. "procedure", "directive", "regulation", "form")
            * target group  (e.g. "undergraduate_students", "graduate_students")
            * important units (e.g. "international_office", "faculty_of_engineering")
        - Use English tags even if the query is in Turkish.
        - Do NOT include extremely generic tags like "document", "general", "other".

        Conventions (not restrictions):
        - If the query is clearly about Erasmus *study* / exchange mobility,
          include "erasmus_study".
        - If the query is clearly about Erasmus *internship* / traineeship,
          include "erasmus_internship".
        - If the query is clearly about student admissions, use something like
          "undergraduate_admission" or "graduate_admission".

        The tag vocabulary is OPEN-ENDED. You may invent any tags that fit the query.

        Output requirements:
        - Output STRICT JSON ONLY, no explanation, no markdown.
        - Make sure the JSON is valid and matches the schema exactly.
    """).strip()

    user_prompt = textwrap.dedent(f"""
        Language: {lang_hint}

        User query:
        {query}
    """).strip()

    raw = call_ollama_json(user_prompt, system_prompt=sys_prompt)
    tags = raw.get("tags", [])
    if not isinstance(tags, list):
        tags = []

    clean_tags: List[str] = []
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

    return clean_tags


# =============== METADATA TAG ACCESS ====================

def get_meta_tags(meta: Dict) -> List[str]:
    """
    Normalize 'tags' in metadata:
    - If list -> lowercase stripped list.
    - If CSV string -> split to list.
    - Else -> [].
    """
    if not meta:
        return []

    tags_val = meta.get("tags", [])
    parts: List[str] = []

    if isinstance(tags_val, list):
        parts = [p for p in tags_val if isinstance(p, str)]
    elif isinstance(tags_val, str):
        parts = tags_val.split(",")
    else:
        parts = []

    out: List[str] = []
    for p in parts:
        t = p.strip().lower()
        if t:
            out.append(t)
    return out


# ------------ FUZZY TAG TOKENIZATION & OVERLAP (GENERIC) ----------

GENERIC_TAG_TOKENS = {
    # very generic Turkish words related to procedures / people / etc.
    "program", "programi", "programina", "proseduru", "prosedürü",
    "yonergesi", "yönergesi", "yönetmelik", "yönetmeliği",
    "basvuru", "başvuru", "basvurulari", "başvuruları",
    "ogrenci", "ögrenci", "ogrencileri", "öğrenci", "öğrencileri",
    "lisans", "lisansustu", "yüksek", "yuksek", "doktorasi", "doktora",
    "genel", "bilgi", "hakkinda", "hakkında",
    # ultra-common English
    "form", "forms", "guide", "guideline", "policy", "policies",
    "procedure", "procedures", "regulation", "regulations",
    "students", "student",
    "plus",  # often useless by itself; 'erasmus' is the real anchor
}


def tokenize_tag(tag: str) -> set:
    """
    Turn 'erasmus_plus_programina' into tokens:
      -> {'erasmus', 'plus', 'programina'}
    Then:
      - filter out super-short tokens
      - filter out ultra-generic tokens
    """
    if not isinstance(tag, str):
        return set()

    t = tag.lower()
    # normalize everything non alnum/underscore into underscores
    t = re.sub(r"[^a-z0-9_]+", "_", t)
    parts = [p for p in t.split("_") if p]

    out = set()
    for p in parts:
        if len(p) <= 2:
            continue
        if p in GENERIC_TAG_TOKENS:
            continue
        out.add(p)
    return out


def soft_tag_overlap(query_tags: List[str], doc_tags: List[str]) -> int:
    """
    Fuzzy overlap between query tags and doc tags:
    - compare at TOKEN level, not whole-tag equality
    - return the integer size of the shared token set
    """
    q_tokens = set()
    for qt in query_tags:
        q_tokens |= tokenize_tag(qt)

    d_tokens = set()
    for dt in doc_tags:
        d_tokens |= tokenize_tag(dt)

    if not q_tokens or not d_tokens:
        return 0

    return len(q_tokens & d_tokens)


# =============== FOLLOWUP / INTENT (LEXICAL) ====================

def get_last_user_query(history: List[Dict]) -> Optional[str]:
    for msg in reversed(history):
        if msg.get("role") == "user":
            text = (msg.get("content") or "").strip()
            if text:
                return text
    return None


def is_followup_semantic(query: str,
                         history: List[Dict],
                         threshold: float = 0.60) -> bool:
    last_q = get_last_user_query(history)
    if not last_q:
        return False

    try:
        q_vec = embed_text_ollama(query)
        last_vec = embed_text_ollama(last_q)
    except Exception as e:
        print(f"[followup-semantic] embedding error: {e}")
        return False

    sim = cosine_sim(q_vec, last_vec)
    print(f"[debug] semantic followup similarity = {sim:.3f}")
    return sim >= threshold


def detect_query_intent_lexical(query: str, lang: Optional[str] = None) -> str:
    if not query:
        return "other"

    q = query.strip().lower()

    # --- generic "count" intent ---
    count_triggers = [
        "kaç tane", "kaç adet", "sayısı kaç", "sayisi kaç",
        "sayısı nedir", "sayisi nedir", "toplam kaç",
    ]
    count_triggers_en = ["how many", "number of", "how much"]

    if any(t in q for t in count_triggers + count_triggers_en):
        return "count_items"

    # --- generic "list names" intent ---
    list_triggers = [
        "isimlerini say", "isimlerini listele", "isimlerini yaz",
        "adlarını say", "adlarini say", "adlarını listele",
        "adlarını yaz", "kategorilerin isimlerini say",
        "kategorilerini listele", "kategori isimlerini say",
    ]
    list_triggers_en = [
        "list the names", "list all", "list all the",
        "say the names", "enumerate the", "enumerate all",
    ]
    if any(t in q for t in list_triggers + list_triggers_en):
        return "list_names"

    if lang == "tr":
        q_nopunct = re.sub(r"[?!.…]+$", "", q).strip()
        if q_nopunct.endswith(" say"):
            if not any(t in q for t in ["sayısı", "sayisi", "kaç tane", "kaç adet", "sayısı kaç", "sayisi kaç"]):
                return "list_names"

    # --- generic "describe" intent ---
    describe_triggers = ["nedir", "ne demek", "açıkla", "detaylandır"]
    describe_triggers_en = ["explain", "describe", "what is", "what are"]

    if any(t in q for t in describe_triggers + describe_triggers_en):
        return "describe"

    return "other"


def is_pronominal_followup_lexical(query: str, lang: Optional[str] = None) -> bool:
    if not query:
        return False

    q_strip = (query or "").strip().lower()

    follow_phrases = [
        "bunu", "şunu", "onu",
        "bunu daha", "şunu daha", "onu daha",
        "daha detaylı", "detaylı açıkla", "detaylandır",
        "madde madde", "maddeler halinde",
        "bir daha açıkla", "tekrar açıkla",
        "kısaca", "özetle", "devam et",
    ]
    pronoun_patterns = [
        r"\b(bu|şu|o)\s+\w+(ya|ye|a|e|ı|i|u|ü|yı|yi|yu|yü|ın|in|un|ün|nın|nin|nun|nün|da|de|ta|te|dan|den|tan|ten)?\b",
    ]

    if any(phr in q_strip for phr in follow_phrases):
        return True

    for pat in pronoun_patterns:
        if re.search(pat, q_strip):
            return True

    return False


# =============== LLM DIALOGUE CLASSIFIER ====================

def classify_dialogue_llm(query: str,
                          history: List[Dict],
                          lang: Optional[str]) -> Dict:
    url = f"{OLLAMA_HOST}/api/chat"

    recent = history[-8:]
    convo_lines = []
    for msg in recent:
        role = msg.get("role", "user")
        content = msg.get("content", "")
        convo_lines.append(f"{role.upper()}: {content}")
    convo_text = "\n".join(convo_lines)

    base_sys = (
        "You are a dialogue classifier for a retrieval-based Q&A assistant.\n"
        "You must output STRICT JSON ONLY, no explanation.\n"
        "Your tasks:\n"
        "1) Decide if the latest user query is a FOLLOW-UP to earlier questions.\n"
        "2) Classify intent into one of: list_names, count_items, describe, other.\n"
        "Schema:\n"
        "{\n"
        '  "intent": "list_names" | "count_items" | "describe" | "other",\n'
        '  "followup": true/false,\n'
        '  "anchor_strategy": "self" | "last_non_followup"\n'
        "}\n"
        'If unsure, use followup=false and intent="other".\n'
    )

    if lang == "tr":
        examples = (
            "TR examples:\n"
            '- "ödül kategorilerinin isimlerini say" -> intent: "list_names"\n'
            '- "kaç tane ödül kategorisi var" -> intent: "count_items"\n'
            '- "bu ödül ne anlama geliyor" -> intent: "describe"\n'
            '- "başvuru formunu nereden bulabilirim" -> intent: "other"\n'
        )
        system_prompt = base_sys + examples
    else:
        examples = (
            "EN examples:\n"
            '- "list the names of all categories" -> "list_names"\n'
            '- "how many categories are there" -> "count_items"\n'
            '- "explain what this award means" -> "describe"\n'
        )
        system_prompt = base_sys + examples

    user_prompt = (
        f"Conversation (oldest to newest):\n{convo_text}\n\n"
        f"Latest user query:\n{query}\n\n"
        "Now output ONLY the JSON:"
    )

    payload = {
        "model": CHAT_MODEL,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_prompt},
        ],
        "stream": False,
        "options": {"temperature": 0.0},
    }

    default = {
        "intent": "other",
        "followup": False,
        "anchor_strategy": "self",
    }

    try:
        resp = requests.post(url, json=payload, timeout=120)
        resp.raise_for_status()
        data = resp.json()
        text = data.get("message", {}).get("content", "").strip()
        if not text:
            return default
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            m = re.search(r"\{.*\}", text, flags=re.DOTALL)
            if not m:
                return default
            parsed = json.loads(m.group(0))
        intent = parsed.get("intent", "other")
        if intent not in {"list_names", "count_items", "describe", "other"}:
            intent = "other"
        followup = bool(parsed.get("followup", False))
        anchor_strategy = parsed.get("anchor_strategy", "self")
        if anchor_strategy not in {"self", "last_non_followup"}:
            anchor_strategy = "self"
        return {
            "intent": intent,
            "followup": followup,
            "anchor_strategy": anchor_strategy,
        }
    except Exception as e:
        print(f"[dialogue-classifier error] {e}")
        return default


# =============== NEGATION EXTRACTION (LLM + HEURISTIC) ====================

def _extract_negated_terms_heuristic(
    query: str,
    lang: Optional[str] = None,
    window_before: int = 6,
    window_after: int = 4,
) -> List[str]:
    """
    Heuristic negation extractor.

    Key idea:
    - Find tokens in a small window around negation markers
      (TR: 'değil', 'hariç', 'dışında'; EN: 'not', 'except', ...).
    - THEN keep only those terms that *do not appear anywhere else*
      in the query outside these negation windows.

    This prevents generic terms like 'erasmus', 'programı', etc.
    from being treated as negated if they also occur in a positive context.
    """
    if not query:
        return []

    q = query.lower()
    raw_tokens = re.findall(r"\w+|[^\w\s]", q)
    word_tokens = [t for t in raw_tokens if re.match(r"\w+", t)]

    if lang is None:
        lang = guess_lang_from_text(query)

    stop_terms = {
        "ve", "veya", "ama", "fakat", "ancak",
        "ile", "için", "icin", "ya", "ya da", "yada",
    }

    if lang == "tr":
        neg_words_before = {"değil", "degil", "hariç", "hariç.", "hariç,", "dışında", "disinda"}
        neg_words_after  = set()
    else:
        neg_words_before = set()
        neg_words_after  = {"not", "except", "excluding", "aside", "other", "other than"}

    cand_positions: List[Tuple[str, int]] = []

    for w_idx, token in enumerate(word_tokens):
        t = token.lower()

        if t in neg_words_before:
            start = max(0, w_idx - window_before)
            for j in range(start, w_idx):
                cand = word_tokens[j].strip().lower()
                if not cand or cand in stop_terms:
                    continue
                cand_positions.append((cand, j))

        if t in neg_words_after:
            end = min(len(word_tokens), w_idx + 1 + window_after)
            for j in range(w_idx + 1, end):
                cand = word_tokens[j].strip().lower()
                if not cand or cand in stop_terms:
                    continue
                cand_positions.append((cand, j))

    if not cand_positions:
        return []

    cand_indices: Dict[str, set] = {}
    for term, idx in cand_positions:
        cand_indices.setdefault(term, set()).add(idx)

    global_indices: Dict[str, set] = {}
    for i, tok in enumerate(word_tokens):
        term = tok.lower()
        global_indices.setdefault(term, set()).add(i)

    neg_terms: List[str] = []
    for term, idxs_in_neg in cand_indices.items():
        outside_positions = global_indices[term] - idxs_in_neg
        if outside_positions:
            continue

        if term in stop_terms or len(term) <= 1:
            continue

        neg_terms.append(term)

    seen = set()
    result: List[str] = []
    for t in neg_terms:
        if t not in seen:
            seen.add(t)
            result.append(t)
    return result


def extract_negated_terms_llm(query: str, lang: Optional[str] = None) -> List[str]:
    """
    Use an LLM (via Ollama) to extract EXPLICITLY NEGATED or EXCLUDED terms
    from the user query.
    """
    if not query:
        return []

    url = f"{OLLAMA_HOST}/api/chat"

    base_sys = (
        "You are a highly precise NEGATION / EXCLUSION extractor for search queries.\n"
        "Your ONLY job is to detect which content words or phrases are explicitly NEGATED,\n"
        "EXCLUDED, or specified as 'not wanted' by the user.\n\n"
        "You MUST follow this JSON schema and output STRICT JSON ONLY:\n"
        "{\n"
        '  "negated_terms": ["term1", "term2", ...]\n'
        "}\n\n"
        "Guidelines:\n"
        "- Only include terms that are clearly under the scope of a negation/exclusion word\n"
        "  such as: not, except, excluding, other than, hariç, değil, dışında, disinda.\n"
        "- Return the terms in LOWERCASE.\n"
        "- Prefer short, searchable units (single words or short noun phrases).\n"
        "- If nothing is explicitly negated, return an empty list.\n"
    )

    if lang == "tr":
        examples = (
            "TR examples:\n"
            '- Soru: \"yurtlar hariç tüm binaları getir\" -> {\"negated_terms\": [\"yurtlar\"]}\n'
            '- Soru: \"spor salonu değil, kütüphane\" -> {\"negated_terms\": [\"spor salonu\"]}\n'
            '- Soru: \"sadece fakülteler\" -> {\"negated_terms\": []}\n'
        )
        system_prompt = base_sys + examples
    else:
        examples = (
            "EN examples:\n"
            '- Q: \"all units except dormitory and sports center\" -> '
            '{\"negated_terms\": [\"dormitory\", \"sports center\"]}\n'
            '- Q: \"not internships, only full-time roles\" -> {\"negated_terms\": [\"internships\"]}\n'
            '- Q: \"list all programs\" -> {\"negated_terms\": []}\n'
        )
        system_prompt = base_sys + examples

    user_prompt = (
        f"User query:\n{query}\n\n"
        'Now return ONLY the JSON object with the field "negated_terms".'
    )

    payload = {
        "model": CHAT_MODEL,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_prompt},
        ],
        "stream": False,
        "options": {"temperature": 0.0},
    }

    try:
        resp = requests.post(url, json=payload, timeout=60)
        resp.raise_for_status()
        data = resp.json()
        text = (data.get("message", {}) or {}).get("content", "").strip()
        if not text:
            return []

        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            m = re.search(r"\{.*\}", text, flags=re.DOTALL)
            if not m:
                return []
            parsed = json.loads(m.group(0))

        terms = parsed.get("negated_terms", [])
        if not isinstance(terms, list):
            return []

        out: List[str] = []
        for t in terms:
            if not isinstance(t, str):
                continue
            t_clean = t.strip().lower()
            if t_clean:
                out.append(t_clean)
        return out
    except Exception as e:
        print(f"[negation-llm error] {e}")
        return []


def extract_negated_terms(query: str, lang: Optional[str] = None, window: int = 3) -> List[str]:
    """
    Negation-aware term extractor used by apply_negation_penalty.
    """
    if not query:
        return []

    if lang is None:
        lang = guess_lang_from_text(query)

    backend = (NEGATION_BACKEND or "llm").lower()
    print(f"[negation] backend={backend}, lang={lang}, query={query!r}")

    if backend == "llm":
        terms = extract_negated_terms_llm(query, lang)
        if terms:
            print(f"[debug] negation terms (llm): {terms}")
            return terms
        print("llm negation extractor returned no terms. Trying heuristic...")

        terms = _extract_negated_terms_heuristic(query, lang, window_before=6, window_after=4)
        if terms:
            print(f"[debug] negation terms (heuristic fallback): {terms}")
        else:
            print("[debug] heuristic fallback also found no negated terms.")
        return terms

    terms = _extract_negated_terms_heuristic(query, lang, window_before=6, window_after=4)
    if terms:
        print(f"[debug] negation terms (heuristic): {terms}")
    else:
        print("[debug] negation heuristic found no terms.")
    return terms


# =============== ANCHOR PICKER (SEMANTIC) ====================

def pick_anchor_query_semantic(query: str,
                               history: List[Dict],
                               max_history_users: int = 10,
                               min_sim: float = 0.62) -> Optional[str]:
    if not history:
        return None

    try:
        q_vec = embed_text_ollama(query)
    except Exception as e:
        print(f"[anchor] error embedding query: {e}")
        return None

    best_sim = -1.0
    best_text = None
    count = 0

    for msg in reversed(history):
        if msg.get("role") != "user":
            continue
        prev_q = (msg.get("content") or "").strip()
        if not prev_q or prev_q == query:
            continue
        try:
            v = embed_text_ollama(prev_q)
        except Exception as e:
            print(f"[anchor] error embedding history query: {e}")
            continue
        sim = cosine_sim(q_vec, v)
        if sim > best_sim:
            best_sim = sim
            best_text = prev_q

        count += 1
        if count >= max_history_users:
            break

    if best_text is not None and best_sim >= min_sim:
        print(f"[debug] anchor chosen (sim={best_sim:.3f}): {repr(best_text)}")
        return best_text

    return None


# =============== MMR + DOC-CENTRIC SELECTION ====================

def mmr_select(candidates: List[Dict],
               doc_embs: Dict[str, np.ndarray],
               query_vec: np.ndarray,
               k: int = TOP_K_FINAL_BASE,
               lambda_mmr: float = 0.7) -> List[Dict]:
    selected: List[Dict] = []
    selected_ids: List[str] = []

    query_sims = {}
    for c in candidates:
        cid = c["chunk_id"]
        emb = doc_embs.get(cid)
        query_sims[cid] = 0.0 if emb is None else cosine_sim(query_vec, emb)

    while len(selected) < min(k, len(candidates)):
        best_cand = None
        best_mmr = -1e9

        for c in candidates:
            cid = c["chunk_id"]
            if cid in selected_ids:
                continue

            rel = 0.5 * c.get("hybrid_score", 0.0) + 0.5 * query_sims.get(cid, 0.0)

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


def select_top_docs_and_chunks(
    candidates: List[Dict],
    max_docs: int = 3,
    chunks_per_doc: int = 4,
) -> List[Dict]:
    """
    Generic doc-centric selection:
    - Group chunks by source_path/doc_path.
    - Score each doc by its best hybrid_score (with mild saturation).
    - Keep top `max_docs` docs.
    - For each kept doc, keep up to `chunks_per_doc` best chunks.
    """
    if not candidates:
        return []

    by_doc: Dict[str, List[Dict]] = {}
    for c in candidates:
        meta = c.get("meta") or {}
        path = meta.get("source_path") or meta.get("doc_path")
        if not path:
            continue
        by_doc.setdefault(path, []).append(c)

    if not by_doc:
        return candidates

    doc_scores = []
    for path, chunks in by_doc.items():
        sorted_chunks = sorted(chunks, key=lambda x: x.get("hybrid_score", 0.0), reverse=True)
        top_scores = [ch.get("hybrid_score", 0.0) for ch in sorted_chunks[:5]]
        score = max(top_scores) if top_scores else 0.0
        doc_scores.append((path, score))

    doc_scores.sort(key=lambda x: x[1], reverse=True)
    keep_paths = {path for path, _ in doc_scores[:max_docs]}

    final_chunks: List[Dict] = []
    for path in keep_paths:
        chunks = by_doc[path]
        chunks_sorted = sorted(chunks, key=lambda x: x.get("hybrid_score", 0.0), reverse=True)
        final_chunks.extend(chunks_sorted[:chunks_per_doc])

    final_chunks.sort(key=lambda x: x.get("hybrid_score", 0.0), reverse=True)
    return final_chunks


# =============== CROSS-ENCODER RERANK ====================

def get_reranker() -> CrossEncoder:
    global RERANKER
    if RERANKER is None:
        RERANKER = CrossEncoder(RERANKER_MODEL_NAME)
    return RERANKER


def cross_encoder_score_pair(query: str, passage: str) -> float:
    model = get_reranker()
    score = model.predict([(query, passage)])[0]
    return float(score)


def cross_encoder_rerank(query: str,
                         candidates: List[Dict],
                         max_candidates: int = CROSS_MAX_CANDIDATES,
                         weight_ce: float = CROSS_WEIGHT) -> List[Dict]:
    """Generic CE + vector hybrid rerank, optional title-overlap bonus. No domain-specific hacks."""
    if not candidates:
        return []

    subset = candidates[:max_candidates].copy()

    # 1) compute CE scores
    for c in subset:
        ce = cross_encoder_score_pair(query, c["text"])
        c["ce_score"] = ce

    ce_vals  = [c.get("ce_score", 0.0) for c in subset]
    vec_vals = [c.get("score", 0.0)    for c in subset]

    max_ce  = max(ce_vals)  if ce_vals  else 1.0
    max_vec = max(vec_vals) if vec_vals else 1.0
    if max_ce <= 0:
        max_ce = 1.0
    if max_vec <= 0:
        max_vec = 1.0

    q_tokens = set(re.findall(r"\w+", (query or "").lower()))

    for c in subset:
        ce_norm  = c.get("ce_score", 0.0) / max_ce
        vec_norm = c.get("score", 0.0)    / max_vec

        hybrid = weight_ce * ce_norm + (1.0 - weight_ce) * vec_norm

        meta = c.get("meta") or {}
        title = (meta.get("title") or "").lower()
        t_tokens = set(re.findall(r"\w+", title))
        overlap = 0.0
        if q_tokens and t_tokens:
            overlap = len(q_tokens & t_tokens) / (len(q_tokens) + 1e-6)
            hybrid *= (1.0 + 0.2 * overlap)

        c["ce_norm"]      = ce_norm
        c["vec_norm"]     = vec_norm
        c["bm25_score"]   = 0.0
        c["vec_score"]    = c.get("score", 0.0)
        c["hybrid_score"] = hybrid

    subset.sort(key=lambda x: x["hybrid_score"], reverse=True)
    return subset


# =============== NEGATION-AWARE PENALTY (GENERIC) ====================

def apply_negation_penalty(
    query: str,
    candidates: List[Dict],
    lang: Optional[str] = None,
    penalty_factor: float = 0.3,
) -> List[Dict]:
    """
    Generic rerank post-processing:
    - If query contains negated terms (X değil, not X, except X...),
      downweight chunks whose titles/headers clearly mention those terms.
    """
    neg_terms = extract_negated_terms(query, lang)
    if not neg_terms:
        return candidates

    neg_terms = set(neg_terms)

    for c in candidates:
        meta = c.get("meta") or {}
        title_blob = " ".join(
            str(meta.get(k, "")) for k in ("title", "h1", "h2", "h3")
        ).lower()

        if any(t in title_blob for t in neg_terms):
            old = c.get("hybrid_score", 0.0)
            c["hybrid_score"] = old * penalty_factor

    candidates.sort(key=lambda x: x.get("hybrid_score", 0.0), reverse=True)
    return candidates


# =============== TAG-AWARE PRIOR (GENERIC, OPEN-ENDED TAGS) ============

def apply_tag_prior(
    query_tags: List[str],
    candidates: List[Dict],
    boost_positive: float = 0.20,
    penalize_negative: float = 0.15,
) -> List[Dict]:
    """
    Use LLM-based OPEN-ENDED tags to bias ranking:
    - If any docs share tags with the query (at token level), chunks from those docs get boosted.
    - Chunks from docs with zero overlap get slightly downweighted.
    - If NO doc has any overlapping tag, do nothing (to avoid mis-routing).

    Now uses fuzzy, token-level overlap:
      "erasmus_plus_programina"  vs  "erasmus_plus"
    share the token "erasmus" -> positive overlap.
    """
    if not candidates or not query_tags:
        return candidates

    # keep raw tags; tokenization is inside soft_tag_overlap
    q_raw = [t for t in query_tags if isinstance(t, str) and t.strip()]
    if not q_raw:
        return candidates

    doc_overlap: Dict[str, int] = {}
    for c in candidates:
        meta = c.get("meta") or {}
        path = meta.get("source_path") or meta.get("doc_path")

        dtags_raw = get_meta_tags(meta)
        overlap = soft_tag_overlap(q_raw, dtags_raw)  # fuzzy token-level overlap
        c["_tag_overlap"] = overlap

        if path:
            prev = doc_overlap.get(path, 0)
            if overlap > prev:
                doc_overlap[path] = overlap

    max_overlap = max(doc_overlap.values()) if doc_overlap else 0
    if max_overlap <= 0:
        # No doc has any overlapping tag -> don't touch ranking
        return candidates

    print(f"[debug] apply_tag_prior: query_tags={q_raw}, max_doc_overlap={max_overlap}")

    for c in candidates:
        base = c.get("hybrid_score", c.get("score", 0.0))
        overlap = c.get("_tag_overlap", 0)
        if overlap > 0:
            factor = 1.0 + boost_positive * min(overlap, 3)
        else:
            factor = 1.0 - penalize_negative
        c["hybrid_score"] = base * factor

    candidates.sort(key=lambda x: x.get("hybrid_score", 0.0), reverse=True)
    return candidates


# =============== CE THRESHOLD FILTER ====================

def filter_by_ce_threshold(
    candidates: List[Dict],
    threshold: float = CE_SCORE_THRESHOLD,
    min_keep: int = 4,
    max_keep: int = 12
) -> List[Dict]:
    if not candidates:
        return []

    def get_use_score(c: Dict) -> float:
        # Prefer the full hybrid score (vector + CE + negation + tags)
        if "hybrid_score" in c:
            return float(c["hybrid_score"])
        if "ce_norm" in c:
            return float(c["ce_norm"])
        return float(c.get("ce_score", 0.0))

    strong = [c for c in candidates if get_use_score(c) >= threshold]

    if len(strong) >= min_keep:
        return strong

    sorted_c = sorted(candidates, key=lambda c: get_use_score(c), reverse=True)
    return sorted_c[:min(max_keep, len(sorted_c))]



# =============== OLLAMA CHAT ====================

def call_ollama_chat(prompt: str, system_prompt: str = "", model: str = CHAT_MODEL) -> str:
    url_chat = f"{OLLAMA_HOST}/api/chat"

    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})

    payload = {
        "model": model,
        "messages": messages,
        "stream": False,
        "options": {
            "temperature": 0.0,
            "top_p": 0.9,
        },
    }

    try:
        print(f"📤 Ollama'ya İstek Gönderiliyor... Model: {model}")
        resp = requests.post(url_chat, json=payload, timeout=300)
        if resp.status_code != 200:
            print(f"❌ OLLAMA HATASI (Status: {resp.status_code}): {resp.text}")
            return f"Ollama Hatası: {resp.text}"

        data = resp.json()
        return data.get("message", {}).get("content", "Boş cevap döndü.")
    except Exception as e:
        print(f"❌ Bağlantı Hatası: {e}")
        return "Üzgünüm, yapay zeka servisine bağlanılamadı."


# =============== CONTEXT BUILD ====================

def build_context(chunks: List[Dict]) -> str:
    parts = []
    for i, ch in enumerate(chunks):
        meta = ch.get("meta", {}) or {}
        path = meta.get("source_path") or meta.get("doc_path", "")
        title = meta.get("title", "")
        lang = meta.get("doc_lang", "")
        header = f"[{i+1}] lang={lang} | title={title} | file={os.path.basename(path)}"
        parts.append(header + "\n" + ch["text"])
    return "\n\n-----\n\n".join(parts)


# =============== MAIN RAG PIPELINE ====================

def doc_level_filter_by_tags_and_negation(
    candidates: List[Dict],
    query_tags: List[str],
    retrieval_query: str,
    lang: Optional[str],
) -> List[Dict]:
    """
    Stronger filter for chroma-mmr:
    - Keep only docs whose tags overlap with query_tags (at token level).
    - If there are docs with overlap, drop all docs with zero overlap.
    - Also, if there are both negated and non-negated docs, drop the negated ones
      (e.g. titles containing 'staj' when 'staj' is negated in the query).
    """
    if not candidates or not query_tags:
        return candidates

    # Normalized query tags
    q_raw = [t for t in query_tags if isinstance(t, str) and t.strip()]
    if not q_raw:
        return candidates

    # 1) Compute tag overlap per doc
    doc_overlap: Dict[str, int] = {}
    for c in candidates:
        meta = c.get("meta") or {}
        path = meta.get("source_path") or meta.get("doc_path")
        if not path:
            continue
        dtags_raw = get_meta_tags(meta)
        overlap = soft_tag_overlap(q_raw, dtags_raw)
        if overlap > 0:
            prev = doc_overlap.get(path, 0)
            if overlap > prev:
                doc_overlap[path] = overlap

    if not doc_overlap:
        # No doc has any overlapping tag -> don't gate by tags
        return candidates

    tagged_paths = set(doc_overlap.keys())

    # 2) Negation-aware: find docs whose titles contain negated terms
    neg_terms = set(extract_negated_terms(retrieval_query, lang))
    doc_has_negated: Dict[str, bool] = {}

    if neg_terms:
        for c in candidates:
            meta = c.get("meta") or {}
            path = meta.get("source_path") or meta.get("doc_path")
            if not path or path not in tagged_paths:
                continue
            title_blob = " ".join(
                str(meta.get(k, "")) for k in ("title", "h1", "h2", "h3")
            ).lower()
            if any(t in title_blob for t in neg_terms):
                doc_has_negated[path] = True

    # 3) Candidate doc set:
    candidate_paths = tagged_paths

    # If we have both negated and non-negated docs, drop the negated ones
    if doc_has_negated:
        non_neg_paths = {p for p in candidate_paths if not doc_has_negated.get(p, False)}
        if non_neg_paths:
            candidate_paths = non_neg_paths

    # 4) Filter candidates
    gated = []
    for c in candidates:
        meta = c.get("meta") or {}
        path = meta.get("source_path") or meta.get("doc_path")
        if path in candidate_paths:
            gated.append(c)

    return gated or candidates


def answer_with_rag(query: str,
                    mode: str,
                    coll,
                    history: List[Dict] = []) -> str:
    q_lang = guess_lang_from_text(query)
    print(f"(auto-detected query language: {q_lang})")

    # ---- dialog classification ----
    dialogue_cls = classify_dialogue_llm(query, history, q_lang)
    intent_llm = dialogue_cls.get("intent", "other")
    followup_llm = bool(dialogue_cls.get("followup", False))
    print(f"[debug] dialogue_cls = {dialogue_cls}")

    intent_lex = detect_query_intent_lexical(query, q_lang)
    intent = intent_lex if intent_lex != "other" else intent_llm
    print(f"[debug] intent = {intent} (lex={intent_lex}, llm={intent_llm})")

    lexical_followup   = is_pronominal_followup_lexical(query, q_lang)
    semantic_followup  = is_followup_semantic(query, history)
    is_followup        = followup_llm or lexical_followup or semantic_followup
    anchor_strategy    = dialogue_cls.get("anchor_strategy", "self")
    print(
        f"[debug] followup = {is_followup} "
        f"(llm={followup_llm}, lexical={lexical_followup}, strategy={anchor_strategy})"
    )

    # ---- anchor query selection (generic) ----
    anchor = None
    if is_followup:
        if lexical_followup or anchor_strategy == "last_non_followup":
            anchor = get_last_user_query(history)
        else:
            anchor = pick_anchor_query_semantic(query, history)

    if anchor is not None:
        print(f"[debug] using anchor query: {repr(anchor)}")
        retrieval_query = f"{anchor}\n\n{query}"
    else:
        retrieval_query = query

    print(f"[debug] retrieval_query = {repr(retrieval_query)}")

    lang_filter = q_lang if q_lang in ("tr", "en") else None
    query_vec = embed_text_ollama(retrieval_query)

    # ---- LLM-based query tags (for semantic gating) ----
    query_tags = infer_query_tags_llm(retrieval_query, q_lang)
    print(f"[debug] query_tags = {query_tags}")

    # ---- retrieval ----
    if mode == "chroma":
        vec_candidates = chroma_search(
            retrieval_query,
            coll,
            query_vec=query_vec,
            top_k=TOP_K_CHROMA,
            lang_filter=lang_filter,
        )

        doc_scores: Dict[str, float] = {}
        doc_tag_hits: Dict[str, int] = {}
        q_raw = query_tags or []

        for c in vec_candidates:
            meta = c.get("meta") or {}
            path = meta.get("source_path") or meta.get("doc_path")
            if not path:
                continue

            doc_scores[path] = doc_scores.get(path, 0.0) + c["score"]

            if q_raw:
                dtags_raw = get_meta_tags(meta)
                overlap = soft_tag_overlap(q_raw, dtags_raw)
                if overlap > 0:
                    prev = doc_tag_hits.get(path, 0)
                    if overlap > prev:
                        doc_tag_hits[path] = overlap

        candidate_paths = set(doc_scores.keys())

        # If we have any docs with tag overlap, restrict competition to those docs only
        if q_raw and doc_tag_hits:
            tag_matched_paths = {p for p, hit in doc_tag_hits.items() if hit > 0}
            if tag_matched_paths:
                candidate_paths = tag_matched_paths
                print(f"[debug] tag-gating active in 'chroma' mode, candidate_docs={len(candidate_paths)}")

        if candidate_paths:
            best_doc_path = max(
                ((p, s) for p, s in doc_scores.items() if p in candidate_paths),
                key=lambda x: x[1]
            )[0]
            doc_chunks = get_all_chunks_for_doc(best_doc_path)
            top_k_final = min(len(doc_chunks), TOP_K_FINAL_MAX)
            retrieved = doc_chunks[:top_k_final]
        else:
            top_k_final = decide_top_k_for_candidates(vec_candidates)
            retrieved = vec_candidates[:top_k_final]

    elif mode == "chroma-mmr":
        vec_results = chroma_search(
            retrieval_query,
            coll,
            query_vec=query_vec,
            top_k=TOP_K_CHROMA,
            lang_filter=lang_filter,
        )
        print("=== DEBUG TOP VECTOR CANDIDATES ===")
        for i, c in enumerate(vec_results[:20], start=1):
            meta = c.get("meta") or {}
            print(
                f"{i:2d}. score={c['score']:.3f} | "
                f"title={meta.get('title')} | "
                f"file={os.path.basename(meta.get('source_path') or meta.get('doc_path',''))}"
            )
        print("===================================")

        # NEW: doc-level tag + negation gating
        vec_results = doc_level_filter_by_tags_and_negation(
            vec_results,
            query_tags=query_tags,
            retrieval_query=retrieval_query,
            lang=q_lang,
        )
        print(f"[debug] after doc-level tag+negation gating, candidates={len(vec_results)}")

        anchor_doc_paths = set()
        if USE_ANCHOR_DOCS and is_followup and anchor:
            try:
                anchor_vec = embed_text_ollama(anchor)
                anchor_results = chroma_search(
                    anchor,
                    coll,
                    query_vec=anchor_vec,
                    top_k=TOP_K_CHROMA,
                    lang_filter=lang_filter,
                )
                for c in anchor_results:
                    meta = c.get("meta") or {}
                    p = meta.get("source_path") or meta.get("doc_path")
                    if p:
                        anchor_doc_paths.add(p)

                print(f"[debug] anchor_doc_paths (len={len(anchor_doc_paths)}):")
                for p in list(anchor_doc_paths)[:10]:
                    print("   ", os.path.basename(p))
            except Exception as e:
                print(f"[anchor-docs error] {e}")
                anchor_doc_paths = set()
        else:
            anchor_doc_paths = set()

        if not vec_results:
            retrieved = []
        else:
            reranked = cross_encoder_rerank(retrieval_query, vec_results)

            # Negation-aware penalty
            reranked = apply_negation_penalty(retrieval_query, reranked, q_lang)

            # Tag-aware prior using open-ended tags (now fuzzy)
            reranked = apply_tag_prior(query_tags, reranked)

            print("=== DEBUG AFTER CE RERANK + TAG PRIOR ===")
            for i, c in enumerate(reranked[:20], start=1):
                meta = c.get("meta") or {}
                print(
                    f"{i:2d}. ce={c.get('ce_score',0):.3f} hyb={c.get('hybrid_score',0):.3f} "
                    f"| title={meta.get('title')}"
                )
            print("===================================")

            high_conf_single_doc = False

            # (1) Very confident CE single-doc mode (generic)
            if reranked:
                top = reranked[0]
                top_ce = top.get("ce_score", 0.0)
                second_ce = reranked[1].get("ce_score", 0.0) if len(reranked) > 1 else 0.0
                if top_ce >= 0.85 and (top_ce - second_ce) >= 0.07:
                    meta_top = top.get("meta") or {}
                    path = meta_top.get("source_path") or meta_top.get("doc_path")
                    if path:
                        print(
                            f"[debug] CE single-doc mode triggered for "
                            f"{os.path.basename(path)} (ce={top_ce:.3f}, gap={top_ce-second_ce:.3f})"
                        )
                        doc_chunks = get_all_chunks_for_doc(path)
                        retrieved = doc_chunks[:TOP_K_FINAL_MAX]
                        high_conf_single_doc = True

            # (2) Intent-aware single-doc mode for list/count
            if not high_conf_single_doc and intent in {"list_names", "count_items"}:
                doc_scores = {}
                for c in reranked:
                    meta = c.get("meta") or {}
                    path = meta.get("source_path") or meta.get("doc_path")
                    if not path:
                        continue
                    doc_scores[path] = doc_scores.get(path, 0.0) + c.get("hybrid_score", 0.0)

                if doc_scores:
                    best_path, best_score = max(doc_scores.items(), key=lambda x: x[1])
                    print(
                        f"[debug] intent-aware single-doc mode active "
                        f"(intent={intent}, best_doc={os.path.basename(best_path)}, "
                        f"score={best_score:.3f})"
                    )
                    doc_chunks = get_all_chunks_for_doc(best_path)
                    retrieved = doc_chunks[:TOP_K_FINAL_MAX]
                    high_conf_single_doc = True

            if not high_conf_single_doc:
                filtered = filter_by_ce_threshold(
                    reranked,
                    threshold=CE_SCORE_THRESHOLD
                )
                if not filtered:
                    filtered = reranked[:TOP_K_FINAL_BASE]

                top_path = None
                if reranked:
                    top_meta = reranked[0].get("meta") or {}
                    top_path = top_meta.get("source_path") or top_meta.get("doc_path")

                if USE_ANCHOR_DOCS and is_followup and anchor_doc_paths and top_path in anchor_doc_paths:
                    print(f"[debug] anchor preference active (top_path in anchor_doc_paths): {os.path.basename(top_path)}")
                    anchor_filtered = []
                    other_filtered = []

                    for c in filtered:
                        meta = c.get("meta") or {}
                        p = meta.get("source_path") or meta.get("doc_path")
                        if p in anchor_doc_paths:
                            anchor_filtered.append(c)
                        else:
                            other_filtered.append(c)

                    MIN_ANCHOR = 4

                    if len(anchor_filtered) >= MIN_ANCHOR:
                        filtered = anchor_filtered[:TOP_K_FINAL_MAX]
                    else:
                        needed = max(MIN_ANCHOR - len(anchor_filtered), 0)
                        filtered = anchor_filtered + other_filtered[:needed]
                else:
                    if is_followup and not USE_ANCHOR_DOCS:
                        print("[debug] followup, but anchor-doc preference is disabled (USE_ANCHOR_DOCS=0)")

                filtered = select_top_docs_and_chunks(
                    filtered,
                    max_docs=3,
                    chunks_per_doc=4,
                )

                candidate_ids = [c["chunk_id"] for c in filtered]
                doc_embs = fetch_doc_embeddings(coll, candidate_ids)

                k = min(len(filtered), TOP_K_FINAL_MAX)
                retrieved = mmr_select(filtered, doc_embs, query_vec,
                                       k=k, lambda_mmr=0.7)
    else:
        raise ValueError("mode must be one of: chroma, chroma-mmr")

    context = build_context(retrieved)
    print("=== DEBUG CONTEXT START ===")
    print(context)
    print("=== DEBUG CONTEXT END ===")

    # ---------- SYSTEM PROMPTS (GENERIC, NO DOMAIN-SPECIFIC HACKS) ----------

    if q_lang == "tr":
        base_tr_sys = (
            "Sen bir kurum içi bilgi sistemine bağlı, TÜRKÇE konuşan profesyonel bir asistansın.\n\n"
            "GENEL KURALLAR:\n"
            "1) Cevabın tamamen akıcı ve düzgün Türkçe olmalı.\n"
            "2) Sadece 'Context' içinde AÇIKÇA yazan bilgilere dayan.\n"
            "3) Context birden fazla farklı prosedür / yönerge / süreç içeriyorsa:\n"
            "   - Önce sorudaki ANA TERİMLERİ (belirli program, süreç, birim adları vb.) belirle.\n"
            "   - Sadece bu terimlerle en çok örtüşen parçaları kullan, diğer prosedürleri YOK SAY.\n"
            "4) Context'te yer almayan tarih, ay, yıl, sayı, kota, oran vb. ayrıntıları UYDURMA.\n"
            "   - Soru tarih soruyorsa ve context'te tarih yoksa, bunu açıkça belirtmek SERBESTTİR.\n"
            "5) Emin olmadığın hiçbir bilgiyi tahmin etme; 'bağlamda bu bilgi yok' demek SERBEST.\n"
        )

        if intent == "list_names":
            sys_prompt = base_tr_sys + (
                "6) Bu soruda amaç İSİMLERİ / KATEGORİLERİ LİSTELEMEKTİR.\n"
                "   - Context içinde ilgili başlık altındaki TÜM adları eksiksiz yazmaya çalış.\n"
                "   - Metinde A., B., C. gibi harfli veya madde madde kategoriler varsa, her birinin adını listele.\n"
                "   - Çıkış formatın: her satırda SADECE bir isim olacak şekilde madde madde liste olsun.\n"
                "   - Ek yorum veya açıklama eklemek zorunda değilsin.\n"
            )
        elif intent == "count_items":
            sys_prompt = base_tr_sys + (
                "6) Bu soruda amaç belirli bir şeyin KAÇ TANE olduğunu bulmaktır.\n"
                "   - Sadece context'te NET olarak sayabildiğin öğeleri say.\n"
                "   - Emin değilsen, 'bağlamdan kesin sayı çıkmıyor' de; tahmin etme.\n"
            )
        else:
            sys_prompt = base_tr_sys + (
                "6) Bu soru genel açıklama sorusudur.\n"
                "   - Context'teki bilgiyi özetleyerek açıkla.\n"
                "   - Context birden fazla alt konuyu içeriyorsa, sadece soruyla direkt ilgili olan kısmı kullan.\n"
                "   - Eğer context sorulan spesifik alt konuyu içermiyorsa, bu durumu açıkça söyle.\n"
            )

    elif q_lang == "en":
        base_en_sys = (
            "You are an assistant for an internal knowledge system.\n"
            "RULES:\n"
            "- Use ONLY the provided context.\n"
            "- If the context contains multiple different procedures/policies, first identify the MAIN terms in the question\n"
            "  (e.g., specific program or process names) and focus ONLY on passages that clearly match those terms.\n"
            "- Do NOT invent dates, months, years, quotas or procedures that do not appear in the context.\n"
            "- If the question is more specific than the context, explain that limitation instead of guessing.\n"
        )
        if intent == "list_names":
            sys_prompt = base_en_sys + (
                "- The user wants a LIST OF NAMES or CATEGORIES.\n"
                "- List every relevant item name from the context, one per line, with minimal or no commentary.\n"
            )
        elif intent == "count_items":
            sys_prompt = base_en_sys + (
                "- The user is asking for a COUNT.\n"
                "- Only give a numeric count if you can clearly derive it from the context.\n"
                "- Otherwise say that the count cannot be determined from the context.\n"
            )
        else:
            sys_prompt = base_en_sys + (
                "- Provide a concise explanation based only on the context.\n"
            )
    else:
        sys_prompt = (
            "You are a helpful assistant answering questions based ONLY on the following context. "
            "Do not invent new facts."
        )

    full_prompt = (
        f"Context:\n{context}\n\n"
        f"Question:\n{query}\n\n"
        "Answer based only on the context above."
    )

    return call_ollama_chat(full_prompt, system_prompt=sys_prompt)


# =============== CLI ====================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        choices=["chroma", "chroma-mmr"],
        default="chroma-mmr",
        help="retrieval mode"
    )
    args = parser.parse_args()

    print(f"Loading Chroma collection from: {CHROMA_DIR}")
    coll = get_chroma_collection()

    load_doc_chunk_stats()

    print("Ready.")
    print(f"Mode = {args.mode}")
    print("Type an empty line to exit.\n")

    history: List[Dict] = []

    while True:
        try:
            q = input("Query> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not q:
            break

        try:
            history.append({"role": "user", "content": q})

            ans = answer_with_rag(
                query=q,
                mode=args.mode,
                coll=coll,
                history=history[:-1],
            )
            print("\n=== ANSWER ===")
            print(ans)
            print("==============\n")

            history.append({"role": "assistant", "content": ans})

        except Exception as e:
            print(f"[error] {e}")
