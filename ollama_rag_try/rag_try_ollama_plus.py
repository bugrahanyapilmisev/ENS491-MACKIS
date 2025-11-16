# rag_try_ollama_plus.py (generic, block-based enumeration extraction)
# RAG: Chroma + BM25 + HyDE + MMR + (optional) cross-encoder rerank
# Block detection for lists, generic noise filtering, multilingual de-dup, language-aware output.

import os, sys, json, pickle, re, argparse
import numpy as np
import requests, chromadb
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

TR_CHARS = "çğıöşüÇĞİÖŞÜ"

# -------------------- utils --------------------
def looks_turkish(s: str) -> bool:
    return any(ch in s for ch in TR_CHARS)

def log(msg: str): print(f"[RAG] {msg}", flush=True)

# -------------------- CLI --------------------
def parse_args():
    p = argparse.ArgumentParser(description="RAG (Chroma + BM25 + MMR + reranker + HyDE)")
    p.add_argument("--ollama-host", default=os.getenv("OLLAMA_HOST", "http://localhost:11434"))
    p.add_argument("--embed-model", default="bge-m3")
    p.add_argument("--embed-dim", type=int, default=1024)
    p.add_argument("--chroma-dir", default=r"C:\\Users\\kosot\\Documents\\bitirme\\preprocessing\\version_plus\\chroma_db_clean_plus")
    p.add_argument("--collection", default="mysu_surecharitasi_bge")
    p.add_argument("--bm25-pickle", default=r"C:\\Users\\kosot\\Documents\\bitirme\\preprocessing\\version_plus\\bm25_index.pkl")
    p.add_argument("--gen-model", default="qwen2.5:7b")
    p.add_argument("--reranker", default="BAAI/bge-reranker-base")
    p.add_argument("--no-rerank", action="store_true")
    p.add_argument("--no-hyde", action="store_true")
    p.add_argument("--topk-vector", type=int, default=40)
    p.add_argument("--topk-bm25", type=int, default=20)
    p.add_argument("--mmr-keep", type=int, default=12)
    p.add_argument("--rerank-keep", type=int, default=12)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--neighbor-span", type=int, default=8)
    p.add_argument("--max-snip", type=int, default=2000)
    p.add_argument("--dedupe-sim", type=float, default=0.82)
    return p.parse_args()

# -------------------- ollama --------------------
def ollama_embed(host, model, dim, text, timeout=180):
    r = requests.post(f"{host}/api/embeddings", json={"model": model, "prompt": text}, timeout=timeout)
    r.raise_for_status()
    v = np.array(r.json()["embedding"], dtype=np.float32)
    v /= (np.linalg.norm(v) + 1e-12)
    assert v.shape[0] == dim, f"embed dim mismatch {v.shape[0]} != {dim}"
    return v

def ollama_generate(host, model, prompt, temp=0.2, timeout=240):
    with requests.post(f"{host}/api/generate",
        json={"model": model, "prompt": prompt, "options": {"temperature": temp}},
        timeout=timeout, stream=True
    ) as r:
        if r.status_code == 404:
            raise RuntimeError(f"Ollama model '{model}' not found. Run: ollama pull {model}\n{r.text}")
        r.raise_for_status()
        out = []
        for line in r.iter_lines(decode_unicode=True):
            if not line: continue
            try:
                obj = json.loads(line)
                chunk = obj.get("response", "")
                if chunk: out.append(chunk)
            except json.JSONDecodeError:
                continue
        return "".join(out)

def tiny_translate_if_needed(host, gen_model, title, prefer_tr):
    if not title: return title
    is_tr = looks_turkish(title)
    if prefer_tr and is_tr: return title
    if (not prefer_tr) and (not is_tr): return title
    tgt = "Turkish" if prefer_tr else "English"
    prompt = f"Translate this item title into {tgt}. Keep it short and official:\n{title}\nAnswer:"
    try: return ollama_generate(host, gen_model, prompt, temp=0.0).splitlines()[0].strip(" -–—")
    except Exception: return title

# -------------------- retrieval pieces --------------------
def load_bm25(path):
    if not os.path.exists(path): raise FileNotFoundError(f"BM25 pickle not found at: {path}")
    with open(path, "rb") as f: blob = pickle.load(f)
    return blob["bm25"], blob["ids"], blob["docs"]

def mmr_select(emb_q, emb_docs, k=10, lambda_mult=0.6):
    selected, cand = [], list(range(len(emb_docs)))
    sim = lambda a,b: float(np.dot(a,b))
    while cand and len(selected) < k:
        if not selected:
            idx = max(cand, key=lambda i: sim(emb_q, emb_docs[i]))
        else:
            def score(i):
                rel = sim(emb_q, emb_docs[i])
                div = max(sim(emb_docs[i], emb_docs[j]) for j in selected)
                return lambda_mult*rel - (1-lambda_mult)*div
            idx = max(cand, key=score)
        selected.append(idx); cand.remove(idx)
    return selected

def load_reranker(name):
    tok = AutoTokenizer.from_pretrained(name)
    model = AutoModelForSequenceClassification.from_pretrained(name)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device).eval()
    return tok, model, device

def rerank(query, passages, tok, model, device, topk=8):
    inputs = tok([query]*len(passages), passages, padding=True, truncation=True,
                 max_length=512, return_tensors="pt").to(device)
    with torch.no_grad():
        scores = model(**inputs).logits.view(-1).detach().cpu().numpy()
    order = np.argsort(-scores)[:topk]
    return order, scores

def multi_query(host, gen_model, base_q, use_hyde=True):
    if not use_hyde: return [base_q]
    p_prompt = f"Paraphrase the question 2 ways (same language). Return as lines only:\n{base_q}\n"
    try:
        paras = [l.strip("- ").strip() for l in ollama_generate(host, gen_model, p_prompt).splitlines() if l.strip()]
        paras = paras[:2] if paras else [base_q]
    except Exception: paras = [base_q]
    h_prompt = f"Write a short hypothetical answer (2–3 sentences, same language):\nQuestion: {base_q}\nAnswer:"
    try: hyde = ollama_generate(host, gen_model, h_prompt).strip()
    except Exception: hyde = ""
    return [base_q] + paras + ([hyde] if hyde else [])

# -------------------- context --------------------
def build_context(cands, keep, max_snip=2000):
    seen, merged = set(), []
    for c in cands:
        if c["id"] in seen: continue
        seen.add(c["id"]); merged.append(c)
        if len(merged) >= keep: break
    ctx, cleaned = "", []
    for i, c in enumerate(merged, 1):
        m = c.get("metadata", {}) or {}
        sect = " > ".join([x for x in [m.get("h1"), m.get("h2"), m.get("h3")] if x]) or (m.get("doctype") or "Document")
        path = m.get("doc_path", ""); title = m.get("title", "")
        snippet = (c["document"] or "")[:max_snip]
        ctx += f"[{i}] TITLE: {title}\nSECTION: {sect}\nPATH: {path}\nTEXT:\n{snippet}\n\n"
        cleaned.append({"id": c["id"], "metadata": {**m, "SECTION_PRINT": sect}})
    return ctx, cleaned

def answer_prompt(query, context):
    return f"""You are a precise policy/document assistant.

Use ONLY the facts in the Context. If the question asks for categories/types/lists, enumerate all items with brief 1–2 line descriptions. Do not add outside knowledge.

Question:
{query}

Context:
{context}

Output format:
- Bullet list of items using the exact official names found in the context, all in the question's language.
- After each bullet, cite (PATH: <full path>, SECTION: <section>).
"""

# -------------------- generic list extraction (block-based) --------------------
ENUM_PATTERNS = [
    r"^\s*[A-Z]\.\s+",      # A. B. C.
    r"^\s*\d+\.\s+",        # 1. 2. 3.
    r"^\s*[ivxIVX]+\.\s+",  # i. ii. iii.
    r"^\s*[-•]\s+",         # - item / • item
]
_ENUM_RE = [re.compile(p) for p in ENUM_PATTERNS]

ACRONYM_LINE_RE = re.compile(r"^\s*[A-Z]{2,}\s*(\([^)]*\))?\s*[:\-–—]?\s*$")

def is_title_like(s: str) -> bool:
    if not s: return False
    if len(s) > 120: return False
    if s.strip().endswith(('.', '?', '!')): return False
    # heuristic: many TitleCase / UPPER words
    words = [w for w in re.findall(r"[A-Za-zÇĞİÖŞÜçğıöşü0-9()&’'/-]+", s)]
    if not words: return False
    upperish = sum(1 for w in words if (w.isupper() or (len(w)>1 and w[0].isupper())))
    return (upperish / max(1,len(words))) >= 0.6

def looks_code_like(s: str) -> bool:
    # Generic code-ish lines (course codes, short acronyms with digits)
    if re.search(r"[A-Z]{2,}\s?\d{2,3}", s): return True
    if len(s.split()) <= 2 and re.match(r"^[A-Z/& -]{2,}$", s): return True
    return False

def looks_boilerplate_heading(s: str) -> bool:
    # Generic plumbing: Related/Relevant/Owner/Forms/Procedures… (language-agnostic-ish)
    return bool(re.search(r"\b(Related|Relevant|Owner|Owners|Forms?|Procedures?|Outputs?|Timing|Unit|Director)\b", s, flags=re.I))

def normalize_title(s: str) -> str:
    s = s.strip()
    s = re.sub(r"^\s*(?:[A-Z]\.|[ivxIVX]+\.|-|\d+\.|•)\s+", "", s)  # strip bullet marker
    s = s.strip("–—-:;·• ")
    return s

def extract_blocks_from_hit(hit, max_items_per_block=40):
    """Return blocks: list of dicts {path, items:[(title,line_idx)], block_key} within this hit."""
    text = (hit.get("document") or "")
    if not text.strip(): return []
    path = (hit.get("metadata") or {}).get("doc_path", "")
    lines = text.splitlines()

    def is_item_line(raw):
        s = raw.strip()
        if not s: return False
        if any(r.match(s) for r in _ENUM_RE): return True
        if ACRONYM_LINE_RE.match(s): return True
        if is_title_like(s): return True
        return False

    blocks = []
    cur = []
    for idx, raw in enumerate(lines):
        s = raw.strip()
        if is_item_line(s):
            t = normalize_title(s)
            if not t: 
                continue
            # generic noise filters (shape-based)
            if looks_boilerplate_heading(t): 
                continue
            if looks_code_like(t):
                continue
            if len(t) < 3:
                continue
            cur.append((t, idx))
            if len(cur) >= max_items_per_block:
                blocks.append({"path": path, "items": cur[:]})
                cur = []
        else:
            if cur:
                blocks.append({"path": path, "items": cur[:]})
                cur = []
    if cur:
        blocks.append({"path": path, "items": cur[:]})
    # keep only multi-item blocks
    return [b for b in blocks if len(b["items"]) >= 2]

def expand_neighbors(coll, hit, span=8, limit=16):
    m = hit.get("metadata") or {}
    path = m.get("doc_path"); idx = m.get("chunk_index")
    if path is None or idx is None: return []
    try:
        want = list(range(max(0, idx-span), idx+span+1))
        res = coll.get(where={"doc_path": path, "chunk_index": {"$in": want}},
                       include=["documents","metadatas","embeddings","ids"])
        neigh=[]
        for i in range(len(res.get("ids",[]))):
            neigh.append({
                "id": res["ids"][i],
                "document": res["documents"][i],
                "metadata": res["metadatas"][i],
                "emb": np.array(res["embeddings"][i], dtype=np.float32) if res.get("embeddings") else None
            })
        uniq, seen=[], set()
        for n in neigh:
            if n["id"] not in seen:
                seen.add(n["id"]); uniq.append(n)
            if len(uniq)>=limit: break
        return uniq
    except Exception:
        return []

# -------------------- block selection & de-dup --------------------
def _cos(a,b): return float(np.dot(a,b))

def pick_best_block(blocks, embed_host, embed_model, embed_dim):
    """Pick the longest block; tie-break by mean pairwise cosine similarity."""
    if not blocks: return None
    blocks = sorted(blocks, key=lambda b: len(b["items"]), reverse=True)
    best = [blocks[0]]
    # collect ties by size
    for b in blocks[1:]:
        if len(b["items"]) == len(best[0]["items"]):
            best.append(b)
        else:
            break
    if len(best) == 1: 
        return best[0]
    # tie-break by coherence
    best_score, best_block = -1.0, None
    for b in best:
        titles = [t for t,_ in b["items"]]
        vecs = [ollama_embed(embed_host, embed_model, embed_dim, t) for t in titles[:20]]  # cap for speed
        if len(vecs) < 2:
            score = 0.0
        else:
            s = 0.0; c = 0
            for i in range(len(vecs)):
                for j in range(i+1, len(vecs)):
                    s += _cos(vecs[i], vecs[j]); c += 1
            score = s / max(1,c)
        if score > best_score:
            best_score, best_block = score, b
    return best_block

def group_semantic(items, embed_host, embed_model, embed_dim, sim_thresh=0.82):
    if not items: return []
    vecs = [ollama_embed(embed_host, embed_model, embed_dim, it["title"]) for it in items]
    used = [False]*len(items); groups=[]
    for i in range(len(items)):
        if used[i]: continue
        g={"lead":i,"members":[i]}; used[i]=True
        for j in range(i+1,len(items)):
            if used[j]: continue
            if _cos(vecs[i], vecs[j]) >= sim_thresh:
                g["members"].append(j); used[j]=True
        groups.append(g)
    return groups

def choose_title(group, items, prefer_tr, host, gen_model):
    def pick(pred):
        for idx in group["members"]:
            if pred(items[idx]["title"]):
                return items[idx]["title"], items[idx].get("path")
        return None, None
    if prefer_tr:
        t,p = pick(lambda t: looks_turkish(t))
        if t: return t,p
        t0 = items[group["lead"]]["title"]; p0 = items[group["lead"]].get("path")
        return tiny_translate_if_needed(host, gen_model, t0, True), p0
    else:
        t,p = pick(lambda t: not looks_turkish(t))
        if t: return t,p
        t0 = items[group["lead"]]["title"]; p0 = items[group["lead"]].get("path")
        return tiny_translate_if_needed(host, gen_model, t0, False), p0

def dedupe_and_localize_block(block, embed_host, embed_model, embed_dim, prefer_tr, gen_host, gen_model, sim_thresh=0.82):
    items = [{"title": t, "path": block["path"]} for t,_ in block["items"]]
    groups = group_semantic(items, embed_host, embed_model, embed_dim, sim_thresh)
    merged=[]
    for g in groups:
        t,p = choose_title(g, items, prefer_tr, gen_host, gen_model)
        t = normalize_title(t)
        if t and len(t) >= 3:
            merged.append({"title": t, "path": p})
    return merged

# -------------------- main --------------------
def main():
    args = parse_args()
    log("starting"); log(f"Chroma dir: {args.chroma_dir}"); log(f"Collection : {args.collection}")
    log(f"BM25 index : {args.bm25_pickle}")

    client = chromadb.PersistentClient(path=args.chroma_dir)
    coll = client.get_or_create_collection(name=args.collection, metadata={"hnsw:space":"cosine"})
    bm25, bm_ids, bm_docs = load_bm25(args.bm25_pickle)

    tok = model = device = None
    use_rerank = not args.no_rerank
    if use_rerank:
        try:
            log(f"loading reranker: {args.reranker} (GPU={'yes' if torch.cuda.is_available() else 'no'}) …")
            tok, model, device = load_reranker(args.reranker)
        except Exception as e:
            log(f"reranker unavailable, continuing without it: {e}")
            use_rerank = False

    query = input("Soru / Question: ").strip()
    if not query: log("empty query; exiting."); return
    prefer_tr = looks_turkish(query)

    log("building multi-query …")
    mq = multi_query(args.ollama_host, args.gen_model,
                     ("(Türkçe yanıtla) " + query) if prefer_tr else query,
                     use_hyde=not args.no_hyde)

    log("vector search …")
    all_vec=[]
    for q in mq:
        qvec = ollama_embed(args.ollama_host, args.embed_model, args.embed_dim, q)
        res = coll.query(query_embeddings=[qvec.tolist()], n_results=args.topk_vector,
                         include=["documents","metadatas","embeddings"])
        if not res["ids"] or not res["ids"][0]: continue
        ids=res["ids"][0]; docs=res["documents"][0]; metas=res["metadatas"][0]
        vembs=[np.array(e,dtype=np.float32) for e in res["embeddings"][0]]
        for i in range(len(ids)):
            all_vec.append({"id":ids[i], "document":docs[i], "metadata":metas[i], "emb":vembs[i]})
    if not all_vec: log("no vector hits from Chroma"); return

    log("BM25 search …")
    simple_tok = lambda s: [t.lower() for t in s.split()]
    bm_scores = bm25.get_scores(simple_tok(query))
    top_bm_idx = np.argsort(-bm_scores)[:args.topk_bm25]
    bm_hits = [{"id": bm_ids[i], "document": bm_docs[i], "metadata": {}, "emb": None} for i in top_bm_idx]

    # merge
    union = {h["id"]:h for h in (all_vec + bm_hits)}
    hits = list(union.values())

    # MMR
    log("MMR selection …")
    emb_q = ollama_embed(args.ollama_host, args.embed_model, args.embed_dim, query)
    emb_docs=[]
    for h in hits:
        if h["emb"] is None:
            h["emb"] = ollama_embed(args.ollama_host, args.embed_model, args.embed_dim, h["document"][:800])
        emb_docs.append(h["emb"])
    mmr_idx = mmr_select(emb_q, emb_docs, k=args.mmr_keep, lambda_mult=0.6)
    mmr_hits = [hits[i] for i in mmr_idx]

    # neighbors
    expanded=[]
    for h in mmr_hits:
        expanded.append(h)
        expanded.extend(expand_neighbors(coll, h, span=args.neighbor_span, limit=16))

    # de-dup by id
    seen, mmr_hits = set(), []
    for h in expanded:
        if h["id"] in seen: continue
        seen.add(h["id"]); mmr_hits.append(h)

    # optional rerank
    if use_rerank:
        log("reranking …")
        order,_ = rerank(query, [h["document"] for h in mmr_hits], tok, model, device, topk=args.rerank_keep)
        final_hits = [mmr_hits[i] for i in order]
    else:
        final_hits = mmr_hits[:args.rerank_keep]

    # ---- block-based extraction ----
    all_blocks=[]
    for h in final_hits:
        all_blocks.extend(extract_blocks_from_hit(h))
    # if nothing, try once more: larger neighbor window (fallback)
    if not all_blocks:
        for h in final_hits:
            all_blocks.extend(extract_blocks_from_hit(h))
    best = pick_best_block(all_blocks, args.ollama_host, args.embed_model, args.embed_dim)

    if best:
        merged = dedupe_and_localize_block(
            best,
            embed_host=args.ollama_host, embed_model=args.embed_model, embed_dim=args.embed_dim,
            prefer_tr=prefer_tr, gen_host=args.ollama_host, gen_model=args.gen_model,
            sim_thresh=args.dedupe_sim
        )
        # keep order as in block; remove exact dup titles
        seen_titles=set(); cleaned=[]
        for it in merged:
            t = it["title"]
            if t.lower() in seen_titles: continue
            seen_titles.add(t.lower())
            cleaned.append(it)
        if len(cleaned) >= 3:
            print("\n=== Answer ===\n")
            for it in cleaned:
                p = it.get("path") or "(no-path)"
                print(f"- {it['title']} (PATH: {p})")
            print("\n=== Sources ===")
            print(f"[1] PATH={best['path']}")
            print()
            return

    # ---- LLM fallback (generic) ----
    log("building context + generating …")
    ctx, used = build_context(final_hits, keep=args.rerank_keep, max_snip=args.max_snip)
    if not ctx.strip():
        print("\n=== Answer ===\nNot found in provided documents.\n"); return
    prompt = answer_prompt(query, ctx)
    out = ollama_generate(args.ollama_host, args.gen_model, prompt, temp=args.temperature)

    print("\n=== Answer ===\n"); print(out)
    print("\n=== Sources ===")
    for i,h in enumerate(used,1):
        m=h.get("metadata",{}) or {}
        path=m.get("doc_path") or "(no-path)"
        sect=m.get("SECTION_PRINT") or "-"
        print(f"[{i}] PATH={path}, SECTION={sect}")
        if m.get("title"): print(f"    TITLE={m['title']}")
    print()

if __name__ == "__main__":
    try: main()
    except KeyboardInterrupt: pass
