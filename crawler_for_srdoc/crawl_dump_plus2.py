# crawler_for_srdoc/crawl_dump_plus2.py
import os, re, csv, time, pathlib, random
from urllib.parse import urljoin, urlparse, urldefrag
from collections import deque
from bs4 import BeautifulSoup
from playwright.sync_api import sync_playwright, TimeoutError as PWTimeout
import langid
# --------- Config ----------
START_URL       = "https://mysu.sabanciuniv.edu/surecharitasi/tr/surec-tablosu"
ALLOWED_NETLOC  = "mysu.sabanciuniv.edu"
ALLOWED_PREFIX  = "/surecharitasi/"
STATE_FILE      = os.path.join(os.path.dirname(__file__), "session.json")  # your saved cookie state
OUT_DIR         = os.path.join(os.path.dirname(__file__), "mysu_dump_plus2")
CATALOG         = os.path.join(OUT_DIR, "catalog.csv")

REQUEST_DELAY_S = 0.6
MAX_PAGES       = 8000
NAV_TIMEOUT_MS  = 45_000
REQ_TIMEOUT_MS  = 60_000
HEAD_TIMEOUT_MS = 25_000

DOC_EXT = {
    ".pdf", ".doc", ".docx", ".xls", ".xlsx", ".ppt", ".pptx",
    ".odt", ".ods", ".zip", ".rar", ".csv", ".txt", ".rtf", ".htm"
}
SKIP_SUBSTRINGS = ("logout", "login")

# --------- Language detection (multi-backend) ----------
TR_CHARS = "çğıöşüÇĞİÖŞÜ"

langid.set_languages(['en', 'tr'])  # constrain to EN/TR for better precision

def url_lang_hint(url: str):
    p = (url or "").lower()
    if "/surecharitasi/tr/" in p: return "tr"
    if "/surecharitasi/en/" in p: return "en"
    return None

def html_lang_attr(html: str) -> str | None:
    # read <html lang="..."> if present
    try:
        from bs4 import BeautifulSoup
        soup = BeautifulSoup(html, "lxml")
        html_tag = soup.find("html")
        if html_tag and html_tag.has_attr("lang"):
            val = (html_tag["lang"] or "").strip().lower()
            if val.startswith("tr"): return "tr"
            if val.startswith("en"): return "en"
    except Exception:
        pass
    return None

def cheap_heuristic_lang(text: str):
    t = (text or "").lower()
    score = sum(x in t for x in [" ve ", " ile ", " bir ", " için ", " veya ", " üniversite "]) + \
            sum(ch in t for ch in TR_CHARS)
    # return (lang, confidence-ish)
    return ("tr", 0.75) if score >= 2 else ("en", 0.60)

def detect_content_lang(text: str, html_raw: str | None = None):
    """
    Returns (lang, confidence) from strongest available signal:
    1) <html lang=""> if present
    2) langid over text
    3) cheap heuristic
    """
    # 1) HTML lang hint
    if html_raw:
        h = html_lang_attr(html_raw)
        if h:  # treat as high-confidence
            return (h, 0.90)

    # 2) langid (3.12-safe)
    if text and len(text) >= 60:
        try:
            l, prob = langid.classify(text[:8000])
            if l in ("tr", "en"):
                # langid prob is uncalibrated; clip into [0.55,0.99] for our gating
                prob = 0.99 if prob > 0.99 else (0.55 if prob < 0.55 else prob)
                return (l, float(prob))
        except Exception:
            pass

    # 3) fallback heuristic
    return cheap_heuristic_lang(text)

# --------- Utils ----------
def sanitize_path(url_path: str) -> str:
    safe = re.sub(r"[^a-zA-Z0-9/_\-.]", "_", url_path).strip("/")
    return safe or "root"

def is_malformed_url(url: str) -> bool:
    return url.count("http://") + url.count("https://") > 1

def ensure_parent(path: str):
    pathlib.Path(path).parent.mkdir(parents=True, exist_ok=True)

def should_visit(url: str) -> bool:
    u = urlparse(url)
    if u.scheme not in ("http","https"): return False
    if u.netloc != ALLOWED_NETLOC: return False
    if not u.path.startswith(ALLOWED_PREFIX): return False
    low = url.lower()
    if any(x in low for x in SKIP_SUBSTRINGS): return False
    return True

def is_document_by_ext(url: str) -> bool:
    ext = os.path.splitext(urlparse(url).path)[1].lower()
    return ext in DOC_EXT

def normalize(url: str) -> str:
    url, _ = urldefrag(url)
    return url

def hash_bytes(b: bytes) -> str:
    import hashlib
    return hashlib.sha256(b).hexdigest()

def filename_from_headers(url: str, headers: dict) -> str:
    cd = headers.get("content-disposition", "") if headers else ""
    m = re.search(r'filename\*?=(?:UTF-8\'\')?"?([^\";]+)"?', cd, flags=re.I)
    if m:
        name = m.group(1)
    else:
        name = os.path.basename(urlparse(url).path) or "download.bin"
    name = re.sub(r"[^a-zA-Z0-9._\- ]", "_", name)
    name = re.sub(r'(\.[A-Za-z0-9]{2,5})\1+$', r'\1', name)
    return name

def shorten_windows_path(path: str, digest: str) -> str:
    if os.name == "nt" and len(path) > 230:
        base_dir = os.path.dirname(path)
        fname = os.path.basename(path)
        stem, ext = os.path.splitext(fname)
        stem = stem[:120]
        fname = f"{stem}_{digest[:8]}{ext}"
        path = os.path.join(base_dir, fname)
    return path

def backoff(attempt: int, base: float = 0.6, cap: float = 5.0) -> float:
    return min(cap, base * (2 ** attempt)) * (0.5 + random.random())

def safe_write_bytes(local_path: str, content: bytes, digest: str) -> str:
    try:
        ensure_parent(local_path)
        with open(local_path, "wb") as f:
            f.write(content)
        return local_path
    except Exception:
        fallback = os.path.join(OUT_DIR, "files", digest[:2], digest[2:4], f"{digest}.bin")
        ensure_parent(fallback)
        with open(fallback, "wb") as f:
            f.write(content)
        return fallback

def safe_write_text(local_path: str, text: str) -> str:
    try:
        ensure_parent(local_path)
        with open(local_path, "w", encoding="utf-8") as f:
            f.write(text)
        return local_path
    except Exception:
        digest = hash_bytes(text.encode("utf-8"))
        fallback = os.path.join(OUT_DIR, "html", digest[:2], digest[2:4], f"{digest}.html")
        ensure_parent(fallback)
        with open(fallback, "w", encoding="utf-8") as f:
            f.write(text)
        return fallback

def quick_pdf_text_for_lang_bytes(blob: bytes) -> str:
    # Very light sniff: first page with pdfminer if available.
    try:
        from pdfminer.high_level import extract_text
        tmp = os.path.join(OUT_DIR, "__tmp_lang_sniff.pdf")
        ensure_parent(tmp)
        with open(tmp, "wb") as f: f.write(blob)
        text = extract_text(tmp, maxpages=1) or ""
        try: os.remove(tmp)
        except: pass
        return text
    except Exception:
        return ""

# --------- HTTP helpers via Playwright context.request ----------
def fetch_with_retries(api, method: str, url: str, timeout_ms: int, max_attempts: int = 3):
    last = None
    for i in range(max_attempts):
        try:
            if method == "GET":  return api.get(url, timeout=timeout_ms)
            if method == "HEAD": return api.head(url, timeout=timeout_ms)
            raise ValueError("Bad HTTP method")
        except Exception as e:
            last = e
            time.sleep(backoff(i))
    raise last

# --------- Crawl ----------
def crawl():
    pathlib.Path(OUT_DIR).mkdir(parents=True, exist_ok=True)
    visited = set()
    q = deque([START_URL])

    header = ["url","local_path","status","content_type","bytes","is_document",
              "url_lang","content_lang","content_lang_conf","lang","lang_mismatch","title"]
    with open(CATALOG, "w", newline="", encoding="utf-8") as fcat, sync_playwright() as p:
        writer = csv.DictWriter(fcat, fieldnames=header)
        writer.writeheader()

        browser = p.chromium.launch(headless=True)
        context = browser.new_context(storage_state=STATE_FILE)
        page = context.new_page()
        api  = context.request

        pages = 0
        while q and pages < MAX_PAGES:
            url = normalize(q.popleft())
            if url in visited or not should_visit(url) or is_malformed_url(url):
                continue
            visited.add(url)

            try:
                # Docs by ext → GET bytes
                if is_document_by_ext(url):
                    r = fetch_with_retries(api, "GET", url, REQ_TIMEOUT_MS)
                    content = r.body()
                    digest  = hash_bytes(content)
                    rel_dir = sanitize_path(os.path.dirname(urlparse(url).path))
                    fname   = filename_from_headers(url, r.headers or {})
                    local_path = os.path.join(OUT_DIR, rel_dir, fname)
                    local_path = shorten_windows_path(local_path, digest)
                    final_path = safe_write_bytes(local_path, content, digest)

                    sniff = quick_pdf_text_for_lang_bytes(content) if (fname.lower().endswith(".pdf")) else ""
                    c_lang, c_conf = detect_content_lang(sniff)
                    u_hint = url_lang_hint(url) or "en"
                    final_lang = c_lang if (len(sniff) >= 200 and c_conf >= 0.55) else u_hint
                    mismatch = (u_hint != c_lang)

                    writer.writerow({
                        "url": url, "local_path": final_path, "status": r.status,
                        "content_type": (r.headers or {}).get("content-type",""),
                        "bytes": len(content), "is_document": True,
                        "url_lang": u_hint, "content_lang": c_lang,
                        "content_lang_conf": f"{c_conf:.2f}", "lang": final_lang,
                        "lang_mismatch": str(bool(mismatch)), "title": ""
                    })
                    pages += 1
                    time.sleep(REQUEST_DELAY_S)
                    continue

                # HEAD probe for HTML/attachments
                pre = None
                try:
                    pre = fetch_with_retries(api, "HEAD", url, HEAD_TIMEOUT_MS)
                except Exception:
                    pre = None

                if pre:
                    pre_ct = (pre.headers or {}).get("content-type","")
                    pre_cd = (pre.headers or {}).get("content-disposition","")
                    if ("attachment" in pre_cd.lower()) or ("text/html" not in pre_ct.lower()):
                        r = fetch_with_retries(api, "GET", url, REQ_TIMEOUT_MS)
                        content = r.body()
                        digest  = hash_bytes(content)
                        rel_dir = sanitize_path(os.path.dirname(urlparse(url).path))
                        fname   = filename_from_headers(url, r.headers or {})
                        local_path = os.path.join(OUT_DIR, rel_dir, fname)
                        local_path = shorten_windows_path(local_path, digest)
                        final_path = safe_write_bytes(local_path, content, digest)

                        sniff = quick_pdf_text_for_lang_bytes(content) if (fname.lower().endswith(".pdf")) else ""
                        c_lang, c_conf = detect_content_lang(sniff)
                        u_hint = url_lang_hint(url) or "en"
                        final_lang = c_lang if (len(sniff) >= 200 and c_conf >= 0.55) else u_hint
                        mismatch = (u_hint != c_lang)

                        writer.writerow({
                            "url": url, "local_path": final_path, "status": r.status,
                            "content_type": (r.headers or {}).get("content-type",""),
                            "bytes": len(content), "is_document": True,
                            "url_lang": u_hint, "content_lang": c_lang,
                            "content_lang_conf": f"{c_conf:.2f}", "lang": final_lang,
                            "lang_mismatch": str(bool(mismatch)), "title": ""
                        })
                        pages += 1
                        time.sleep(REQUEST_DELAY_S)
                        continue

                # Navigate HTML
                resp = page.goto(url, wait_until="domcontentloaded", timeout=NAV_TIMEOUT_MS)
                status = resp.status if resp else None
                ct     = (resp.headers.get("content-type") if resp else "") or "text/html"
                final_url = resp.url if resp else page.url
                html = page.content()

                soup = BeautifulSoup(html, "lxml")
                for tag in soup(["script","style","noscript"]):
                    tag.decompose()
                main_text = soup.get_text(" ", strip=True)
                title_tag = soup.find("title")
                title_txt = title_tag.get_text(strip=True) if title_tag else ""

                u_hint = url_lang_hint(final_url) or "en"
                c_lang, c_conf = detect_content_lang(main_text, html_raw=html)
                final_lang = c_lang if (len(main_text) >= 400 and c_conf >= 0.55) else u_hint
                mismatch = (u_hint != c_lang)

                rel_path = sanitize_path(urlparse(final_url).path)
                if rel_path.endswith("/"): rel_path += "index.html"
                if not os.path.splitext(rel_path)[1]: rel_path += ".html"
                if rel_path.lower().endswith(".htm"): rel_path += "l"
                local_path = os.path.join(OUT_DIR, rel_path)
                local_path = shorten_windows_path(local_path, "0"*64)
                final_path = safe_write_text(local_path, html)

                writer.writerow({
                    "url": final_url, "local_path": final_path, "status": status,
                    "content_type": ct, "bytes": len(html.encode("utf-8")), "is_document": False,
                    "url_lang": u_hint, "content_lang": c_lang,
                    "content_lang_conf": f"{c_conf:.2f}", "lang": final_lang,
                    "lang_mismatch": str(bool(mismatch)), "title": title_txt
                })

                # discover links
                for a in soup.select("a[href]"):
                    href = (a.get("href") or "").strip()
                    if not href: continue
                    nxt = urljoin(final_url, href)
                    if should_visit(nxt) and not is_malformed_url(nxt):
                        q.append(nxt)

                pages += 1
                time.sleep(REQUEST_DELAY_S)

            except PWTimeout:
                print(f"[timeout] {url}")
            except Exception as e:
                print(f"[error] {url}: {e}")

        browser.close()
    print(f"Done. Crawled {pages} pages → {OUT_DIR}\nCatalog: {CATALOG}")

# -------------------------
# Main
# -------------------------
if __name__ == "__main__":
    crawl()

