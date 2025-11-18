# crawl_mysu_surecharitasi.py
import os, re, csv, time, pathlib, random, errno
from urllib.parse import urljoin, urlparse, urldefrag
from collections import deque
from bs4 import BeautifulSoup
from playwright.sync_api import sync_playwright, TimeoutError as PWTimeout

# -------------------------
# Config
# -------------------------
START_URL       = "https://mysu.sabanciuniv.edu/surecharitasi/tr/surec-tablosu"
ALLOWED_NETLOC  = "mysu.sabanciuniv.edu"
ALLOWED_PREFIX  = "/surecharitasi/"
STATE_FILE = r"C:\\Users\\kosot\\Documents\\bitirme\\crawler_for_srdoc\\session.json"  # your saved storage_state
OUT_DIR         = "mysu_dump2"
CATALOG         = os.path.join(OUT_DIR, "catalog.csv")

REQUEST_DELAY_S = 0.6
MAX_PAGES       = 5000
NAV_TIMEOUT_MS  = 45_000
REQ_TIMEOUT_MS  = 60_000
HEAD_TIMEOUT_MS = 30_000

DOC_EXT = {
    ".pdf", ".doc", ".docx", ".xls", ".xlsx", ".ppt", ".pptx",
    ".odt", ".ods", ".zip", ".rar", ".csv", ".txt", ".rtf", ".htm"
}
SKIP_SUBSTRINGS = ("logout", "login")

# -------------------------
# Utils
# -------------------------
def sanitize_path(url_path: str) -> str:
    # Keep / as separators, remove anything unsafe for filesystem
    safe = re.sub(r"[^a-zA-Z0-9/_\-.]", "_", url_path).strip("/")
    return safe or "root"

def is_malformed_url(url: str) -> bool:
    # skip hrefs like 'http://...http://...'
    return url.count("http://") + url.count("https://") > 1

def ensure_parent_dir(path: str):
    pathlib.Path(path).parent.mkdir(parents=True, exist_ok=True)

def should_visit(url: str) -> bool:
    u = urlparse(url)
    if u.scheme not in ("http", "https"):
        return False
    if u.netloc != ALLOWED_NETLOC:
        return False
    if not u.path.startswith(ALLOWED_PREFIX):
        return False
    low = url.lower()
    if any(x in low for x in SKIP_SUBSTRINGS):
        return False
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
    # sanitize filename
    name = re.sub(r"[^a-zA-Z0-9._\- ]", "_", name)
    # normalize accidental double extensions like '.pdf.docx.pdf'
    name = re.sub(r'(\.[A-Za-z0-9]{2,5})\1+$', r'\1', name)
    return name

def shorten_windows_path(path: str, digest: str) -> str:
    # Avoid old Windows MAX_PATH issues (~260). Trim the *filename* part.
    if os.name == "nt" and len(path) > 230:
        base_dir = os.path.dirname(path)
        fname = os.path.basename(path)
        stem, ext = os.path.splitext(fname)
        stem = stem[:120]  # keep something readable
        fname = f"{stem}_{digest[:8]}{ext}"
        path = os.path.join(base_dir, fname)
    return path

def backoff(attempt: int, base: float = 0.6, cap: float = 5.0) -> float:
    # exponential backoff with jitter
    return min(cap, base * (2 ** attempt)) * (0.5 + random.random())

def safe_write_bytes(local_path: str, content: bytes, digest: str) -> str:
    """
    Try to write to local_path. If parent dirs are missing or path is problematic,
    fall back to a short hashed path under OUT_DIR/files/.
    """
    try:
        ensure_parent_dir(local_path)
        with open(local_path, "wb") as f:
            f.write(content)
        return local_path
    except OSError as e:
        # Fallback: bucket by hash
        fallback_dir = os.path.join(OUT_DIR, "files", digest[:2], digest[2:4])
        pathlib.Path(fallback_dir).mkdir(parents=True, exist_ok=True)
        # keep the original extension if any
        _, ext = os.path.splitext(local_path)
        fallback_path = os.path.join(fallback_dir, f"{digest}{ext or '.bin'}")
        with open(fallback_path, "wb") as f:
            f.write(content)
        print(f"[warn] fallback path used for write: {fallback_path} (reason: {e})")
        return fallback_path

def safe_write_text(local_path: str, text: str, digest_hint: str = "") -> str:
    """
    Same idea as safe_write_bytes but for text/HTML.
    """
    try:
        ensure_parent_dir(local_path)
        with open(local_path, "w", encoding="utf-8") as f:
            f.write(text)
        return local_path
    except OSError as e:
        dh = digest_hint or hash_bytes(text.encode("utf-8"))
        fallback_dir = os.path.join(OUT_DIR, "html", dh[:2], dh[2:4])
        pathlib.Path(fallback_dir).mkdir(parents=True, exist_ok=True)
        fallback_path = os.path.join(fallback_dir, f"{dh}.html")
        with open(fallback_path, "w", encoding="utf-8") as f:
            f.write(text)
        print(f"[warn] fallback path used for write (HTML): {fallback_path} (reason: {e})")
        return fallback_path

# -------------------------
# Core download helpers
# -------------------------
def save_binary(url: str, r, out_dir: str, writer):
    status  = r.status
    headers = r.headers or {}
    ctype   = headers.get("content-type", "")
    content = r.body()
    digest  = hash_bytes(content)

    rel_dir = sanitize_path(os.path.dirname(urlparse(url).path))
    fname   = filename_from_headers(url, headers)

    # Build and possibly shorten
    local_path = os.path.join(out_dir, rel_dir, fname)
    local_path = shorten_windows_path(local_path, digest)

    # Safe write with fallback
    final_path = safe_write_bytes(local_path, content, digest)

    writer.writerow({
        "url": url,
        "local_path": final_path,
        "status": status,
        "content_type": ctype,
        "sha256": digest,
        "bytes": len(content),
        "is_document": True
    })
    return final_path, len(content)

def fetch_with_retries(api, method: str, url: str, timeout_ms: int, max_attempts: int = 3):
    last_err = None
    for attempt in range(max_attempts):
        try:
            if method == "GET":
                return api.get(url, timeout=timeout_ms)
            elif method == "HEAD":
                return api.head(url, timeout=timeout_ms)
            else:
                raise ValueError("Unsupported method")
        except Exception as e:
            last_err = e
            time.sleep(backoff(attempt))
    raise last_err

# -------------------------
# Crawl
# -------------------------
def crawl():
    pathlib.Path(OUT_DIR).mkdir(parents=True, exist_ok=True)
    visited = set()
    q = deque([START_URL])

    with open(CATALOG, "w", newline="", encoding="utf-8") as fcat, sync_playwright() as p:
        writer = csv.DictWriter(
            fcat,
            fieldnames=["url", "local_path", "status", "content_type", "sha256", "bytes", "is_document"]
        )
        writer.writeheader()

        browser = p.chromium.launch(headless=True)
        context = browser.new_context(storage_state=STATE_FILE)
        page = context.new_page()
        api  = context.request

        pages_crawled = 0

        while q and pages_crawled < MAX_PAGES:
            url = normalize(q.popleft())
            if url in visited or not should_visit(url) or is_malformed_url(url):
                continue
            visited.add(url)

            try:
                # 1) By extension → direct GET
                if is_document_by_ext(url):
                    r = fetch_with_retries(api, "GET", url, REQ_TIMEOUT_MS)
                    save_binary(url, r, OUT_DIR, writer)
                    pages_crawled += 1
                    time.sleep(REQUEST_DELAY_S)
                    continue

                # 2) No extension (or unknown) → HEAD preflight
                pre = None
                try:
                    pre = fetch_with_retries(api, "HEAD", url, HEAD_TIMEOUT_MS)
                except Exception:
                    pre = None  # Some servers disallow HEAD

                if pre:
                    pre_ct = (pre.headers or {}).get("content-type", "")
                    pre_cd = (pre.headers or {}).get("content-disposition", "")
                    if ("attachment" in pre_cd.lower()) or ("text/html" not in pre_ct.lower()):
                        r = fetch_with_retries(api, "GET", url, REQ_TIMEOUT_MS)
                        save_binary(url, r, OUT_DIR, writer)
                        pages_crawled += 1
                        time.sleep(REQUEST_DELAY_S)
                        continue

                # 3) It’s HTML → navigate, save, discover
                resp = page.goto(url, wait_until="domcontentloaded", timeout=NAV_TIMEOUT_MS)
                status = resp.status if resp else None
                ctype  = (resp.headers.get("content-type") if resp else "") or ""
                url_final = resp.url if resp else page.url

                html = page.content()
                rel_path = sanitize_path(urlparse(url_final).path)
                # Use .html extension
                if rel_path.endswith("/"):
                    rel_path += "index.html"
                elif not os.path.splitext(rel_path)[1]:
                    rel_path += ".html"
                elif rel_path.lower().endswith(".htm"):
                    rel_path += "l"  # turn .htm into .html to avoid ambiguity

                local_path = os.path.join(OUT_DIR, rel_path)
                local_path = shorten_windows_path(local_path, "0"*64)

                final_path = safe_write_text(local_path, html)

                writer.writerow({
                    "url": url_final,
                    "local_path": final_path,
                    "status": status,
                    "content_type": ctype or "text/html",
                    "sha256": "",
                    "bytes": len(html.encode("utf-8")),
                    "is_document": False
                })

                # Discover links (BFS)
                soup = BeautifulSoup(html, "lxml")
                for a in soup.select("a[href]"):
                    href = (a.get("href") or "").strip()
                    if not href:
                        continue
                    target = urljoin(url_final, href)
                    if should_visit(target) and not is_malformed_url(target):
                        q.append(target)

                pages_crawled += 1
                time.sleep(REQUEST_DELAY_S)

            except PWTimeout:
                print(f"[timeout] {url}")
            except Exception as e:
                print(f"[error] {url}: {e}")

        browser.close()

    print(f"Done. Crawled pages: {pages_crawled}, saved to {OUT_DIR}, catalog at {CATALOG}")

# -------------------------
# Main
# -------------------------
if __name__ == "__main__":
    crawl()
