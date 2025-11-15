# crawl_mysu_surecharitasi.py
import os, re, csv, time, hashlib, pathlib
from urllib.parse import urljoin, urlparse, urldefrag
from collections import deque
from bs4 import BeautifulSoup
from playwright.sync_api import sync_playwright, TimeoutError as PWTimeout

START_URL = "https://mysu.sabanciuniv.edu/surecharitasi/tr/surec-tablosu"
ALLOWED_NETLOC = "mysu.sabanciuniv.edu"
ALLOWED_PREFIX = "/surecharitasi/"
# ✅ FIX 1: make the path a raw string (or use forward slashes)
STATE_FILE = r"C:\\Users\\kosot\\OneDrive\\Masaüstü\\CS\\bitirme\\crawler_for_srdoc\\session.json"
OUT_DIR = "mysu_dump"
CATALOG = os.path.join(OUT_DIR, "catalog.csv")

DOC_EXT = {
    ".pdf", ".doc", ".docx", ".xls", ".xlsx", ".ppt", ".pptx",
    ".odt", ".ods", ".zip", ".rar", ".csv", ".txt"
}

REQUEST_DELAY_S = 0.7
MAX_PAGES = 5000

def sanitize_path(url_path: str) -> str:
    safe = re.sub(r"[^a-zA-Z0-9/_\-.]", "_", url_path).strip("/")
    return safe or "root"

def ensure_parent_dir(path):
    pathlib.Path(path).parent.mkdir(parents=True, exist_ok=True)

def should_visit(url):
    u = urlparse(url)
    return (u.scheme in ("http","https")
            and u.netloc == ALLOWED_NETLOC
            and u.path.startswith(ALLOWED_PREFIX))

def is_document(url):
    ext = os.path.splitext(urlparse(url).path)[1].lower()
    return ext in DOC_EXT

def normalize(url):
    url, _ = urldefrag(url)
    return url

def hash_bytes(b):
    import hashlib as _h
    return _h.sha256(b).hexdigest()

def crawl():
    pathlib.Path(OUT_DIR).mkdir(parents=True, exist_ok=True)
    visited = set()
    q = deque([START_URL])

    with open(CATALOG, "w", newline="", encoding="utf-8") as fcat, sync_playwright() as p:
        writer = csv.DictWriter(fcat, fieldnames=[
            "url", "local_path", "status", "content_type", "sha256", "bytes", "is_document"
        ])
        writer.writeheader()

        browser = p.chromium.launch(headless=True)
        context = browser.new_context(storage_state=STATE_FILE)
        page = context.new_page()

        pages_crawled = 0
        while q and pages_crawled < MAX_PAGES:
            url = normalize(q.popleft())
            if url in visited or not should_visit(url):
                continue
            visited.add(url)

            try:
                # ✅ FIX 2: use the Response returned by goto()
                resp = page.goto(url, wait_until="domcontentloaded", timeout=45_000)
                time.sleep(REQUEST_DELAY_S)

                status = resp.status if resp else None
                ctype = (resp.headers.get("content-type") if resp else "") or ""
                url_final = resp.url if resp else page.url

                # Download if it looks like a document by extension or content-type
                if is_document(url_final) or ("application/" in ctype and "html" not in ctype):
                    bin_data = page.evaluate("""async () => {
                        const res = await fetch(window.location.href, { credentials: 'include' });
                        const buf = await res.arrayBuffer();
                        return Array.from(new Uint8Array(buf));
                    }""")
                    content = bytes(bin_data)
                    digest = hash_bytes(content)
                    rel_path = sanitize_path(urlparse(url_final).path)
                    if rel_path.endswith("/"):
                        rel_path += "index.bin"
                    local_path = os.path.join(OUT_DIR, rel_path)
                    ensure_parent_dir(local_path)
                    with open(local_path, "wb") as outf:
                        outf.write(content)

                    writer.writerow({
                        "url": url_final,
                        "local_path": local_path,
                        "status": status,
                        "content_type": ctype,
                        "sha256": digest,
                        "bytes": len(content),
                        "is_document": True
                    })
                    pages_crawled += 1
                    continue

                # Otherwise, treat as HTML: save and discover links
                html = page.content()
                rel_path = sanitize_path(urlparse(url_final).path)
                if rel_path.endswith("/"):
                    rel_path += "index.html"
                elif not os.path.splitext(rel_path)[1]:
                    rel_path += ".html"
                local_path = os.path.join(OUT_DIR, rel_path)
                ensure_parent_dir(local_path)
                with open(local_path, "w", encoding="utf-8") as outf:
                    outf.write(html)

                writer.writerow({
                    "url": url_final,
                    "local_path": local_path,
                    "status": status,
                    "content_type": ctype or "text/html",
                    "sha256": "",
                    "bytes": len(html.encode("utf-8")),
                    "is_document": False
                })

                soup = BeautifulSoup(html, "lxml")
                for a in soup.select("a[href]"):
                    href = a.get("href", "").strip()
                    if not href:
                        continue
                    target = urljoin(url_final, href)
                    if should_visit(target):
                        q.append(target)

                pages_crawled += 1

            except PWTimeout:
                print(f"[timeout] {url}")
            except Exception as e:
                print(f"[error] {url}: {e}")

        browser.close()
    print(f"Done. Crawled pages: {pages_crawled}, saved to {OUT_DIR}, catalog at {CATALOG}")

if __name__ == "__main__":
    crawl()
