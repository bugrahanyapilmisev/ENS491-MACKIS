"""
preprocess_docs.py

- Reads raw crawled documents (HTML, PDF, DOC, DOCX, CSV, etc.)
- Cleans and normalizes text.
- For Sabancı "süreç haritası" HTML, removes boilerplate header (URL, Type, Ara, etc.).
- Converts legacy .doc/.doc_20 to PDF via LibreOffice, then parses as PDF.
- Writes one JSON per document into PREPROCESSED_ROOT.

JSON schema:
{
  "source_path": "relative/path/from_RAW_ROOT/psr-c220-0102.html",
  "title": "Dersten Çekilme Prosedürü (PSR-C220-0102) | Yönerge",
  "lang": "tr" | "en",
  "html_lang": "tr" | "en" | "",
  "text": "<full cleaned body text>"
}
"""

import os, re, json, pathlib, subprocess
from typing import List, Tuple, Optional

import pandas as pd
from bs4 import BeautifulSoup
from pypdf import PdfReader
from dotenv import load_dotenv

# Load .env file (ROOT_DIR, PREPROCESSING_PATH, etc.)
load_dotenv()

try:
    import docx  # for .docx
except ImportError:
    docx = None

try:
    from langdetect import detect as ld_detect
except ImportError:
    ld_detect = None

# ---------------- CONFIG ----------------

BASE_DIR = os.getenv("PROJECT_ROOT") or os.getcwd()
RAW_ROOT = os.getenv("ROOT_DIR") or os.path.join(BASE_DIR, "crawler_for_srdoc")
PREPROCESSING_DIR = os.getenv("PREPROCESSING_PATH") or os.path.join(BASE_DIR, "preprocessing")
PRE_ROOT = os.getenv("PREPROCESSED_ROOT") or os.path.join(PREPROCESSING_DIR, "preprocessed_docs")
CONVERTED_DOC_DIR = os.path.join(PRE_ROOT, "_converted_doc")

os.makedirs(PRE_ROOT, exist_ok=True)
os.makedirs(CONVERTED_DOC_DIR, exist_ok=True)

SUPPORTED_EXTS = {
    ".html", ".htm", ".edu",
    ".pdf",
    ".doc", ".doc_20",
    ".docx",
    ".csv",            # keep CSV (tabular but often cleaner than Excel)
}
# Explicitly discard Excel-like binaries
SKIP_EXTS = {".bin", ".xlsx", ".xls"}

TR_DIACRITICS = "çğıöşüÇĞİÖŞÜ"
MIN_TEXT_LEN = 50


# ---------------- UTILS ----------------
def normalize_ws(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def glue_codes(text: str) -> str:
    # merge lines where a dash is used for hyphenation at end of line
    return re.sub(r"-\n", "", text)


def fix_pdf_wrapping(text: str) -> str:
    """
    Fix line-wrapping issues in PDF-extracted text.
    - Joins lines where there isn't a clear sentence boundary.
    - Keeps lines that start with capital letters / numbers as potential new sentences.
    """
    text = glue_codes(text)
    text = re.sub(
        r"(?<![\.!?:])\n(?!\s*[A-Z" + TR_DIACRITICS + r"0-9])",
        " ",
        text,
    )
    return normalize_ws(text)


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
    html_tag = soup.find("html")
    if html_tag is None:
        return ""
    lang_attr = html_tag.get("lang") or html_tag.get("xml:lang")
    if not lang_attr:
        return ""
    code = lang_attr.split("-")[0].lower()
    if code in ("tr", "en"):
        return code
    return ""


# ---------------- HTML BOILERPLATE STRIPPING ----------------

BOILER_PATTERNS = [
    r"^url\s*:",
    r"^type \(logical\)\s*:",
    r"^file type\s*:",
    r"^language\s*:",
    r"^parse status\s*:",
    r"^sabancı üniversitesi süreçleri$",
    r"^arama formu$",
    r"^ara$",
    r"^english$",
    r"^türkçe$",
    r"^a-z bookmarks$",
    r"^künye bilgisini göster$",
    r"^ilgili birimler",
    r"^süreç sahibi",
    r"^süreç sorumluları",
    r"^süreç tedarikçileri",
    r"^süreç girdisi",
    r"^süreç müşterileri",
    r"^performans göstergeleri",
    r"^raporlama periyodu",
]

BOILER_REGEXES = [re.compile(pat, re.IGNORECASE) for pat in BOILER_PATTERNS]


def is_boilerplate_line(line: str) -> bool:
    stripped = line.strip()
    if not stripped:
        return False
    for rx in BOILER_REGEXES:
        if rx.match(stripped):
            return True
    return False


def strip_mysu_header(full_text: str) -> str:
    """
    Remove the big metadata header block you see in the screenshots
    (URL, Type, Ara, English, Türkçe, etc.), and keep only the body.
    """
    lines = [normalize_ws(l.rstrip()) for l in full_text.splitlines()]

    filtered: List[str] = []
    for ln in lines:
        if is_boilerplate_line(ln):
            continue
        filtered.append(ln)

    body_start = 0
    hints = (
        "amaç", "kapsam", "tanımlar", "bu prosedür",
        "uygulama adımları", "öğrenci", "isteğe bağlı yaz dönemi"
    )
    for i, ln in enumerate(filtered):
        low = ln.lower()
        if any(low.startswith(h) for h in hints):
            body_start = i
            break

    body = "\n".join(filtered[body_start:])
    return normalize_ws(body)


# ---------------- LOADERS ----------------

def extract_html_body(path: str) -> Tuple[str, str, str]:
    """
    Returns: (body_text, title, html_lang)
    """
    html = pathlib.Path(path).read_text(encoding="utf-8", errors="ignore")
    soup = BeautifulSoup(html, "lxml")

    html_lang = detect_html_lang_from_soup(soup)

    # remove layout junk
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
    body = soup.select_one("main") or soup.select_one("article") or soup.body or soup

    full_text = body.get_text("\n")
    full_text = normalize_ws(full_text)
    full_text = glue_codes(full_text)

    # apply Sabancı-specific header stripping
    body_text = strip_mysu_header(full_text)
    return body_text, title, html_lang


def load_pdf_body(path: str) -> Tuple[str, str]:
    reader = PdfReader(path)
    pages = [(p.extract_text() or "") for p in reader.pages]
    text = "\n".join(pages)
    text = fix_pdf_wrapping(text)
    title = os.path.splitext(os.path.basename(path))[0]
    return text, title


def load_docx_body(path: str) -> Tuple[str, str]:
    if docx is None:
        raise RuntimeError("python-docx is not installed; cannot read .docx")
    d = docx.Document(path)
    parts = []
    for p in d.paragraphs:
        if p.text:
            parts.append(p.text)
    text = normalize_ws("\n".join(parts))
    title = os.path.splitext(os.path.basename(path))[0]
    return text, title


def load_csv_body(path: str) -> Tuple[str, str]:
    df = pd.read_csv(path)
    text = normalize_ws(df.to_string(index=False))
    title = os.path.splitext(os.path.basename(path))[0]
    return text, title


# ---------------- DOC -> PDF CONVERSION ----------------

def convert_doc_to_pdf(path: str) -> str:
    """
    Convert a .doc/.doc_20 file to PDF using LibreOffice/soffice.
    Returns the output PDF path, or "" if conversion failed.
    """
    os.makedirs(CONVERTED_DOC_DIR, exist_ok=True)

    base = os.path.splitext(os.path.basename(path))[0]
    out_pdf = os.path.join(CONVERTED_DOC_DIR, base + ".pdf")

    # If we already converted it before, reuse it
    if os.path.exists(out_pdf):
        return out_pdf

    # Pick the correct executable
    if os.name == "nt":
        # Adjust if your LibreOffice is installed in a different folder:
        soffice = r"C:\\Program Files\\LibreOffice\\program\soffice.exe"
    else:
        soffice = "libreoffice"

    cmd = [
        soffice,
        "--headless",
        "--convert-to",
        "pdf",
        "--outdir",
        CONVERTED_DOC_DIR,
        path,
    ]

    try:
        print(f"[convert] DOC -> PDF: {path}")
        result = subprocess.run(
            cmd,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

        if not os.path.exists(out_pdf):
            print(f"[warn] expected PDF not found after conversion: {out_pdf}")
            return ""
        return out_pdf

    except FileNotFoundError:
        # soffice/libreoffice not found
        print(
            "[error] LibreOffice/soffice executable not found.\n"
            "Please check that LibreOffice is installed and the path in convert_doc_to_pdf is correct."
        )
        return ""
    except subprocess.CalledProcessError as e:
        print(f"[error] converting DOC to PDF failed for {path}: {e}")
        return ""
    except Exception as e:
        print(f"[error] unexpected error during DOC->PDF conversion for {path}: {e}")
        return ""


# ---------------- PREPROCESS ONE FILE ----------------

def preprocess_one(path: str) -> Optional[dict]:
    ext = os.path.splitext(path)[1].lower()
    rel_source = os.path.relpath(path, RAW_ROOT).replace("\\", "/")

    text = ""
    title = ""
    html_lang = ""

    if ext in (".html", ".htm", ".edu"):
        text, title, html_lang = extract_html_body(path)
    elif ext == ".pdf":
        text, title = load_pdf_body(path)
    elif ext in (".doc", ".doc_20"):
        pdf_path = convert_doc_to_pdf(path)
        if not pdf_path:
            return None
        text, title = load_pdf_body(pdf_path)
    elif ext == ".docx":
        text, title = load_docx_body(path)
    elif ext == ".csv":
        text, title = load_csv_body(path)
    else:
        print(f"[skip-unsupported] {path}")
        return None

    text = normalize_ws(text)

    if len(text) < MIN_TEXT_LEN:
        print(f"[skip-short] {path} ({len(text)} chars)")
        return None

    lang = guess_lang_from_text(text)

    data = {
        "source_path": rel_source,
        "title": title,
        "lang": lang,
        "html_lang": html_lang,
        "text": text,
    }
    return data


def save_preprocessed_json(rel_source: str, data: dict):
    rel_no_ext = os.path.splitext(rel_source)[0]
    out_path = os.path.join(PRE_ROOT, rel_no_ext + ".json").replace("\\", "/")
    out_dir = os.path.dirname(out_path)
    os.makedirs(out_dir, exist_ok=True)
    pathlib.Path(out_path).write_text(
        json.dumps(data, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"[ok] {rel_source} -> {out_path}")


# ---------------- MAIN ----------------

def main():
    total = 0
    processed = 0

    for root, _, files in os.walk(RAW_ROOT):
        for fn in files:
            path = os.path.join(root, fn)
            ext = os.path.splitext(fn)[1].lower()
            total += 1
            if path[-11:] == "catalog.csv":
                print("catalogu geçtim")
                continue
            
            if ext in SKIP_EXTS:
                print(f"[skip-bin/excel] {path}")
                continue
            if ext not in SUPPORTED_EXTS:
                print(f"[skip-unknown-ext] {path}")
                continue

            try:
                data = preprocess_one(path)
                if data is None:
                    continue
                save_preprocessed_json(data["source_path"], data)
                processed += 1
            except Exception as e:
                print(f"[error] {path}: {e}")

    print(f"\nDone. total files seen={total}, preprocessed={processed}")


if __name__ == "__main__":
    main()
