#!/usr/bin/env python
"""
Stage 1: Build a parsed dataset from crawler output.

Reads:
  - crawler_for_srdoc/mysu_dump_plus2/catalog.csv
  - the files referenced in that catalog

Writes:
  - preprocessing/parsed_dataset.jsonl
    (one JSON object per document, with clean text for HTML/PDF/DOCX/Excel/CSV)
"""

import csv
import json
import os
import re
from pathlib import Path
from typing import Dict, Any, Tuple

from bs4 import BeautifulSoup
from pypdf import PdfReader

# Optional deps for non-HTML
try:
    import docx  # for .docx
except ImportError:
    docx = None

try:
    import textract  # optional, for legacy .doc
except ImportError:
    textract = None

import pandas as pd


# --- CONFIG -----------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parents[1]

CRAWLER_ROOT = REPO_ROOT / "crawler_for_srdoc" / "mysu_dump_plus2"
CATALOG_PATH = CRAWLER_ROOT / "catalog.csv"

PARSED_DATASET_PATH = REPO_ROOT / "preprocessing" / "parsed_dataset.jsonl"


# --- BASIC HELPERS ----------------------------------------------------------


def guess_type_from_url(url: str) -> str:
    """Infer SR document type from URL path (logical type: surec, yonerge, ...)."""
    u = url.lower()
    if "/yonerge/" in u:
        return "yonerge"
    if "/prosedur/" in u:
        return "prosedur"
    if "/surec/" in u:
        return "surec"
    if "/form/" in u:
        return "form"
    return "generic_html"


def map_local_path(local_path_str: str) -> Path:
    """
    Map the local_path from catalog.csv (which may contain an absolute path
    from another machine) to the corresponding path under this repo's CRAWLER_ROOT.
    """
    lp = Path(local_path_str)

    # Relative → from catalog folder
    if not lp.is_absolute():
        return (CATALOG_PATH.parent / lp).resolve()

    parts = lp.parts
    if "mysu_dump_plus2" in parts:
        idx = parts.index("mysu_dump_plus2")
        sub = Path(*parts[idx + 1 :])
        return (CRAWLER_ROOT / sub).resolve()

    return lp


# --- TEXT NORMALIZATION (copied from build_chroma_store_plus2) -------------


TR_DIACRITICS = "çğıöşüÇĞİÖŞÜ"


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


# --- LOADERS (aligned with build_chroma_store_plus2) ------------------------


def load_pdf_text(path: str) -> Tuple[str, str]:
    reader = PdfReader(path)
    pages = [(p.extract_text() or "") for p in reader.pages]
    text = "\n".join(pages)
    text = fix_pdf_wrapping(text)
    title = ""
    return text, title


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
    df = pd.read_csv(path)
    text = normalize_ws_keep_diacritics(df.to_string(index=False))
    title = os.path.splitext(os.path.basename(path))[0]
    return text, title


def load_non_html_text(local_path: Path, doctype: str) -> Tuple[str, str]:
    ext = doctype.lower()
    if ext == "pdf":
        return load_pdf_text(str(local_path))
    if ext == "docx":
        return load_docx_text(str(local_path))
    if ext == "doc":
        return load_legacy_doc_text(str(local_path))
    if ext in {"xls", "xlsx"}:
        return load_excel_text(str(local_path))
    if ext == "csv":
        return load_csv_text(str(local_path))
    return "", ""


# --- HTML LOADER (simple full-page clean text) ------------------------------


def load_html_and_clean(local_path: Path) -> Dict[str, Any]:
    """
    Load an HTML file and return:
      - title
      - full_clean_text
    """
    try:
        raw = local_path.read_text(encoding="utf-8", errors="ignore")
    except Exception as e:
        return {
            "parse_status": "error",
            "parse_error": f"read_error: {e}",
            "title": "",
            "full_clean_text": "",
        }

    soup = BeautifulSoup(raw, "lxml")

    # Remove script/style/noscript
    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()

    # Title: <title> or h1#page-title
    title_tag = soup.find("title")
    title = title_tag.get_text(strip=True) if title_tag else ""
    h1 = soup.find("h1", id="page-title")
    if h1:
        h1_text = h1.get_text(strip=True)
        if h1_text:
            title = h1_text

    full_text = soup.get_text("\n", strip=True)

    return {
        "parse_status": "ok",
        "parse_error": "",
        "title": title,
        "full_clean_text": full_text,
    }


# --- MAIN -------------------------------------------------------------------


def build_parsed_corpus() -> None:
    """Read catalog.csv and build parsed_dataset.jsonl."""
    if not CRAWLER_ROOT.exists():
        raise SystemExit(f"CRAWLER_ROOT not found: {CRAWLER_ROOT}")

    if not CATALOG_PATH.exists():
        raise SystemExit(f"catalog.csv not found: {CATALOG_PATH}")

    PARSED_DATASET_PATH.parent.mkdir(parents=True, exist_ok=True)

    print(f"[*] Crawler root:   {CRAWLER_ROOT}")
    print(f"[*] Catalog path:   {CATALOG_PATH}")
    print(f"[*] Dataset output: {PARSED_DATASET_PATH}")

    next_id = 1

    with CATALOG_PATH.open("r", encoding="utf-8", errors="ignore") as f_in, \
            PARSED_DATASET_PATH.open("w", encoding="utf-8") as f_out:

        reader = csv.DictReader(f_in)

        for row in reader:
            if not row:
                continue

            url = (row.get("url") or "").strip()
            local_path_str = (row.get("local_path") or "").strip()
            status = (row.get("status") or "").strip()
            content_type = (row.get("content_type") or "").strip().lower()
            bytes_str = (row.get("bytes") or "").strip()
            is_document_str = (row.get("is_document") or "").strip()
            url_lang = (row.get("url_lang") or "").strip()
            content_lang = (row.get("content_lang") or "").strip()
            lang_field = (row.get("lang") or "").strip()
            lang_mismatch = (row.get("lang_mismatch") or "").strip()
            title_from_catalog = (row.get("title") or "").strip()

            lang = lang_field or content_lang or url_lang or "unknown"

            local_path = map_local_path(local_path_str)
            doctype = local_path.suffix.lstrip(".").lower()  # html, pdf, docx, ...

            doc_type = guess_type_from_url(url)
            is_document = is_document_str.lower() == "true"

            try:
                n_bytes = int(bytes_str)
            except ValueError:
                n_bytes = None

            record: Dict[str, Any] = {
                "id": next_id,
                "url": url,
                "local_path": str(local_path),
                "http_status": status,
                "content_type": content_type,
                "bytes": n_bytes,
                "is_document": is_document,
                "type": doc_type,         # logical type: yonerge/surec/...
                "doctype": doctype,       # file extension: html/pdf/docx/...
                "url_lang": url_lang,
                "content_lang": content_lang,
                "lang": lang,
                "lang_mismatch": lang_mismatch,
                "title": title_from_catalog,
                "full_clean_text": "",
                "parse_status": "",
                "parse_error": "",
            }

            # --- HTML ---
            if not is_document and "text/html" in content_type:
                if not local_path.exists():
                    record["parse_status"] = "error"
                    record["parse_error"] = "file_missing"
                else:
                    parsed = load_html_and_clean(local_path)
                    title_final = parsed["title"] or title_from_catalog
                    record.update(
                        title=title_final,
                        full_clean_text=parsed["full_clean_text"],
                        parse_status=parsed["parse_status"],
                        parse_error=parsed["parse_error"],
                    )

            # --- Non-HTML documents (PDF, DOCX, DOC, XLS, XLSX, CSV) ---
            elif is_document and doctype in {"pdf", "docx", "doc", "xls", "xlsx", "csv"}:
                if not local_path.exists():
                    record["parse_status"] = "error"
                    record["parse_error"] = "file_missing"
                else:
                    try:
                        text, title2 = load_non_html_text(local_path, doctype)
                        if text:
                            record["full_clean_text"] = text
                            record["parse_status"] = "ok"
                            record["parse_error"] = ""
                            if title2:
                                record["title"] = title2
                        else:
                            record["parse_status"] = "error"
                            record["parse_error"] = "empty_text"
                    except Exception as e:
                        record["parse_status"] = "error"
                        record["parse_error"] = f"non_html_parse_error: {e}"

            # --- Other or unsupported types ---
            else:
                record["parse_status"] = "skipped_unsupported"
                record["parse_error"] = ""

            json.dump(record, f_out, ensure_ascii=False)
            f_out.write("\n")

            next_id += 1

    print(f"[*] Done. Wrote {next_id - 1} records to {PARSED_DATASET_PATH}")


if __name__ == "__main__":
    build_parsed_corpus()
