"""
preprocess_docs.py - IMPROVED VERSION

Key improvements over v1:
1. Hierarchical section extraction (preserves document structure)
2. Better boilerplate removal
3. Metadata extraction (procedure codes, dates, responsible units)
4. Document summary generation
5. Better language detection
6. Structured JSON output with sections

JSON schema:
{
  "source_path": "relative/path/to/doc.html",
  "title": "Document Title",
  "lang": "tr" | "en",
  "doc_type": "procedure" | "directive" | "regulation" | "form" | "other",
  "metadata": {
    "procedure_code": "PIRO-C420-0101",
    "effective_date": "2014-04-16",
    "update_date": "2025-09-24",
    "responsible_unit": "IRO / Uluslararası İlişkiler Ofisi",
    "approver": "MH / Mütevelli Heyeti"
  },
  "summary": "Auto-generated document summary...",
  "sections": [
    {"header": "Amaç", "level": 1, "text": "..."},
    {"header": "Kapsam", "level": 1, "text": "..."},
    {"header": "Uygulama Adımları", "level": 1, "text": "..."},
    {"header": "1. Tanıtım, Çağrı ve Başvuru", "level": 2, "text": "..."}
  ],
  "full_text": "Complete cleaned text for backward compatibility"
}
"""

import os
import re
import json
import pathlib
import subprocess
import shutil
from typing import List, Tuple, Optional, Dict, Any
import pandas as pd
from bs4 import BeautifulSoup
from pypdf import PdfReader
from dotenv import load_dotenv

load_dotenv()

try:
    import docx
except ImportError:
    docx = None

try:
    from langdetect import detect as ld_detect
except ImportError:
    ld_detect = None

# ================= CONFIG =================

BASE_DIR = os.getenv("PROJECT_ROOT") or os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# Use mysu_dump_plus2 specifically for all raw files
RAW_ROOT = os.getenv("ROOT_DIR") or os.path.join(BASE_DIR, "crawler_for_srdoc", "mysu_dump_plus2")
PREPROCESSING_DIR = os.getenv("PREPROCESSING_PATH") or os.path.join(BASE_DIR, "preprocessing")
PRE_ROOT = os.getenv("PREPROCESSED_ROOT_V2") or os.path.join(PREPROCESSING_DIR, "preprocessed_docs_v2")
CONVERTED_DOC_DIR = os.path.join(PRE_ROOT, "_converted_doc")

os.makedirs(PRE_ROOT, exist_ok=True)
os.makedirs(CONVERTED_DOC_DIR, exist_ok=True)

# Supported file extensions - includes all document types
SUPPORTED_EXTS = {".html", ".htm", ".edu", ".pdf", ".doc", ".doc_20", ".docx", ".csv"}
# Skip binary and Excel files
SKIP_EXTS = {".bin", ".xlsx", ".xls"}

# LibreOffice path for Windows (adjust if needed)
LIBREOFFICE_PATH = os.getenv("LIBREOFFICE_PATH") or r"C:\Program Files\LibreOffice\program\soffice.exe"

TR_DIACRITICS = "çğıöşüÇĞİÖŞÜ"
MIN_TEXT_LEN = 50

# ================= UTILITIES =================

def normalize_ws(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()

def glue_codes(text: str) -> str:
    return re.sub(r"-\n", "", text)

def fix_pdf_wrapping(text: str) -> str:
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

# ================= METADATA EXTRACTION =================

def extract_procedure_metadata(text: str) -> Dict[str, Any]:
    """Extract structured metadata from procedure documents."""
    metadata = {}
    
    # Procedure code patterns
    code_patterns = [
        r"Prosedür\s*No[:\s]+([A-Z0-9\-]+(?:V\d+)?)",
        r"Yönerge\s*No[:\s]+([A-Z0-9\-]+(?:V\d+)?)",
        r"\(([PFIS][A-Z]{2,4}-[A-Z0-9\-]+)\)",
    ]
    for pat in code_patterns:
        m = re.search(pat, text, re.IGNORECASE)
        if m:
            metadata["procedure_code"] = m.group(1).upper()
            break
    
    # Date patterns
    date_patterns = [
        (r"Yürürlük\s*Tarihi[:\s]+(\d{2}[-/]\d{2}[-/]\d{4})", "effective_date"),
        (r"Güncelleme\s*Tarihi[:\s]+(\d{2}[-/]\d{2}[-/]\d{4})", "update_date"),
        (r"Effective\s*Date[:\s]+(\d{2}[-/]\d{2}[-/]\d{4})", "effective_date"),
        (r"Update\s*Date[:\s]+(\d{2}[-/]\d{2}[-/]\d{4})", "update_date"),
    ]
    for pat, key in date_patterns:
        m = re.search(pat, text, re.IGNORECASE)
        if m:
            metadata[key] = m.group(1)
    
    # Responsible unit
    unit_patterns = [
        r"İlgili\s*Birim\s*/\s*Sahibi[:\s]+([^\n]+?)(?:Onaylayan|$)",
        r"Related\s*Unit[:\s]+([^\n]+?)(?:Approver|$)",
    ]
    for pat in unit_patterns:
        m = re.search(pat, text, re.IGNORECASE)
        if m:
            metadata["responsible_unit"] = normalize_ws(m.group(1))
            break
    
    # Approver
    approver_patterns = [
        r"Onaylayan[:\s]+([^\n]+?)(?:İlgili|Amaç|$)",
        r"Approver[:\s]+([^\n]+?)(?:Related|Purpose|$)",
    ]
    for pat in approver_patterns:
        m = re.search(pat, text, re.IGNORECASE)
        if m:
            metadata["approver"] = normalize_ws(m.group(1))
            break
    
    return metadata

def detect_doc_type(title: str, text: str) -> str:
    """Detect document type from title and content."""
    title_lower = (title or "").lower()
    text_lower = (text or "")[:2000].lower()
    
    if "prosedür" in title_lower or "procedure" in title_lower:
        return "procedure"
    elif "yönerge" in title_lower or "directive" in title_lower:
        return "directive"
    elif "yönetmelik" in title_lower or "regulation" in title_lower:
        return "regulation"
    elif "form" in title_lower:
        return "form"
    elif "prosedür no" in text_lower or "procedure no" in text_lower:
        return "procedure"
    elif "yönerge no" in text_lower:
        return "directive"
    
    return "other"

# ================= SECTION EXTRACTION =================

# Patterns for Turkish/English section headers
SECTION_PATTERNS_TR = [
    r"^(Amaç)\s*$",
    r"^(Kapsam)\s*$",
    r"^(Tanımlar(?:\s*/\s*Kısaltmalar)?)\s*$",
    r"^(Uygulama\s*Adımları)[:\s]*$",
    r"^(Sorumluluk(?:lar)?)\s*$",
    r"^(İlgili\s*(?:Formlar|Yönergeler|Dokümanlar))\s*$",
    r"^(\d+\.\s*.+)$",  # Numbered sections like "1. Tanıtım"
    r"^(\d+\.\d+\.?\s*.+)$",  # Sub-sections like "1.1. Alt Başlık"
]

SECTION_PATTERNS_EN = [
    r"^(Purpose)\s*$",
    r"^(Scope)\s*$",
    r"^(Definitions(?:\s*/\s*Abbreviations)?)\s*$",
    r"^(Application\s*Steps)[:\s]*$",
    r"^(Responsibilities)\s*$",
    r"^(Related\s*(?:Forms|Directives|Documents))\s*$",
    r"^(\d+\.\s*.+)$",
    r"^(\d+\.\d+\.?\s*.+)$",
]

def extract_sections_from_text(text: str, lang: str = "tr") -> List[Dict]:
    """
    Extract hierarchical sections from document text.
    Returns list of {"header": str, "level": int, "text": str}
    """
    patterns = SECTION_PATTERNS_TR if lang == "tr" else SECTION_PATTERNS_EN
    
    lines = text.split("\n")
    sections = []
    current_section = {"header": "Introduction", "level": 0, "text": ""}
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
        
        is_header = False
        header_text = ""
        level = 1
        
        for pat in patterns:
            m = re.match(pat, line, re.IGNORECASE)
            if m:
                is_header = True
                header_text = m.group(1).strip()
                
                # Determine level
                if re.match(r"^\d+\.\d+", header_text):
                    level = 2
                elif re.match(r"^\d+\.", header_text):
                    level = 1
                else:
                    level = 1
                break
        
        if is_header:
            # Save previous section if it has content
            if current_section["text"].strip():
                sections.append(current_section)
            current_section = {"header": header_text, "level": level, "text": ""}
        else:
            current_section["text"] += " " + line
    
    # Don't forget the last section
    if current_section["text"].strip():
        sections.append(current_section)
    
    # Clean up section texts
    for sec in sections:
        sec["text"] = normalize_ws(sec["text"])
    
    return sections

# ================= BOILERPLATE REMOVAL =================

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
    r"^a-z bookmarks?$",
    r"^künye bilgisini göster$",
    r"^ilgili birimler$",
    r"^süreç sahibi$",
    r"^süreç sorumluları$",
    r"^süreç tedarikçileri$",
    r"^süreç girdisi$",
    r"^süreç müşterileri$",
    r"^performans göstergeleri$",
    r"^raporlama periyodu$",
    r"^copyright.*sabancı",
    r"^all rights reserved",
    r"^tüm hakları saklıdır",
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

def strip_boilerplate(text: str) -> str:
    """Remove boilerplate lines from text."""
    lines = text.split("\n")
    filtered = [l for l in lines if not is_boilerplate_line(l)]
    return "\n".join(filtered)

# ================= SUMMARY GENERATION =================

def generate_document_summary(title: str, sections: List[Dict], max_len: int = 500) -> str:
    """
    Generate a concise summary from document structure.
    Uses Amaç (Purpose) and Kapsam (Scope) sections if available.
    """
    summary_parts = []
    
    # Add title
    if title:
        summary_parts.append(f"Belge: {title}")
    
    # Look for Purpose/Scope sections
    purpose_keywords = ["amaç", "purpose", "amacı"]
    scope_keywords = ["kapsam", "scope"]
    
    for sec in sections:
        header_lower = sec["header"].lower()
        
        for kw in purpose_keywords:
            if kw in header_lower:
                purpose_text = sec["text"][:300]
                summary_parts.append(f"Amaç: {purpose_text}")
                break
        
        for kw in scope_keywords:
            if kw in header_lower:
                scope_text = sec["text"][:200]
                summary_parts.append(f"Kapsam: {scope_text}")
                break
    
    summary = " | ".join(summary_parts)
    
    if len(summary) > max_len:
        summary = summary[:max_len] + "..."
    
    return summary if summary else "Belge özeti mevcut değil."

# ================= HTML EXTRACTION =================

def detect_html_lang_from_soup(soup: BeautifulSoup) -> str:
    html_tag = soup.find("html")
    if html_tag is None:
        return ""
    lang_attr = html_tag.get("lang") or html_tag.get("xml:lang")
    if not lang_attr:
        return ""
    code = lang_attr.split("-")[0].lower()
    return code if code in ("tr", "en") else ""

def extract_html_body(path: str) -> Tuple[str, str, str, List[Dict]]:
    """
    Extract body text with section structure.
    Returns: (full_text, title, html_lang, sections)
    """
    html = pathlib.Path(path).read_text(encoding="utf-8", errors="ignore")
    soup = BeautifulSoup(html, "lxml")
    
    html_lang = detect_html_lang_from_soup(soup)
    
    # Remove junk tags
    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()
    for sel in ["nav", "footer", "header", ".menu", ".navbar", ".breadcrumb", 
                ".sidebar", "#cookie-banner", ".social-share", ".pagination"]:
        for node in soup.select(sel):
            node.decompose()
    
    title = (soup.title.string.strip() if soup.title and soup.title.string else "")
    body = soup.select_one("main") or soup.select_one("article") or soup.body or soup
    
    full_text = body.get_text("\n")
    full_text = strip_boilerplate(full_text)
    full_text = normalize_ws(full_text)
    
    # Detect language for section extraction
    lang = html_lang or guess_lang_from_text(full_text)
    sections = extract_sections_from_text(full_text, lang)
    
    return full_text, title, html_lang, sections

# ================= PDF EXTRACTION =================

def load_pdf_body(path: str) -> Tuple[str, str, List[Dict]]:
    """Extract PDF with section detection."""
    reader = PdfReader(path)
    pages = [(p.extract_text() or "") for p in reader.pages]
    text = "\n".join(pages)
    text = fix_pdf_wrapping(text)
    title = os.path.splitext(os.path.basename(path))[0]
    
    lang = guess_lang_from_text(text)
    sections = extract_sections_from_text(text, lang)
    
    return text, title, sections

# ================= DOCX EXTRACTION =================

def load_docx_body(path: str) -> Tuple[str, str, List[Dict]]:
    """Extract DOCX with structure."""
    if docx is None:
        raise RuntimeError("python-docx is not installed")
    
    d = docx.Document(path)
    parts = []
    for p in d.paragraphs:
        if p.text:
            parts.append(p.text)
    text = "\n".join(parts)
    text = normalize_ws(text)
    title = os.path.splitext(os.path.basename(path))[0]
    
    lang = guess_lang_from_text(text)
    sections = extract_sections_from_text(text, lang)
    
    return text, title, sections

# ================= DOC -> PDF CONVERSION =================

def convert_doc_to_pdf(path: str) -> Optional[str]:
    """
    Convert a .doc/.doc_20 file to PDF using LibreOffice.
    Returns the output PDF path, or None if conversion failed.
    """
    os.makedirs(CONVERTED_DOC_DIR, exist_ok=True)
    
    base = os.path.splitext(os.path.basename(path))[0]
    out_pdf = os.path.join(CONVERTED_DOC_DIR, base + ".pdf")
    
    # If already converted, reuse it
    if os.path.exists(out_pdf):
        return out_pdf
    
    # Check if LibreOffice exists
    if not os.path.exists(LIBREOFFICE_PATH):
        print(f"[warn] LibreOffice not found at {LIBREOFFICE_PATH}")
        return None
    
    cmd = [
        LIBREOFFICE_PATH,
        "--headless",
        "--convert-to", "pdf",
        "--outdir", CONVERTED_DOC_DIR,
        path,
    ]
    
    try:
        print(f"  [convert] DOC -> PDF: {os.path.basename(path)}")
        result = subprocess.run(
            cmd,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=120,  # 2 minute timeout
        )
        if os.path.exists(out_pdf):
            return out_pdf
        else:
            print(f"  [warn] PDF not created for {path}")
            return None
    except subprocess.TimeoutExpired:
        print(f"  [timeout] DOC conversion timed out: {path}")
        return None
    except subprocess.CalledProcessError as e:
        print(f"  [error] DOC conversion failed: {e}")
        return None
    except Exception as e:
        print(f"  [error] DOC conversion error: {e}")
        return None

# ================= MAIN PREPROCESSING =================

def preprocess_one(path: str) -> Optional[Dict]:
    """
    Preprocess a single document with enhanced metadata extraction.
    """
    ext = os.path.splitext(path)[1].lower()
    rel_source = os.path.relpath(path, RAW_ROOT).replace("\\", "/")
    
    text = ""
    title = ""
    html_lang = ""
    sections = []
    
    try:
        if ext in (".html", ".htm", ".edu"):
            text, title, html_lang, sections = extract_html_body(path)
        elif ext == ".pdf":
            text, title, sections = load_pdf_body(path)
        elif ext == ".docx":
            text, title, sections = load_docx_body(path)
        elif ext in (".doc", ".doc_20"):
            # Convert DOC to PDF using LibreOffice, then extract
            pdf_path = convert_doc_to_pdf(path)
            if pdf_path and os.path.exists(pdf_path):
                text, title, sections = load_pdf_body(pdf_path)
                # Use original doc title
                title = os.path.splitext(os.path.basename(path))[0]
            else:
                print(f"  [skip] Could not convert DOC: {path}")
                return None
        elif ext == ".csv":
            df = pd.read_csv(path)
            text = normalize_ws(df.to_string(index=False))
            title = os.path.splitext(os.path.basename(path))[0]
            sections = [{"header": "Data", "level": 1, "text": text}]
        else:
            print(f"[skip-unsupported] {path}")
            return None
    except Exception as e:
        print(f"[error] {path}: {e}")
        return None
    
    text = normalize_ws(text)
    
    if len(text) < MIN_TEXT_LEN:
        print(f"[skip-short] {path} ({len(text)} chars)")
        return None
    
    lang = html_lang or guess_lang_from_text(text)
    doc_type = detect_doc_type(title, text)
    metadata = extract_procedure_metadata(text)
    summary = generate_document_summary(title, sections)
    
    data = {
        "source_path": rel_source,
        "title": title,
        "lang": lang,
        "html_lang": html_lang,
        "doc_type": doc_type,
        "metadata": metadata,
        "summary": summary,
        "sections": sections,
        "full_text": text,  # Backward compatible
    }
    
    return data

def save_preprocessed_json(rel_source: str, data: Dict):
    """Save preprocessed data to JSON."""
    rel_no_ext = os.path.splitext(rel_source)[0]
    out_path = os.path.join(PRE_ROOT, rel_no_ext + ".json").replace("\\", "/")
    out_dir = os.path.dirname(out_path)
    os.makedirs(out_dir, exist_ok=True)
    
    pathlib.Path(out_path).write_text(
        json.dumps(data, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"[ok] {rel_source} -> {out_path}")

# ================= MAIN =================

def main():
    import time
    start_time = time.time()
    
    # Collect all files first
    all_files = []
    for root, _, files in os.walk(RAW_ROOT):
        for fn in files:
            path = os.path.join(root, fn)
            ext = os.path.splitext(fn)[1].lower()
            
            if "catalog.csv" in path:
                continue
            if ext in SKIP_EXTS:
                continue
            if ext not in SUPPORTED_EXTS:
                continue
            
            all_files.append(path)
    
    total = len(all_files)
    processed = 0
    skipped = 0
    errors = 0
    
    print(f"\n{'='*60}")
    print(f"PREPROCESSING - {total} files to process")
    print(f"Source: {RAW_ROOT}")
    print(f"Output: {PRE_ROOT}")
    print(f"{'='*60}\n")
    
    for i, path in enumerate(all_files, 1):
        ext = os.path.splitext(path)[1].lower()
        elapsed = time.time() - start_time
        rate = i / elapsed if elapsed > 0 else 0
        eta = (total - i) / rate if rate > 0 else 0
        
        print(f"\n[{i}/{total}] ({processed} ok, {skipped} skip, {errors} err) "
              f"ETA: {eta/60:.1f}min | {os.path.basename(path)}")
        
        try:
            data = preprocess_one(path)
            if data is None:
                skipped += 1
                continue
            save_preprocessed_json(data["source_path"], data)
            processed += 1
        except Exception as e:
            print(f"  [error] {e}")
            errors += 1
    
    elapsed = time.time() - start_time
    print(f"\n{'='*60}")
    print(f"DONE in {elapsed/60:.1f} minutes")
    print(f"  Total files: {total}")
    print(f"  Processed:   {processed}")
    print(f"  Skipped:     {skipped}")
    print(f"  Errors:      {errors}")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()
