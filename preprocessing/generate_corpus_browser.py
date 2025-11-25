#!/usr/bin/env python
"""
Stage 2: Generate a static mini "document browser" from parsed_dataset.jsonl.

Reads:
  - preprocessing/parsed_dataset.jsonl

Writes:
  - preprocessing/corpus_browser/index.html
  - preprocessing/corpus_browser/doc_<id>.html  (one per document)
"""

import html
import json
from pathlib import Path
from typing import List, Dict, Any


REPO_ROOT = Path(__file__).resolve().parents[1]

PARSED_DATASET_PATH = REPO_ROOT / "preprocessing" / "parsed_dataset.jsonl"
BROWSER_ROOT = REPO_ROOT / "preprocessing" / "corpus_browser"


def load_corpus() -> List[Dict[str, Any]]:
    docs: List[Dict[str, Any]] = []
    with PARSED_DATASET_PATH.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            docs.append(obj)
    # sort by id for stable ordering
    docs.sort(key=lambda d: d.get("id", 0))
    return docs


def write_doc_page(doc: Dict[str, Any]) -> None:
    doc_id = doc["id"]
    title = doc.get("title") or f"Document {doc_id}"
    url = doc.get("url", "")
    doc_type = doc.get("type", "")
    doctype = doc.get("doctype", "")
    lang = doc.get("lang", "")
    status = doc.get("parse_status", "")
    full_text = doc.get("full_clean_text", "")

    # Escape HTML special chars in text
    title_h = html.escape(title)
    url_h = html.escape(url)
    type_h = html.escape(doc_type)
    doctype_h = html.escape(doctype or "")
    lang_h = html.escape(lang)
    status_h = html.escape(status)
    text_h = html.escape(full_text)

    out_path = BROWSER_ROOT / f"doc_{doc_id:06d}.html"

    html_doc = f"""<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8">
  <title>{title_h}</title>
</head>
<body>
  <h1>{title_h}</h1>
  <p><strong>URL:</strong> {url_h}</p>
  <p><strong>Type (logical):</strong> {type_h}</p>
  <p><strong>File type:</strong> {doctype_h}</p>
  <p><strong>Language:</strong> {lang_h}</p>
  <p><strong>Parse status:</strong> {status_h}</p>
  <hr>
  <pre>{text_h}</pre>
</body>
</html>
"""
    out_path.write_text(html_doc, encoding="utf-8")


def write_index_page(docs: List[Dict[str, Any]]) -> None:
    rows_html = []

    for doc in docs:
        doc_id = doc["id"]
        title = doc.get("title") or f"Document {doc_id}"
        doc_type = doc.get("type", "")
        doctype = doc.get("doctype", "")
        lang = doc.get("lang", "")
        status = doc.get("parse_status", "")

        title_h = html.escape(title)
        type_h = html.escape(doc_type)
        doctype_h = html.escape(doctype or "")
        lang_h = html.escape(lang)
        status_h = html.escape(status)

        link = f"doc_{doc_id:06d}.html"

        row = f"""
      <tr>
        <td>{doc_id}</td>
        <td>{title_h}</td>
        <td>{type_h}</td>
        <td>{doctype_h}</td>
        <td>{lang_h}</td>
        <td>{status_h}</td>
        <td><a href="{link}">Open</a></td>
      </tr>
"""
        rows_html.append(row)

    table_html = "\n".join(rows_html)

    html_index = f"""<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8">
  <title>Parsed Dataset Browser</title>
  <style>
    body {{
      font-family: system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      padding: 16px;
    }}
    table {{ border-collapse: collapse; width: 100%; }}
    th, td {{ border: 1px solid #ccc; padding: 4px 8px; }}
    th {{ background-color: #f0f0f0; }}
    tr:nth-child(even) {{ background-color: #fafafa; }}
    input.filter-input {{
      width: 95%;
      box-sizing: border-box;
      padding: 2px 4px;
      font-size: 12px;
    }}
    .filters-help {{
      font-size: 12px;
      color: #555;
      margin-bottom: 8px;
    }}
  </style>
  <script>
    function setupFilters() {{
      var table = document.getElementById("docs-table");
      if (!table) return;

      var tbody = table.tBodies[0];
      if (!tbody) return;

      var rows = Array.prototype.slice.call(tbody.rows);
      var inputs = document.querySelectorAll(".filter-input");

      function applyFilters() {{
        var filters = Array.prototype.map.call(inputs, function(inp) {{
          return inp.value.trim().toLowerCase();
        }});

        rows.forEach(function(row) {{
          var cells = row.cells;
          var visible = true;

          // filters correspond to columns:
          // 0: ID, 1: Title, 2: Type, 3: File type, 4: Lang, 5: Parse status
          for (var i = 0; i < filters.length; i++) {{
            var f = filters[i];
            if (!f) continue;  // empty filter -> ignore

            if (i >= cells.length) continue;

            var cellText = cells[i].textContent || "";
            cellText = cellText.toLowerCase();

            if (cellText.indexOf(f) === -1) {{
              visible = false;
              break;
            }}
          }}

          row.style.display = visible ? "" : "none";
        }});
      }}

      inputs.forEach(function(inp) {{
        inp.addEventListener("input", applyFilters);
      }});
    }}

    window.addEventListener("DOMContentLoaded", setupFilters);
  </script>
</head>
<body>
  <h1>Parsed Dataset Browser</h1>
  <p>Total documents: {len(docs)}</p>
  <p class="filters-help">
    Filters work on ID, Title, Type, File type, Lang, and Parse status.
    All non-empty filters are combined (logical AND).
  </p>
  <table id="docs-table">
    <thead>
      <tr>
        <th>ID</th>
        <th>Title</th>
        <th>Type</th>
        <th>File type</th>
        <th>Lang</th>
        <th>Parse status</th>
        <th>View</th>
      </tr>
      <tr>
        <th><input class="filter-input" placeholder="Filter ID"></th>
        <th><input class="filter-input" placeholder="Filter title"></th>
        <th><input class="filter-input" placeholder="Filter type"></th>
        <th><input class="filter-input" placeholder="Filter file type"></th>
        <th><input class="filter-input" placeholder="Filter lang"></th>
        <th><input class="filter-input" placeholder="Filter status"></th>
        <th></th>
      </tr>
    </thead>
    <tbody>
{table_html}
    </tbody>
  </table>
</body>
</html>
"""

    index_path = BROWSER_ROOT / "index.html"
    index_path.write_text(html_index, encoding="utf-8")


def generate_corpus_browser() -> None:
    if not PARSED_DATASET_PATH.exists():
        raise SystemExit(f"parsed_dataset.jsonl not found: {PARSED_DATASET_PATH}")

    BROWSER_ROOT.mkdir(parents=True, exist_ok=True)

    print(f"[*] Loading dataset from {PARSED_DATASET_PATH} ...")
    docs = load_corpus()
    print(f"[*] Loaded {len(docs)} documents")

    print(f"[*] Writing document pages to {BROWSER_ROOT} ...")
    for doc in docs:
        write_doc_page(doc)

    print("[*] Writing index.html ...")
    write_index_page(docs)

    print(f"[*] Done. Open {BROWSER_ROOT / 'index.html'} in your browser.")


if __name__ == "__main__":
    generate_corpus_browser()
