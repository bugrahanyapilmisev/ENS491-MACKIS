import os
import json
import pathlib

import pandas as pd
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

BASE_DIR = os.getenv("PROJECT_ROOT") or os.getcwd()
PREPROCESSING_DIR = os.getenv("PREPROCESSING_PATH") or os.path.join(BASE_DIR, "preprocessing")
PRE_ROOT = os.getenv("PREPROCESSED_ROOT") or os.path.join(PREPROCESSING_DIR, "preprocessed_docs")
CHUNKS_PARQUET = os.path.join(PREPROCESSING_DIR, "chunks_plus2.parquet")


@st.cache_data
def load_docs_index():
    rows = []
    for root, _, files in os.walk(PRE_ROOT):
        for fn in files:
            if not fn.endswith(".json"):
                continue
            full = os.path.join(root, fn)
            rel = os.path.relpath(full, PRE_ROOT).replace("\\", "/")
            try:
                data = json.loads(pathlib.Path(full).read_text(encoding="utf-8"))
            except Exception:
                continue
            rows.append(
                {
                    "rel_path": rel,
                    "source_path": data.get("source_path", ""),
                    "title": data.get("title", ""),
                    "lang": data.get("lang", ""),
                    "html_lang": data.get("html_lang", ""),
                    "text_len": len(data.get("text", "") or ""),
                }
            )
    if not rows:
        return pd.DataFrame(columns=["rel_path", "source_path", "title", "lang", "html_lang", "text_len"])
    df = pd.DataFrame(rows)
    return df.sort_values("source_path")


@st.cache_data
def load_chunks_df():
    if not os.path.exists(CHUNKS_PARQUET):
        return None
    try:
        return pd.read_parquet(CHUNKS_PARQUET)
    except Exception:
        return None


def main():
    st.title("Preprocessed Docs Browser")

    st.write(f"**PRE_ROOT:** `{PRE_ROOT}`")
    st.write(f"**CHUNKS_PARQUET:** `{CHUNKS_PARQUET}`")

    df_docs = load_docs_index()
    if df_docs.empty:
        st.warning("No preprocessed JSON files found. Check PRE_ROOT and run preprocess_docs.py first.")
        return

    # ---- Filters ----
    with st.sidebar:
        st.header("Filters")
        langs = ["(all)"] + sorted([x for x in df_docs["lang"].unique().tolist() if x])
        lang_sel = st.selectbox("Language", langs)
        search_q = st.text_input("Search in title / source_path (substring)", "")

    df_filtered = df_docs.copy()
    if lang_sel != "(all)":
        df_filtered = df_filtered[df_filtered["lang"] == lang_sel]

    if search_q:
        q = search_q.lower()
        mask = (
            df_filtered["title"].str.lower().str.contains(q, na=False)
            | df_filtered["source_path"].str.lower().str.contains(q, na=False)
        )
        df_filtered = df_filtered[mask]

    st.subheader(f"Documents ({len(df_filtered)}/{len(df_docs)})")
    st.dataframe(df_filtered.reset_index(drop=True))

    if df_filtered.empty:
        return

    # ---- Pick a doc ----
    idx = st.number_input(
        "Pick row index to inspect",
        min_value=0,
        max_value=len(df_filtered) - 1,
        value=0,
        step=1,
    )
    row = df_filtered.iloc[int(idx)]

    st.markdown("---")
    st.subheader("Selected document")
    st.write(f"**Source path:** `{row['source_path']}`")
    st.write(f"**Rel JSON path:** `{row['rel_path']}`")
    st.write(f"**Title:** {row['title']}")
    st.write(f"**Lang / HTML lang:** {row['lang']} / {row['html_lang']}")
    st.write(f"**Text length:** {row['text_len']} characters")

    json_path = os.path.join(PRE_ROOT, row["rel_path"])
    try:
        data = json.loads(pathlib.Path(json_path).read_text(encoding="utf-8"))
    except Exception as e:
        st.error(f"Could not read JSON: {e}")
        return

    with st.expander("Raw JSON"):
        st.json(data)

    # ---- Text preview ----
    st.subheader("Text preview")
    full_text = data.get("text", "") or ""
    max_chars = min(20000, len(full_text))
    if max_chars <= 0:
        st.info("No text in this document.")
    else:
        default = min(3000, max_chars)
        preview_len = st.slider(
            "Preview length (characters)",
            min_value=100,
            max_value=max_chars,
            value=default,
            step=100,
        )
        st.text(full_text[:preview_len])

    # ---- Show chunks for this doc (if built) ----
    chunks_df = load_chunks_df()
    if chunks_df is not None:
        st.markdown("---")
        st.subheader("Chunks for this document (if built)")
        possible = chunks_df[chunks_df["source_path"] == row["source_path"]]
        if possible.empty:
            st.info("No chunks found for this document in chunks_plus2.parquet.")
        else:
            doc_id = possible.iloc[0]["doc_id"]
            df_doc_chunks = chunks_df[chunks_df["doc_id"] == doc_id].sort_values("chunk_index")
            st.write(f"{len(df_doc_chunks)} chunks")
            st.dataframe(df_doc_chunks[["chunk_id", "chunk_index", "lang", "html_lang"]])
            cidx = st.number_input(
                "Chunk index to preview",
                min_value=0,
                max_value=max(0, len(df_doc_chunks) - 1),
                value=0,
                step=1,
                key="chunk_index_input",
            )
            chunk_row = df_doc_chunks.iloc[int(cidx)]
            st.text(chunk_row["text"])


if __name__ == "__main__":
    main()
