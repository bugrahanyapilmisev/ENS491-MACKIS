# build_bm25_plus2.py
import os
import re
import pickle
from typing import List
import numpy as np
import pandas as pd
from rank_bm25 import BM25Okapi

# MUST match your existing config
CHECKPOINT_DIR = r"C:\\Users\\kosot\\Documents\\bitirme\\preprocessing\\checkpoints_plus2"
CHUNK_PARQUET  = os.path.join(CHECKPOINT_DIR, "chunks_plus2.parquet")

BM25DIR = r"C:\\Users\\kosot\\Documents\\bitirme\\preprocessing"
BM25_PKL       = os.path.join(BM25DIR, "bm25_plus2.pkl")

os.makedirs(CHECKPOINT_DIR, exist_ok=True)

TOKEN_RE = re.compile(r"\w+", flags=re.UNICODE)

def tokenize(text: str) -> List[str]:
    text = (text or "").lower()
    return TOKEN_RE.findall(text)

def main():
    print(f"Loading chunks from: {CHUNK_PARQUET}")
    df = pd.read_parquet(CHUNK_PARQUET)

    # safety checks
    required_cols = {"chunk_id", "content", "doc_path", "title", "doc_lang"}
    missing = required_cols - set(df.columns)
    if missing:
        raise RuntimeError(f"Missing columns in chunks parquet: {missing}")

    texts      = df["content"].astype(str).tolist()
    chunk_ids  = df["chunk_id"].astype(str).tolist()
    doc_paths  = df["doc_path"].astype(str).tolist()
    titles     = df["title"].astype(str).tolist()
    doc_langs  = df["doc_lang"].astype(str).tolist()

    print(f"Total chunks: {len(texts)}")

    # Tokenize all docs
    tokenized_corpus = []
    for i, t in enumerate(texts):
        tokenized = tokenize(t)
        tokenized_corpus.append(tokenized)
        if (i + 1) % 10000 == 0:
            print(f"[tokenize] {i+1}/{len(texts)}")

    print("Building BM25 index...")
    bm25 = BM25Okapi(tokenized_corpus)
    print("BM25 index built.")

    bm25_pack = {
        "bm25": bm25,
        "chunk_ids": chunk_ids,
        "texts": texts,
        "doc_paths": doc_paths,
        "titles": titles,
        "doc_langs": doc_langs,
    }

    with open(BM25_PKL, "wb") as f:
        pickle.dump(bm25_pack, f, protocol=pickle.HIGHEST_PROTOCOL)

    print(f"✅ BM25 index saved to: {BM25_PKL}")
    print(f"Indexed chunks: {len(chunk_ids)}")

if __name__ == "__main__":
    main()
