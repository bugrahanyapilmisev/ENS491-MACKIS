# build_bm25_from_parquet.py
import os, json, pickle
import pyarrow.dataset as ds
from rank_bm25 import BM25Okapi  # pip install rank-bm25

PARQUET_DIR   = r"C:\\Users\\kosot\\Documents\\bitirme\\preprocessing\\version_plus\\checkpoints_plus\\parquet_bge"
OUT_PICKLE    = r"C:\\Users\\kosot\\Documents\\bitirme\\preprocessing\\version_plus\\bm25_index.pkl"

def iter_rows(parquet_dir):
    d = ds.dataset(parquet_dir, format="parquet")
    tbl = d.to_table()  # loads all; if memory tight, read file-by-file with ds.dataset(parquet_dir).files
    data = tbl.to_pydict()
    for i in range(len(data["id"])):
        yield data["id"][i], data["document"][i]

def simple_tokenize(s: str):
    return [t.lower() for t in s.split()]

def main():
    ids, docs, corpus = [], [], []
    for _id, _doc in iter_rows(PARQUET_DIR):
        ids.append(_id); docs.append(_doc); corpus.append(simple_tokenize(_doc))

    bm25 = BM25Okapi(corpus)
    with open(OUT_PICKLE, "wb") as f:
        pickle.dump({"ids": ids, "docs": docs, "bm25": bm25}, f)
    print("Saved BM25:", OUT_PICKLE, f"(docs={len(ids)})")

if __name__ == "__main__":
    main()
