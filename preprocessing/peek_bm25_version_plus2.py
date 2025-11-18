# peek_bm25_version_plus2.py
import argparse, pickle, collections

def main():
    ap = argparse.ArgumentParser("Peek BM25 pickle")
    ap.add_argument("--bm25-pickle", default="C:\\Users\\kosot\\Documents\\bitirme\\preprocessing\\bm25_plus2.pkl")
    args = ap.parse_args()

    with open(args.bm25_pickle, "rb") as f:
        blob = pickle.load(f)
    print(blob.keys())
    n = len(blob["doc_langs"])
    print(f"N docs: {n}")
    langs = collections.Counter(blob.get("doc_langs", []))
    print("Lang counts:", dict(langs))

if __name__ == "__main__":
    main()
