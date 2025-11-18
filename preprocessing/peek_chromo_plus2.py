# peek_chromo_plus2.py
import argparse, chromadb, random, collections

def main():
    ap = argparse.ArgumentParser("Peek Chroma collection")
    ap.add_argument("--chroma-dir", default="C:\\Users\\kosot\\Documents\\bitirme\\preprocessing\\chroma_db_mysu")
    ap.add_argument("--collection", default="mysu_surecharitasi_bge_m3_v1")
    ap.add_argument("--sample", type=int, default=5)
    args = ap.parse_args()

    client = chromadb.PersistentClient(path=args.chroma_dir)
    coll = client.get_or_create_collection(name=args.collection)

    # Chroma doesn't have a count API; approximate via get(limit) if supported, else rely on persisted ids
    all_ids = coll.get(include=["metadatas"], limit=1000000).get("ids", [])
    metas = coll.get(ids=all_ids[:min(len(all_ids),20000)], include=["metadatas"]).get("metadatas", [])
    cnt = len(all_ids)
    langs = collections.Counter([(m.get("content_lang") or "?") for m in metas])

    print(f"Total vectors: {cnt}")
    print("Lang (first 20k sample):", dict(langs))

    if all_ids:
        for _ in range(min(args.sample, len(all_ids))):
            i = random.randint(0, len(all_ids)-1)
            doc = coll.get(ids=[all_ids[i]], include=["documents","metadatas"])
            print("\n--- sample ---")
            print(doc["metadatas"][0].get("doc_path"))
            print("lang=", doc["metadatas"][0].get("content_lang"),
                  "chunk_index=", doc["metadatas"][0].get("chunk_index"))
            print((doc["documents"][0] or "")[:250], "…")

if __name__ == "__main__":
    main()
