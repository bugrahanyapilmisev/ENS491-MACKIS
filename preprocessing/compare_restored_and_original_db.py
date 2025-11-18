# compare_chroma_dbs_plus2.py
import os
import json
import numpy as np
import chromadb

# ==== EDIT HERE ====
CHROMA_DIR_ORIG = r"C:\\Users\\kosot\\Documents\\bitirme\\preprocessing\\chroma_db_mysu"
CHROMA_DIR_NEW  = r"C:\\Users\\kosot\\Documents\\bitirme\\preprocessing\\chroma_db_mysu_RESTORE"
COLL_NAME       = "mysu_surecharitasi_bge_m3_v1"

BATCH           = 1000
MAX_MISMATCHES_TO_SHOW = 10
# ====================


def load_collection(path, coll_name):
    client = chromadb.PersistentClient(path=path)
    coll = client.get_collection(name=coll_name)
    total = coll.count()
    print(f"[{path}] collection '{coll_name}' -> count={total}")

    ids_list = []
    docs_dict = {}
    metas_dict = {}
    embs_dict = {}

    offset = 0
    while offset < total:
        n = min(BATCH, total - offset)
        # 'ids' include listine yazılmıyor, zaten her zaman geliyor
        res = coll.get(
            limit=n,
            offset=offset,
            include=["embeddings", "documents", "metadatas"],
        )
        for i, cid in enumerate(res["ids"]):
            ids_list.append(cid)
            docs_dict[cid] = res["documents"][i]
            metas_dict[cid] = res["metadatas"][i]
            embs_dict[cid] = np.array(res["embeddings"][i], dtype=np.float32)
        offset += n
        print(f"  loaded {offset}/{total} from {path}")
    return ids_list, docs_dict, metas_dict, embs_dict


def main():
    # --- load both collections fully into memory ---
    ids1, docs1, metas1, embs1 = load_collection(CHROMA_DIR_ORIG, COLL_NAME)
    ids2, docs2, metas2, embs2 = load_collection(CHROMA_DIR_NEW,  COLL_NAME)

    # 1) Count check
    if len(ids1) != len(ids2):
        print(f"❌ Count mismatch: orig={len(ids1)} new={len(ids2)}")
        return
    else:
        print(f"✅ Same vector count: {len(ids1)}")

    # 2) ID set check
    set1 = set(ids1)
    set2 = set(ids2)
    if set1 != set2:
        print("❌ ID sets differ!")
        print("  In orig but not in new (first 10):", list(set1 - set2)[:10])
        print("  In new but not in orig (first 10):", list(set2 - set1)[:10])
        return
    else:
        print("✅ Same ID set")

    # 3) Detailed content check (docs, metas, embeddings)
    mismatches = 0

    for cid in ids1:
        d1 = docs1[cid]
        d2 = docs2[cid]
        if d1 != d2:
            mismatches += 1
            print(f"❌ Document mismatch for id={cid}")
            if mismatches >= MAX_MISMATCHES_TO_SHOW:
                break
            continue

        m1 = metas1[cid]
        m2 = metas2[cid]
        if m1 != m2:
            mismatches += 1
            print(f"❌ Metadata mismatch for id={cid}")
            print("  orig:", json.dumps(m1, ensure_ascii=False)[:300])
            print("  new :", json.dumps(m2, ensure_ascii=False)[:300])
            if mismatches >= MAX_MISMATCHES_TO_SHOW:
                break
            continue

        v1 = embs1[cid]
        v2 = embs2[cid]
        if not np.allclose(v1, v2, atol=1e-6):
            mismatches += 1
            diff = np.max(np.abs(v1 - v2))
            print(f"❌ Embedding mismatch for id={cid}, max abs diff={diff}")
            if mismatches >= MAX_MISMATCHES_TO_SHOW:
                break

    if mismatches == 0:
        print("✅ All docs, metadatas and embeddings match within tolerance.")
    else:
        print(f"⚠ Found {mismatches} mismatches (shown up to {MAX_MISMATCHES_TO_SHOW}).")


if __name__ == "__main__":
    main()
