"""
Diagnose why certain docs fail tagging — fetch their text from ChromaDB and show it.
"""
import chromadb

CHROMA_DIR = r"C:\bitirme3\ENS491-MACKIS\creating_database\chroma_db_v2"
COLL_NAME  = "mysu_v3_qwen3"

# Sample of docs that showed ERR (no tags generated)
FAILING_SRCS = [
    "108.html", "1109.html", "1143.html", "458.html", "470.html",
    "499.html", "506.html", "569.html", "581.html", "583.html",
    "586.html", "61.html", "71.html", "723.html",
    "corrective-action-procedure-pgs-s620-03-01.html",
    "call-management-procedure-pit-s140-0101.html",
    "cip-101-course-procedure-pcip-a220-01-01.html",
    "change-management-procedure-pgs-s620-03-02.html",
    "architectural-and-construction-works-procedur",
    "authority-index-creation-and-development-proc",
]

client = chromadb.PersistentClient(path=CHROMA_DIR)
coll   = client.get_collection(COLL_NAME)

result = coll.get(include=["documents", "metadatas"])
ids    = result["ids"]
docs   = result["documents"]
metas  = result["metadatas"]

# Build source_path -> chunks map
src_map = {}
for i in range(len(ids)):
    src = metas[i].get("source_path", "")
    basename = src.split("/")[-1].split("\\")[-1]
    for fail in FAILING_SRCS:
        if fail in basename or fail in src:
            if src not in src_map:
                src_map[src] = []
            src_map[src].append(docs[i] or "")

# Print results
found = 0
for fail in FAILING_SRCS:
    match = next((s for s in src_map if fail in s), None)
    if not match:
        print(f"\n❌ NOT FOUND in ChromaDB: {fail}")
        continue
    texts = src_map[match]
    combined = " ".join(texts)
    print(f"\n{'='*60}")
    print(f"📄 {fail}")
    print(f"   Chunks: {len(texts)} | Total chars: {len(combined)}")
    print(f"   Preview: {combined[:300].strip()!r}")
    found += 1

print(f"\n\nFound {found}/{len(FAILING_SRCS)} failing docs in ChromaDB")
