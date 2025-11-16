# peek_bm25_plus.py
import pickle

PICKLE = r"C:\\Users\\kosot\\Documents\\bitirme\\preprocessing\\version_plus\\bm25_index.pkl"
with open(PICKLE, "rb") as f:
    data = pickle.load(f)

bm25, ids, docs = data["bm25"], data["ids"], data["docs"]

def tok(x): return [t.lower() for t in x.split()]
query = "Erasmus ders verme hareketliliği belgeler"
scores = bm25.get_scores(tok(query))
top = sorted(zip(scores, ids, docs), reverse=True)[:5]
for s, i, d in top:
    print(round(float(s),3), i, (d[:160] + ("..." if len(d) > 160 else "")).replace("\n"," "))
