# check_lang_dist.py
import pandas as pd
from collections import Counter

CHUNKS = r"C:\\Users\\kosot\\Documents\\bitirme\\preprocessing\\checkpoints_plus2\\chunks_plus2.parquet"

df = pd.read_parquet(CHUNKS)

doc_lang_by_path = (
    df.groupby("doc_path")["doc_lang"]
      .first()
      .fillna("?")
      .tolist()
)

print("Doc-level doc_lang counts:", Counter(doc_lang_by_path))
print("Total docs:", len(doc_lang_by_path))
