"""
select_focused_docs.py - Select documents for focused hybrid KG testing

Creates selected_docs.json with key documents + noise documents.
"""

import os
import json
import pandas as pd
from collections import defaultdict

# Load chunks
# Load chunks
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(CURRENT_DIR)
CHUNKS_PATH = os.path.join(ROOT_DIR, "creating_database", "checkpoints_v2", "chunks_v2.parquet")

print("Loading chunks...")
df = pd.read_parquet(CHUNKS_PATH)
print(f"Total chunks: {len(df)}")

# Get unique documents
docs = df.groupby('source_path').agg({
    'title': 'first',
    'content': lambda x: ' '.join(x.astype(str)),
    'doc_type': 'first',
    'tags': 'first',
    'chunk_id': 'count'
}).reset_index()
docs.columns = ['source_path', 'title', 'content_preview', 'doc_type', 'tags', 'chunk_count']
docs['content_preview'] = docs['content_preview'].str[:500]

print(f"Unique documents: {len(docs)}")

# Keywords for each category
CATEGORY_KEYWORDS = {
    "erasmus": ["erasmus", "staj", "internship", "exchange", "hareketlilik", "mobility"],
    "library": ["kütüphane", "library", "ödünç", "kitap", "ill", "borrowing"],
    "discipline": ["disiplin", "discipline", "ceza", "penalty", "soruşturma"],
    "scholarship": ["burs", "scholarship", "mali destek", "financial"],
    "registration": ["kayıt", "mezuniyet", "registration", "graduation"],
    "graduate": ["lisansüstü", "graduate", "yüksek lisans", "doktora", "tez"],
}

# Noise categories (unrelated to test questions)
NOISE_KEYWORDS = ["ihale", "satın alma", "insan kaynakları", "it", "bilgi teknoloji", "ulaşım", "araç"]

def match_category(title, content_preview):
    """Match document to category based on keywords."""
    text = f"{title} {content_preview}".lower()
    
    for category, keywords in CATEGORY_KEYWORDS.items():
        for kw in keywords:
            if kw in text:
                return category
    
    # Check if noise
    for kw in NOISE_KEYWORDS:
        if kw in text:
            return "noise"
    
    return "other"

# Categorize documents
docs['category'] = docs.apply(lambda row: match_category(row['title'], row['content_preview']), axis=1)

# Print category distribution
print("\nCategory distribution:")
for cat, count in docs['category'].value_counts().items():
    print(f"  {cat}: {count}")

# Select documents
selected = defaultdict(list)

# Select key documents (max 15 per category)
for category in CATEGORY_KEYWORDS.keys():
    cat_docs = docs[docs['category'] == category].head(15)
    for _, row in cat_docs.iterrows():
        selected[category].append({
            "source_path": row['source_path'],
            "title": row['title'],
            "chunk_count": int(row['chunk_count'])
        })
    print(f"Selected {len(selected[category])} docs for {category}")

# Select noise documents (max 30)
noise_docs = docs[docs['category'] == 'noise'].head(30)
for _, row in noise_docs.iterrows():
    selected['noise'].append({
        "source_path": row['source_path'],
        "title": row['title'],
        "chunk_count": int(row['chunk_count'])
    })
print(f"Selected {len(selected['noise'])} noise docs")

# Create output
output = {
    "description": "Selected documents for focused hybrid KG testing",
    "total_categories": len(CATEGORY_KEYWORDS),
    "total_key_docs": sum(len(v) for k, v in selected.items() if k != 'noise'),
    "total_noise_docs": len(selected['noise']),
    "documents": dict(selected)
}

# Calculate total chunks
all_paths = []
for category, doc_list in selected.items():
    for doc in doc_list:
        all_paths.append(doc['source_path'])

total_chunks = df[df['source_path'].isin(all_paths)]['chunk_id'].count()
output['total_chunks'] = int(total_chunks)

# Save
output_path = os.path.join(PREPROCESSING_DIR, "selected_docs.json")
with open(output_path, "w", encoding="utf-8") as f:
    json.dump(output, f, ensure_ascii=False, indent=2)

print(f"\nSaved to {output_path}")
print(f"Total chunks to process: {total_chunks}")
