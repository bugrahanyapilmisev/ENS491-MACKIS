"""
select_focused_docs.py - Select ALL documents for full KG building

Creates selected_docs.json with ALL documents categorized by topic.
No documents are excluded — the KG should cover the entire corpus.
"""

import os
import json
import pandas as pd
from collections import defaultdict

# Load chunks
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(CURRENT_DIR)
CHUNKS_PATH = os.path.join(ROOT_DIR, "creating_database", "checkpoints_v2", "chunks_v3.parquet")

print("Loading chunks...")
df = pd.read_parquet(CHUNKS_PATH)
print(f"Total chunks: {len(df)}")

# Get unique documents
docs = df.groupby('source_path').agg({
    'title': 'first',
    'content': lambda x: ' '.join(x.astype(str)),
    'doc_type': 'first',
    'chunk_id': 'count'
}).reset_index()
docs.columns = ['source_path', 'title', 'content_preview', 'doc_type', 'chunk_count']
docs['content_preview'] = docs['content_preview'].str[:500]

print(f"Unique documents: {len(docs)}")

# Expanded keywords for comprehensive categorization
CATEGORY_KEYWORDS = {
    "erasmus": ["erasmus", "staj", "internship", "exchange", "hareketlilik",
                "mobility", "değişim programı", "exchange program"],
    "library": ["kütüphane", "library", "ödünç", "kitap", "ill", "borrowing",
                "bilgi merkezi", "information center", "circulation"],
    "discipline": ["disiplin", "discipline", "ceza", "penalty", "soruşturma",
                   "investigation", "uzaklaştırma", "suspension"],
    "scholarship": ["burs", "scholarship", "mali destek", "financial aid",
                    "bağış", "donation"],
    "registration": ["kayıt", "mezuniyet", "registration", "graduation",
                     "diploma", "transkript", "transcript", "ilişik kesme"],
    "graduate": ["lisansüstü", "graduate", "yüksek lisans", "doktora", "tez",
                 "thesis", "enstitü", "institute", "master", "phd"],
    "undergraduate": ["lisans", "undergraduate", "çift anadal", "yandal",
                      "minor", "double major", "ders", "course", "kredi"],
    "housing": ["yurt", "dormitory", "konaklama", "housing", "residence",
                "lojman"],
    "food_services": ["yemek", "kafeterya", "cafeteria", "food", "yiyecek",
                      "içecek"],
    "transportation": ["ulaşım", "servis", "shuttle", "araç", "transport",
                       "otopark", "parking"],
    "health_safety": ["sağlık", "güvenlik", "isg", "acil", "safety", "health",
                      "iş sağlığı", "occupational", "yangın", "fire"],
    "it_services": ["bilgi teknoloji", "bilişim", "yazılım", "laptop",
                    "network", "ağ", "e-posta", "mysu"],
    "hr_employment": ["insan kaynakları", "personel", "atama", "izin",
                      "çalışan", "human resources", "appointment", "leave"],
    "procurement": ["satın alma", "ihale", "procurement", "tender", "tedarik",
                    "supply"],
    "research": ["araştırma", "proje", "research", "fund", "fon", "patent",
                 "sınai mülkiyet", "intellectual property"],
    "quality_management": ["kalite", "quality", "iso", "akreditasyon",
                          "accreditation", "iç denetim", "audit"],
    "student_council": ["öğrenci konseyi", "student council", "öğrenci birliği",
                       "student union", "kulüp", "club"],
    "environment": ["çevre", "environment", "atık", "waste", "sera gazı",
                    "greenhouse", "enerji", "energy", "sürdürülebilirlik"],
}

def match_category(title, content_preview):
    """Match document to category based on keywords."""
    text = f"{title} {content_preview}".lower()

    for category, keywords in CATEGORY_KEYWORDS.items():
        for kw in keywords:
            if kw in text:
                return category

    return "other"

# Categorize documents
docs['category'] = docs.apply(
    lambda row: match_category(row['title'], row['content_preview']), axis=1
)

# Print category distribution
print("\nCategory distribution:")
for cat, count in docs['category'].value_counts().items():
    print(f"  {cat}: {count}")

# Select ALL documents — no filtering, no limits
selected = defaultdict(list)

for _, row in docs.iterrows():
    category = row['category']
    selected[category].append({
        "source_path": row['source_path'],
        "title": row['title'],
        "chunk_count": int(row['chunk_count'])
    })

# Create output
output = {
    "description": "ALL documents for full KG building",
    "total_categories": len(CATEGORY_KEYWORDS),
    "total_docs": sum(len(v) for v in selected.values()),
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
output_path = os.path.join(CURRENT_DIR, "selected_docs.json")
with open(output_path, "w", encoding="utf-8") as f:
    json.dump(output, f, ensure_ascii=False, indent=2)

print(f"\nSaved to {output_path}")
print(f"Total documents: {output['total_docs']}")
print(f"Total chunks to process: {total_chunks}")
