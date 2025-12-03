# build_graph.py
import sys
import os
import pandas as pd
from tqdm import tqdm

# Path ayarları (backend modüllerini bulması için)
sys.path.append(os.path.join(os.getcwd(), "backend"))

from database import SessionLocal
from services.graph_service import GraphService
from dotenv import load_dotenv

load_dotenv("backend/.env")

CHUNK_PATH = os.path.join(os.getenv("PREPROCESSING_PATH"), "checkpoints_plus2", "chunks_plus2.parquet")

def main():
    print("🚀 Knowledge Graph İnşası Başlıyor...")
    
    if not os.path.exists(CHUNK_PATH):
        print(f"❌ Parquet dosyası bulunamadı: {CHUNK_PATH}")
        return

    # Veriyi Oku
    df = pd.read_parquet(CHUNK_PATH)
    print(f"📄 Toplam {len(df)} chunk var. İşlem başlıyor...")

    db = SessionLocal()
    graph_service = GraphService(db)

    # Örnek olarak ilk 50 chunk'ı işleyelim (Hepsini yapmak saatler sürer, önce test et)
    sample_df = df.head(50) 

    for index, row in tqdm(sample_df.iterrows(), total=len(sample_df)):
        text = row['content']
        # Metadata'dan varsa doc_id alabiliriz, şimdilik None geçiyoruz
        
        # 1. LLM ile Çıkarım Yap
        triples = graph_service.extract_triples_from_text(text)
        
        # 2. Veritabanına Kaydet
        if triples:
            graph_service.save_triples_to_db(triples)

    print("✅ Graph İnşası Tamamlandı!")
    db.close()

if __name__ == "__main__":
    main()