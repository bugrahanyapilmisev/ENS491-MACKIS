# backend/services/rag_service.py
import os
import sys
from typing import List, Dict

# Kendi dizinimizdeki modülü import edebilmek için
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Senin 700 satırlık doslandan gerekli fonksiyonları çekiyoruz
from rag_core import (
    get_chroma_collection, 
    load_bm25_index, 
    load_doc_chunk_stats, 
    answer_with_rag,
    CHROMA_DIR, 
    BM25_PKL
)

class RAGService:
    def __init__(self):
        print("🚀 Gelişmiş RAG Motoru (MMR + Hybrid) Yükleniyor...")
        
        # 1. Senin kodundaki yükleme fonksiyonlarını kullanıyoruz
        # Dosya yollarının (Path) doğru olduğundan emin ol!
        if not os.path.exists(CHROMA_DIR) or not os.path.exists(BM25_PKL):
            print("⚠️ UYARI: Chroma veya BM25 dosyaları bulunamadı! rag_core.py içindeki yolları kontrol et.")
        
        self.coll = get_chroma_collection()
        self.bm25_pack = load_bm25_index()
        load_doc_chunk_stats() # Global değişkeni doldurur
        
        print("✅ RAG Motoru Hazır!")

    def query(self, user_query: str, history: List[Dict] = []):
        """
        FastAPI'den gelen isteği senin orijinal fonksiyonuna iletir.
        """
        print(f"🔍 Analiz Ediliyor (chroma-mmr): {user_query}")
        
        try:
            # Senin gelişmiş fonksiyonunu çağırıyoruz
            # mode="hybrid-mmr" olarak sabitledim, istersen değiştirebilirsin.
            answer_text = answer_with_rag(
                query=user_query,
                mode="chroma-mmr", 
                bm25_pack=self.bm25_pack,
                coll=self.coll,
                history=history # Az önce eklediğimiz parametre
            )
            
            # Senin kodun şu an sadece 'answer' dönüyor, kaynakları (sources) return etmiyor.
            # Eğer kaynakları da Frontend'de göstermek istersen rag_core.py'yi 
            # (answer, retrieved_docs) döndürecek şekilde güncellemen gerekir.
            # Şimdilik kaynakları boş dönüyoruz hata vermesin diye.
            sources = [] 
            
            return answer_text, sources

        except Exception as e:
            print(f"❌ RAG Core Hatası: {e}")
            return "Üzgünüm, sistemi çalıştırırken teknik bir hata oluştu.", []

# Global Instance
rag_engine = RAGService()