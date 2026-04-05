# backend/services/rag_service.py
import os
import sys
from typing import List, Dict

# Kendi dizinimizdeki modülü import edebilmek için path ayarı
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# --- DÜZELTME: BM25 İMPORTLARINI KALDIRDIK ---
# Yeni rag_core.py dosyasında artık sadece bunlar var:
from rag_core import (
    get_chroma_collection,
    load_doc_chunk_stats,
    answer_with_rag,
    CHROMA_DIR
)

class RAGService:
    def __init__(self):
        print("🚀 Gelişmiş RAG Motoru (Chroma + Cross-Encoder) Yükleniyor...")
        
        # 1. Dosya Kontrolü
        if not os.path.exists(CHROMA_DIR):
            print(f"⚠️ UYARI: ChromaDB klasörü bulunamadı: {CHROMA_DIR}")
            print("Lütfen .env dosyasındaki PREPROCESSING_PATH ayarını kontrol et.")
        
        # 2. ChromaDB Bağlantısı
        try:
            self.coll = get_chroma_collection()
            print("✅ ChromaDB Bağlantısı Başarılı.")
        except Exception as e:
            print(f"❌ ChromaDB Hatası: {e}")
            self.coll = None

        # 3. İstatistikleri Yükle (BM25 artık yok)
        try:
            load_doc_chunk_stats()
            print("✅ Döküman İstatistikleri Yüklendi.")
        except Exception as e:
            print(f"⚠️ İstatistik yükleme uyarısı: {e}")
        
        print("✅ RAG Motoru Hazır!")

    def query(self, user_query: str, history: List[Dict] = []):
        """
        FastAPI'den gelen isteği rag_core'a iletir.
        """
        print(f"🔍 Analiz Ediliyor (Reranker): {user_query}")
        
        if not self.coll:
            return "Veritabanı bağlantısı olmadığı için cevap veremiyorum.", []

        try:
            # --- DÜZELTME: PARAMETRELERİ GÜNCELLEDİK ---
            # Yeni answer_with_rag fonksiyonu 'bm25_pack' parametresi ALMIYOR.
            answer_text = answer_with_rag(
                query=user_query,
                mode="chroma-mmr",  # Yeni sistemin varsayılan modu
                coll=self.coll,
                history=history
            )
            
            # Not: Şu anki rag_core.py sadece metin (string) dönüyor.
            # Kaynakları (sources) da döndürmek istersen rag_core.py'yi düzenlemen gerekir.
            # Şimdilik boş liste dönüyoruz.
            sources = [] 
            
            return answer_text, sources

        except Exception as e:
            print(f"❌ RAG Core Hatası: {e}")
            # Hatanın detayını konsola bas ki görelim
            import traceback
            traceback.print_exc()
            return "Üzgünüm, sistemi çalıştırırken teknik bir hata oluştu.", []

# Global Instance
rag_engine = RAGService()