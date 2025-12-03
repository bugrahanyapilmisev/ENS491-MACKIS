# backend/database.py
import os
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base
from dotenv import load_dotenv 

# .env dosyasını yükle
load_dotenv()

# URL'i .env dosyasından çek
SQLALCHEMY_DATABASE_URL = os.getenv("DATABASE_URL")

if not SQLALCHEMY_DATABASE_URL:
    raise ValueError("DATABASE_URL .env dosyasında bulunamadı!")

# Supabase/Postgres için bağlantı ayarı
# GÜNCELLENEN KISIM BURASI 👇
engine = create_engine(
    SQLALCHEMY_DATABASE_URL,
    pool_pre_ping=True,  # <-- EN ÖNEMLİSİ: Bağlantı koparsa otomatik tekrar dener
    pool_size=10,        # Havuzda tutulacak bağlantı sayısı
    max_overflow=20,     # Yoğunlukta açılacak ekstra bağlantı limiti
    pool_recycle=1800    # Bağlantıları 30 dakikada bir yenile (bayatlamayı önler)
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()