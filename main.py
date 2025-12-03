# backend/main.py
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from database import engine, Base
import models
from routers import chat, auth

# Tabloları oluştur
Base.metadata.create_all(bind=engine)

app = FastAPI()

# --- GÜNCELLENMİŞ CORS AYARLARI ---
# Frontend'in gelebileceği tüm adresleri buraya yazıyoruz
origins = [
    "http://localhost:5173",      # Vite varsayılanı
    "http://127.0.0.1:5173",      # Vite bazen IP ile çalışır
    "http://localhost:3000",      # Alternatif
    "http://127.0.0.1:3000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # <--- YILDIZ KOYDUK (Her yerden gelen isteği kabul et)
    allow_credentials=True,
    allow_methods=["*"],  # Tüm metodlara izin ver (GET, POST, OPTIONS vs.)
    allow_headers=["*"],  # Tüm başlıklara izin ver
)
# ----------------------------------

app.include_router(chat.router)
app.include_router(auth.router)

@app.get("/")
def read_root():
    return {"status": "MACKIS RAG Backend Aktif 🚀"}