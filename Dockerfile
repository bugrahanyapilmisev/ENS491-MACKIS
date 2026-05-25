# ── Backend Dockerfile (FastAPI) ──────────────────────────────────────────────
FROM python:3.10-slim

# Sistem bağımlılıkları (psycopg2 için libpq gerekli)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libpq-dev \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Önce bağımlılıkları kopyala (cache'den faydalanmak için)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Uygulama kaynak kodunu kopyala
COPY . .

# Dışarıya 8000 portunu aç
EXPOSE 8000

# Üretim için uvicorn ile başlat
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]

