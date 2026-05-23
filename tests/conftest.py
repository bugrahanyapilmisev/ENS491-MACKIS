# tests/conftest.py
"""
Shared pytest fixtures for MACKIS backend tests.

Strateji:
  - Supabase'deki mevcut 'postgres' veritabanına bağlanılır (tablolar zaten var).
  - create_all / drop_all ÇAĞRILMAZ — production verisi asla bozulmaz.
  - Her test fonksiyonu kendi transaction'ında çalışır; bittikten sonra
    ROLLBACK yapılır → hiçbir veri DB'ye commit edilmez, DB temiz kalır.
  - RAG engine test ortamında hiç başlatılmaz (mock ile kapatılır).
  - Bağlantı: Supabase Direct Connection (port 5432), PgBouncer değil.
"""

import os
import uuid
from datetime import timedelta

import pytest
from dotenv import load_dotenv
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
from fastapi.testclient import TestClient

# ── .env.test'i yükle (üretim .env'i ezmeden) ────────────────────────────────
load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), "..", ".env.test"), override=True)

# ── Env var'ları modüller import edilmeden ÖNCE set et ───────────────────────
TEST_DATABASE_URL = os.environ.get("TEST_DATABASE_URL")
if not TEST_DATABASE_URL:
    raise EnvironmentError(
        ".env.test dosyasında TEST_DATABASE_URL bulunamadı.\n"
        "Supabase Dashboard → Settings → Database → Direct connection URL'ini kullanın.\n"
        "Format: postgresql://postgres:SIFRE@db.XXXX.supabase.co:5432/postgres"
    )

# JWT_SECRET_KEY utils.py import edilmeden önce ortamda olmalı
os.environ.setdefault("JWT_SECRET_KEY", os.environ["JWT_SECRET_KEY"])
os.environ["DATABASE_URL"] = TEST_DATABASE_URL  # database.py bunu okur

# ── Şimdi proje modüllerini import edebiliriz ─────────────────────────────────
from database import Base, get_db   # noqa: E402
from main import app                # noqa: E402
import models                       # noqa: E402
from utils import create_access_token, get_password_hash  # noqa: E402


# =============================================================================
# SESSION-SCOPED: Engine + Tablo Oluşturma
# =============================================================================

@pytest.fixture(scope="session")
def engine():
    """
    Test session başında bir kez çalışır:
      - Supabase Direct Connection (port 5432) ile bağlan.
      - Tablolar Supabase'de zaten mevcut; create_all / drop_all ÇAĞRILMAZ.
      - drop_all çağrılsaydı production verileri silinirdi — kesinlikle yapılmaz.
    """
    eng = create_engine(
        TEST_DATABASE_URL,
        pool_pre_ping=True,
    )
    yield eng
    eng.dispose()


# =============================================================================
# FUNCTION-SCOPED: Transaction Rollback Isolation
# =============================================================================

@pytest.fixture()
def db(engine):
    """
    Her test için ayrı bir transaction açar.
    Test bitince ROLLBACK yapılır → DB bir sonraki test için temiz.
    Bu yöntem drop/create'den çok daha hızlıdır.
    """
    connection = engine.connect()
    transaction = connection.begin()

    TestSession = sessionmaker(bind=connection, autocommit=False, autoflush=False)
    session = TestSession()

    yield session

    session.close()
    transaction.rollback()
    connection.close()


# =============================================================================
# FUNCTION-SCOPED: FastAPI TestClient
# =============================================================================

@pytest.fixture()
def client(db, mocker):
    """
    FastAPI TestClient:
      - get_db dependency'si test session'ına yönlendirilir.
      - RAG engine mock'lanır (ChromaDB'ye bağlanmaya çalışmaz).
    """
    # RAG engine'i global olarak mock'la — chat endpoint'i çağıran testler
    # bu mock'u dilediği gibi override edebilir.
    mocker.patch(
        "routers.chat.rag_engine.query",
        return_value=(
            "Mock cevap: test yanıtı.",
            [],
            {"retrieved_chunks": [], "context_chunks": [], "analysis": {"language": "tr"}},
        ),
    )

    def override_get_db():
        yield db

    app.dependency_overrides[get_db] = override_get_db
    with TestClient(app, raise_server_exceptions=True) as c:
        yield c
    app.dependency_overrides.clear()


# =============================================================================
# TEST VERİSİ: Kullanıcı & Token
# =============================================================================

@pytest.fixture()
def test_user(db):
    """Standart öğrenci kullanıcısı — her testte sıfırdan oluşturulur."""
    user = models.User(
        user_id=uuid.uuid4(),
        email="student@sabanciuniv.edu",
        display_name="Test Öğrencisi",
        password_hash=get_password_hash("password123"),
        role="undergrad",
        status="active",
    )
    db.add(user)
    db.commit()
    db.refresh(user)
    return user


@pytest.fixture()
def admin_user(db):
    """Admin kullanıcısı."""
    user = models.User(
        user_id=uuid.uuid4(),
        email="admin@sabanciuniv.edu",
        display_name="Test Admin",
        password_hash=get_password_hash("adminpass"),
        role="admin",
        status="active",
    )
    db.add(user)
    db.commit()
    db.refresh(user)
    return user


@pytest.fixture()
def other_user(db):
    """Başka bir kullanıcı — izolasyon testleri için."""
    user = models.User(
        user_id=uuid.uuid4(),
        email="other@sabanciuniv.edu",
        display_name="Başka Kullanıcı",
        password_hash=get_password_hash("otherpass"),
        role="undergrad",
        status="active",
    )
    db.add(user)
    db.commit()
    db.refresh(user)
    return user


@pytest.fixture()
def auth_token(test_user):
    """test_user için geçerli JWT token."""
    return create_access_token(
        data={"sub": test_user.email, "user_id": str(test_user.user_id)},
        expires_delta=timedelta(minutes=30),
    )


@pytest.fixture()
def auth_headers(auth_token):
    """Authorization header dict'i — client.get/post'a doğrudan verilebilir."""
    return {"Authorization": f"Bearer {auth_token}"}


@pytest.fixture()
def admin_token(admin_user):
    return create_access_token(
        data={"sub": admin_user.email, "user_id": str(admin_user.user_id)},
        expires_delta=timedelta(minutes=30),
    )


@pytest.fixture()
def admin_headers(admin_token):
    return {"Authorization": f"Bearer {admin_token}"}
