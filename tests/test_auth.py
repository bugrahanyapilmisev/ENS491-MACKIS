# tests/test_auth.py
"""
POST /auth/login endpoint testleri.

Kapsam:
  - Başarılı giriş → 200 + access_token
  - Yanlış şifre → 401
  - Olmayan kullanıcı → 401
  - Admin kullanıcı → is_admin: True
  - password_hash None olan kullanıcı → 401
  - Eksik alan (email yok) → 422 (FastAPI validation)
"""

import uuid

import pytest

import models
from utils import get_password_hash


# =============================================================================
# Başarılı giriş senaryoları
# =============================================================================

class TestLoginSuccess:

    def test_returns_200(self, client, test_user):
        res = client.post("/auth/login", json={
            "email": "student@sabanciuniv.edu",
            "password": "password123",
        })
        assert res.status_code == 200

    def test_response_contains_access_token(self, client, test_user):
        res = client.post("/auth/login", json={
            "email": "student@sabanciuniv.edu",
            "password": "password123",
        })
        data = res.json()
        assert "access_token" in data
        assert len(data["access_token"]) > 10

    def test_token_type_is_bearer(self, client, test_user):
        res = client.post("/auth/login", json={
            "email": "student@sabanciuniv.edu",
            "password": "password123",
        })
        assert res.json()["token_type"] == "bearer"

    def test_user_name_returned(self, client, test_user):
        res = client.post("/auth/login", json={
            "email": "student@sabanciuniv.edu",
            "password": "password123",
        })
        assert res.json()["user_name"] == "Test Öğrencisi"

    def test_is_admin_false_for_student(self, client, test_user):
        res = client.post("/auth/login", json={
            "email": "student@sabanciuniv.edu",
            "password": "password123",
        })
        assert res.json()["is_admin"] is False


# =============================================================================
# Admin girişi
# =============================================================================

class TestAdminLogin:

    def test_is_admin_true_for_admin_role(self, client, admin_user):
        res = client.post("/auth/login", json={
            "email": "admin@sabanciuniv.edu",
            "password": "adminpass",
        })
        assert res.status_code == 200
        assert res.json()["is_admin"] is True

    def test_admin_also_gets_token(self, client, admin_user):
        res = client.post("/auth/login", json={
            "email": "admin@sabanciuniv.edu",
            "password": "adminpass",
        })
        assert "access_token" in res.json()


# =============================================================================
# Hatalı giriş senaryoları
# =============================================================================

class TestLoginFailure:

    def test_wrong_password_returns_401(self, client, test_user):
        res = client.post("/auth/login", json={
            "email": "student@sabanciuniv.edu",
            "password": "yanlisSifre",
        })
        assert res.status_code == 401

    def test_wrong_password_no_token_in_response(self, client, test_user):
        res = client.post("/auth/login", json={
            "email": "student@sabanciuniv.edu",
            "password": "yanlisSifre",
        })
        assert "access_token" not in res.json()

    def test_nonexistent_user_returns_401(self, client):
        res = client.post("/auth/login", json={
            "email": "kimse@sabanciuniv.edu",
            "password": "herhangi",
        })
        assert res.status_code == 401

    def test_user_without_password_hash_returns_401(self, client, db):
        """password_hash alanı None olan kullanıcı login yapamamalı."""
        user = models.User(
            user_id=uuid.uuid4(),
            email="nohash@sabanciuniv.edu",
            display_name="Hash Yok",
            password_hash=None,
            role="undergrad",
            status="active",
        )
        db.add(user)
        db.commit()

        res = client.post("/auth/login", json={
            "email": "nohash@sabanciuniv.edu",
            "password": "birSey",
        })
        assert res.status_code == 401

    def test_empty_password_returns_401(self, client, test_user):
        res = client.post("/auth/login", json={
            "email": "student@sabanciuniv.edu",
            "password": "",
        })
        # Boş şifre bcrypt doğrulamasını geçemez
        assert res.status_code == 401

    def test_missing_email_field_returns_422(self, client):
        """Zorunlu alan eksikse FastAPI 422 döner."""
        res = client.post("/auth/login", json={"password": "birSey"})
        assert res.status_code == 422

    def test_missing_password_field_returns_422(self, client):
        res = client.post("/auth/login", json={"email": "a@b.com"})
        assert res.status_code == 422


# =============================================================================
# Güvenlik — timing-safe davranış (dolaylı kontrol)
# =============================================================================

class TestLoginSecurity:

    def test_error_message_is_generic(self, client, test_user):
        """Yanlış şifre ve olmayan kullanıcı aynı hata mesajını döndürmeli
        (kullanıcı numaralandırma saldırısını önlemek için)."""
        res_wrong_pass = client.post("/auth/login", json={
            "email": "student@sabanciuniv.edu",
            "password": "yanlisSifre",
        })
        res_no_user = client.post("/auth/login", json={
            "email": "hicyok@sabanciuniv.edu",
            "password": "herhangi",
        })
        # Her ikisi de 401 ve aynı detail mesajı
        assert res_wrong_pass.status_code == res_no_user.status_code == 401
        assert res_wrong_pass.json()["detail"] == res_no_user.json()["detail"]
