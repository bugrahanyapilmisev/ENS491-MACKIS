# tests/test_utils.py
"""
utils.py için birim testler.

Kapsam:
  - verify_password   → doğru / yanlış / boş şifre
  - get_password_hash → hash'in bcrypt formatında olması
  - create_access_token → payload içeriği, expiry
  - get_current_user_id → geçerli token, süresi dolmuş, bozuk, user_id eksik
"""

import os
import uuid
from datetime import timedelta

import pytest
from fastapi import HTTPException
from jose import jwt, JWTError

# conftest.py env'leri zaten set etti; güvenle import edebiliriz
from utils import (
    verify_password,
    get_password_hash,
    create_access_token,
    get_current_user_id,
)

SECRET_KEY = os.environ["JWT_SECRET_KEY"]
ALGORITHM = os.environ.get("JWT_ALGORITHM", "HS256")


# =============================================================================
# verify_password
# =============================================================================

class TestVerifyPassword:

    def test_correct_password_returns_true(self):
        hashed = get_password_hash("dogruSifre123")
        assert verify_password("dogruSifre123", hashed) is True

    def test_wrong_password_returns_false(self):
        hashed = get_password_hash("dogruSifre123")
        assert verify_password("yanlisSifre", hashed) is False

    def test_empty_password_returns_false(self):
        hashed = get_password_hash("birSifre")
        assert verify_password("", hashed) is False

    def test_case_sensitive(self):
        hashed = get_password_hash("Buyuk")
        assert verify_password("buyuk", hashed) is False  # küçük harf farklı

    def test_hash_as_string_and_bytes_both_work(self):
        """DB'den bazen str, bazen bytes gelebilir — ikisi de kabul edilmeli."""
        hashed_str = get_password_hash("test123")
        hashed_bytes = hashed_str.encode("utf-8")
        assert verify_password("test123", hashed_str) is True
        assert verify_password("test123", hashed_bytes) is True


# =============================================================================
# get_password_hash
# =============================================================================

class TestGetPasswordHash:

    def test_returns_string(self):
        result = get_password_hash("herhangiSifre")
        assert isinstance(result, str)

    def test_starts_with_bcrypt_prefix(self):
        result = get_password_hash("herhangiSifre")
        assert result.startswith("$2b$") or result.startswith("$2a$")

    def test_two_hashes_are_different(self):
        """Bcrypt salt kullandığından aynı şifrenin iki hash'i farklı olmalı."""
        h1 = get_password_hash("ayniSifre")
        h2 = get_password_hash("ayniSifre")
        assert h1 != h2

    def test_hash_verifiable(self):
        """Hash oluşturulup tekrar verify_password ile doğrulanabilmeli."""
        sifre = "Makk1s@Test"
        hashed = get_password_hash(sifre)
        assert verify_password(sifre, hashed) is True


# =============================================================================
# create_access_token
# =============================================================================

class TestCreateAccessToken:

    def test_token_is_string(self):
        token = create_access_token({"sub": "a@b.com", "user_id": "uuid-1"})
        assert isinstance(token, str)
        assert len(token) > 10

    def test_payload_user_id_preserved(self):
        uid = str(uuid.uuid4())
        token = create_access_token({"sub": "a@b.com", "user_id": uid})
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        assert payload["user_id"] == uid

    def test_payload_sub_preserved(self):
        token = create_access_token({"sub": "ogrenci@su.edu", "user_id": "x"})
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        assert payload["sub"] == "ogrenci@su.edu"

    def test_token_has_expiry_field(self):
        token = create_access_token({"sub": "a@b.com", "user_id": "x"})
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        assert "exp" in payload

    def test_expired_token_raises_on_decode(self):
        """Süresi -1 saniye olan token decode edilemez."""
        token = create_access_token(
            {"sub": "a@b.com", "user_id": "x"},
            expires_delta=timedelta(seconds=-1),
        )
        with pytest.raises(JWTError):
            jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])

    def test_custom_expiry_respected(self):
        """60 dakika expiry ayarlandığında payload'daki exp değeri yakın olmalı."""
        import time
        token = create_access_token(
            {"sub": "a@b.com", "user_id": "x"},
            expires_delta=timedelta(minutes=60),
        )
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        now = int(time.time())
        # exp yaklaşık 60 dakika sonrası olmalı (±5 sn tolerans)
        assert abs(payload["exp"] - (now + 3600)) < 5


# =============================================================================
# get_current_user_id
# =============================================================================

class TestGetCurrentUserId:

    def test_valid_token_returns_user_id(self):
        uid = str(uuid.uuid4())
        token = create_access_token({"sub": "a@b.com", "user_id": uid})
        result = get_current_user_id(token)
        assert result == uid

    def test_expired_token_raises_401(self):
        token = create_access_token(
            {"sub": "a@b.com", "user_id": "x"},
            expires_delta=timedelta(seconds=-1),
        )
        with pytest.raises(HTTPException) as exc_info:
            get_current_user_id(token)
        assert exc_info.value.status_code == 401

    def test_invalid_token_string_raises_401(self):
        with pytest.raises(HTTPException) as exc_info:
            get_current_user_id("bu.gecersiz.bir.token")
        assert exc_info.value.status_code == 401

    def test_empty_token_raises_401(self):
        with pytest.raises(HTTPException) as exc_info:
            get_current_user_id("")
        assert exc_info.value.status_code == 401

    def test_token_without_user_id_raises_401(self):
        """user_id alanı olmayan token reddedilmeli."""
        token = create_access_token({"sub": "a@b.com"})  # user_id yok
        with pytest.raises(HTTPException) as exc_info:
            get_current_user_id(token)
        assert exc_info.value.status_code == 401

    def test_token_signed_with_wrong_key_raises_401(self):
        """Farklı secret ile imzalanmış token reddedilmeli."""
        token = jwt.encode(
            {"sub": "a@b.com", "user_id": "x"},
            "yanlis-secret-key",
            algorithm=ALGORITHM,
        )
        with pytest.raises(HTTPException) as exc_info:
            get_current_user_id(token)
        assert exc_info.value.status_code == 401
