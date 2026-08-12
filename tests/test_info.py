# tests/test_info.py
"""
GET /api/suggestions ve GET /api/stats endpoint testleri.

Kapsam:
  - /api/suggestions → 6 öğe, doğru alanlar, Türkçe içerik
  - /api/stats → document_count ve topic_count döner
  - /api/stats → gerçek DB kayıtlarını sayar
  - /api/stats → DB hatasında 500 değil 200 + sıfırlar döner (graceful degradation)
"""

import uuid

import pytest

import models


# =============================================================================
# GET /api/suggestions
# =============================================================================

class TestSuggestions:

    def test_returns_200(self, client):
        res = client.get("/api/suggestions")
        assert res.status_code == 200

    def test_returns_exactly_six_suggestions(self, client):
        data = res = client.get("/api/suggestions").json()
        assert len(data["suggestions"]) == 6

    def test_each_suggestion_has_question_field(self, client):
        suggestions = client.get("/api/suggestions").json()["suggestions"]
        for s in suggestions:
            assert "question" in s, f"'question' alanı eksik: {s}"

    def test_each_suggestion_has_hint_field(self, client):
        suggestions = client.get("/api/suggestions").json()["suggestions"]
        for s in suggestions:
            assert "hint" in s, f"'hint' alanı eksik: {s}"

    def test_each_suggestion_has_emoji_field(self, client):
        suggestions = client.get("/api/suggestions").json()["suggestions"]
        for s in suggestions:
            assert "emoji" in s, f"'emoji' alanı eksik: {s}"

    def test_questions_are_nonempty_strings(self, client):
        suggestions = client.get("/api/suggestions").json()["suggestions"]
        for s in suggestions:
            assert isinstance(s["question"], str) and len(s["question"]) > 0

    def test_response_has_suggestions_key(self, client):
        data = client.get("/api/suggestions").json()
        assert "suggestions" in data

    def test_content_is_sabanci_specific(self, client):
        """Sorular Sabancı Üniversitesi'ne özgü anahtar kelimeler içermeli."""
        questions = [
            s["question"]
            for s in client.get("/api/suggestions").json()["suggestions"]
        ]
        full_text = " ".join(questions).lower()
        # En az birinde Sabancı bağlamına ait bir kelime geçmeli
        sabanci_keywords = ["erasmus", "burs", "kütüphane", "mezuniyet", "anadal", "uyarı"]
        assert any(kw in full_text for kw in sabanci_keywords), (
            f"Sorularda Sabancı bağlamına ait kelime bulunamadı: {questions}"
        )


# =============================================================================
# GET /api/stats
# =============================================================================

class TestStats:

    def test_returns_200(self, client):
        res = client.get("/api/stats")
        assert res.status_code == 200

    def test_response_has_document_count_key(self, client):
        data = client.get("/api/stats").json()
        assert "document_count" in data

    def test_response_has_topic_count_key(self, client):
        data = client.get("/api/stats").json()
        assert "topic_count" in data

    def test_counts_are_integers(self, client):
        data = client.get("/api/stats").json()
        assert isinstance(data["document_count"], int)
        assert isinstance(data["topic_count"], int)

    def test_counts_are_non_negative(self, client):
        data = client.get("/api/stats").json()
        assert data["document_count"] >= 0
        assert data["topic_count"] >= 0

    def test_document_count_reflects_db(self, client, db):
        """DB'ye doküman eklenince document_count artmalı."""
        before = client.get("/api/stats").json()["document_count"]

        doc = models.Document(
            source_type="url",
            source_uri="https://mysu.sabanciuniv.edu/test",
            title="Test Dokümanı",
            lang="tr",
        )
        db.add(doc)
        db.commit()

        after = client.get("/api/stats").json()["document_count"]
        assert after == before + 1

    def test_topic_count_reflects_kg_nodes(self, client, db):
        """KGNode (topic label) eklenince topic_count artmalı."""
        before = client.get("/api/stats").json()["topic_count"]

        # Önce bir doküman gerekiyor (KGNode FK)
        doc = models.Document(
            source_type="html",
            source_uri="https://mysu.sabanciuniv.edu/konu",
            title="KG Kaynak",
            lang="tr",
        )
        db.add(doc)
        db.commit()

        node = models.KGNode(
            label="topic",
            name="Burs",
            doc_id=doc.document_id,
        )
        db.add(node)
        db.commit()

        after = client.get("/api/stats").json()["topic_count"]
        assert after == before + 1

    def test_graceful_degradation_on_db_error(self, client, mocker):
        """DB'de hata olursa 500 değil 200 + sıfırlar dönmeli."""
        mocker.patch(
            "routers.info.func.count",
            side_effect=Exception("Simüle edilmiş DB hatası"),
        )
        res = client.get("/api/stats")
        assert res.status_code == 200
        data = res.json()
        assert data["document_count"] == 0
        assert data["topic_count"] == 0

    def test_response_does_not_expose_chunk_count(self, client):
        """chunk_count teknik detay — frontend'e gönderilmemeli."""
        data = client.get("/api/stats").json()
        assert "chunk_count" not in data

    def test_response_does_not_expose_kg_edge_count(self, client):
        """kg_edge_count teknik detay — frontend'e gönderilmemeli."""
        data = client.get("/api/stats").json()
        assert "kg_edge_count" not in data
