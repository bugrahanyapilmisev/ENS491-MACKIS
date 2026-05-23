# tests/test_chat_router.py
"""
chat router testleri.

Kapsam:
  A. Yardımcı fonksiyonlar (_resolve_chunk_info, _load_conversation_history)
  B. POST /chat  — yeni sohbet, mevcut sohbet, yetki kontrolü, kimlik doğrulama
  C. GET /chat/history — boş, kendi verisi, başka kullanıcıya sızıntı yok, kimlik
"""

import uuid

import pytest

import models
from routers.chat import _resolve_chunk_info, _load_conversation_history


# =============================================================================
# A. Yardımcı Fonksiyonlar
# =============================================================================

class TestResolveChunkInfo:
    """_resolve_chunk_info(db, chroma_chunk_id) → (chunk_id, source_uri)"""

    def _seed_doc_and_chunk(self, db, hash_val="abc123"):
        doc = models.Document(
            source_type="url",
            source_uri="https://mysu.sabanciuniv.edu/erasmus",
            title="Erasmus Rehberi",
            lang="tr",
        )
        db.add(doc)
        db.commit()

        chunk = models.Chunk(
            document_id=doc.document_id,
            ordinal=1,
            content="Erasmus başvuru şartları...",
            hash=hash_val,
        )
        db.add(chunk)
        db.commit()
        return doc, chunk

    def test_returns_chunk_id_and_url_when_found(self, db):
        doc, chunk = self._seed_doc_and_chunk(db)
        cid, url = _resolve_chunk_info(db, "abc123")
        assert cid == chunk.chunk_id
        assert url == "https://mysu.sabanciuniv.edu/erasmus"

    def test_returns_none_none_when_not_found(self, db):
        cid, url = _resolve_chunk_info(db, "hicyok-hash")
        assert cid is None
        assert url is None

    def test_returns_none_none_for_empty_string(self, db):
        cid, url = _resolve_chunk_info(db, "")
        assert cid is None
        assert url is None

    def test_different_hashes_resolve_independently(self, db):
        doc1, chunk1 = self._seed_doc_and_chunk(db, hash_val="hash-A")
        doc2 = models.Document(
            source_type="pdf",
            source_uri="https://mysu.sabanciuniv.edu/burs",
            title="Burs Rehberi",
            lang="tr",
        )
        db.add(doc2)
        db.commit()
        chunk2 = models.Chunk(
            document_id=doc2.document_id,
            ordinal=1,
            content="Burs başvurusu...",
            hash="hash-B",
        )
        db.add(chunk2)
        db.commit()

        cid_a, url_a = _resolve_chunk_info(db, "hash-A")
        cid_b, url_b = _resolve_chunk_info(db, "hash-B")

        assert cid_a == chunk1.chunk_id
        assert cid_b == chunk2.chunk_id
        assert url_a != url_b


class TestLoadConversationHistory:
    """_load_conversation_history(db, conversation_id, limit) → list[dict]"""

    def test_returns_empty_for_new_conversation(self, db, test_user):
        conv = models.Conversation(user_id=test_user.user_id, title="Boş Sohbet")
        db.add(conv)
        db.commit()
        result = _load_conversation_history(db, conv.conversation_id)
        assert result == []

    def test_returns_messages_in_chronological_order(self, db, test_user):
        conv = models.Conversation(user_id=test_user.user_id, title="Sıralı Sohbet")
        db.add(conv)
        db.commit()

        msg1 = models.ChatMessage(
            conversation_id=conv.conversation_id,
            user_id=test_user.user_id,
            role="user",
            content="İlk mesaj",
        )
        msg2 = models.ChatMessage(
            conversation_id=conv.conversation_id,
            role="assistant",
            content="İlk cevap",
        )
        db.add_all([msg1, msg2])
        db.commit()

        result = _load_conversation_history(db, conv.conversation_id)
        assert len(result) == 2
        assert result[0]["role"] == "user"
        assert result[1]["role"] == "assistant"

    def test_result_dicts_have_role_and_content_keys(self, db, test_user):
        conv = models.Conversation(user_id=test_user.user_id, title="Dict Test")
        db.add(conv)
        db.commit()
        msg = models.ChatMessage(
            conversation_id=conv.conversation_id,
            user_id=test_user.user_id,
            role="user",
            content="Merhaba",
        )
        db.add(msg)
        db.commit()

        result = _load_conversation_history(db, conv.conversation_id)
        assert "role" in result[0]
        assert "content" in result[0]

    def test_limit_is_respected(self, db, test_user):
        conv = models.Conversation(user_id=test_user.user_id, title="Limit Test")
        db.add(conv)
        db.commit()

        for i in range(15):
            db.add(models.ChatMessage(
                conversation_id=conv.conversation_id,
                user_id=test_user.user_id,
                role="user",
                content=f"Mesaj {i}",
            ))
        db.commit()

        result = _load_conversation_history(db, conv.conversation_id, limit=5)
        assert len(result) <= 5


# =============================================================================
# B. POST /chat
# =============================================================================

class TestChatEndpoint:

    def test_authenticated_request_returns_200(self, client, auth_headers):
        res = client.post("/chat", json={"query": "Erasmus şartları nelerdir?"}, headers=auth_headers)
        assert res.status_code == 200

    def test_response_contains_answer(self, client, auth_headers):
        res = client.post("/chat", json={"query": "Burs bilgisi"}, headers=auth_headers)
        data = res.json()
        assert "answer" in data
        assert isinstance(data["answer"], str)
        assert len(data["answer"]) > 0

    def test_new_conversation_is_created(self, client, auth_headers):
        """conversation_id verilmezse yeni sohbet oluşturulmalı."""
        res = client.post("/chat", json={"query": "Yeni sohbet"}, headers=auth_headers)
        data = res.json()
        assert "conversation_id" in data
        assert isinstance(data["conversation_id"], int)
        assert data["conversation_id"] > 0

    def test_response_contains_query_id(self, client, auth_headers):
        res = client.post("/chat", json={"query": "Sorgu ID testi"}, headers=auth_headers)
        assert "query_id" in res.json()

    def test_response_contains_message_id(self, client, auth_headers):
        res = client.post("/chat", json={"query": "Mesaj ID testi"}, headers=auth_headers)
        assert "message_id" in res.json()

    def test_response_contains_confidence(self, client, auth_headers):
        res = client.post("/chat", json={"query": "Güven skoru testi"}, headers=auth_headers)
        assert "confidence" in res.json()

    def test_existing_conversation_is_reused(self, client, auth_headers, test_user, db):
        """Var olan conversation_id verildiğinde yeni sohbet oluşturulmamalı."""
        conv = models.Conversation(user_id=test_user.user_id, title="Mevcut Sohbet")
        db.add(conv)
        db.commit()

        res = client.post("/chat", json={
            "query": "Devam eden sohbet",
            "conversation_id": conv.conversation_id,
        }, headers=auth_headers)

        assert res.status_code == 200
        assert res.json()["conversation_id"] == conv.conversation_id

    def test_cross_user_conversation_returns_403(self, client, auth_headers, other_user, db):
        """Başka kullanıcıya ait sohbete erişim 403 döndürmeli."""
        other_conv = models.Conversation(user_id=other_user.user_id, title="Başkasının Sohbeti")
        db.add(other_conv)
        db.commit()

        res = client.post("/chat", json={
            "query": "Yasak erişim",
            "conversation_id": other_conv.conversation_id,
        }, headers=auth_headers)

        assert res.status_code == 403

    def test_unauthenticated_request_returns_401(self, client):
        """Token olmadan istek 401 döndürmeli."""
        res = client.post("/chat", json={"query": "Yetkisiz"})
        assert res.status_code == 401

    def test_empty_query_returns_422(self, client, auth_headers):
        """Boş query FastAPI validation'ı geçemez."""
        res = client.post("/chat", json={}, headers=auth_headers)
        assert res.status_code == 422

    def test_conversation_title_set_from_query(self, client, auth_headers, db):
        """Yeni sohbetin başlığı ilk mesajın ilk 50 karakterinden oluşmalı."""
        query = "Erasmus değişim programı için minimum GNO kaç olmalı?"
        res = client.post("/chat", json={"query": query}, headers=auth_headers)
        conv_id = res.json()["conversation_id"]

        conv = db.query(models.Conversation).filter(
            models.Conversation.conversation_id == conv_id
        ).first()
        assert conv is not None
        assert conv.title.startswith(query[:30])

    def test_user_message_saved_to_db(self, client, auth_headers, db):
        """Kullanıcı mesajı chat_messages tablosuna kaydedilmeli."""
        query = "DB'ye kaydediliyor mu?"
        res = client.post("/chat", json={"query": query}, headers=auth_headers)
        conv_id = res.json()["conversation_id"]

        user_msgs = db.query(models.ChatMessage).filter(
            models.ChatMessage.conversation_id == conv_id,
            models.ChatMessage.role == "user",
        ).all()
        assert len(user_msgs) >= 1
        assert any(m.content == query for m in user_msgs)

    def test_assistant_message_saved_to_db(self, client, auth_headers, db):
        """Asistan cevabı chat_messages tablosuna kaydedilmeli."""
        res = client.post("/chat", json={"query": "Asistan cevabı kaydı"}, headers=auth_headers)
        conv_id = res.json()["conversation_id"]

        ai_msgs = db.query(models.ChatMessage).filter(
            models.ChatMessage.conversation_id == conv_id,
            models.ChatMessage.role == "assistant",
        ).all()
        assert len(ai_msgs) >= 1


# =============================================================================
# C. GET /chat/history
# =============================================================================

class TestChatHistory:

    def test_unauthenticated_returns_401(self, client):
        res = client.get("/chat/history")
        assert res.status_code == 401

    def test_authenticated_returns_200(self, client, auth_headers):
        res = client.get("/chat/history", headers=auth_headers)
        assert res.status_code == 200

    def test_empty_history_returns_empty_list(self, client, auth_headers):
        res = client.get("/chat/history", headers=auth_headers)
        assert res.json() == []

    def test_returns_own_conversations(self, client, auth_headers, test_user, db):
        conv = models.Conversation(user_id=test_user.user_id, title="Kendi Sohbetim")
        db.add(conv)
        db.commit()

        data = client.get("/chat/history", headers=auth_headers).json()
        assert len(data) == 1
        assert data[0]["title"] == "Kendi Sohbetim"

    def test_does_not_return_other_users_conversations(self, client, auth_headers, other_user, db):
        """Başka kullanıcının sohbetleri listelenmemeli."""
        other_conv = models.Conversation(
            user_id=other_user.user_id,
            title="Gizli Sohbet",
        )
        db.add(other_conv)
        db.commit()

        data = client.get("/chat/history", headers=auth_headers).json()
        titles = [c["title"] for c in data]
        assert "Gizli Sohbet" not in titles

    def test_conversation_has_required_fields(self, client, auth_headers, test_user, db):
        """Her sohbet id, title, timestamp, preview, messages içermeli."""
        conv = models.Conversation(user_id=test_user.user_id, title="Alan Testi")
        db.add(conv)
        db.commit()

        data = client.get("/chat/history", headers=auth_headers).json()
        conv_data = data[0]
        assert "id" in conv_data
        assert "title" in conv_data
        assert "timestamp" in conv_data
        assert "preview" in conv_data
        assert "messages" in conv_data

    def test_messages_list_is_included(self, client, auth_headers, test_user, db):
        conv = models.Conversation(user_id=test_user.user_id, title="Mesajlı Sohbet")
        db.add(conv)
        db.commit()

        msg = models.ChatMessage(
            conversation_id=conv.conversation_id,
            user_id=test_user.user_id,
            role="user",
            content="Merhaba",
        )
        db.add(msg)
        db.commit()

        data = client.get("/chat/history", headers=auth_headers).json()
        messages = data[0]["messages"]
        assert len(messages) == 1
        assert messages[0]["role"] == "user"
        assert messages[0]["content"] == "Merhaba"

    def test_archived_conversations_excluded(self, client, auth_headers, test_user, db):
        """archived=True olan sohbetler listelenmemeli."""
        active_conv = models.Conversation(
            user_id=test_user.user_id, title="Aktif Sohbet", archived=False
        )
        archived_conv = models.Conversation(
            user_id=test_user.user_id, title="Arşivlendi", archived=True
        )
        db.add_all([active_conv, archived_conv])
        db.commit()

        data = client.get("/chat/history", headers=auth_headers).json()
        titles = [c["title"] for c in data]
        assert "Aktif Sohbet" in titles
        assert "Arşivlendi" not in titles

    def test_multiple_conversations_returned(self, client, auth_headers, test_user, db):
        db.add_all([
            models.Conversation(user_id=test_user.user_id, title="Sohbet 1"),
            models.Conversation(user_id=test_user.user_id, title="Sohbet 2"),
            models.Conversation(user_id=test_user.user_id, title="Sohbet 3"),
        ])
        db.commit()

        data = client.get("/chat/history", headers=auth_headers).json()
        assert len(data) == 3
