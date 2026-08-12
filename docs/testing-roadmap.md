# MACKIS — Testing Roadmap

> **Scope:** Backend unit & integration tests (pytest) + Frontend component tests (Vitest + React Testing Library) + End-to-end tests (Playwright).
> Every critical code path in `routers/`, `utils.py`, `services/agents/`, and all React components is covered.

---

## 1. Tooling & Installation

### 1.1 Backend (Python)

```bash
# From ENS491-MACKIS/
pip install pytest pytest-asyncio httpx pytest-mock pytest-cov --break-system-packages
```

| Package | Purpose |
|---|---|
| `pytest` | Test runner |
| `pytest-asyncio` | Async FastAPI test support |
| `httpx` / `TestClient` | Hit FastAPI endpoints in-process (no real server needed) |
| `pytest-mock` | `mocker` fixture — mock RAG engine, DB sessions, JWT |
| `pytest-cov` | Coverage report → `pytest --cov=. --cov-report=html` |

### 1.2 Frontend (TypeScript)

```bash
# From ENS491-MACKIS/frontend/
npm install -D vitest @vitest/ui jsdom \
  @testing-library/react @testing-library/user-event @testing-library/jest-dom \
  msw@2
```

| Package | Purpose |
|---|---|
| `vitest` | Fast Vite-native test runner (drop-in for Jest) |
| `@testing-library/react` | Render components, query DOM |
| `@testing-library/user-event` | Simulate keyboard, click, submit |
| `@testing-library/jest-dom` | Extra matchers (`toBeInTheDocument`, etc.) |
| `msw` v2 | Mock Service Worker — intercepts `fetch`/`axios` at the network layer |

Add to `vite.config.ts`:
```ts
test: {
  environment: 'jsdom',
  globals: true,
  setupFiles: ['./src/__tests__/setup.ts'],
},
```

### 1.3 End-to-End (Playwright)

```bash
npm install -D @playwright/test
npx playwright install chromium
```

---

## 2. File Structure

```
ENS491-MACKIS/
├── tests/                            ← backend
│   ├── conftest.py                   ← shared fixtures (test DB, test client, auth token)
│   ├── test_utils.py                 ← JWT, password hashing
│   ├── test_auth.py                  ← POST /auth/login
│   ├── test_info.py                  ← GET /api/suggestions, /api/stats
│   ├── test_chat_router.py           ← POST /chat, GET /chat/history
│   └── test_query_analysis.py        ← QueryAnalysisAgent logic
│
├── frontend/src/__tests__/           ← frontend
│   ├── setup.ts                      ← jest-dom matchers + MSW server
│   ├── mocks/
│   │   └── handlers.ts               ← MSW request handlers
│   ├── api.test.ts                   ← lib/api.ts functions
│   ├── LoginPage.test.tsx
│   ├── ChatMessage.test.tsx
│   ├── KnowledgeBaseStats.test.tsx
│   ├── SourceCard.test.tsx
│   └── App.test.tsx
│
└── e2e/                              ← playwright
    ├── playwright.config.ts
    └── tests/
        ├── login.spec.ts
        └── chat.spec.ts
```

---

## 3. Backend Unit Tests

### 3.1 `tests/conftest.py` — Shared Fixtures

```python
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from database import Base, get_db
from main import app
from utils import create_access_token, get_password_hash
import models

# ── In-memory SQLite for isolation (no real Postgres needed) ──
SQLALCHEMY_TEST_URL = "sqlite:///:memory:"

@pytest.fixture(scope="session")
def engine():
    eng = create_engine(SQLALCHEMY_TEST_URL, connect_args={"check_same_thread": False})
    Base.metadata.create_all(bind=eng)
    yield eng
    Base.metadata.drop_all(bind=eng)

@pytest.fixture()
def db(engine):
    TestSession = sessionmaker(bind=engine)
    session = TestSession()
    yield session
    session.rollback()
    session.close()

@pytest.fixture()
def client(db):
    """TestClient with DB override and mocked RAG engine."""
    app.dependency_overrides[get_db] = lambda: db
    yield TestClient(app)
    app.dependency_overrides.clear()

@pytest.fixture()
def test_user(db):
    user = models.User(
        email="student@sabanciuniv.edu",
        display_name="Test Student",
        password_hash=get_password_hash("password123"),
        role="student",
    )
    db.add(user)
    db.commit()
    db.refresh(user)
    return user

@pytest.fixture()
def auth_token(test_user):
    return create_access_token({"sub": test_user.email, "user_id": str(test_user.user_id)})

@pytest.fixture()
def auth_headers(auth_token):
    return {"Authorization": f"Bearer {auth_token}"}
```

---

### 3.2 `tests/test_utils.py` — JWT & Password

```python
# ── verify_password ──────────────────────────────────────────
def test_verify_password_correct():
    hash_ = get_password_hash("mySecret")
    assert verify_password("mySecret", hash_) is True

def test_verify_password_wrong():
    hash_ = get_password_hash("mySecret")
    assert verify_password("wrongPassword", hash_) is False

def test_verify_password_empty():
    hash_ = get_password_hash("mySecret")
    assert verify_password("", hash_) is False

# ── create_access_token ──────────────────────────────────────
def test_token_contains_user_id():
    token = create_access_token({"sub": "a@b.com", "user_id": "uuid-123"})
    payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
    assert payload["user_id"] == "uuid-123"

def test_token_expires():
    token = create_access_token({"sub": "a@b.com"}, expires_delta=timedelta(seconds=-1))
    with pytest.raises(JWTError):
        jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])

# ── get_current_user_id ──────────────────────────────────────
def test_get_current_user_id_valid(auth_token):
    uid = get_current_user_id(auth_token)
    assert uid is not None

def test_get_current_user_id_invalid():
    with pytest.raises(HTTPException) as exc:
        get_current_user_id("not.a.real.token")
    assert exc.value.status_code == 401

def test_get_current_user_id_missing_user_id_field():
    token = create_access_token({"sub": "a@b.com"})  # no user_id key
    with pytest.raises(HTTPException) as exc:
        get_current_user_id(token)
    assert exc.value.status_code == 401
```

---

### 3.3 `tests/test_auth.py` — `POST /auth/login`

```python
def test_login_success(client, test_user):
    res = client.post("/auth/login", json={"email": "student@sabanciuniv.edu", "password": "password123"})
    assert res.status_code == 200
    data = res.json()
    assert "access_token" in data
    assert data["token_type"] == "bearer"
    assert data["user_name"] == "Test Student"
    assert data["is_admin"] is False

def test_login_wrong_password(client, test_user):
    res = client.post("/auth/login", json={"email": "student@sabanciuniv.edu", "password": "wrong"})
    assert res.status_code == 401

def test_login_user_not_found(client):
    res = client.post("/auth/login", json={"email": "nobody@sabanciuniv.edu", "password": "anything"})
    assert res.status_code == 401

def test_login_admin_flag(client, db):
    admin = models.User(
        email="admin@sabanciuniv.edu",
        display_name="Admin",
        password_hash=get_password_hash("adminpass"),
        role="admin",
    )
    db.add(admin); db.commit()
    res = client.post("/auth/login", json={"email": "admin@sabanciuniv.edu", "password": "adminpass"})
    assert res.status_code == 200
    assert res.json()["is_admin"] is True

def test_login_no_password_hash(client, db):
    user = models.User(email="nohash@su.edu", display_name="X", password_hash=None, role="student")
    db.add(user); db.commit()
    res = client.post("/auth/login", json={"email": "nohash@su.edu", "password": "anything"})
    assert res.status_code == 401
```

---

### 3.4 `tests/test_info.py` — `/api/suggestions` & `/api/stats`

```python
def test_suggestions_returns_six(client):
    res = client.get("/api/suggestions")
    assert res.status_code == 200
    data = res.json()
    assert len(data["suggestions"]) == 6

def test_suggestions_fields(client):
    res = client.get("/api/suggestions")
    for s in res.json()["suggestions"]:
        assert "question" in s
        assert "hint" in s
        assert "emoji" in s

def test_stats_returns_expected_keys(client, db):
    res = client.get("/api/stats")
    assert res.status_code == 200
    data = res.json()
    assert "document_count" in data
    assert "topic_count" in data

def test_stats_counts_documents(client, db):
    # Seed one document
    doc = models.Document(title="Test Doc", source_uri="https://example.com")
    db.add(doc); db.commit()
    res = client.get("/api/stats")
    assert res.json()["document_count"] >= 1

def test_stats_graceful_on_db_error(client, mocker):
    mocker.patch("routers.info.func.count", side_effect=Exception("DB down"))
    res = client.get("/api/stats")
    assert res.status_code == 200          # graceful — never 500
    assert res.json()["document_count"] == 0
```

---

### 3.5 `tests/test_chat_router.py` — Chat Endpoint

```python
# ── Helper resolvers ────────────────────────────────────────
def test_resolve_chunk_info_found(db):
    doc = models.Document(title="D", source_uri="https://x.com")
    db.add(doc); db.commit()
    chunk = models.Chunk(document_id=doc.document_id, hash="abc123", content="text")
    db.add(chunk); db.commit()
    cid, url = _resolve_chunk_info(db, "abc123")
    assert cid == chunk.chunk_id
    assert url == "https://x.com"

def test_resolve_chunk_info_not_found(db):
    cid, url = _resolve_chunk_info(db, "doesnotexist")
    assert cid is None and url is None

def test_resolve_chunk_info_empty_string(db):
    cid, url = _resolve_chunk_info(db, "")
    assert cid is None and url is None

# ── POST /chat ───────────────────────────────────────────────
def test_chat_creates_new_conversation(client, auth_headers, mocker):
    mocker.patch("routers.chat.rag_engine.query",
                 return_value=("Cevap", [], {"retrieved_chunks": [], "context_chunks": [], "analysis": {}}))
    res = client.post("/chat", json={"query": "Erasmus şartları nelerdir?"}, headers=auth_headers)
    assert res.status_code == 200
    data = res.json()
    assert "answer" in data
    assert "conversation_id" in data
    assert data["conversation_id"] > 0

def test_chat_reuses_existing_conversation(client, auth_headers, test_user, db, mocker):
    conv = models.Conversation(user_id=test_user.user_id, title="Test Sohbet")
    db.add(conv); db.commit()
    mocker.patch("routers.chat.rag_engine.query",
                 return_value=("Cevap", [], {"retrieved_chunks": [], "context_chunks": [], "analysis": {}}))
    res = client.post("/chat", json={"query": "devam?", "conversation_id": conv.conversation_id}, headers=auth_headers)
    assert res.status_code == 200
    assert res.json()["conversation_id"] == conv.conversation_id

def test_chat_403_wrong_conversation(client, auth_headers, db):
    other = models.User(email="other@su.edu", display_name="O", password_hash="x", role="student")
    db.add(other); db.commit()
    conv = models.Conversation(user_id=other.user_id, title="Other's chat")
    db.add(conv); db.commit()
    res = client.post("/chat", json={"query": "hack?", "conversation_id": conv.conversation_id}, headers=auth_headers)
    assert res.status_code == 403

def test_chat_unauthenticated(client):
    res = client.post("/chat", json={"query": "test"})
    assert res.status_code == 401

# ── GET /chat/history ────────────────────────────────────────
def test_chat_history_empty(client, auth_headers):
    res = client.get("/chat/history", headers=auth_headers)
    assert res.status_code == 200
    assert res.json() == []

def test_chat_history_returns_own_conversations(client, auth_headers, test_user, db):
    conv = models.Conversation(user_id=test_user.user_id, title="Sohbet 1")
    db.add(conv); db.commit()
    res = client.get("/chat/history", headers=auth_headers)
    data = res.json()
    assert len(data) == 1
    assert data[0]["title"] == "Sohbet 1"
    assert "messages" in data[0]
    assert "preview" in data[0]

def test_chat_history_does_not_leak_other_users(client, auth_headers, db):
    other = models.User(email="other2@su.edu", display_name="Other", password_hash="x", role="student")
    db.add(other); db.commit()
    db.add(models.Conversation(user_id=other.user_id, title="Secret")); db.commit()
    res = client.get("/chat/history", headers=auth_headers)
    assert all(c["title"] != "Secret" for c in res.json())

def test_chat_history_unauthenticated(client):
    res = client.get("/chat/history")
    assert res.status_code == 401
```

---

### 3.6 `tests/test_query_analysis.py` — Agent Logic

```python
# Uses real QueryAnalysisAgent with mocked LLM + embedding services

def make_agent(mocker) -> QueryAnalysisAgent:
    emb = mocker.MagicMock()
    emb.embed.return_value = [0.1] * 1024
    llm = mocker.MagicMock()
    llm.chat.return_value = '{"tags": ["erasmus"], "negations": []}'
    config = RAGConfig.from_env()
    return QueryAnalysisAgent(emb, llm, config)

def test_language_detection_turkish(mocker):
    agent = make_agent(mocker)
    result = agent.detect_language("Erasmus programına nasıl başvururum?")
    assert result == "tr"

def test_language_detection_english(mocker):
    agent = make_agent(mocker)
    result = agent.detect_language("How do I apply for the Erasmus program?")
    assert result == "en"

def test_negation_cue_detected(mocker):
    agent = make_agent(mocker)
    assert agent._query_has_negation_cue("Erasmus hariç diğer burslar nelerdir?") is True

def test_negation_cue_not_detected(mocker):
    agent = make_agent(mocker)
    assert agent._query_has_negation_cue("Burs başvurusu nasıl yapılır?") is False

def test_negation_cue_english(mocker):
    agent = make_agent(mocker)
    assert agent._query_has_negation_cue("Scholarships excluding Erasmus") is True
```

---

## 4. Frontend Unit Tests

### 4.1 `setup.ts`

```ts
import '@testing-library/jest-dom'
import { server } from './mocks/server'

beforeAll(() => server.listen())
afterEach(() => server.resetHandlers())
afterAll(() => server.close())
```

### 4.2 `mocks/handlers.ts` (MSW)

```ts
import { http, HttpResponse } from 'msw'

export const handlers = [
  http.post('/auth/login', () => HttpResponse.json({
    access_token: 'fake-token',
    token_type: 'bearer',
    user_name: 'Test Student',
    is_admin: false,
  })),

  http.get('/api/stats', () => HttpResponse.json({
    document_count: 42,
    topic_count: 17,
  })),

  http.get('/api/suggestions', () => HttpResponse.json({
    suggestions: Array(6).fill({ question: 'Q?', hint: 'H', emoji: '📚' }),
  })),

  http.get('/chat/history', () => HttpResponse.json([])),

  http.post('/chat', () => HttpResponse.json({
    answer: 'Test yanıtı',
    sources: [],
    conversation_id: 1,
    query_id: 1,
    message_id: 1,
    confidence: 0.95,
  })),
]
```

---

### 4.3 `LoginPage.test.tsx`

```ts
// What we test:
// ✅ Renders email + password inputs
// ✅ Empty field → shows "Please fill in all fields" error, does NOT call API
// ✅ Invalid email format → shows email validation error
// ✅ Correct credentials → onLogin callback is called with (email, name, isAdmin)
// ✅ 401 from backend → shows error message
// ✅ Toggle to sign-up mode → name field appears

test('renders email and password inputs', () => { ... })
test('empty fields show validation error without API call', async () => { ... })
test('invalid email format shows error', async () => { ... })
test('successful login calls onLogin with correct args', async () => { ... })
test('401 response shows error message to user', async () => { ... })
test('toggle to signup shows name field', async () => { ... })
test('submit button shows loading state during request', async () => { ... })
```

---

### 4.4 `ChatMessage.test.tsx`

```ts
// What we test:
// ✅ User message: renders content, shows "You" label, no sources section
// ✅ Assistant message: renders "MACKIS" label, renders markdown (bold, links)
// ✅ Confidence badge: ≥90% → green, 75–89% → blue, <75% → yellow
// ✅ Sources section hidden when sources=[]
// ✅ Sources collapsible: click trigger → source cards appear
// ✅ Multiple sources: "Referenced N documents"

test('renders user message without sources section', () => { ... })
test('renders assistant label as MACKIS', () => { ... })
test('renders markdown bold text correctly', () => { ... })
test('renders links with target=_blank', () => { ... })
test('shows green confidence badge for ≥90%', () => { ... })
test('shows yellow confidence badge for <75%', () => { ... })
test('hides sources when sources array is empty', () => { ... })
test('shows sources count and expands on click', async () => { ... })
test('renders correct source count in trigger label', () => { ... })
```

---

### 4.5 `KnowledgeBaseStats.test.tsx`

```ts
// What we test:
// ✅ Loading spinner shown initially
// ✅ After fetch resolves: document_count and topic_count displayed
// ✅ Large numbers formatted with + suffix (e.g. "42+" not just "42")
// ✅ Error state: "Veriler yüklenemedi." shown on network failure

test('shows loading spinner on mount', () => { ... })
test('displays document count from API', async () => { ... })
test('displays topic count from API', async () => { ... })
test('formats counts ≥1000 with + suffix', async () => { ... })
test('shows error message on fetch failure', async () => { ... })
```

---

### 4.6 `SourceCard.test.tsx`

```ts
// What we test:
// ✅ Renders title and excerpt
// ✅ When url provided: renders as anchor tag with correct href
// ✅ When url is null: no broken link rendered
// ✅ Index badge shown (e.g. "1", "2")

test('renders title and excerpt', () => { ... })
test('renders clickable link when url is provided', () => { ... })
test('renders no anchor when url is null', () => { ... })
```

---

### 4.7 `api.test.ts` — API Functions

```ts
// What we test:
// ✅ loginUser: success returns { access_token, user_name, is_admin }
// ✅ loginUser: 401 throws AxiosError
// ✅ sendMessageToRAG: success returns RAGResponse shape
// ✅ sendMessageToRAG: passes conversation_id in payload when provided
// ✅ fetchChatHistory: success returns array of conversations

test('loginUser success returns token', async () => { ... })
test('loginUser 401 throws error', async () => { ... })
test('sendMessageToRAG returns answer and sources', async () => { ... })
test('sendMessageToRAG omits conversation_id when not provided', async () => { ... })
test('fetchChatHistory returns empty array', async () => { ... })
```

---

### 4.8 `App.test.tsx` — Routing Logic

```ts
// What we test:
// ✅ Unauthenticated: renders LoginPage (not chat UI)
// ✅ After login as student: renders chat interface (not LoginPage, not AdminDashboard)
// ✅ After login as admin: renders AdminDashboard
// ✅ Logout: goes back to LoginPage and clears user state

test('shows LoginPage when not logged in', () => { ... })
test('shows chat UI after student login', async () => { ... })
test('shows AdminDashboard after admin login', async () => { ... })
test('logout returns to LoginPage', async () => { ... })
```

---

## 5. End-to-End Tests (Playwright)

> These run against the **real** running stack (Vite on :3000, FastAPI on :8000).

### 5.1 `e2e/tests/login.spec.ts`

```ts
test('login page is visible on first load', async ({ page }) => { ... })
test('invalid credentials shows error message', async ({ page }) => { ... })
test('valid credentials navigates to chat screen', async ({ page }) => { ... })
test('logout returns to login screen', async ({ page }) => { ... })
```

### 5.2 `e2e/tests/chat.spec.ts`

```ts
test('can type and submit a question', async ({ page }) => { ... })
test('response message shows MACKIS avatar', async ({ page }) => { ... })
test('response appears without page reload', async ({ page }) => { ... })
test('sources collapsible opens on click', async ({ page }) => { ... })
test('new conversation appears in sidebar', async ({ page }) => { ... })
test('KnowledgeBaseStats card shows non-zero counts', async ({ page }) => { ... })
```

---

## 6. Coverage Targets

| Layer | Target | Priority |
|---|---|---|
| `utils.py` | 100% | Critical (auth) |
| `routers/auth.py` | 100% | Critical |
| `routers/info.py` | 95%+ | High |
| `routers/chat.py` | 85%+ | High |
| `services/agents/query_analysis_agent.py` | 80%+ | Medium |
| Frontend components | 85%+ | High |
| `lib/api.ts` | 95%+ | High |
| E2E happy paths | All green | Gate to ship |

Run coverage:
```bash
# Backend
pytest tests/ --cov=. --cov-report=html --ignore=crawler_for_srdoc

# Frontend
npx vitest run --coverage
```

---

## 7. Implementation Order (Sprint Plan)

| Sprint | What to implement | Est. effort |
|---|---|---|
| **1** | conftest.py, test_utils.py, test_auth.py | 1 day |
| **2** | test_info.py, test_chat_router.py | 1–2 days |
| **3** | test_query_analysis.py | 1 day |
| **4** | Frontend setup.ts + MSW handlers + api.test.ts | 1 day |
| **5** | LoginPage.test.tsx, ChatMessage.test.tsx | 1 day |
| **6** | KnowledgeBaseStats.test.tsx, SourceCard.test.tsx, App.test.tsx | 1 day |
| **7** | Playwright setup + login.spec.ts + chat.spec.ts | 1–2 days |
| **8** | Coverage review, fix gaps, CI hook (GitHub Actions) | 1 day |

**Total estimated: ~9 days of focused work.**

---

## 8. Notes & Known Constraints

- **ChromaDB is unavailable in CI** — all tests that touch the RAG engine must mock `rag_engine.query` via `mocker.patch`. The `conftest.py` can do this globally so no individual test needs to repeat it.
- **pgvector is not available in SQLite** — the `Vector(1024)` column on `KGNode` must be skipped or the model monkey-patched in the test session. Use `TESTING=true` env var to conditionally disable the pgvector column during tests.
- **JWT_SECRET_KEY must be set** — add `JWT_SECRET_KEY=test-secret` to a `.env.test` file and load it in `conftest.py` before importing `utils`.
- **MSW in Vitest** — requires `@testing-library/msw` + `server.ts` with `setupServer(...handlers)`. Axios must be configured to NOT use the Vite proxy base URL in tests; set `baseURL: ''` in a test-specific axios instance, or mock the module entirely.
