# MACKIS — Multi-Agent Conversational Knowledge & Information System

A Retrieval-Augmented Generation (RAG) chatbot for Sabancı University, built for the ENS 491/492 Senior Design Project. MACKIS answers student questions about university policies, regulations, and academic procedures using a hybrid retrieval pipeline backed by a PostgreSQL/pgvector database on Supabase.

---

## Table of Contents

1. [System Overview](#system-overview)
2. [Prerequisites](#prerequisites)
3. [Project Structure](#project-structure)
4. [Local Setup — Step by Step](#local-setup--step-by-step)
   - [1. Clone the Repository](#1-clone-the-repository)
   - [2. Backend Setup](#2-backend-setup)
   - [3. Environment Variables](#3-environment-variables)
   - [4. Database Setup (Supabase)](#4-database-setup-supabase)
   - [5. ChromaDB & BM25 Index (Local Vector Store)](#5-chromadb--bm25-index-local-vector-store)
   - [6. Run the Backend](#6-run-the-backend)
   - [7. Frontend Setup](#7-frontend-setup)
   - [8. Run the Frontend](#8-run-the-frontend)
5. [Running Tests](#running-tests)
6. [Docker (Recommended)](#docker-recommended)
7. [API Reference](#api-reference)
8. [Architecture Overview](#architecture-overview)
9. [Troubleshooting](#troubleshooting)

---

## System Overview

| Layer | Technology |
|---|---|
| Frontend | React 18 + TypeScript + Vite + Tailwind CSS |
| Backend | FastAPI (Python 3.11+) |
| Database & Auth | Supabase (PostgreSQL) |
| Vector Search | pgvector on Supabase + local ChromaDB fallback |
| Keyword Search | BM25 (rank-bm25, persisted as `.pkl`) |
| Embeddings | Qwen3-Embedding-8B via OpenRouter API |
| LLM | Qwen3-32B via OpenRouter API |
| Reranker | BAAI/bge-reranker-v2-m3 (local, via sentence-transformers) |

---

## Prerequisites

Make sure the following are installed on your machine before starting:

- **Python 3.11 or higher** — [python.org](https://www.python.org/downloads/)
- **Node.js 18 or higher** + **npm** — [nodejs.org](https://nodejs.org/)
- **Git** — [git-scm.com](https://git-scm.com/)
- **A Supabase account** (free tier is sufficient) — [supabase.com](https://supabase.com/)
- **An OpenRouter account** for LLM and embedding API access — [openrouter.ai](https://openrouter.ai/)
- *(Optional)* **Docker & Docker Compose** — only needed for the Docker path

> **Note:** The reranker model (`BAAI/bge-reranker-v2-m3`) is downloaded automatically from Hugging Face the first time the backend starts. This requires an internet connection and about 1.1 GB of disk space.

---

## Project Structure

```
ENS491-MACKIS/
├── main.py                    # FastAPI application entry point
├── database.py                # SQLAlchemy engine & session setup
├── models.py                  # Database ORM models
├── schemas.py                 # Pydantic request/response schemas
├── utils.py                   # Shared utilities (JWT, hashing, etc.)
├── requirements.txt           # Python dependencies
├── .env.example               # Environment variable template
│
├── routers/
│   ├── auth.py                # POST /auth/login, /auth/register
│   ├── chat.py                # POST /chat, GET /chat/history
│   └── info.py                # GET /api/stats, /api/suggestions
│
├── services/
│   ├── rag_service.py         # Top-level RAG orchestrator
│   ├── kg_service.py          # Knowledge graph queries (PostgreSQL)
│   ├── config/settings.py     # Centralized RAG configuration
│   ├── core/
│   │   ├── embedding_service.py   # Qwen3 embeddings via OpenRouter
│   │   ├── vector_store.py        # ChromaDB vector store wrapper
│   │   ├── pgvector_store.py      # pgvector (Supabase) vector store
│   │   ├── bm25_service.py        # BM25 keyword index
│   │   ├── db_data_loader.py      # PostgreSQL chunk data loader
│   │   └── llm_service.py         # LLM calls via OpenRouter
│   └── agents/
│       ├── query_analysis_agent.py   # Query understanding & expansion
│       ├── retrieval_agent.py        # Hybrid retrieval (BM25 + vector)
│       ├── ranking_agent.py          # Cross-encoder reranking + MMR
│       └── generation_agent.py       # Answer generation
│
├── creating_database/
│   ├── chroma_db_v2/          # Local ChromaDB vector store (chunk embeddings)
│   └── checkpoints_v2/
│       └── bm25_index_v3.pkl  # Pre-built BM25 index
│
├── frontend/                  # React frontend (copy of MACKIS_frontend)
│   ├── src/
│   │   ├── App.tsx
│   │   ├── components/
│   │   ├── lib/
│   │   └── api/
│   ├── package.json
│   └── vite.config.ts
│
├── tests/                     # Backend pytest test suite
├── scripts/                   # Data migration & utility scripts
└── docs/                      # Architecture documentation
```

---

## Local Setup — Step by Step

### 1. Clone the Repository

```bash
git clone <repository-url>
cd ENS491-MACKIS
```

---

### 2. Backend Setup

Create and activate a Python virtual environment, then install dependencies.

**macOS / Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

**Windows (Command Prompt):**
```cmd
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

> **Tip:** The first `pip install` will take a few minutes. The `sentence-transformers` package is large.

---

### 3. Environment Variables

Copy the template and fill in your credentials:

```bash
cp .env.example .env
```

Open `.env` in any text editor and update the following required fields:

```env
# --- REQUIRED ---

# Supabase connection string (Session Mode pooler, port 6543)
DATABASE_URL=postgresql://postgres.<PROJECT_REF>:<DB_PASSWORD>@aws-0-<REGION>.pooler.supabase.com:6543/postgres

# A random secret key for JWT tokens (generate with: openssl rand -hex 32)
JWT_SECRET_KEY=<your-secret>

# Your OpenRouter API key
OPENROUTER_API_KEY=sk-or-v1-<your-key>

# --- PATHS (set to absolute paths on your machine) ---

# Path to the preprocessing directory
PREPROCESSING_PATH=/absolute/path/to/ENS491-MACKIS/preprocessing

# Path to the local ChromaDB directory (chunk embeddings)
CHROMA_DIR_V2=/absolute/path/to/ENS491-MACKIS/creating_database/chroma_db_v2

# Path to the BM25 index pickle file
BM25_INDEX_PATH=/absolute/path/to/ENS491-MACKIS/creating_database/checkpoints_v2/bm25_index_v3.pkl

CHECKPOINT_DIR_V2=/absolute/path/to/ENS491-MACKIS/creating_database/checkpoints_v2
```

Everything else in `.env.example` can be left at its default value for a standard local setup.

---

### 4. Database Setup (Supabase)

MACKIS uses Supabase as its primary database for user accounts, conversation history, document chunks, and vector search.

**Step 1 — Create a Supabase project**

Go to [supabase.com](https://supabase.com), create a new project, and note your project's **Database URL** from *Settings → Database → Connection string → Session mode*.

**Step 2 — Enable the pgvector extension**

In the Supabase dashboard, open the **SQL Editor** and run:

```sql
CREATE EXTENSION IF NOT EXISTS vector;
```

**Step 3 — Create the application tables**

Run the backend once to let SQLAlchemy create all tables automatically:

```bash
# Make sure your virtual environment is active and .env is filled in
python main.py
# Or simply: uvicorn main:app --reload
```

SQLAlchemy will create the `users`, `conversations`, `messages`, `documents`, `chunks`, `chunk_embeddings`, and `kg_nodes` tables on first startup.

**Step 4 — Create an admin user**

Use the Supabase SQL editor (or `psql`) to insert your first user:

```sql
-- Replace the values below with your own
INSERT INTO users (email, hashed_password, name, is_admin)
VALUES (
  'admin@sabanciuniv.edu',
  -- Generate a bcrypt hash first (see note below)
  '$2b$12$<your-bcrypt-hash>',
  'Admin User',
  true
);
```

To generate a bcrypt hash for your password, run this one-liner with your virtual environment active:

```bash
python -c "from passlib.context import CryptContext; ctx = CryptContext(schemes=['bcrypt']); print(ctx.hash('your_password_here'))"
```

---

### 5. ChromaDB & BM25 Index (Local Vector Store)

The chunk embeddings and keyword index are stored locally and are **not** generated from scratch during startup — they must be present before running the application.

**Required files:**

| File | Description |
|---|---|
| `creating_database/chroma_db_v2/` | ChromaDB directory containing chunk embeddings (Qwen3-Embedding-8B) |
| `creating_database/checkpoints_v2/bm25_index_v3.pkl` | Pre-built BM25 index over 17,788 document chunks |

These files are large and are not included in the Git repository. Obtain them from the project team (shared drive / external storage) and place them in the paths shown above.

> If the paths in your `.env` are correct, the backend will log `✅ BM25 Index Loaded: 17788 documents` on startup, confirming the index was found.

**Verify ChromaDB collection name:**

The collection inside `chroma_db_v2` must match `COLL_NAME_V3` in your `.env` (default: `mysu_v3_qwen3`). If the collection name differs, update `.env` accordingly.

---

### 6. Run the Backend

With the virtual environment active and `.env` filled in:

```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

The API will be available at `http://localhost:8000`.

You can verify it is running by opening `http://localhost:8000` in a browser — you should see:

```json
{"status": "MACKIS RAG Backend Aktif 🚀"}
```

Interactive API documentation is available at `http://localhost:8000/docs`.

> **Note:** On the very first request that triggers retrieval, the reranker model (`BAAI/bge-reranker-v2-m3`) will be downloaded from Hugging Face (~1.1 GB). Subsequent startups use the cached model.

---

### 7. Frontend Setup

```bash
cd frontend
npm install
```

Create a frontend `.env` file to point it at the backend:

```bash
# frontend/.env
VITE_API_BASE_URL=http://localhost:8000
```

> **Note:** If the file `frontend/.env` already exists in the repository, verify that `VITE_API_BASE_URL` is set to `http://localhost:8000`.

---

### 8. Run the Frontend

```bash
# Inside the frontend/ directory
npm run dev
```

The app will be available at `http://localhost:5173` (Vite default).

Open your browser and navigate to `http://localhost:5173`. Log in with the admin credentials you created in Step 4.

---

## Running Tests

### Backend (pytest)

```bash
# From the project root, with venv active
pytest tests/ -v
```

To run with coverage:

```bash
pytest tests/ --cov=. --cov-report=term-missing
```

> Backend tests use a separate SQLite in-memory database and mock the RAG pipeline. No Supabase connection is needed.

### Frontend — Unit Tests (Vitest)

```bash
cd frontend
npm test
```

### Frontend — End-to-End Tests (Playwright)

```bash
cd frontend

# Install Playwright browsers (first time only)
npx playwright install

# Run all E2E tests (headless)
npm run e2e

# Run with browser UI visible
npm run e2e:headed

# Open Playwright interactive UI
npm run e2e:ui
```

> E2E tests expect both the backend (`localhost:8000`) and the frontend dev server (`localhost:5173`) to be running.

---

## Docker (Recommended)

Docker is the easiest way to run the full stack. Make sure Docker Desktop is installed and running.

**Step 1 — Prepare your `.env`**

Follow [Step 3](#3-environment-variables) above to create a filled-in `.env` file at the project root.

**Step 2 — Make sure the local data files are in place**

The ChromaDB directory and BM25 index must exist at the paths defined in `.env` before starting the containers (they are mounted into the backend container as volumes).

**Step 3 — Build and start**

```bash
# From the project root
docker compose up --build
```

This will:
- Build the backend image (Python 3.11, installs all requirements)
- Build the frontend image (Node 18, builds Vite app, serves via Nginx)
- Start both services

| Service | URL |
|---|---|
| Frontend | http://localhost:3000 |
| Backend API | http://localhost:8000 |
| API Docs | http://localhost:8000/docs |

**Stopping:**

```bash
docker compose down
```

> **Note:** The Docker setup is functional but may receive further updates. If you run into issues, the manual local setup described above is the most reliable path.

---

## API Reference

### Authentication

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/auth/login` | Login with email + password. Returns a JWT token. |
| `POST` | `/auth/register` | Register a new user account. |

**Login request body:**
```json
{
  "email": "user@sabanciuniv.edu",
  "password": "your_password"
}
```

**Login response:**
```json
{
  "access_token": "<jwt>",
  "token_type": "bearer",
  "user_name": "User Name",
  "is_admin": false
}
```

All subsequent requests must include the token as a Bearer header:
```
Authorization: Bearer <access_token>
```

### Chat

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/chat` | Send a message, get an answer with sources. |
| `GET` | `/chat/history` | Retrieve all past conversations for the logged-in user. |

**Chat request body:**
```json
{
  "query": "What are the Erasmus eligibility requirements?",
  "conversation_id": 42
}
```

`conversation_id` is optional. Omit it to start a new conversation; include it to continue an existing one.

**Chat response:**
```json
{
  "answer": "To be eligible for Erasmus...",
  "sources": [
    {
      "chunk_id": 1234,
      "title": "Erasmus+ Program Guide",
      "excerpt": "Students must have completed at least...",
      "score": 0.87,
      "url": null
    }
  ],
  "conversation_id": 42,
  "query_id": 101,
  "message_id": 202,
  "confidence": 0.82
}
```

### Info

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/api/stats` | Returns document and chunk counts for the knowledge base. |
| `GET` | `/api/suggestions` | Returns suggested questions for the UI. |

---

## Architecture Overview

```
User Query
    │
    ▼
┌──────────────────────┐
│  Query Analysis Agent │  ← Query expansion, tag inference, negation detection
└──────────┬───────────┘
           │
    ┌──────┴──────┐
    ▼             ▼
BM25 Search   Vector Search
(local .pkl)  (ChromaDB local → pgvector Supabase)
    │             │
    └──────┬──────┘
           ▼
   Hybrid Score Fusion (RRF + weighted combination)
           │
           ▼
┌──────────────────────┐
│   Ranking Agent       │  ← Cross-encoder rerank, tag boost, MMR diversity, source penalty
│  (bge-reranker-v2-m3)│
└──────────┬───────────┘
           │
           ▼
   Top-K Chunks Selected
           │
           ▼
┌──────────────────────┐
│  Generation Agent     │  ← Qwen3-32B via OpenRouter, context assembly, citation
└──────────┬───────────┘
           │
           ▼
   Answer + Sources → Frontend
```

---

## Troubleshooting

**Blank white screen on the frontend**

Open the browser developer console (F12) and check for JavaScript errors. Most commonly caused by a missing or incorrect `VITE_API_BASE_URL` in `frontend/.env`, or the backend not running.

**`BM25 index not found` error on startup**

The `BM25_INDEX_PATH` in your `.env` points to a file that does not exist. Verify the absolute path is correct and the `.pkl` file has been placed there.

**`Collection not found` error (ChromaDB)**

The `COLL_NAME_V3` in your `.env` does not match the collection that exists inside `CHROMA_DIR_V2`. Check the collection name with:

```python
import chromadb
client = chromadb.PersistentClient(path="/path/to/chroma_db_v2")
print([c.name for c in client.list_collections()])
```

**Reranker download is slow or fails**

The `BAAI/bge-reranker-v2-m3` model (~1.1 GB) is downloaded from Hugging Face on first use. Make sure you have a stable internet connection. The model is cached at `~/.cache/huggingface/` after the first download.

**`401 Unauthorized` on API requests**

Your JWT token has expired (default: 24 hours). Log out and log back in to get a new token.

**Database connection errors**

Verify that `DATABASE_URL` in your `.env` uses the **Session Mode pooler** URL from Supabase (port `6543`), not the direct connection (port `5432`). Direct connections are blocked on most Supabase free-tier projects.
