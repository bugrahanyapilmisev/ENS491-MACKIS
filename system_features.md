# ENS491-MACKIS System Documentation (Comprehensive)

This document serves as the definitive technical reference for the `ENS491-MACKIS` system, a high-performance RAG (Retrieval-Augmented Generation) engine designed for university administrative domains.

---

## 1. Data Ingestion & Preprocessing (`preprocessing/`)

The system transforms raw unstructured data into semantic knowledge.

### 1.1. Multi-Format Ingestion (`preprocess_docs.py`)
*   **HTML & PDF**: Native parsing using `BeautifulSoup` and `pypdf`.
*   **Legacy Formats**: Automatically converts `.doc` and `.doc_20` files to PDF using **LibreOffice** (headless mode) before extraction.
*   **Boilerplate Removal**: Uses regex patterns to strip navigational elements ("Back to Menu", "Copyright", "Search") to reduce noise.

### 1.2. Structure Preservation
Instead of flat text, documents are parsed hierarchically:
*   **Section Extraction**: Identifies headers like "Amaç", "Kapsam", "1.1. Başvuru" using language-specific regex rules (`SECTION_PATTERNS_TR/EN`).
*   **Metadata Extraction**: Regex extraction of:
    *   **Procedure Codes** (e.g., `PIRO-C420`)
    *   **Dates** (Effective Date, Update Date)
    *   **Responsible Units** (e.g., "International Office")
    *   **Document Type** (Directive, Regulation, Form)
*   **Auto-Summary**: Generates specific summaries by concatenating "Purpose" and "Scope" sections.

### 1.3. Developer Tools
*   **Document Browser** (`browse_preprocessed_docs.py`): A **Streamlit** application to inspect preprocessed JSONs and chunk boundaries visually.

---

## 2. Knowledge Graph Architecture (`creating_kg/`)

The system uses a **Hybrid Knowledge Graph** to ground answers in verified facts.

*   **Extraction**: An LLM extracts `(Entity) -> [Relation] -> (Entity)` triples.
*   **Validation Layer** (`validate_kg_facts.py`):
    *   **LLM Validator**: A specialized prompt critiques every extracted fact. It rejects "dates" misinterpreted as "durations" or "placeholders" extracted as "values".
*   **Pattern-Based Validation** (`rebuild_kg_index_pattern.py`): Rebuilds the embedding index only from facts that pass strict regex validation rules.
*   **Focused Selection** (`select_focused_docs.py`): A utility to create balanced datasets (e.g., 15 Erasmus + 15 Library + 30 Noise docs) for rapid iteration.

---

## 3. Vector Database & Storage (`creating_database/` & `models.py`)

### 3.1. Advanced Chunking strategy (`build_chroma_store.py`)
*   **Contextual Embeddings**: 
    *   Chunks are not embedded in isolation. 
    *   Format: `[Document Title] [Section Header] Content...`
    *   This ensures a chunk describing "GPA=2.5" is strictly bound to "Erasmus" in the vector space.
*   **Independent Semantic Tagging** (`update_tags_v2.py`):
    *   An LLM assigns open-ended tags (e.g., `student_disciplinary_action`, `scholarship_criteria`) to documents.
    *   These tags are stored in **ChromaDB Metadata** for filtering.

### 3.2. Data Models & Traceability (`models.py`)
The system uses **PostgreSQL** with `pgvector` for long-term storage and full traceability:
*   **`QuerySession`**: Tracks user sessions.
*   **`QueryEvent`**: Records every user question.
*   **`RetrievalHit`**: Records *exactly* which chunks were found (Rank 1..10) and their scores.
*   **`Answer`**: Stores the final AI response.
*   **Result**: Complete audibility of *why* the AI gave a specific answer.

---

## 4. The RAG Core (`services/rag_core.py`)

The heart of the system is a 4-Stage "Recall-Rerank-Synthesize" pipeline.

### Stage 1: Query Intelligence
*   **Intent Detection**: Distinguishes between `list_names` ("List all forms"), `count_items` ("How many..."), `describe`, and `other`.
*   **Negation Extraction** (`extract_negated_terms_llm`):
    *   The LLM specifically extracts terms the user wants to *exclude* (e.g., "scholarships **other than** Erasmus").
    *   **Negation Penalty**: Documents matching these terms are aggressively penalised (score * 0.3).
*   **Query Expansion**: Rewrites the query into 3 variations to handle synonyms.
*   **HyDE (Hypothetical Document Embeddings)**: Hallucinates a "perfect answer" and embeds it to find semantically similar real documents.

### Stage 2: Hybrid Search (Recall)
*   **Vector Search**: Finds semantic matches via **BGE-M3** embeddings.
*   **BM25 Search**: Finds exact keyword matches (vital for codes like `MATH-102`).
*   **Reciprocal Rank Fusion (RRF)**: Merges the two lists, boosting documents that appear in both.
*   **Multi-Query**: Runs this search for *all* expanded query variations.

### Stage 3: Cross-Encoder Reranking (Precision)
*   **Model**: `BAAI/bge-reranker-v2-m3`.
*   **Logic**: Re-scores the top ~50 pairs (Query, Document) to value logical consistency and penalize irrelevant semantic matches (e.g., distinguishing "can" from "cannot").
*   **Tag Boosting**: Chunks with metadata tags matching the query topic get a multiplier boost.

### Stage 4: Generation
*   **Context Assembly**: Combines top text chunks + **Verified KG Facts**.
*   **Citation**: The prompt enforces citing the `Source Path` for every claim.
*   **Hallucination Check** (`verify_answer_numbers`): A strict regex-based safety check ensures any number/date in the final answer actually exists in the context.

---

## 5. Testing & Validation (`testing/`)

*   **`test_rag_detailed.py`**: A comprehensive test suite with 30+ Golden Questions categorized by domain (Erasmus, Library, Discipline).
    *   **Metrics**: Measures **Keyword Coverage** (percentage of expected keywords present in the answer) and **Latency**.
    *   **Output**: Generates detailed reports (`rag_test_output.txt`) showing expected vs. actual answers.

---

## 6. Access Layers (`routers/` & `services/rag_service.py`)

*   **`rag_service.py`**: A clean implementation of the Facade pattern, initializing the heavy engines (Chroma, BM25) once and exposing a simple `.query()` method.
*   **FastAPI Router** (`routers/chat.py`):
    *   Handles **User Authentication** (Guest/User).
    *   Manages **Conversation History**.
    *   Logs **Query Events** and **Retrieval Hits** to the PostgreSQL database for analytics.

## Summary table of Technologies

| Component | Technology | File(s) |
| :--- | :--- | :--- |
| **Parsing** | BeautifulSoup, pypdf, LibreOffice | `preprocess_docs.py` |
| **Embeddings** | BAAI/bge-m3 | `build_chroma_store.py` |
| **Vector DB** | ChromaDB & pgvector | `rag_core.py`, `models.py` |
| **Graph** | Validated Hybrid KG | `creating_kg/` |
| **Reranking** | Cross-Encoder (bge-reranker-v2-m3) | `rag_core.py` |
| **Validation** | Regex & LLM Critics | `validate_kg_facts.py`, `rag_core.py` |
| **Testing** | Golden Dataset Suite | `test_rag_detailed.py` |
