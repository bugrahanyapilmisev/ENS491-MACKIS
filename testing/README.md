# MACKIS — RAG Evaluation Suite

This directory contains the end-to-end evaluation framework for the MACKIS
Retrieval-Augmented Generation system.

---

## Quick start

```bash
# Full multi-metric evaluation (default)
python testing/test_rag_detailed.py

# Legacy keyword-coverage-only mode (backward compat)
python testing/test_rag_detailed.py --no-eval

# Ask a single question
python testing/test_rag_detailed.py -q "Erasmus stajı için minimum GNO nedir?"
```

All terminal output is mirrored to `testing/rag_test_output.txt`.  
Evaluation reports (JSON + table) are saved under `testing/test_results/`.

---

## Files

| File | Purpose |
|---|---|
| `evaluator.py` | `Evaluator` class — implements all metrics |
| `test_rag_detailed.py` | Test runner — 30 golden questions, calls Evaluator |
| `rag_test_output.txt` | Latest run's terminal output |
| `test_results/` | Timestamped JSON + TXT reports |

---

## Metrics

### 1. Answer Similarity (weight 0.30)

**Tool:** Cosine similarity via `BAAI/bge-m3` embeddings (Ollama).  
**Fallback:** `paraphrase-multilingual-MiniLM-L12-v2` (sentence-transformers) if
Ollama is unreachable.

Measures _semantic closeness_ between the system's answer and the ground-truth
expected answer.  Handles paraphrasing and Turkish/English bilingual Q&A
correctly (unlike exact-match or keyword checks).

> **Why 0.30?** Similarity is a strong primary signal but can be gamed by a
> verbose answer that reproduces expected vocabulary without being faithful.
> Other metrics constrain it.

---

### 2. Faithfulness (weight 0.30)

**Tool:** LLM judge — Ollama (`qwen2.5:7b` or configured `CHAT_MODEL`).

Asks: *"Is every factual claim in the system answer grounded in the retrieved
context?"*

Prompt template (chain-of-thought, output JSON `{score:0-10, unsupported_claims:[…], reasoning:…}`):

```
Given:
  QUESTION: …
  RETRIEVED CONTEXT: …  (top-10 chunks from search_only)
  SYSTEM ANSWER: …

Score 10 = fully grounded; 0 = fully hallucinated.
```

**Faithfulness gate:** If faithfulness < 0.35, the composite score is multiplied
by `(faithfulness / 0.35)²`.  A score of 0.2 therefore reduces the composite
by ×0.33, making hallucination catastrophic regardless of other metric values.

> **Why 0.30 and gated?** In a university regulatory system, telling a student
> the wrong GPA threshold or internship duration is actively harmful.  The gate
> ensures that hallucinating answers can never pass.

---

### 3. Answer Relevance (weight 0.15)

**Tool:** LLM judge — Ollama.

Asks: *"Does the answer address the question, regardless of factual correctness?"*

Catches evasive / off-topic answers that are superficially fluent.

> **Why 0.15?** An answer that is faithful and factually accurate but slightly
> off-topic should still score well overall.  Relevance is the weakest constraint.

---

### 4. Factual Accuracy (weight 0.25)

**Tool:** Regex extraction + exact-match check.

Extracts all numbers, decimal values, percentages, and dates from the expected
(ground-truth) answer and checks how many appear verbatim in the system answer.

Examples of extracted facts: `2.20`, `2.5`, `10 adet`, `30 gün`, `1 ay`.

Fallback: if the expected answer contains no extractable numbers, character
4-gram overlap is used.

> **Why 0.25 (higher than default 0.10)?** MACKIS is a regulatory Q&A system.
> GPA thresholds, credit counts, deadlines, and durations appear in almost every
> useful answer.  Being numerically precise is more important than sounding fluent.

---

### 5. Keyword Coverage (kept as legacy metric — not weighted in composite)

Simple token overlap between expected and actual answers, Turkish-diacritic-aware.

Threshold for "pass" in legacy mode: ≥ 50 %.

**Known issues (why it is no longer primary):**

| Issue | Example |
|---|---|
| False positives | Common stop-words inflate score |
| False negatives | Correct paraphrase receives 0 |
| Turkish diacritics | "ınternship" ≠ "internship" |
| No faithfulness | 100 % match possible while hallucinating |

---

## Composite Score Formula

```
composite = 0.30 × answer_similarity
           + 0.30 × faithfulness
           + 0.25 × factual_accuracy
           + 0.15 × answer_relevance

# Faithfulness gate
if faithfulness < 0.35:
    composite × = (faithfulness / 0.35)²
```

**Pass threshold:** composite ≥ **0.70**

---

## Old vs New — Side-by-Side Comparison

Every run prints a comparison table:

```
█ OLD metric (keyword coverage ≥ 0.50) :  18/30 passed
█ NEW metric (composite      ≥ 0.70) :  24/30 passed
```

The full per-question table also includes both `KwCov` and `Compose` columns.

---

## Report outputs

After each full run two files are written to `test_results/`:

```
eval_report_20260219_143500.json   ← machine-readable, full detail
eval_table_20260219_143500.txt     ← copy of the pretty-printed table
```

### JSON schema

```jsonc
{
  "generated_at": "20260219_143500",
  "pass_threshold": 0.70,
  "composite_formula": "...",
  "overall": {
    "total": 30,
    "passed": 24,
    "pass_rate": 0.800,
    "avg_answer_similarity": 0.712,
    "avg_faithfulness": 0.681,
    "avg_answer_relevance": 0.743,
    "avg_factual_accuracy": 0.668,
    "avg_keyword_coverage": 0.543,   // legacy
    "avg_composite_score": 0.697,
    "avg_latency": 8.3
  },
  "by_category": { "Erasmus": { "n": 4, "passed": 3, … }, … },
  "questions": [
    {
      "question_id": "Q1",
      "category": "Erasmus",
      "question": "Erasmus staj için minimum GNO?",
      "expected": "Lisans 2.20, Lisansüstü 2.5",
      "actual": "…",
      "latency": 7.2,
      "passed": true,
      "scores": {
        "answer_similarity": 0.812,
        "faithfulness": 0.900,
        "answer_relevance": 0.800,
        "factual_accuracy": 1.000,
        "keyword_coverage": 0.643,   // legacy
        "composite_score": 0.872
      },
      "notes": ["faithfulness note: …", "factual note: found 3/3 facts"]
    }
  ]
}
```

---

## Architecture

```
test_rag_detailed.py
        │
        ├─► RAGPipeline.answer(question)          → answer text + latency
        ├─► RAGPipeline.search_only(question)     → context chunks
        │
        └─► Evaluator.evaluate(question, answer, expected, context)
                │
                ├─ _score_answer_similarity()  Ollama embed → cosine
                ├─ _score_faithfulness()       Ollama LLM judge
                ├─ _score_answer_relevance()   Ollama LLM judge
                ├─ _score_factual_accuracy()   regex
                └─ _score_keyword_coverage()   token overlap (legacy)
```

All evaluation is **fully offline** — only the local Ollama server is called.
No OpenAI / Cohere / external API keys are required.

---

## Configuration

The `Evaluator` reads from the same `.env` as the main pipeline:

| Variable | Default | Effect |
|---|---|---|
| `OLLAMA_HOST` | `http://localhost:11434` | Ollama server URL |
| `EMBED_MODEL` | `bge-m3` | Embedding model for similarity |
| `CHAT_MODEL` | `qwen2.5:7b` | LLM judge model |

Override at instantiation:

```python
from testing.evaluator import Evaluator
ev = Evaluator(
    ollama_host="http://localhost:11434",
    embed_model="bge-m3",
    chat_model="llama3.1:latest",
    pass_threshold=0.70,
)
```

---

## Adding new test questions

Edit the `TEST_QUESTIONS` list in `test_rag_detailed.py`:

```python
{
    "id": "Q31",
    "category": "MyCategory",
    "question": "Your question here?",
    "expected_answer": "The ground-truth answer (used for all metrics).",
    "source": "Document name / code",
},
```

The `expected_answer` is used for:
- **Answer Similarity** (semantic embedding target)
- **Factual Accuracy** (numbers/dates extracted from here)
- **Keyword Coverage** (legacy token overlap)

Make it as precise as possible, including the actual numbers.
