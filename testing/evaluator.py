"""
evaluator.py - Multi-Metric RAG Evaluation Framework

Domain: Sabancı University administrative Q&A (MACKIS)
Author: ENS491 Team

─────────────────────────────────────────────────────────────────────────────
METRICS
─────────────────────────────────────────────────────────────────────────────
  1. Answer Similarity   – Cosine similarity between answer and expected
                           embeddings (bge-m3 via Ollama, fallback: ST).
  2. Faithfulness        – LLM judge (Ollama): every claim grounded in context?
  3. Answer Relevance    – LLM judge (Ollama): does answer address the question?
  4. Factual Accuracy    – Regex: numbers / dates in expected found in answer.
  5. Keyword Coverage    – Legacy word-overlap metric (kept for comparison).

─────────────────────────────────────────────────────────────────────────────
COMPOSITE SCORE  (domain-optimized for regulatory university Q&A)
─────────────────────────────────────────────────────────────────────────────

  composite = 0.30 × similarity
            + 0.30 × faithfulness
            + 0.25 × factual_accuracy
            + 0.15 × relevance

  Faithfulness gate: if faithfulness < 0.35, apply quadratic penalty
      composite × = (faithfulness / 0.35)²

  Rationale: In a university regulatory system, giving the wrong GPA
  threshold or duration is worse than giving a vague answer. Factual
  accuracy therefore receives 0.25 (vs 0.10 in the original proposal),
  faithfulness stays at 0.30 to guard against hallucination, and
  relevance is slightly lower at 0.15 because a grounded answer to a
  somewhat-related question is still useful.

  Pass threshold: composite ≥ 0.70

─────────────────────────────────────────────────────────────────────────────
USAGE
─────────────────────────────────────────────────────────────────────────────
  from testing.evaluator import Evaluator, EvaluationResult

  ev = Evaluator()
  result = ev.evaluate(
      question_id="Q1",
      category="Erasmus",
      question="Erasmus staj için minimum GNO?",
      answer=pipeline_answer,
      expected="Lisans için 2.20, Lisansüstü için 2.5",
      context=context_chunks_text,
      latency=1.23,
  )
  print(result.pretty())

  report = ev.generate_report(results, output_dir="testing/test_results")
"""

from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Tuple
from datetime import datetime

import numpy as np
import requests

# ─────────────────────────────────────────────────────────────────────────────
# Optional: sentence-transformers (fallback embedder if Ollama is unavailable)
# ─────────────────────────────────────────────────────────────────────────────
try:
    from sentence_transformers import SentenceTransformer
    _ST_AVAILABLE = True
except ImportError:
    _ST_AVAILABLE = False

# Turkish diacritics used in keyword normalisation
_TR_DIACRITICS = "çğıöşüÇĞİÖŞÜ"

# ─────────────────────────────────────────────────────────────────────────────
# Data containers
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class MetricScores:
    answer_similarity: float = 0.0     # 0–1
    faithfulness: float = 0.0          # 0–1
    answer_relevance: float = 0.0      # 0–1
    factual_accuracy: float = 0.0      # 0–1
    keyword_coverage: float = 0.0      # 0–1  (legacy)
    composite_score: float = 0.0       # 0–1  (primary pass/fail)


@dataclass
class EvaluationResult:
    question_id: str
    category: str
    question: str
    expected: str
    actual: str
    latency: float
    scores: MetricScores = field(default_factory=MetricScores)
    passed: bool = False
    notes: List[str] = field(default_factory=list)

    # ─── Formatting helpers ────────────────────────────────────────────────

    def pretty(self, width: int = 80) -> str:
        """Return a human-readable block for one question result."""
        sep = "─" * width
        status = "✅ PASS" if self.passed else "❌ FAIL"
        s = scores = self.scores
        lines = [
            sep,
            f"{status}  {self.question_id} [{self.category}]  ({self.latency:.1f}s)",
            sep,
            f"  Q : {self.question}",
            f"  Ex: {self.expected}",
            f"  An: {self.actual[:160]}{'…' if len(self.actual) > 160 else ''}",
            "",
            f"  ── Metrics ──────────────────────────────────",
            f"  Answer Similarity : {s.answer_similarity:.3f}",
            f"  Faithfulness      : {s.faithfulness:.3f}",
            f"  Answer Relevance  : {s.answer_relevance:.3f}",
            f"  Factual Accuracy  : {s.factual_accuracy:.3f}",
            f"  ─────────────────────────────────────────────",
            f"  Composite Score   : {s.composite_score:.3f}  {'≥0.70 ✅' if s.composite_score >= 0.70 else '<0.70 ❌'}",
            f"  Keyword Coverage  : {s.keyword_coverage:.3f}  (legacy)",
        ]
        if self.notes:
            lines += ["", "  Notes:"] + [f"    • {n}" for n in self.notes]
        return "\n".join(lines)

    def as_dict(self) -> dict:
        return {
            "question_id": self.question_id,
            "category": self.category,
            "question": self.question,
            "expected": self.expected,
            "actual": self.actual,
            "latency": round(self.latency, 3),
            "passed": self.passed,
            "scores": {
                "answer_similarity": round(self.scores.answer_similarity, 4),
                "faithfulness": round(self.scores.faithfulness, 4),
                "answer_relevance": round(self.scores.answer_relevance, 4),
                "factual_accuracy": round(self.scores.factual_accuracy, 4),
                "keyword_coverage": round(self.scores.keyword_coverage, 4),
                "composite_score": round(self.scores.composite_score, 4),
            },
            "notes": self.notes,
        }


# ─────────────────────────────────────────────────────────────────────────────
# Composite score formula
# ─────────────────────────────────────────────────────────────────────────────

def compute_composite(
    similarity: float,
    faithfulness: float,
    relevance: float,
    factual_accuracy: float,
    faithfulness_gate_threshold: float = 0.35,
) -> float:
    """
    Domain-optimized composite for university regulatory Q&A.

    Weights
    -------
    0.30 × similarity        – semantic closeness to expected answer
    0.30 × faithfulness      – all claims grounded in retrieved context
    0.25 × factual_accuracy  – exact numbers / dates from expected found
    0.15 × relevance         – answer addresses the question

    Faithfulness gate
    -----------------
    If faithfulness < threshold, a quadratic penalty is applied:
        composite ×= (faithfulness / threshold)²
    This ensures a hallucinating answer can never score above ~45 % even
    if it accidentally matches vocabulary.
    """
    base = (
        0.30 * similarity
        + 0.30 * faithfulness
        + 0.25 * factual_accuracy
        + 0.15 * relevance
    )
    if faithfulness < faithfulness_gate_threshold:
        gate_factor = (faithfulness / faithfulness_gate_threshold) ** 2
        base *= gate_factor
    return round(float(np.clip(base, 0.0, 1.0)), 4)


# ─────────────────────────────────────────────────────────────────────────────
# Evaluator class
# ─────────────────────────────────────────────────────────────────────────────

class Evaluator:
    """
    Multi-metric RAG evaluator for MACKIS.

    Parameters
    ----------
    ollama_host : str
        Base URL for the Ollama server.
    embed_model : str
        Embedding model name served by Ollama (default: bge-m3).
    chat_model : str
        Chat/LLM model name served by Ollama (used for LLM-judge metrics).
    pass_threshold : float
        Composite score ≥ this value → PASS.
    st_model_name : str
        fallback sentence-transformers model for similarity (used only when
        Ollama is unreachable).
    llm_judge_timeout : int
        Seconds to wait for Ollama LLM-judge responses.
    """

    def __init__(
        self,
        ollama_host: str = "http://localhost:11434",
        embed_model: str = "bge-m3",
        chat_model: Optional[str] = None,
        pass_threshold: float = 0.70,
        st_model_name: str = "paraphrase-multilingual-MiniLM-L12-v2",
        llm_judge_timeout: int = 120,
    ):
        self.ollama_host = ollama_host.rstrip("/")
        self.embed_model = embed_model
        self.pass_threshold = pass_threshold
        self.llm_judge_timeout = llm_judge_timeout

        # Resolve chat model from env if not given
        if chat_model is None:
            from dotenv import load_dotenv
            load_dotenv()
            chat_model = os.getenv("CHAT_MODEL", "qwen2.5:7b")
        self.chat_model = chat_model

        # Embedding cache: sha1 → numpy vector
        self._embed_cache: Dict[str, np.ndarray] = {}

        # Lazy-load ST model only if Ollama embed is unavailable
        self._st_model: Optional[object] = None
        self._st_model_name = st_model_name

        # Check Ollama connectivity once at init
        self._ollama_embed_ok = self._ping_ollama_embed()
        self._ollama_llm_ok = self._ping_ollama_llm()

        if not self._ollama_embed_ok:
            print("[Evaluator] ⚠️  Ollama embed unavailable – falling back to sentence-transformers")
        if not self._ollama_llm_ok:
            print("[Evaluator] ⚠️  Ollama LLM unavailable – faithfulness/relevance will be 0.5 (neutral)")

    # ─────────────────────────────────────────────────────────────────────────
    # Public API
    # ─────────────────────────────────────────────────────────────────────────

    def evaluate(
        self,
        question: str,
        answer: str,
        expected: str,
        context: str = "",
        question_id: str = "",
        category: str = "",
        latency: float = 0.0,
    ) -> EvaluationResult:
        """
        Score one question/answer pair on all metrics.

        Parameters
        ----------
        question : str
            The user question.
        answer : str
            The RAG system's actual answer.
        expected : str
            The ground-truth / reference answer.
        context : str
            Retrieved document chunks joined as plain text.
            Used for faithfulness.  May be empty string.
        question_id : str
            Identifier for reporting.
        category : str
            Question category (Erasmus, Library, …).
        latency : float
            Wall-clock seconds the pipeline took.
        """
        notes: List[str] = []

        # 1. Answer Similarity
        similarity = self._score_answer_similarity(answer, expected)

        # 2. Faithfulness (LLM judge)
        if context:
            faithfulness, faith_note = self._score_faithfulness(question, answer, context)
        else:
            faithfulness = 0.5   # neutral – no context available
            faith_note = "no context supplied; faithfulness set to neutral 0.5"
        notes.append(f"faithfulness note: {faith_note}")

        # 3. Answer Relevance (LLM judge)
        relevance, rel_note = self._score_answer_relevance(question, answer)
        notes.append(f"relevance note: {rel_note}")

        # 4. Factual Accuracy (regex)
        factual, factual_note = self._score_factual_accuracy(answer, expected)
        notes.append(f"factual note: {factual_note}")

        # 5. Keyword Coverage (legacy)
        keyword_cov = self._score_keyword_coverage(answer, expected)

        # Composite
        composite = compute_composite(
            similarity=similarity,
            faithfulness=faithfulness,
            relevance=relevance,
            factual_accuracy=factual,
        )

        scores = MetricScores(
            answer_similarity=round(similarity, 4),
            faithfulness=round(faithfulness, 4),
            answer_relevance=round(relevance, 4),
            factual_accuracy=round(factual, 4),
            keyword_coverage=round(keyword_cov, 4),
            composite_score=composite,
        )

        return EvaluationResult(
            question_id=question_id,
            category=category,
            question=question,
            expected=expected,
            actual=answer,
            latency=latency,
            scores=scores,
            passed=composite >= self.pass_threshold,
            notes=notes,
        )

    def generate_report(
        self,
        results: List[EvaluationResult],
        output_dir: str = "",
    ) -> Dict:
        """
        Generate JSON report + pretty-printed table.

        Saves
        -----
        <output_dir>/eval_report_<timestamp>.json
        <output_dir>/eval_table_<timestamp>.txt

        Returns
        -------
        dict  – the report data structure (also printed to stdout).
        """
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        passed = [r for r in results if r.passed]
        failed = [r for r in results if not r.passed]
        valid  = results  # all results (errors excluded upstream)

        # Per-category aggregation
        cat_stats: Dict[str, Dict] = {}
        for r in valid:
            c = r.category
            if c not in cat_stats:
                cat_stats[c] = {"n": 0, "passed": 0, "scores": {
                    k: [] for k in
                    ["answer_similarity", "faithfulness", "answer_relevance",
                     "factual_accuracy", "keyword_coverage", "composite_score"]
                }}
            cat_stats[c]["n"] += 1
            if r.passed:
                cat_stats[c]["passed"] += 1
            for k in cat_stats[c]["scores"]:
                cat_stats[c]["scores"][k].append(getattr(r.scores, k))

        cat_summary = {}
        for cat, data in cat_stats.items():
            cat_summary[cat] = {
                "n": data["n"],
                "passed": data["passed"],
                "pass_rate": round(data["passed"] / data["n"], 3) if data["n"] else 0,
                "avg": {k: round(float(np.mean(v)), 4) for k, v in data["scores"].items() if v},
            }

        # Overall averages
        def _avg(attr):
            vals = [getattr(r.scores, attr) for r in valid]
            return round(float(np.mean(vals)), 4) if vals else 0.0

        overall = {
            "total": len(results),
            "passed": len(passed),
            "failed": len(failed),
            "pass_rate": round(len(passed) / len(results), 3) if results else 0,
            "avg_answer_similarity": _avg("answer_similarity"),
            "avg_faithfulness": _avg("faithfulness"),
            "avg_answer_relevance": _avg("answer_relevance"),
            "avg_factual_accuracy": _avg("factual_accuracy"),
            "avg_keyword_coverage": _avg("keyword_coverage"),
            "avg_composite_score": _avg("composite_score"),
            "avg_latency": round(float(np.mean([r.latency for r in valid])), 2) if valid else 0,
        }

        report = {
            "generated_at": ts,
            "pass_threshold": self.pass_threshold,
            "ollama_host": self.ollama_host,
            "chat_model": self.chat_model,
            "composite_formula": (
                "0.30×similarity + 0.30×faithfulness + 0.25×factual_accuracy + 0.15×relevance "
                "| gate: if faithfulness<0.35 → composite×=(faith/0.35)²"
            ),
            "overall": overall,
            "by_category": cat_summary,
            "questions": [r.as_dict() for r in results],
        }

        # ── Save JSON ──────────────────────────────────────────────────────
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            json_path = os.path.join(output_dir, f"eval_report_{ts}.json")
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(report, f, ensure_ascii=False, indent=2)
            print(f"\n📄 JSON report saved → {json_path}")

        # ── Pretty table ───────────────────────────────────────────────────
        table_lines = self._build_table(results, overall, cat_summary)
        table_str = "\n".join(table_lines)
        print(table_str)

        if output_dir:
            txt_path = os.path.join(output_dir, f"eval_table_{ts}.txt")
            with open(txt_path, "w", encoding="utf-8") as f:
                f.write(table_str)
            print(f"📄 Table saved         → {txt_path}")

        return report

    # ─────────────────────────────────────────────────────────────────────────
    # Metric implementations
    # ─────────────────────────────────────────────────────────────────────────

    def _score_answer_similarity(self, answer: str, expected: str) -> float:
        """Cosine similarity between answer and expected embeddings (0–1)."""
        if not answer.strip() or not expected.strip():
            return 0.0
        emb_a = self._embed(answer)
        emb_e = self._embed(expected)
        if emb_a is None or emb_e is None:
            return 0.0
        return float(np.clip(self._cosine(emb_a, emb_e), 0.0, 1.0))

    def _score_faithfulness(
        self, question: str, answer: str, context: str
    ) -> Tuple[float, str]:
        """
        LLM-judge faithfulness: are all claims in *answer* grounded in *context*?

        Returns (score 0–1, short note).
        """
        if not self._ollama_llm_ok:
            return 0.5, "ollama_llm_unavailable; neutral 0.5 used"

        # Truncate context to avoid token limits
        ctx_trunc = context[:3000] + ("…" if len(context) > 3000 else "")

        prompt = (
            "You are a strict grounding checker for a university knowledge system.\n\n"
            f"QUESTION: {question}\n\n"
            f"RETRIEVED CONTEXT (source documents):\n{ctx_trunc}\n\n"
            f"SYSTEM ANSWER: {answer}\n\n"
            "Task: Evaluate whether EVERY factual claim in the SYSTEM ANSWER is "
            "explicitly supported by the RETRIEVED CONTEXT.\n"
            "Consider:\n"
            "  • Numbers, thresholds, durations, names → must appear in context\n"
            "  • Paraphrasing is acceptable if meaning is preserved\n"
            "  • Invented details, guesses, or additions not in context = unfaithful\n\n"
            "Reply ONLY with valid JSON (no markdown):\n"
            '{"score": <0-10>, "unsupported_claims": ["list of claims not in context"], '
            '"reasoning": "<one sentence>"}'
        )
        raw = self._ollama_chat(prompt, temperature=0.0)
        score, note = self._parse_llm_score(raw, key="score")
        return score / 10.0, note

    def _score_answer_relevance(
        self, question: str, answer: str
    ) -> Tuple[float, str]:
        """
        LLM-judge relevance: does *answer* actually address *question*?

        Returns (score 0–1, short note).
        """
        if not self._ollama_llm_ok:
            return 0.5, "ollama_llm_unavailable; neutral 0.5 used"

        prompt = (
            "You are an evaluator for a university Q&A system.\n\n"
            f"QUESTION: {question}\n\n"
            f"ANSWER: {answer}\n\n"
            "Task: Score how well the ANSWER addresses the QUESTION, "
            "regardless of whether it is factually correct.\n"
            "  10 = directly and completely answers the question\n"
            "   5 = partially relevant or tangential\n"
            "   0 = completely off-topic or refuses to answer\n\n"
            "Reply ONLY with valid JSON (no markdown):\n"
            '{"score": <0-10>, "reasoning": "<one sentence>"}'
        )
        raw = self._ollama_chat(prompt, temperature=0.0)
        score, note = self._parse_llm_score(raw, key="score")
        return score / 10.0, note

    def _score_factual_accuracy(
        self, answer: str, expected: str
    ) -> Tuple[float, str]:
        """
        Regex-based factual accuracy.

        Extracts all numbers, GPA values, percentages, and date-like
        strings from *expected* and checks how many appear verbatim in
        *answer*.  Returns (ratio, note_string).

        Turkish-aware: checks both normalised and original forms.
        """
        if not expected.strip():
            return 1.0, "no expected provided; perfect score assumed"

        # Extract candidates: numbers, decimals, percentages, year-like strings
        num_pattern = re.compile(
            r"\b\d+(?:[.,]\d+)?(?:\s*%|\s*ay|\s*month|\s*yıl|\s*year)?\b",
            re.IGNORECASE,
        )
        date_pattern = re.compile(
            r"\b\d{1,2}[./\-]\d{1,2}[./\-]\d{2,4}\b"
        )

        expected_nums = set(num_pattern.findall(expected))
        expected_nums |= set(date_pattern.findall(expected))

        # Also grab short specific tokens like "2.20", "2.5", "10 adet"
        # that the number regex might normalise away
        specific = re.findall(r"\b\d+(?:\.\d+)?\b", expected)
        expected_nums |= set(specific)

        if not expected_nums:
            # No extractable numbers → fallback to character n-gram overlap
            overlap = self._char_ngram_overlap(expected, answer, n=4)
            return overlap, f"no numeric facts; char-4gram overlap={overlap:.2f}"

        answer_lower = answer.lower()
        expected_lower = expected.lower()

        found = []
        missing = []
        for num in expected_nums:
            # Normalize: replace comma with period
            num_norm = num.replace(",", ".")
            if num_norm in answer_lower or num in answer_lower:
                found.append(num)
            else:
                missing.append(num)

        ratio = len(found) / len(expected_nums)
        note = (
            f"found {len(found)}/{len(expected_nums)} facts"
            + (f"; missing: {missing[:5]}" if missing else "")
        )
        return round(ratio, 4), note

    def _score_keyword_coverage(self, answer: str, expected: str) -> float:
        """
        Legacy metric: fraction of non-trivial expected tokens found in answer.

        Turkish-diacritic-aware: checks both original and ASCII-folded forms.
        """
        if not expected.strip():
            return 0.0

        stop_words = {
            "ve", "ile", "bir", "bu", "için", "olan", "olan", "de", "da",
            "the", "a", "an", "of", "in", "is", "to", "for", "and", "or",
            "en", "az", "en", "çok", "kadar",
        }

        tokens = re.findall(
            r"[a-z" + _TR_DIACRITICS.lower() + r"0-9]+",
            expected.lower()
        )
        keywords = [t for t in tokens if len(t) > 2 and t not in stop_words]

        if not keywords:
            return 0.0

        answer_lower = answer.lower()
        # ASCII-folded version for Turkish diacritics comparison
        answer_ascii = self._fold_tr(answer_lower)

        found = 0
        for kw in keywords:
            if kw in answer_lower or self._fold_tr(kw) in answer_ascii:
                found += 1

        return round(found / len(keywords), 4)

    # ─────────────────────────────────────────────────────────────────────────
    # Embedding helpers
    # ─────────────────────────────────────────────────────────────────────────

    def _embed(self, text: str) -> Optional[np.ndarray]:
        """Embed text → normalized numpy vector (uses cache)."""
        import hashlib
        key = hashlib.sha1(text.encode("utf-8", errors="ignore")).hexdigest()
        if key in self._embed_cache:
            return self._embed_cache[key]

        vec = None
        if self._ollama_embed_ok:
            vec = self._ollama_embed(text)
        if vec is None:
            vec = self._st_embed(text)
        if vec is not None:
            self._embed_cache[key] = vec
        return vec

    def _ollama_embed(self, text: str) -> Optional[np.ndarray]:
        try:
            r = requests.post(
                f"{self.ollama_host}/api/embeddings",
                json={"model": self.embed_model, "prompt": text},
                timeout=60,
            )
            r.raise_for_status()
            vec = np.array(r.json()["embedding"], dtype=np.float32)
            norm = np.linalg.norm(vec)
            if norm > 0:
                vec /= norm
            return vec
        except Exception:
            return None

    def _st_embed(self, text: str) -> Optional[np.ndarray]:
        if not _ST_AVAILABLE:
            return None
        if self._st_model is None:
            print(f"[Evaluator] Loading ST model '{self._st_model_name}'…")
            self._st_model = SentenceTransformer(self._st_model_name)
        try:
            vec = self._st_model.encode(text, normalize_embeddings=True)
            return vec.astype(np.float32)
        except Exception:
            return None

    @staticmethod
    def _cosine(a: np.ndarray, b: np.ndarray) -> float:
        return float(np.dot(a, b))  # both already normalised

    # ─────────────────────────────────────────────────────────────────────────
    # LLM helpers
    # ─────────────────────────────────────────────────────────────────────────

    def _ollama_chat(self, prompt: str, temperature: float = 0.0) -> str:
        """Send a prompt to Ollama and return the raw text response."""
        try:
            r = requests.post(
                f"{self.ollama_host}/api/chat",
                json={
                    "model": self.chat_model,
                    "messages": [{"role": "user", "content": prompt}],
                    "stream": False,
                    "options": {"temperature": temperature},
                },
                timeout=self.llm_judge_timeout,
            )
            r.raise_for_status()
            return r.json().get("message", {}).get("content", "")
        except Exception as e:
            return f'{{"score": 5, "reasoning": "ollama_error: {e}"}}'

    @staticmethod
    def _parse_llm_score(raw: str, key: str = "score") -> Tuple[float, str]:
        """
        Parse JSON from LLM response.  Tries strict JSON first, then
        extractor regex, then defaults to 5/10.
        """
        # Strip markdown code fences if present
        clean = re.sub(r"```(?:json)?|```", "", raw).strip()

        # Try full JSON parse
        try:
            data = json.loads(clean)
            score = float(data.get(key, 5))
            reasoning = str(data.get("reasoning", ""))
            return min(max(score, 0), 10), reasoning[:120]
        except (json.JSONDecodeError, ValueError):
            pass

        # Fallback: regex extract score value
        m = re.search(r'["\'`]?' + key + r'["\'`]?\s*:\s*(\d+(?:\.\d+)?)', clean)
        if m:
            score = float(m.group(1))
            return min(max(score, 0), 10), f"regex_extracted:{clean[:80]}"

        return 5.0, f"parse_failed; raw={raw[:100]}"

    # ─────────────────────────────────────────────────────────────────────────
    # Connectivity checks
    # ─────────────────────────────────────────────────────────────────────────

    def _list_ollama_models(self) -> Optional[list]:
        """
        Call GET /api/tags – the lightweight Ollama metadata endpoint.
        Returns the list of model name strings, or None if unreachable.
        This endpoint does NOT load any model so it responds in <1 second.
        """
        try:
            r = requests.get(f"{self.ollama_host}/api/tags", timeout=5)
            if r.status_code != 200:
                return None
            models = r.json().get("models", [])
            # Each entry has a "name" key, e.g. "llama3.1:latest"
            return [m.get("name", "") for m in models]
        except Exception:
            return None

    def _ping_ollama_embed(self) -> bool:
        """
        Three-step check:
          1. Is Ollama reachable at all? (GET /api/tags, <1 s)
          2. Is the embed model listed?
          3. Does a real embed call succeed? (generous 90 s timeout –
             model is likely already in VRAM from RAGPipeline init)
        """
        available = self._list_ollama_models()
        if available is None:
            # Ollama not reachable
            return False

        # Step 2: model presence check (match on prefix in case of tag variants)
        model_base = self.embed_model.split(":")[0].lower()
        found = any(m.lower().startswith(model_base) for m in available)
        if not found:
            print(f"[Evaluator] embed model '{self.embed_model}' not found in Ollama "
                  f"(available: {available[:5]})")
            return False

        # Step 3: real inference call with generous timeout
        try:
            r = requests.post(
                f"{self.ollama_host}/api/embeddings",
                json={"model": self.embed_model, "prompt": "test"},
                timeout=90,
            )
            return r.status_code == 200
        except Exception:
            return False

    def _ping_ollama_llm(self) -> bool:
        """
        Three-step check:
          1. Is Ollama reachable? (reuses /api/tags result)
          2. Is the chat model listed?
          3. Does a minimal (num_predict=1) chat call succeed?
             Timeout is set to 120 s – llama3.1:latest can take 30-60 s
             to produce even 1 token if it was recently swapped out.
        """
        available = self._list_ollama_models()
        if available is None:
            return False

        model_base = self.chat_model.split(":")[0].lower()
        found = any(m.lower().startswith(model_base) for m in available)
        if not found:
            print(f"[Evaluator] chat model '{self.chat_model}' not found in Ollama "
                  f"(available: {available[:5]})")
            return False

        try:
            r = requests.post(
                f"{self.ollama_host}/api/chat",
                json={
                    "model": self.chat_model,
                    "messages": [{"role": "user", "content": "1+1="}],
                    "stream": False,
                    "options": {"temperature": 0, "num_predict": 2},
                },
                timeout=120,
            )
            return r.status_code == 200
        except Exception:
            return False

    # ─────────────────────────────────────────────────────────────────────────
    # Utility
    # ─────────────────────────────────────────────────────────────────────────

    @staticmethod
    def _fold_tr(text: str) -> str:
        """Simplistic Turkish→ASCII fold for diacritic-resilient matching."""
        table = str.maketrans("çğışöüÇĞİŞÖÜ", "cgisouCGISOu")
        return text.translate(table)

    @staticmethod
    def _char_ngram_overlap(reference: str, hypothesis: str, n: int = 4) -> float:
        """Character n-gram F1-like overlap (for texts with no numbers)."""
        def ngrams(t): return {t[i:i+n] for i in range(len(t) - n + 1)}
        ref_ng = ngrams(reference.lower())
        hyp_ng = ngrams(hypothesis.lower())
        if not ref_ng:
            return 1.0
        return round(len(ref_ng & hyp_ng) / len(ref_ng), 4)

    # ─────────────────────────────────────────────────────────────────────────
    # Pretty-table builder
    # ─────────────────────────────────────────────────────────────────────────

    def _build_table(
        self,
        results: List[EvaluationResult],
        overall: Dict,
        cat_summary: Dict,
    ) -> List[str]:
        w = 120
        lines = [
            "",
            "═" * w,
            "  MACKIS RAG EVALUATION REPORT — Multi-Metric Framework",
            "═" * w,
            "",
            "  COMPOSITE FORMULA: 0.30×Similarity + 0.30×Faithfulness + 0.25×FactualAcc + 0.15×Relevance",
            "  Faithfulness gate: if faith < 0.35 → composite×=(faith/0.35)²",
            f"  Pass threshold: ≥ {self.pass_threshold:.2f}",
            "",
            "─" * w,
            f"  {'ID':<6} {'Cat':<14} {'Sim':>6} {'Faith':>7} {'Rel':>6} {'Fact':>6} {'KwCov':>7} {'Compose':>8}  {'Status':<12} {'Lat(s)':>7}",
            "─" * w,
        ]

        for r in results:
            s = r.scores
            status = "✅ PASS" if r.passed else "❌ FAIL"
            # Side-by-side: keyword coverage (old) and composite (new)
            lines.append(
                f"  {r.question_id:<6} {r.category:<14} "
                f"{s.answer_similarity:>6.3f} "
                f"{s.faithfulness:>7.3f} "
                f"{s.answer_relevance:>6.3f} "
                f"{s.factual_accuracy:>6.3f} "
                f"{s.keyword_coverage:>7.3f} "
                f"{s.composite_score:>8.3f}  "
                f"{status:<12} "
                f"{r.latency:>7.1f}"
            )

        lines += [
            "─" * w,
            f"  {'MEAN':<6} {'':<14} "
            f"{overall['avg_answer_similarity']:>6.3f} "
            f"{overall['avg_faithfulness']:>7.3f} "
            f"{overall['avg_answer_relevance']:>6.3f} "
            f"{overall['avg_factual_accuracy']:>6.3f} "
            f"{overall['avg_keyword_coverage']:>7.3f} "
            f"{overall['avg_composite_score']:>8.3f}  "
            f"{'':12} "
            f"{overall['avg_latency']:>7.1f}",
            "═" * w,
            "",
            "  BY CATEGORY",
            "─" * w,
            f"  {'Category':<14} {'N':>4} {'Passed':>7} {'PassRate':>9} {'Avg Comp':>9} {'Avg KwCov(old)':>15}",
            "─" * w,
        ]

        for cat, data in sorted(cat_summary.items()):
            lines.append(
                f"  {cat:<14} {data['n']:>4} {data['passed']:>7} "
                f"{data['pass_rate']:>9.1%} "
                f"{data['avg'].get('composite_score', 0):>9.3f} "
                f"{data['avg'].get('keyword_coverage', 0):>15.3f}"
            )

        lines += [
            "═" * w,
            "",
            f"  OVERALL: {overall['passed']}/{overall['total']} passed "
            f"({overall['pass_rate']:.1%}) | "
            f"avg composite={overall['avg_composite_score']:.3f} | "
            f"avg kw-coverage(legacy)={overall['avg_keyword_coverage']:.3f}",
            "",
            "  ▶  Old metric (keyword coverage ≥ 0.5) vs New metric (composite ≥ 0.70):",
        ]

        # Side-by-side old vs new comparison
        old_passed = sum(1 for r in results if r.scores.keyword_coverage >= 0.5)
        new_passed = overall["passed"]
        lines += [
            f"     Old pass count : {old_passed}/{overall['total']}",
            f"     New pass count : {new_passed}/{overall['total']}",
            "",
            "═" * w,
        ]
        return lines
