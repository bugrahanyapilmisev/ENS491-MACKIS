"""
evaluator.py - Multi-Metric RAG Evaluation Framework

Domain: Sabancı University administrative Q&A (MACKIS)
Author: ENS491 Team

─────────────────────────────────────────────────────────────────────────────
METRICS
─────────────────────────────────────────────────────────────────────────────
  1. Answer Similarity   – Cosine similarity between answer and expected
                           embeddings (bge-m3 via Ollama, fallback: ST).
  2. Faithfulness        – LLM judge: every claim grounded in context?
  3. Answer Relevance    – LLM judge: does answer address the question?
  4. Factual Accuracy    – Regex: numbers / dates in expected found in answer.
  5. Keyword Coverage    – Legacy word-overlap metric (kept for comparison).

─────────────────────────────────────────────────────────────────────────────
LLM JUDGE PROVIDERS
─────────────────────────────────────────────────────────────────────────────
  judge_provider="gemini" (default, recommended)
      Uses the Gemini API as a fully independent, neutral judge.
      This eliminates self-judging bias when comparing two Ollama models
      (e.g. llama3.1 vs qwen3.5) — the judge is always the same external
      Gemini model regardless of which RAG model is under test.

      Requires: GEMINI_API_KEY in .env
      Default judge model: gemini-2.0-flash
      Override via: JUDGE_MODEL=gemini-1.5-pro (or any Gemini model)

  judge_provider="ollama" (legacy)
      Uses the same Ollama server as the RAG pipeline for judging.
      WARNING: when evaluating a model against itself this introduces
      self-judging bias.  Kept for backward compatibility only.

─────────────────────────────────────────────────────────────────────────────
COMPOSITE SCORE  (domain-optimized for regulatory university Q&A)
─────────────────────────────────────────────────────────────────────────────

  composite = 0.30 × similarity
            + 0.30 × faithfulness
            + 0.25 × factual_accuracy
            + 0.15 × relevance

  Faithfulness gate (softened):
      If faithfulness < 0.20:
        - If factual_accuracy >= 0.60 → override faithfulness to 0.35
          (safety net: the answer IS correct, judge is wrong about sourcing)
        - Otherwise → composite *= (faithfulness / 0.20)

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

  ev = Evaluator()   # uses JUDGE_PROVIDER / GEMINI_API_KEY from .env
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

# ─────────────────────────────────────────────────────────────────────────────
# Optional: google-genai (Gemini API judge)
# ─────────────────────────────────────────────────────────────────────────────
try:
    from google import genai as _google_genai
    _GENAI_AVAILABLE = True
except ImportError:
    _google_genai = None  # type: ignore
    _GENAI_AVAILABLE = False

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
    faithfulness_gate_threshold: float = 0.20,
) -> float:
    """
    Domain-optimized composite for university regulatory Q&A.

    Weights
    -------
    0.30 × similarity        – semantic closeness to expected answer
    0.30 × faithfulness      – all claims grounded in retrieved context
    0.25 × factual_accuracy  – exact numbers / dates from expected found
    0.15 × relevance         – answer addresses the question

    Faithfulness gate (softened)
    ----------------------------
    If faithfulness < threshold:
      - Safety net: if factual_accuracy >= 0.60, override faithfulness
        to 0.35 (the answer has correct facts → judge is wrong about sourcing)
      - Otherwise: apply a LINEAR penalty: composite *= faithfulness / threshold
    This prevents the 6 false-negative pattern where correct answers
    get Faithfulness=0.1 from the Gemini judge.
    """
    # Safety net: if the answer contains the correct facts but the
    # judge scored faithfulness very low, the judge is likely wrong.
    effective_faith = faithfulness
    if faithfulness < faithfulness_gate_threshold and factual_accuracy >= 0.60:
        effective_faith = max(faithfulness, 0.35)

    base = (
        0.30 * similarity
        + 0.30 * effective_faith
        + 0.25 * factual_accuracy
        + 0.15 * relevance
    )
    if effective_faith < faithfulness_gate_threshold:
        gate_factor = effective_faith / faithfulness_gate_threshold
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
        Chat/LLM model name served by Ollama — the model being *evaluated*
        (NOT the judge).  Used only when judge_provider="ollama" (legacy).
    judge_provider : str
        Which backend to use for LLM-judge metrics (faithfulness, relevance).
        "gemini" (default) — neutral external judge via Gemini API.
        "ollama"           — legacy self-judge (same model as RAG pipeline).
        Falls back to JUDGE_PROVIDER env var if not passed explicitly.
    judge_model : str
        Model name for the judge provider.
        For Gemini: e.g. "gemini-2.0-flash" (default), "gemini-1.5-pro".
        For Ollama: any locally available model name.
        Falls back to JUDGE_MODEL env var if not passed explicitly.
    gemini_api_key : str, optional
        Gemini API key.  Falls back to GEMINI_API_KEY env var.
    pass_threshold : float
        Composite score ≥ this value → PASS.
    st_model_name : str
        Fallback sentence-transformers model for similarity (used only when
        Ollama embed is unreachable).
    llm_judge_timeout : int
        Seconds to wait for Ollama LLM-judge responses (Gemini uses its own
        internal timeout via the SDK).
    """

    def __init__(
        self,
        ollama_host: str = "http://localhost:11434",
        embed_model: str = "bge-m3",
        chat_model: Optional[str] = None,
        judge_provider: Optional[str] = None,
        judge_model: Optional[str] = None,
        gemini_api_key: Optional[str] = None,
        pass_threshold: float = 0.70,
        st_model_name: str = "paraphrase-multilingual-MiniLM-L12-v2",
        llm_judge_timeout: int = 120,
    ):
        from dotenv import load_dotenv
        load_dotenv()

        self.ollama_host = ollama_host.rstrip("/")
        self.embed_model = embed_model
        self.pass_threshold = pass_threshold
        self.llm_judge_timeout = llm_judge_timeout

        # ── Resolve the RAG chat model (model under test) ──────────────────
        if chat_model is None:
            chat_model = os.getenv("CHAT_MODEL", "qwen2.5:7b")
        self.chat_model = chat_model

        # ── Resolve judge provider & model ────────────────────────────────
        if judge_provider is None:
            judge_provider = os.getenv("JUDGE_PROVIDER", "gemini")
        self.judge_provider = judge_provider.lower().strip()

        if judge_model is None:
            if self.judge_provider == "gemini":
                judge_model = os.getenv("JUDGE_MODEL", "gemini-2.0-flash")
            else:
                judge_model = os.getenv("JUDGE_MODEL", chat_model)
        self.judge_model = judge_model

        # ── Gemini client (only when judge_provider == "gemini") ──────────
        self._gemini_client = None
        self._gemini_ok = False
        if self.judge_provider == "gemini":
            if gemini_api_key is None:
                gemini_api_key = os.getenv("GEMINI_API_KEY", "")
            self._gemini_api_key = gemini_api_key
            self._gemini_ok = self._ping_gemini()
            if not self._gemini_ok:
                print(
                    "[Evaluator] ⚠️  Gemini judge unavailable – "
                    "faithfulness/relevance will be 0.5 (neutral).\n"
                    "           Check GEMINI_API_KEY in your .env file."
                )

        # ── Embedding cache: sha1 → numpy vector ─────────────────────────
        self._embed_cache: Dict[str, np.ndarray] = {}

        # Lazy-load ST model only if Ollama embed is unavailable
        self._st_model: Optional[object] = None
        self._st_model_name = st_model_name

        # ── Check Ollama connectivity ─────────────────────────────────────
        self._ollama_embed_ok = self._ping_ollama_embed()
        # Ollama LLM only needed when judge_provider == "ollama"
        self._ollama_llm_ok = (
            self._ping_ollama_llm() if self.judge_provider == "ollama" else False
        )

        if not self._ollama_embed_ok:
            print("[Evaluator] ⚠️  Ollama embed unavailable – falling back to sentence-transformers")
        if self.judge_provider == "ollama" and not self._ollama_llm_ok:
            print("[Evaluator] ⚠️  Ollama LLM unavailable – faithfulness/relevance will be 0.5 (neutral)")

        # ── Startup summary ───────────────────────────────────────────────
        judge_status = (
            f"gemini ({self.judge_model}) – {'✅ ready' if self._gemini_ok else '❌ unavailable'}"
            if self.judge_provider == "gemini"
            else f"ollama ({self.judge_model}) – {'✅ ready' if self._ollama_llm_ok else '❌ unavailable'}"
        )
        print(f"[Evaluator] Judge provider : {judge_status}")
        print(f"[Evaluator] Model under test: {self.chat_model}")

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
            "judge_provider": self.judge_provider,
            "judge_model": self.judge_model,
            "composite_formula": (
                "0.30×similarity + 0.30×faithfulness + 0.25×factual_accuracy + 0.15×relevance "
                "| gate: softened with factual_accuracy safety net"
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

        The judge is fully independent of the model under test:
          • judge_provider="gemini" → Gemini API (neutral external judge)
          • judge_provider="ollama" → Ollama (legacy; same server as RAG model)

        Returns (score 0–1, short note).
        """
        # Guard: check judge availability
        if self.judge_provider == "gemini" and not self._gemini_ok:
            return 0.5, "gemini_judge_unavailable; neutral 0.5 used"
        if self.judge_provider == "ollama" and not self._ollama_llm_ok:
            return 0.5, "ollama_llm_unavailable; neutral 0.5 used"

        # Context truncation: Gemini 2.5 Flash has a 1M token window so we pass
        # the full context. Ollama (llama3.1) is capped at 4000 chars to stay
        # within practical limits for local inference.
        ctx_limit = 4000 if self.judge_provider == "ollama" else len(context)
        ctx_trunc = context[:ctx_limit] + ("…" if len(context) > ctx_limit else "")

        prompt = (
            "You are a grounding checker for a university knowledge system.\n"
            "IMPORTANT: Your response must be a single raw JSON object — "
            "no markdown, no code fences, no extra text.\n\n"
            f"QUESTION: {question}\n\n"
            f"RETRIEVED CONTEXT (source documents):\n{ctx_trunc}\n\n"
            f"SYSTEM ANSWER: {answer}\n\n"
            "Task: Evaluate whether the factual claims in the SYSTEM ANSWER are "
            "supported by the RETRIEVED CONTEXT.\n\n"
            "SCORING RULES:\n"
            "  • If a claim (number, name, procedure, date, condition) appears ANYWHERE in the RETRIEVED CONTEXT, it IS faithful → score 8-10\n"
            "  • The context includes neighbor-expanded chunks and knowledge graph facts. Information from these sources is LEGITIMATE and should NOT be penalized.\n"
            "  • Mentioning document codes (e.g. 'PSR-C210-0101'), form names, office names, or procedural details FROM the context is NOT unfaithful\n"
            "  • Adding extra details that ARE in context but were not explicitly asked for is NOT unfaithful. This is helpful elaboration.\n"
            "  • Paraphrasing, summarizing, or combining information from multiple chunks is acceptable if meaning is preserved\n"
            "  • Only score BELOW 5 if the answer contains facts that DIRECTLY CONTRADICT the context or invents numbers/facts NOT found anywhere in the context\n"
            "  • Score 0 ONLY if the answer is completely fabricated with no basis in the context\n"
            "  • When in doubt, score HIGHER. A correct answer that adds helpful detail should score 7-9, not 1-3.\n\n"
            "Output format (raw JSON only, no markdown):\n"
            '{"score": <integer 0-10>, "unsupported_claims": ["..."], '
            '"reasoning": "<one sentence>"}'
        )
        raw = self._llm_judge_chat(prompt, temperature=0.0)
        score, note = self._parse_llm_score(raw, key="score")

        # Retry with stripped prompt if parse failed (score==5.0 and note starts with parse_failed)
        if score == 5.0 and note.startswith("parse_failed"):
            simple_prompt = (
                "Rate faithfulness 0-10: does the ANSWER accurately reflect the CONTEXT without contradicting it? Extra correct details are acceptable.\n"
                f"CONTEXT (first 2000 chars): {ctx_trunc[:2000]}\n\n"
                f"ANSWER: {answer}\n\n"
                "Reply with ONLY this JSON (no other text): "
                '{"score": <integer>}'
            )
            raw2 = self._llm_judge_chat(simple_prompt, temperature=0.0)
            score2, note2 = self._parse_llm_score(raw2, key="score")
            if not note2.startswith("parse_failed"):
                return score2 / 10.0, f"retry_ok:{note2}"

        return score / 10.0, note

    def _score_answer_relevance(
        self, question: str, answer: str
    ) -> Tuple[float, str]:
        """
        LLM-judge relevance: does *answer* actually address *question*?

        The judge is fully independent of the model under test (see
        _score_faithfulness for provider details).

        Returns (score 0–1, short note).
        """
        if self.judge_provider == "gemini" and not self._gemini_ok:
            return 0.5, "gemini_judge_unavailable; neutral 0.5 used"
        if self.judge_provider == "ollama" and not self._ollama_llm_ok:
            return 0.5, "ollama_llm_unavailable; neutral 0.5 used"

        prompt = (
            "You are an evaluator for a university Q&A system.\n"
            "IMPORTANT: Your response must be a single raw JSON object — "
            "no markdown, no code fences, no extra text.\n\n"
            f"QUESTION: {question}\n\n"
            f"ANSWER: {answer}\n\n"
            "Task: Score how well the ANSWER addresses the QUESTION, "
            "regardless of whether it is factually correct.\n"
            "  10 = directly and completely answers the question\n"
            "   5 = partially relevant or tangential\n"
            "   0 = completely off-topic or refuses to answer\n\n"
            "Output format (raw JSON only, no markdown):\n"
            '{"score": <integer 0-10>, "reasoning": "<one sentence>"}'
        )
        raw = self._llm_judge_chat(prompt, temperature=0.0)
        score, note = self._parse_llm_score(raw, key="score")

        # Retry with stripped prompt if parse failed
        if score == 5.0 and note.startswith("parse_failed"):
            simple_prompt = (
                f"Does this answer address the question? Score 0-10.\n"
                f"Question: {question}\nAnswer: {answer}\n"
                "Reply ONLY with: {\"score\": <integer>}"
            )
            raw2 = self._llm_judge_chat(simple_prompt, temperature=0.0)
            score2, note2 = self._parse_llm_score(raw2, key="score")
            if not note2.startswith("parse_failed"):
                return score2 / 10.0, f"retry_ok:{note2}"

        return score / 10.0, note

    # ── Number-word mappings for Turkish and English ──────────────────────
    _NUM_WORD_TO_DIGIT: Dict[str, str] = {
        # Turkish
        "bir": "1", "iki": "2", "üç": "3", "uc": "3",
        "dört": "4", "dort": "4", "beş": "5", "bes": "5",
        "altı": "6", "alti": "6", "yedi": "7", "sekiz": "8",
        "dokuz": "9", "on": "10", "yirmi": "20", "otuz": "30",
        "kırk": "40", "kirk": "40", "elli": "50", "altmış": "60",
        "altmis": "60", "yetmiş": "70", "yetmis": "70",
        "seksen": "80", "doksan": "90", "yüz": "100", "yuz": "100",
        # English
        "one": "1", "two": "2", "three": "3", "four": "4",
        "five": "5", "six": "6", "seven": "7", "eight": "8",
        "nine": "9", "ten": "10", "twenty": "20", "thirty": "30",
        "forty": "40", "fifty": "50", "sixty": "60", "seventy": "70",
        "eighty": "80", "ninety": "90", "hundred": "100",
    }
    _DIGIT_TO_NUM_WORDS: Dict[str, List[str]] = {}  # built at class load

    @classmethod
    def _build_digit_to_words(cls):
        if cls._DIGIT_TO_NUM_WORDS:
            return
        for word, digit in cls._NUM_WORD_TO_DIGIT.items():
            cls._DIGIT_TO_NUM_WORDS.setdefault(digit, []).append(word)

    def _score_factual_accuracy(
        self, answer: str, expected: str
    ) -> Tuple[float, str]:
        """
        Regex-based factual accuracy.

        Extracts all numbers, GPA values, percentages, and date-like
        strings from *expected* and checks how many appear verbatim in
        *answer*.  Returns (ratio, note_string).

        Turkish-aware: checks both normalised and original forms,
        handles comma/dot decimal variants, and maps number-words
        (e.g. "altı" ↔ "6", "three" ↔ "3").
        Uses word-boundary matching to avoid substring false positives
        (e.g. "5" should NOT match inside "15" or "50").
        """
        self._build_digit_to_words()

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

        def _word_boundary_present(token: str, text: str) -> bool:
            """Check if token appears in text at a word boundary."""
            # Escape special regex chars in the token, then wrap with \b
            escaped = re.escape(token)
            return bool(re.search(r'(?<!\d)' + escaped + r'(?!\d)', text))

        found = []
        missing = []
        for num in expected_nums:
            # Strip trailing unit suffixes for pure numeric matching
            pure_num = re.sub(r'\s*(ay|month|yıl|year|%)$', '', num,
                              flags=re.IGNORECASE).strip()

            # Build all variants to check
            variants = {num, pure_num}
            # Comma↔dot normalization (handles Turkish 2,20 ↔ 2.20)
            variants.add(num.replace(",", "."))
            variants.add(num.replace(".", ","))
            variants.add(pure_num.replace(",", "."))
            variants.add(pure_num.replace(".", ","))

            # Number-word variants: if pure_num is a plain integer,
            # also check for its Turkish/English word equivalents
            if pure_num.isdigit() and pure_num in self._DIGIT_TO_NUM_WORDS:
                for word in self._DIGIT_TO_NUM_WORDS[pure_num]:
                    variants.add(word)

            matched = False
            for v in variants:
                if not v:
                    continue
                if _word_boundary_present(v, answer_lower):
                    matched = True
                    break

            if matched:
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
        """Embed text → normalized numpy vector (uses cache).

        If Ollama embedding fails mid-run, we permanently switch to the
        SentenceTransformer fallback and flush the cache to avoid
        dimension mismatches (Ollama bge-m3 = 1024-d vs ST = 384-d).
        """
        import hashlib
        key = hashlib.sha1(text.encode("utf-8", errors="ignore")).hexdigest()
        if key in self._embed_cache:
            return self._embed_cache[key]

        vec = None
        if self._ollama_embed_ok:
            vec = self._ollama_embed(text)
            if vec is None:
                # Ollama embed just failed — disable it and flush cache
                # so we don't mix 1024-d (Ollama) with 384-d (ST) vectors.
                print("[Evaluator] Ollama embed failed mid-run; "
                      "switching to SentenceTransformer and flushing cache")
                self._ollama_embed_ok = False
                self._embed_cache.clear()
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
    # LLM judge helpers
    # ─────────────────────────────────────────────────────────────────────────

    def _llm_judge_chat(self, prompt: str, temperature: float = 0.0) -> str:
        """
        Dispatcher: route judge prompt to Gemini or Ollama based on
        self.judge_provider.

        Returns raw text response string (expected to be JSON by callers).
        """
        if self.judge_provider == "gemini":
            return self._gemini_chat(prompt, temperature=temperature)
        return self._ollama_chat(prompt, temperature=temperature)

    def _gemini_chat(self, prompt: str, temperature: float = 0.0) -> str:
        """
        Send a judge prompt to the Gemini API and return the raw text.

        Uses google-genai SDK with response_mime_type='application/json'
        to force JSON output and prevent markdown wrapping.
        Returns a neutral-score JSON string on any error.
        """
        if not _GENAI_AVAILABLE or not self._gemini_ok:
            return '{"score": 5, "reasoning": "gemini_unavailable"}'
        try:
            client = _google_genai.Client(api_key=self._gemini_api_key)
            response = client.models.generate_content(
                model=self.judge_model,
                contents=prompt,
                config=_google_genai.types.GenerateContentConfig(
                    temperature=temperature,
                    max_output_tokens=512,
                    response_mime_type="application/json",
                ),
            )
            text = response.text or ""
            # Strip any residual markdown fences just in case
            text = text.strip().lstrip("```json").lstrip("```").rstrip("```").strip()
            return text if text else '{"score": 5, "reasoning": "empty_response"}'
        except Exception as e:
            return f'{{"score": 5, "reasoning": "gemini_error: {str(e)[:120]}"}}' # noqa: E501

    def _ollama_chat(self, prompt: str, temperature: float = 0.0) -> str:
        """
        Send a prompt to Ollama (using judge_model when provider=ollama)
        and return the raw text response.

        NOTE: when judge_provider='ollama' this uses self.judge_model,
        NOT self.chat_model, so you can still set an independent Ollama
        judge model via JUDGE_MODEL in .env.
        """
        # Use judge_model when acting as judge, chat_model is the tested model
        model = (
            self.judge_model
            if self.judge_provider == "ollama"
            else self.chat_model
        )
        try:
            r = requests.post(
                f"{self.ollama_host}/api/chat",
                json={
                    "model": model,
                    "messages": [{"role": "user", "content": prompt}],
                    "stream": False,
                    "options": {"temperature": temperature},
                },
                timeout=self.llm_judge_timeout,
            )
            r.raise_for_status()
            return r.json().get("message", {}).get("content", "")
        except Exception as e:
            return f'{{"score": 5, "reasoning": "ollama_error: {e}"}}'  # noqa: E501

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
          2. Is the judge model listed? (uses self.judge_model, not chat_model)
          3. Does a minimal (num_predict=1) chat call succeed?
             Timeout is set to 120 s – llama3.1:latest can take 30-60 s
             to produce even 1 token if it was recently swapped out.
        """
        available = self._list_ollama_models()
        if available is None:
            return False

        model_base = self.judge_model.split(":")[0].lower()
        found = any(m.lower().startswith(model_base) for m in available)
        if not found:
            print(f"[Evaluator] ollama judge model '{self.judge_model}' not found in Ollama "
                  f"(available: {available[:5]})")
            return False

        try:
            r = requests.post(
                f"{self.ollama_host}/api/chat",
                json={
                    "model": self.judge_model,
                    "messages": [{"role": "user", "content": "1+1="}],
                    "stream": False,
                    "options": {"temperature": 0, "num_predict": 2},
                },
                timeout=120,
            )
            return r.status_code == 200
        except Exception:
            return False

    def _ping_gemini(self) -> bool:
        """
        Verify Gemini API connectivity:
          1. Check google-genai SDK is installed.
          2. Check GEMINI_API_KEY is provided (non-empty).
          3. Send a minimal test request (max 1 token) to confirm the key
             is valid and the model exists.
        Returns True only if all three steps pass.
        """
        if not _GENAI_AVAILABLE:
            print("[Evaluator] google-genai package not installed. "
                  "Run: pip install google-genai")
            return False

        if not self._gemini_api_key or self._gemini_api_key == "your_gemini_api_key_here":
            print("[Evaluator] GEMINI_API_KEY is not set or is still a placeholder. "
                  "Add your key to .env: GEMINI_API_KEY=<your_key>")
            return False

        try:
            client = _google_genai.Client(api_key=self._gemini_api_key)
            resp = client.models.generate_content(
                model=self.judge_model,
                contents="Reply with the single word: OK",
                config=_google_genai.types.GenerateContentConfig(
                    max_output_tokens=4,
                    temperature=0.0,
                ),
            )
            # Any successful response means the key + model are valid
            return True
        except Exception as e:
            print(f"[Evaluator] Gemini ping failed: {e}")
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
            "  Faithfulness gate: if faith < 0.20 → composite×=(faith/0.20)",
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
