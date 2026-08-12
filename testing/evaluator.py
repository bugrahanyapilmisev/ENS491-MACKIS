"""
evaluator.py - RAGAS-Based RAG Evaluation Framework

Domain: Sabancı University administrative Q&A (MACKIS)
Author: ENS491 Team

─────────────────────────────────────────────────────────────────────────────
METRICS  (powered by RAGAS library)
─────────────────────────────────────────────────────────────────────────────
  1. Faithfulness        – RAGAS: are all claims in the answer grounded in
                           the retrieved context?  Claim-level decomposition.
  2. Answer Relevancy    – RAGAS: does the answer address the question?
  3. Answer Correctness  – RAGAS: semantic + factual alignment with expected
                           answer.  Replaces old Answer Similarity + Factual
                           Accuracy.
  4. Context Recall      – RAGAS: can the ground truth be attributed to the
                           retrieved context?

─────────────────────────────────────────────────────────────────────────────
COMPOSITE SCORE
─────────────────────────────────────────────────────────────────────────────

  composite = 0.30 × faithfulness
            + 0.30 × answer_correctness
            + 0.25 × context_recall
            + 0.15 × answer_relevancy

  Pass threshold: composite ≥ 0.70

─────────────────────────────────────────────────────────────────────────────
LLM JUDGE
─────────────────────────────────────────────────────────────────────────────
  Uses Gemini via RAGAS's built-in LLM wrapper (llm_factory).
  Requires: GEMINI_API_KEY in .env
  Default judge model: gemini-2.5-flash

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
      context_chunks=["chunk1 text", "chunk2 text", ...],
      latency=1.23,
  )
  print(result.pretty())

  report = ev.generate_report(results, output_dir="testing/test_results")
"""

from __future__ import annotations

import json
import os
import time
import asyncio
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Tuple
from datetime import datetime

import numpy as np

from dotenv import load_dotenv
load_dotenv()


# ─────────────────────────────────────────────────────────────────────────────
# RAGAS imports
# ─────────────────────────────────────────────────────────────────────────────

import warnings
warnings.filterwarnings('ignore', category=DeprecationWarning)

from ragas import SingleTurnSample, EvaluationDataset, evaluate as ragas_evaluate
from ragas.metrics import Faithfulness, AnswerRelevancy, AnswerCorrectness, ContextRecall
from langchain_openai import ChatOpenAI, OpenAIEmbeddings


# ─────────────────────────────────────────────────────────────────────────────
# Data containers
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class MetricScores:
    faithfulness: float = 0.0        # 0–1  RAGAS Faithfulness
    answer_relevancy: float = 0.0    # 0–1  RAGAS AnswerRelevancy
    answer_correctness: float = 0.0  # 0–1  RAGAS AnswerCorrectness
    context_recall: float = 0.0      # 0–1  RAGAS ContextRecall
    composite_score: float = 0.0     # 0–1  weighted average (primary pass/fail)

    # Legacy aliases for backward compatibility with compare_models.py
    @property
    def answer_similarity(self) -> float:
        return self.answer_correctness

    @property
    def factual_accuracy(self) -> float:
        return self.answer_correctness

    @property
    def answer_relevance(self) -> float:
        return self.answer_relevancy

    @property
    def keyword_coverage(self) -> float:
        return 0.0  # Removed, always 0


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
        sep = "-" * width
        status = "[PASS]" if self.passed else "[FAIL]"
        s = self.scores
        lines = [
            sep,
            f"{status}  {self.question_id} [{self.category}]  ({self.latency:.1f}s)",
            sep,
            f"  Q : {self.question}",
            f"  Ex: {self.expected}",
            f"  An: {self.actual[:160]}{'…' if len(self.actual) > 160 else ''}",
            "",
            f"  -- RAGAS Metrics ----------------------------------",
            f"  Faithfulness       : {s.faithfulness:.3f}",
            f"  Answer Relevancy   : {s.answer_relevancy:.3f}",
            f"  Answer Correctness : {s.answer_correctness:.3f}",
            f"  Context Recall     : {s.context_recall:.3f}",
            f"  -------------------------------------------------",
            f"  Composite Score    : {s.composite_score:.3f}  {'>=0.70 PASS' if s.composite_score >= 0.70 else '<0.70 FAIL'}",
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
                "faithfulness": round(self.scores.faithfulness, 4),
                "answer_relevancy": round(self.scores.answer_relevancy, 4),
                "answer_correctness": round(self.scores.answer_correctness, 4),
                "context_recall": round(self.scores.context_recall, 4),
                "composite_score": round(self.scores.composite_score, 4),
            },
            "notes": self.notes,
        }


# ─────────────────────────────────────────────────────────────────────────────
# Composite score formula
# ─────────────────────────────────────────────────────────────────────────────

def compute_composite(
    faithfulness: float,
    answer_correctness: float,
    context_recall: float,
    answer_relevancy: float,
) -> float:
    """
    Weighted composite for university regulatory Q&A.

    Weights
    -------
    0.30 × faithfulness       – all claims grounded in retrieved context
    0.30 × answer_correctness – semantic + factual alignment with expected
    0.25 × context_recall     – retrieved context covers ground truth
    0.15 × answer_relevancy   – answer addresses the question
    """
    base = (
        0.30 * faithfulness
        + 0.30 * answer_correctness
        + 0.25 * context_recall
        + 0.15 * answer_relevancy
    )
    return round(float(np.clip(base, 0.0, 1.0)), 4)


# ─────────────────────────────────────────────────────────────────────────────
# Evaluator class
# ─────────────────────────────────────────────────────────────────────────────

class Evaluator:
    """
    RAGAS-based RAG evaluator for MACKIS.

    Parameters
    ----------
    ollama_host : str
        (Kept for interface compatibility; not used by RAGAS.)
    embed_model : str
        (Kept for interface compatibility; not used by RAGAS.)
    chat_model : str
        The model being *evaluated* (NOT the judge). Used for reporting.
    judge_provider : str
        Always "gemini" for RAGAS. Kept for interface compatibility.
    judge_model : str
        Gemini model name for RAGAS judge. Default: "gemini-2.5-flash".
    gemini_api_key : str, optional
        Gemini API key. Falls back to GEMINI_API_KEY env var.
    pass_threshold : float
        Composite score ≥ this value → PASS.
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
        # Legacy kwargs (ignored, kept for backward compat)
        st_model_name: str = "",
        llm_judge_timeout: int = 120,
    ):
        load_dotenv()

        self.pass_threshold = pass_threshold

        # ── Resolve the RAG chat model (model under test) ──────────────────
        if chat_model is None:
            chat_model = os.getenv("CHAT_MODEL", "qwen3:32b")
        self.chat_model = chat_model

        # ── Resolve judge model ────────────────────────────────────────────
        self.judge_provider = os.getenv("JUDGE_PROVIDER","openrouter")
        if judge_model is None:
            judge_model = os.getenv("JUDGE_MODEL", "meta-llama/llama-3.3-70b-instruct")
        self.judge_model = judge_model

        # ── Gemini API key ────────────────────────────────────────────────
        if gemini_api_key is None:
            gemini_api_key = os.getenv("GEMINI_API_KEY", "")
        self._gemini_api_key = gemini_api_key

        # ── Initialize RAGAS LLM wrapper ──────────────────────────────────
        self._ragas_llm = None
        self._ragas_metrics = None
        self._ragas_ok = False

        try:
            # Set the API key for google-genai
            os.environ["GOOGLE_API_KEY"] = self._gemini_api_key

            # ── Initialize LLM Judge based on Provider ──
            if self.judge_provider.lower() == "gemini":
                from langchain_google_genai import ChatGoogleGenerativeAI
                self._ragas_llm = ChatGoogleGenerativeAI(
                    model=self.judge_model,
                    google_api_key=self._gemini_api_key,
                    temperature=0.0
                )
            else:
                # Default to OpenRouter
                self._ragas_llm = ChatOpenAI(
                    model=self.judge_model,
                    api_key=os.environ.get("OPENROUTER_API_KEY"),
                    base_url="https://openrouter.ai/api/v1",
                    temperature=0.0
                )

            # ── Always use OpenRouter Embeddings to avoid Google 404 bug ──
            judge_embedder = os.environ.get("JUDGE_EMBEDDER", "openai/text-embedding-3-small")
            self._ragas_embeddings = OpenAIEmbeddings(
                model=judge_embedder,
                api_key=os.environ.get("OPENROUTER_API_KEY"),
                base_url="https://openrouter.ai/api/v1",
                max_retries=10,
                timeout=60
            )

            self._ragas_metrics = [
                Faithfulness(),
                AnswerRelevancy(),
                AnswerCorrectness(),
                ContextRecall(),
            ]
            self._ragas_ok = True
            print(f"[Evaluator] RAGAS initialized with Gemini ({self.judge_model}) -- OK")
        except Exception as e:
            print(f"[Evaluator] WARNING: RAGAS init failed: {e}")
            print(f"[Evaluator] All metrics will be neutral (0.5)")

        # ── Legacy compat attributes ──────────────────────────────────────
        self._ollama_embed_ok = False
        self._ollama_llm_ok = False
        self._gemini_ok = self._ragas_ok

        # ── Startup summary ───────────────────────────────────────────────
        status = "READY" if self._ragas_ok else "UNAVAILABLE"
        print(f"[Evaluator] Judge provider : gemini ({self.judge_model}) -- {status}")
        print(f"[Evaluator] Model under test: {self.chat_model}")
        print(f"[Evaluator] Metrics: RAGAS Faithfulness, AnswerRelevancy, "
              f"AnswerCorrectness, ContextRecall")

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
        Score one question/answer pair using RAGAS metrics.

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
            Will be split into individual chunks for RAGAS.
        question_id : str
            Identifier for reporting.
        category : str
            Question category (Erasmus, Library, …).
        latency : float
            Wall-clock seconds the pipeline took.
        """
        notes: List[str] = []

        if not self._ragas_ok:
            # Fallback: neutral scores
            scores = MetricScores(
                faithfulness=0.5,
                answer_relevancy=0.5,
                answer_correctness=0.5,
                context_recall=0.5,
                composite_score=0.5,
            )
            notes.append("RAGAS unavailable; neutral scores used")
            return EvaluationResult(
                question_id=question_id,
                category=category,
                question=question,
                expected=expected,
                actual=answer,
                latency=latency,
                scores=scores,
                passed=False,
                notes=notes,
            )

        # ── Split context into chunks for RAGAS ───────────────────────────
        # The context is joined by "\n\n-----\n\n" from _build_context_from_chunks
        if context:
            chunks = [c.strip() for c in context.split("\n\n-----\n\n") if c.strip()]
        else:
            chunks = ["No context provided."]

        # ── Create RAGAS sample ───────────────────────────────────────────
        sample = SingleTurnSample(
            user_input=question,
            response=answer,
            retrieved_contexts=chunks,
            reference=expected,
        )

        # ── Run RAGAS evaluation ──────────────────────────────────────────
        try:
            dataset = EvaluationDataset(samples=[sample])
            result = ragas_evaluate(
                dataset=dataset,
                metrics=self._ragas_metrics,
                llm=self._ragas_llm,
                embeddings=self._ragas_embeddings,
            )

            # Extract scores from RAGAS result
            df = result.to_pandas()
            row = df.iloc[0]

            faithfulness = float(row.get("faithfulness", 0.5))
            answer_relevancy = float(row.get("answer_relevancy", 0.5))
            answer_correctness = float(row.get("answer_correctness", 0.5))
            context_recall = float(row.get("context_recall", 0.5))

            # Handle NaN values (RAGAS sometimes returns NaN)
            if np.isnan(faithfulness):
                faithfulness = 0.5
                notes.append("faithfulness=NaN, set to 0.5")
            if np.isnan(answer_relevancy):
                answer_relevancy = 0.5
                notes.append("answer_relevancy=NaN, set to 0.5")
            if np.isnan(answer_correctness):
                answer_correctness = 0.5
                notes.append("answer_correctness=NaN, set to 0.5")
            if np.isnan(context_recall):
                context_recall = 0.5
                notes.append("context_recall=NaN, set to 0.5")

            notes.append(f"ragas_ok: all 4 metrics computed successfully")

        except Exception as e:
            # On error, use neutral scores
            faithfulness = 0.5
            answer_relevancy = 0.5
            answer_correctness = 0.5
            context_recall = 0.5
            notes.append(f"ragas_error: {str(e)[:200]}")

        # ── Compute composite ─────────────────────────────────────────────
        composite = compute_composite(
            faithfulness=faithfulness,
            answer_correctness=answer_correctness,
            context_recall=context_recall,
            answer_relevancy=answer_relevancy,
        )

        scores = MetricScores(
            faithfulness=round(faithfulness, 4),
            answer_relevancy=round(answer_relevancy, 4),
            answer_correctness=round(answer_correctness, 4),
            context_recall=round(context_recall, 4),
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
        valid  = results

        # Per-category aggregation
        cat_stats: Dict[str, Dict] = {}
        for r in valid:
            c = r.category
            if c not in cat_stats:
                cat_stats[c] = {"n": 0, "passed": 0, "scores": {
                    k: [] for k in
                    ["faithfulness", "answer_relevancy",
                     "answer_correctness", "context_recall", "composite_score"]
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
            "avg_faithfulness": _avg("faithfulness"),
            "avg_answer_relevancy": _avg("answer_relevancy"),
            "avg_answer_correctness": _avg("answer_correctness"),
            "avg_context_recall": _avg("context_recall"),
            "avg_composite_score": _avg("composite_score"),
            "avg_latency": round(float(np.mean([r.latency for r in valid])), 2) if valid else 0,
            # Legacy aliases for compare_models.py
            "avg_answer_similarity": _avg("answer_correctness"),
            "avg_factual_accuracy": _avg("answer_correctness"),
            "avg_answer_relevance": _avg("answer_relevancy"),
            "avg_keyword_coverage": 0.0,
        }

        report = {
            "generated_at": ts,
            "pass_threshold": self.pass_threshold,
            "evaluation_framework": "RAGAS",
            "chat_model": self.chat_model,
            "judge_provider": self.judge_provider,
            "judge_model": self.judge_model,
            "composite_formula": (
                "0.30×faithfulness + 0.30×answer_correctness "
                "+ 0.25×context_recall + 0.15×answer_relevancy"
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
            "=" * w,
            "  MACKIS RAG EVALUATION REPORT -- RAGAS Framework",
            "=" * w,
            "",
            "  COMPOSITE: 0.30*Faithfulness + 0.30*AnswerCorrectness + 0.25*ContextRecall + 0.15*AnswerRelevancy",
            f"  Pass threshold: >= {self.pass_threshold:.2f}",
            "",
            "-" * w,
            f"  {'ID':<6} {'Cat':<14} {'Faith':>7} {'AnsCorr':>8} {'CtxRec':>7} {'AnsRel':>7} {'Compose':>8}  {'Status':<12} {'Lat(s)':>7}",
            "-" * w,
        ]

        for r in results:
            s = r.scores
            status = "[PASS]" if r.passed else "[FAIL]"
            lines.append(
                f"  {r.question_id:<6} {r.category:<14} "
                f"{s.faithfulness:>7.3f} "
                f"{s.answer_correctness:>8.3f} "
                f"{s.context_recall:>7.3f} "
                f"{s.answer_relevancy:>7.3f} "
                f"{s.composite_score:>8.3f}  "
                f"{status:<12} "
                f"{r.latency:>7.1f}"
            )

        lines += [
            "-" * w,
            f"  {'MEAN':<6} {'':<14} "
            f"{overall['avg_faithfulness']:>7.3f} "
            f"{overall['avg_answer_correctness']:>8.3f} "
            f"{overall['avg_context_recall']:>7.3f} "
            f"{overall['avg_answer_relevancy']:>7.3f} "
            f"{overall['avg_composite_score']:>8.3f}  "
            f"{'':12} "
            f"{overall['avg_latency']:>7.1f}",
            "=" * w,
            "",
            "  BY CATEGORY",
            "-" * w,
            f"  {'Category':<14} {'N':>4} {'Passed':>7} {'PassRate':>9} {'Avg Comp':>9}",
            "-" * w,
        ]

        for cat, data in sorted(cat_summary.items()):
            lines.append(
                f"  {cat:<14} {data['n']:>4} {data['passed']:>7} "
                f"{data['pass_rate']:>9.1%} "
                f"{data['avg'].get('composite_score', 0):>9.3f}"
            )

        lines += [
            "=" * w,
            "",
            f"  OVERALL: {overall['passed']}/{overall['total']} passed "
            f"({overall['pass_rate']:.1%}) | "
            f"avg composite={overall['avg_composite_score']:.3f}",
            "",
            "=" * w,
        ]
        return lines
