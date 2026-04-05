"""
compare_models.py - Side-by-side comparison of two Ollama models on the
                    MACKIS RAG evaluation suite.

Key design guarantees (fairness):
  • ONE shared Evaluator instance → same Gemini judge for both models
  • ONE shared embed backend     → identical similarity scoring conditions
  • Same 45 test questions       → identical inputs for both models
  • Results saved as JSON + table for both runs

Usage
-----
  # Compare with explicit model names (recommended):
  .venv\\Scripts\\python.exe testing\\compare_models.py \\
      --model-a llama3.1:latest \\
      --model-b qwen3.5:latest

  # Use .env CHAT_MODEL as model-a, specify model-b:
  .venv\\Scripts\\python.exe testing\\compare_models.py \\
      --model-b qwen3.5:latest

  # Skip questions already answered, resume from question N:
  .venv\\Scripts\\python.exe testing\\compare_models.py \\
      --model-a llama3.1:latest --model-b qwen3.5:latest \\
      --start-from 10

Output
------
  testing/test_results/compare_<timestamp>/
    model_a_report.json
    model_b_report.json
    comparison_table.txt       ← side-by-side table
    comparison_summary.json    ← machine-readable winner summary
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import sys
import time
from datetime import datetime
from typing import Dict, List, Optional, Tuple

# ── make sure project root is importable ───────────────────────────────────────
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ── Tee: duplicate stdout to file ─────────────────────────────────────────────
class Tee:
    def __init__(self, console, fh):
        self.console = console
        self.fh = fh
    def write(self, obj):
        try:
            self.console.write(obj)
        except (UnicodeEncodeError, UnicodeDecodeError):
            self.console.write(obj.encode("ascii", "replace").decode("ascii"))
        self.fh.write(obj)
    def flush(self):
        self.console.flush()
        self.fh.flush()

# ── Load test questions (reuse list from test_rag_detailed) ───────────────────
from testing.test_rag_detailed import TEST_QUESTIONS, _build_context_from_chunks, _wrap


def _load_pipeline(model_name: str):
    """Create a RAGPipeline with the given chat model."""
    from services.pipeline.rag_pipeline import RAGPipeline
    from services.config.settings import RAGConfig

    # Override CHAT_MODEL in env so RAGConfig.from_env() picks it up
    os.environ["CHAT_MODEL"] = model_name
    config = RAGConfig.from_env()
    # Double-check the override took effect
    assert config.ollama.chat_model == model_name, (
        f"Config did not pick up model name override: "
        f"got {config.ollama.chat_model!r}, expected {model_name!r}"
    )
    pipeline = RAGPipeline(config)
    return pipeline, config


def _run_model(
    pipeline,
    model_name: str,
    questions: List[Dict],
    start_from: int = 1,
) -> List[Dict]:
    """
    Run all questions through the pipeline and return raw answer records.
    Does NOT call the evaluator — scoring happens afterwards in one shared batch.

    Returns list of dicts: {id, category, question, expected, answer, context, latency}
    """
    records = []
    n = len(questions)

    print()
    print("=" * 80)
    print(f"  MODEL: {model_name}")
    print(f"  Running {n} questions…")
    print("=" * 80)

    for i, test in enumerate(questions, 1):
        if i < start_from:
            records.append(None)  # placeholder so index alignment is preserved
            continue

        qid      = test["id"]
        category = test.get("category", "Other")
        question = test["question"]
        expected = test["expected_answer"]

        print(f"\n[{i:02d}/{n}] {qid} [{category}]")
        print(f"  Q: {question[:100]}{'…' if len(question) > 100 else ''}")

        t0 = time.time()
        try:
            answer  = pipeline.answer(question, history=[])
            latency = time.time() - t0
        except Exception as e:
            latency = time.time() - t0
            print(f"  !! PIPELINE ERROR: {e}")
            records.append({
                "id": qid, "category": category,
                "question": question, "expected": expected,
                "answer": f"[PIPELINE ERROR: {e}]",
                "context": "", "latency": latency, "error": True,
            })
            continue

        # Retrieve context for faithfulness scoring
        try:
            chunks  = pipeline.search_only(question, top_k=10)
            context = _build_context_from_chunks(chunks)
        except Exception:
            context = ""

        print(f"  A: {answer[:120]}{'…' if len(answer) > 120 else ''}  ({latency:.1f}s)")
        records.append({
            "id": qid, "category": category,
            "question": question, "expected": expected,
            "answer": answer, "context": context,
            "latency": latency, "error": False,
        })
        
        # ⏱️ Give local Ollama / GPU a small rest to flush VRAM and prevent thermal throttling
        time.sleep(3)

    return records


def _evaluate_records(evaluator, records: List[Dict], model_name: str):
    """Score a list of answer records using the shared Evaluator."""
    from testing.evaluator import EvaluationResult

    results = []
    n = len([r for r in records if r and not r.get("error")])
    done = 0

    print()
    print("=" * 80)
    print(f"  SCORING with Gemini judge: {model_name}")
    print(f"  Embed backend : {'Ollama bge-m3' if evaluator._ollama_embed_ok else 'SentenceTransformers'}")
    print("=" * 80)

    for rec in records:
        if rec is None:
            continue   # skipped by start_from
        if rec.get("error"):
            continue   # pipeline error — no meaningful score to assign

        done += 1
        sys.stdout.write(f"  [{done:02d}/{n}] Scoring {rec['id']}… ")
        sys.stdout.flush()

        result = evaluator.evaluate(
            question_id=rec["id"],
            category=rec["category"],
            question=rec["question"],
            answer=rec["answer"],
            expected=rec["expected"],
            context=rec["context"],
            latency=rec["latency"],
        )
        results.append(result)
        s = result.scores
        print(
            f"sim={s.answer_similarity:.2f}  "
            f"faith={s.faithfulness:.2f}  "
            f"rel={s.answer_relevance:.2f}  "
            f"fact={s.factual_accuracy:.2f}  "
            f"comp={s.composite_score:.2f}"
        )

    # Override evaluator's internal chat_model label in the report
    evaluator.chat_model = model_name
    return results


def _build_comparison_table(
    results_a: list, results_b: list,
    model_a: str, model_b: str,
    out_dir: str,
) -> str:
    """Build and print a side-by-side comparison table. Returns the table string."""
    import numpy as np

    def _avg(results, attr):
        vals = [getattr(r.scores, attr) for r in results]
        return round(float(np.mean(vals)), 4) if vals else 0.0

    metrics = [
        ("answer_similarity", "Similarity"),
        ("faithfulness",      "Faithfulness"),
        ("answer_relevance",  "Relevance"),
        ("factual_accuracy",  "FactualAcc"),
        ("composite_score",   "Composite"),
    ]

    # Overall averages
    avg_a = {m: _avg(results_a, m) for m, _ in metrics}
    avg_b = {m: _avg(results_b, m) for m, _ in metrics}
    lat_a = round(float(np.mean([r.latency for r in results_a])), 2) if results_a else 0
    lat_b = round(float(np.mean([r.latency for r in results_b])), 2) if results_b else 0
    pass_a = sum(1 for r in results_a if r.passed)
    pass_b = sum(1 for r in results_b if r.passed)

    W = 128
    col_w = 18

    ma = model_a[:col_w]
    mb = model_b[:col_w]

    lines = [
        "",
        "═" * W,
        "  MACKIS RAG  ──  MODEL COMPARISON REPORT",
        "  (Judge: Gemini 2.5 Flash  |  Embedding: bge-m3 via Ollama / ST fallback)",
        "═" * W,
        "",
        f"  {'Model A':<{col_w}}  {ma}",
        f"  {'Model B':<{col_w}}  {mb}",
        "",
        "─" * W,
        f"  {'Metric':<20} {'Model A':>{col_w}} {'Model B':>{col_w}} {'Winner':>12}",
        "─" * W,
    ]

    winners = {}
    for attr, label in metrics:
        va = avg_a[attr]
        vb = avg_b[attr]
        if va > vb + 0.005:
            winner = "A  ◀"
        elif vb > va + 0.005:
            winner = "   ▶ B"
        else:
            winner = "  tie"
        winners[attr] = winner
        lines.append(
            f"  {label:<20} {va:>{col_w}.4f} {vb:>{col_w}.4f} {winner:>12}"
        )

    lines += [
        "─" * W,
        f"  {'Pass rate':<20} {pass_a:>{col_w-1}}/{len(results_a)} "
        f"{pass_b:>{col_w-1}}/{len(results_b)}",
        f"  {'Avg latency (s)':<20} {lat_a:>{col_w}.2f} {lat_b:>{col_w}.2f}",
        "═" * W,
    ]

    # Per-question side-by-side
    if results_a and results_b:
        # Build lookup by question_id for model B
        b_by_id = {r.question_id: r for r in results_b}

        lines += [
            "",
            "  PER-QUESTION DETAILS",
            "─" * W,
            f"  {'ID':<6} {'Cat':<14} "
            f"{'A_Comp':>8} {'A_Pass':<7} "
            f"{'B_Comp':>8} {'B_Pass':<7} "
            f"{'Delta(A-B)':>10}  Note",
            "─" * W,
        ]

        for ra in results_a:
            rb = b_by_id.get(ra.question_id)
            if rb is None:
                continue
            delta = ra.scores.composite_score - rb.scores.composite_score
            pa = "✅" if ra.passed else "❌"
            pb = "✅" if rb.passed else "❌"
            sign = "+" if delta >= 0 else ""
            note = ""
            if abs(delta) < 0.01:
                note = "≈ tie"
            elif delta > 0:
                note = "A better"
            else:
                note = "B better"
            lines.append(
                f"  {ra.question_id:<6} {ra.category:<14} "
                f"{ra.scores.composite_score:>8.3f} {pa:<7} "
                f"{rb.scores.composite_score:>8.3f} {pb:<7} "
                f"{sign}{delta:>9.3f}  {note}"
            )
        lines.append("═" * W)

    # Overall winner
    n_wins_a = sum(1 for _, w in winners.items() if "A" in w and "B" not in w)
    n_wins_b = sum(1 for _, w in winners.items() if "B" in w and "A" not in w)
    if n_wins_a > n_wins_b:
        overall = f"OVERALL WINNER → Model A ({model_a})"
    elif n_wins_b > n_wins_a:
        overall = f"OVERALL WINNER → Model B ({model_b})"
    else:
        overall = "OVERALL → TIE"
    lines += ["", f"  {overall}", "═" * W, ""]

    table_str = "\n".join(lines)
    print(table_str)

    txt_path = os.path.join(out_dir, "comparison_table.txt")
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(table_str)
    print(f"  Table saved → {txt_path}")

    return table_str


def _save_comparison_summary(
    results_a, results_b,
    model_a: str, model_b: str,
    out_dir: str,
    ts: str,
):
    import numpy as np

    def _ovr(results):
        if not results:
            return {}
        return {
            "pass_rate":          round(sum(1 for r in results if r.passed) / len(results), 3),
            "avg_similarity":     round(float(np.mean([r.scores.answer_similarity  for r in results])), 4),
            "avg_faithfulness":   round(float(np.mean([r.scores.faithfulness       for r in results])), 4),
            "avg_relevance":      round(float(np.mean([r.scores.answer_relevance   for r in results])), 4),
            "avg_factual_acc":    round(float(np.mean([r.scores.factual_accuracy   for r in results])), 4),
            "avg_composite":      round(float(np.mean([r.scores.composite_score    for r in results])), 4),
            "avg_latency":        round(float(np.mean([r.latency                   for r in results])), 2),
        }

    summary = {
        "generated_at": ts,
        "model_a": model_a,
        "model_b": model_b,
        "model_a_overall": _ovr(results_a),
        "model_b_overall": _ovr(results_b),
        "per_question": [
            {
                "id":         ra.question_id,
                "category":   ra.category,
                "question":   ra.question,
                "model_a": {
                    "answer":    ra.actual,
                    "latency":   ra.latency,
                    "scores":    ra.as_dict()["scores"],
                    "passed":    ra.passed,
                },
                "model_b": {
                    "answer":    rb.actual if (rb := {r.question_id: r for r in results_b}.get(ra.question_id)) else "",
                    "latency":   rb.latency if rb else 0,
                    "scores":    rb.as_dict()["scores"] if rb else {},
                    "passed":    rb.passed if rb else False,
                },
            }
            for ra in results_a
        ],
    }

    path = os.path.join(out_dir, "comparison_summary.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"  JSON saved  → {path}")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    from dotenv import load_dotenv
    load_dotenv()

    parser = argparse.ArgumentParser(
        description="Compare two Ollama models on the MACKIS RAG evaluation suite."
    )
    parser.add_argument(
        "--model-a", default=None,
        help="Model A name (default: CHAT_MODEL from .env, currently llama3.1:latest)",
    )
    parser.add_argument(
        "--model-b", default="qwen3.5:latest",
        help="Model B name (default: qwen3.5:latest)",
    )
    parser.add_argument(
        "--start-from", type=int, default=1,
        help="Start from question N (1-based). Useful for resuming.",
    )
    parser.add_argument(
        "--questions", type=int, default=0,
        help="Limit to first N questions (0 = all 45).",
    )
    args = parser.parse_args()

    # Resolve model A from env if not given
    model_a = args.model_a or os.getenv("CHAT_MODEL", "llama3.1:latest")
    model_b = args.model_b

    # Limit questions if requested
    questions = TEST_QUESTIONS
    if args.questions > 0:
        questions = questions[: args.questions]

    # ── Output directory ──────────────────────────────────────────────────────
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.join(
        os.path.dirname(__file__), "test_results", f"compare_{ts}"
    )
    os.makedirs(out_dir, exist_ok=True)

    # ── Tee output to file ────────────────────────────────────────────────────
    log_path = os.path.join(out_dir, "compare_log.txt")
    log_file = open(log_path, "w", encoding="utf-8")
    _orig_stdout = sys.stdout
    _orig_stderr = sys.stderr
    sys.stdout = Tee(_orig_stdout, log_file)
    sys.stderr = Tee(_orig_stderr, log_file)

    print("=" * 80)
    print("  MACKIS RAG — MODEL COMPARISON")
    print(f"  Model A : {model_a}")
    print(f"  Model B : {model_b}")
    print(f"  Questions: {len(questions)}")
    print(f"  Output dir: {out_dir}")
    print("=" * 80)

    # ── Load ONE shared Evaluator (guarantees identical judge + embed for both) ─
    print("\n[1/5] Initializing shared Evaluator…")
    from testing.evaluator import Evaluator
    evaluator = Evaluator()  # reads JUDGE_PROVIDER, JUDGE_MODEL, GEMINI_API_KEY from .env

    embed_backend = (
        f"Ollama {evaluator.embed_model}"
        if evaluator._ollama_embed_ok
        else f"SentenceTransformers ({evaluator._st_model_name})"
    )
    print(f"       Embed backend : {embed_backend}")

    # ── Run Model A ───────────────────────────────────────────────────────────
    print(f"\n[2/5] Running pipeline with Model A: {model_a}")
    pipeline_a, _ = _load_pipeline(model_a)
    records_a = _run_model(pipeline_a, model_a, questions, args.start_from)

    # ── Run Model B ───────────────────────────────────────────────────────────
    print(f"\n[3/5] Running pipeline with Model B: {model_b}")
    pipeline_b, _ = _load_pipeline(model_b)
    records_b = _run_model(pipeline_b, model_b, questions, args.start_from)

    # ── Score both with the shared Evaluator ─────────────────────────────────
    print(f"\n[4/5] Scoring Model A answers with Gemini judge…")
    results_a = _evaluate_records(evaluator, records_a, model_a)

    print(f"\n      Scoring Model B answers with Gemini judge…")
    results_b = _evaluate_records(evaluator, records_b, model_b)

    # ── Save individual reports ───────────────────────────────────────────────
    print(f"\n[5/5] Generating reports…\n")
    evaluator.chat_model = model_a
    evaluator.generate_report(results_a, output_dir=out_dir)

    evaluator.chat_model = model_b
    evaluator.generate_report(results_b, output_dir=out_dir)

    # ── Side-by-side comparison ───────────────────────────────────────────────
    _build_comparison_table(results_a, results_b, model_a, model_b, out_dir)
    _save_comparison_summary(results_a, results_b, model_a, model_b, out_dir, ts)

    print(f"\n  All output files in: {out_dir}")
    print(f"  Full log:             {log_path}")

    # Restore stdout
    sys.stdout = _orig_stdout
    sys.stderr = _orig_stderr
    log_file.close()


if __name__ == "__main__":
    main()
