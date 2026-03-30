"""
diagnose_retrieval.py - Retrieval vs Generation Diagnostic Tool

PURPOSE:
  Identifies whether failing questions are due to:
  (A) RETRIEVAL MISS — expected values are NOT in the retrieved chunks
  (B) GENERATION MISS — expected values ARE in chunks but LLM didn't use them

For each test question, this script:
  1. Runs the full pipeline (query analysis → retrieval → ranking)
  2. Captures the EXACT chunks that would go to generation
  3. Extracts expected numbers/keywords from the expected answer
  4. Checks which values appear in the chunks vs the final answer
  5. Produces a diagnostic report

Run:  python testing/diagnose_retrieval.py
      python testing/diagnose_retrieval.py --only-failing   (skip passing questions)
      python testing/diagnose_retrieval.py --ids Q1 Q4 Q19  (specific questions only)
"""

import os
import sys
import re
import json
import time
from datetime import datetime
from typing import Dict, List, Optional, Set, Tuple
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv()

# Import test questions
from testing.test_rag_detailed import TEST_QUESTIONS


# ─────────────────────────────────────────────────────────────────────────────
# Value Extraction
# ─────────────────────────────────────────────────────────────────────────────

def extract_expected_values(expected: str) -> Dict[str, List[str]]:
    """
    Extract testable values from expected answer.
    Returns dict with categories of values.
    """
    values = {
        "numbers": [],       # Pure numbers: 2.20, 60, 15
        "durations": [],     # Duration phrases: 2 ay, 60 gün, 6 yarıyıl
        "keywords": [],      # Domain-specific keywords
    }
    
    # Numbers (including decimals)
    nums = re.findall(r'\b\d+(?:[.,]\d+)?\b', expected)
    values["numbers"] = list(set(nums))
    
    # Duration patterns (Turkish and English)
    dur_patterns = [
        r'\d+\s*(?:ay|month|gün|day|hafta|week|yıl|year|yarıyıl|semester|dönem|saat|hour)',
        r'\d+\s*(?:iş günü|business day|work day)',
    ]
    for pat in dur_patterns:
        matches = re.findall(pat, expected, re.IGNORECASE)
        values["durations"].extend(matches)
    
    # Domain keywords (significant terms from expected answer)
    stop_words = {
        "ve", "ile", "bir", "bu", "için", "olan", "de", "da", "dir", "dır",
        "the", "a", "an", "of", "in", "is", "to", "for", "and", "or",
        "en", "az", "çok", "kadar", "olarak", "gibi", "her", "tüm",
        "can", "may", "must", "should", "from", "with", "that", "this",
        "are", "not", "also", "only", "they", "their", "been", "have",
        "gerekir", "yapılır", "edilir", "olmalı", "olan", "ise",
    }
    words = re.findall(r'[a-zA-ZçğıöşüÇĞİÖŞÜ]{4,}', expected.lower())
    keywords = [w for w in words if w not in stop_words]
    # Deduplicate while preserving order
    seen = set()
    for kw in keywords:
        if kw not in seen:
            seen.add(kw)
            values["keywords"].append(kw)
    
    return values


def check_values_in_text(values: Dict[str, List[str]], text: str) -> Dict[str, Dict]:
    """
    Check which expected values appear in text.
    Returns per-category found/missing breakdown.
    """
    text_lower = text.lower()
    results = {}
    
    for category, items in values.items():
        found = []
        missing = []
        for item in items:
            item_lower = item.lower()
            # Check original and comma/dot variants for numbers
            variants = {item_lower}
            variants.add(item_lower.replace(",", "."))
            variants.add(item_lower.replace(".", ","))
            
            if any(v in text_lower for v in variants):
                found.append(item)
            else:
                missing.append(item)
        
        results[category] = {
            "found": found,
            "missing": missing,
            "total": len(items),
            "ratio": len(found) / max(len(items), 1),
        }
    
    return results


# ─────────────────────────────────────────────────────────────────────────────
# Pipeline Instrumentation
# ─────────────────────────────────────────────────────────────────────────────

def run_instrumented_pipeline(pipeline, query: str) -> Dict:
    """
    Run the pipeline with full instrumentation.
    Returns all intermediate artifacts for diagnosis.
    """
    from services.pipeline.rag_pipeline import KG_AVAILABLE, kg_query_hybrid
    
    config = pipeline.config
    
    # 1. Query Analysis
    analysis = pipeline.query_agent.analyze(query, [])
    language = analysis["language"]
    intent = analysis["intent"]
    query_tags = analysis["tags"]
    negated_terms = analysis["negated_terms"]
    expanded_queries = analysis.get("expanded_queries", [query])
    
    # 2. Build retrieval query
    retrieval_query = pipeline.query_agent.build_retrieval_query(query, analysis)
    
    # 3. Retrieval
    skip_hyde = intent in {"count_items", "list_names"}
    candidates = pipeline.retrieval_agent.retrieve(
        retrieval_query,
        language=language,
        use_expansion=config.features.use_query_expansion,
        use_hyde=config.features.use_hyde and not skip_hyde,
        pre_expanded=expanded_queries if config.features.use_query_expansion else None
    )
    
    # 4. Ranking
    reranked = pipeline.ranking_agent.rerank(
        retrieval_query,
        candidates,
        query_tags=query_tags,
        negated_terms=negated_terms
    )
    
    # 5. Filter by threshold
    strong = pipeline.ranking_agent.filter_by_threshold(reranked)
    
    # 6. Document-level diversity + chunk cap
    max_ctx = config.retrieval.max_docs_context
    max_per_doc = max(3, max_ctx // 3)
    
    if intent in {"list_names", "count_items"} and reranked:
        best_chunk = reranked[0]
        best_meta = best_chunk.get("meta") or {}
        best_path = best_meta.get("source_path")
        
        if best_path:
            retrieved = pipeline.data_loader.get_all_chunks_for_doc(best_path)
            retrieved = retrieved[:max_ctx]
        else:
            retrieved = pipeline.ranking_agent.document_level_select(
                strong, max_chunks=max_ctx, max_per_doc=max_per_doc
            )
    else:
        retrieved = pipeline.ranking_agent.document_level_select(
            strong, max_chunks=max_ctx, max_per_doc=max_per_doc
        )
    
    # 7. KG facts
    kg_facts = ""
    if KG_AVAILABLE and kg_query_hybrid:
        try:
            kg_facts = kg_query_hybrid(query, lang=language or "tr")
        except Exception:
            pass
    
    # 8. Build context (what the LLM actually sees)
    context = pipeline.generation_agent.build_context(retrieved)
    
    # 9. Generate answer
    system_prompt = pipeline.generation_agent._get_system_prompt(language)
    full_prompt = pipeline.generation_agent._build_prompt(query, context, kg_facts)
    answer = pipeline.llm_service.chat(full_prompt, system_prompt, temperature=0.0)
    answer = pipeline.generation_agent.verify_answer_numbers(answer, context, language)
    
    return {
        "analysis": analysis,
        "total_candidates": len(candidates),
        "after_rerank": len(reranked),
        "after_threshold": len(strong),
        "final_chunks": retrieved,
        "final_chunk_count": len(retrieved),
        "context_text": context,
        "kg_facts": kg_facts,
        "answer": answer,
        # Top-10 reranked for inspection
        "top_reranked": [
            {
                "rank": i + 1,
                "score": c.get("hybrid_score", 0),
                "ce_score": c.get("ce_score", 0),
                "title": (c.get("meta") or {}).get("title", "")[:60],
                "section": (c.get("meta") or {}).get("section_header", "")[:40],
                "source_path": (c.get("meta") or {}).get("source_path", "")[:80],
                "text_preview": c.get("text", "")[:150],
            }
            for i, c in enumerate(reranked[:10])
        ],
        # Source documents of final chunks
        "final_sources": list(set(
            (c.get("meta") or {}).get("source_path", "unknown")
            for c in retrieved
        )),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Diagnosis Logic
# ─────────────────────────────────────────────────────────────────────────────

def diagnose_question(pipeline, test: Dict) -> Dict:
    """
    Full diagnosis for one question.
    """
    qid = test["id"]
    question = test["question"]
    expected = test["expected_answer"]
    
    print(f"\n{'='*80}")
    print(f"  {qid} [{test.get('category', '')}]")
    print(f"{'='*80}")
    print(f"  Q: {question[:100]}...")
    
    t0 = time.time()
    result = run_instrumented_pipeline(pipeline, question)
    latency = time.time() - t0
    
    answer = result["answer"]
    context_text = result["context_text"]
    kg_facts = result["kg_facts"]
    
    # Full text that the LLM could draw from
    full_llm_input = context_text
    if kg_facts:
        full_llm_input = kg_facts + "\n\n" + context_text
    
    # Extract expected values
    expected_vals = extract_expected_values(expected)
    
    # Check values in: (a) retrieved chunks, (b) final answer
    in_chunks = check_values_in_text(expected_vals, full_llm_input)
    in_answer = check_values_in_text(expected_vals, answer)
    
    # Classify failure type
    num_in_chunks = in_chunks["numbers"]
    num_in_answer = in_answer["numbers"]
    
    total_expected_nums = num_in_chunks["total"]
    nums_in_context = num_in_chunks["ratio"]
    nums_in_final = num_in_answer["ratio"]
    
    kw_in_chunks = in_chunks["keywords"]
    kw_in_answer = in_answer["keywords"]
    
    # Classification logic
    if total_expected_nums == 0:
        # No numbers to check; classify by keywords
        if kw_in_chunks["ratio"] < 0.4:
            diagnosis = "RETRIEVAL_MISS"
            diagnosis_detail = "Key expected keywords not found in retrieved chunks"
        elif kw_in_answer["ratio"] < 0.4:
            diagnosis = "GENERATION_MISS"
            diagnosis_detail = "Keywords in chunks but LLM didn't include them"
        else:
            diagnosis = "LIKELY_OK"
            diagnosis_detail = "Most keywords present in both chunks and answer"
    elif nums_in_context < 0.5:
        diagnosis = "RETRIEVAL_MISS"
        diagnosis_detail = f"Only {num_in_chunks['found']}/{total_expected_nums} expected numbers found in retrieved chunks"
    elif nums_in_final < 0.5 and nums_in_context >= 0.5:
        diagnosis = "GENERATION_MISS"
        diagnosis_detail = f"Numbers in chunks ({num_in_chunks['found']}) but missing from answer ({num_in_answer['missing']})"
    elif nums_in_final >= 0.5 and nums_in_context >= 0.5:
        diagnosis = "LIKELY_OK"
        diagnosis_detail = f"Most numbers present in both chunks and answer"
    else:
        diagnosis = "MIXED"
        diagnosis_detail = "Partial retrieval and generation issues"
    
    # Print diagnosis
    icon = {
        "RETRIEVAL_MISS": "🔍❌",
        "GENERATION_MISS": "🤖❌", 
        "LIKELY_OK": "✅",
        "MIXED": "⚠️",
    }.get(diagnosis, "❓")
    
    print(f"\n  {icon} DIAGNOSIS: {diagnosis}")
    print(f"     {diagnosis_detail}")
    print(f"     Latency: {latency:.1f}s")
    
    print(f"\n  📊 Numbers ({total_expected_nums} expected):")
    if num_in_chunks["found"]:
        print(f"     In chunks  ✅: {num_in_chunks['found']}")
    if num_in_chunks["missing"]:
        print(f"     NOT in chunks ❌: {num_in_chunks['missing']}")
    if num_in_answer["found"]:
        print(f"     In answer  ✅: {num_in_answer['found']}")
    if num_in_answer["missing"]:
        print(f"     NOT in answer ❌: {num_in_answer['missing']}")
    
    print(f"\n  📄 Context ({result['final_chunk_count']} chunks from {len(result['final_sources'])} docs):")
    for src in result["final_sources"]:
        name = os.path.basename(src) if src else "unknown"
        print(f"     • {name}")
    
    print(f"\n  📋 Top-5 Reranked Candidates:")
    for c in result["top_reranked"][:5]:
        print(f"     {c['rank']:2d}. [{c['score']:.3f}] {c['title']}")
    
    if kg_facts:
        print(f"\n  🔗 KG Facts: {len(kg_facts)} chars")
    else:
        print(f"\n  🔗 KG Facts: None")
    
    return {
        "id": qid,
        "category": test.get("category", ""),
        "question": question,
        "expected": expected,
        "answer_preview": answer[:200],
        "diagnosis": diagnosis,
        "diagnosis_detail": diagnosis_detail,
        "latency": round(latency, 1),
        "numbers": {
            "expected": expected_vals["numbers"],
            "in_chunks": in_chunks["numbers"],
            "in_answer": in_answer["numbers"],
        },
        "keywords": {
            "in_chunks_ratio": round(kw_in_chunks["ratio"], 3),
            "in_answer_ratio": round(kw_in_answer["ratio"], 3),
            "missing_from_chunks": kw_in_chunks["missing"][:10],
            "missing_from_answer": kw_in_answer["missing"][:10],
        },
        "pipeline": {
            "total_candidates": result["total_candidates"],
            "after_rerank": result["after_rerank"],
            "after_threshold": result["after_threshold"],
            "final_chunks": result["final_chunk_count"],
            "sources": result["final_sources"],
            "kg_available": bool(kg_facts),
        },
        "top_reranked": result["top_reranked"][:5],
    }


# ─────────────────────────────────────────────────────────────────────────────
# Report Generation
# ─────────────────────────────────────────────────────────────────────────────

def generate_report(diagnostics: List[Dict], output_dir: str):
    """Generate summary report from diagnostics."""
    
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Classify
    retrieval_miss = [d for d in diagnostics if d["diagnosis"] == "RETRIEVAL_MISS"]
    generation_miss = [d for d in diagnostics if d["diagnosis"] == "GENERATION_MISS"]
    likely_ok = [d for d in diagnostics if d["diagnosis"] == "LIKELY_OK"]
    mixed = [d for d in diagnostics if d["diagnosis"] == "MIXED"]
    
    # Print summary
    print("\n" + "=" * 80)
    print("  RETRIEVAL DIAGNOSTIC SUMMARY")
    print("=" * 80)
    
    total = len(diagnostics)
    print(f"\n  Total questions diagnosed: {total}")
    print(f"  🔍❌ RETRIEVAL MISS    : {len(retrieval_miss):2d} ({len(retrieval_miss)/total*100:.0f}%)")
    print(f"  🤖❌ GENERATION MISS   : {len(generation_miss):2d} ({len(generation_miss)/total*100:.0f}%)")
    print(f"  ⚠️  MIXED              : {len(mixed):2d} ({len(mixed)/total*100:.0f}%)")
    print(f"  ✅  LIKELY OK           : {len(likely_ok):2d} ({len(likely_ok)/total*100:.0f}%)")
    
    if retrieval_miss:
        print(f"\n  ── RETRIEVAL MISSES (wrong chunks retrieved) ──")
        for d in retrieval_miss:
            print(f"     {d['id']:5s} [{d['category']:14s}] {d['diagnosis_detail'][:60]}")
            if d["numbers"]["in_chunks"]["missing"]:
                print(f"           Missing numbers: {d['numbers']['in_chunks']['missing']}")
    
    if generation_miss:
        print(f"\n  ── GENERATION MISSES (info in chunks, LLM didn't use it) ──")
        for d in generation_miss:
            print(f"     {d['id']:5s} [{d['category']:14s}] {d['diagnosis_detail'][:60]}")
            if d["numbers"]["in_answer"]["missing"]:
                print(f"           LLM missed numbers: {d['numbers']['in_answer']['missing']}")
    
    if mixed:
        print(f"\n  ── MIXED ISSUES ──")
        for d in mixed:
            print(f"     {d['id']:5s} [{d['category']:14s}] {d['diagnosis_detail'][:60]}")
    
    # Per-category breakdown
    cat_diag = defaultdict(lambda: {"retrieval": 0, "generation": 0, "ok": 0, "mixed": 0})
    for d in diagnostics:
        cat = d["category"]
        if d["diagnosis"] == "RETRIEVAL_MISS":
            cat_diag[cat]["retrieval"] += 1
        elif d["diagnosis"] == "GENERATION_MISS":
            cat_diag[cat]["generation"] += 1
        elif d["diagnosis"] == "LIKELY_OK":
            cat_diag[cat]["ok"] += 1
        else:
            cat_diag[cat]["mixed"] += 1
    
    print(f"\n  ── PER-CATEGORY BREAKDOWN ──")
    print(f"  {'Category':<16} {'Ret.Miss':>8} {'Gen.Miss':>8} {'Mixed':>6} {'OK':>4}")
    print(f"  {'─'*16} {'─'*8} {'─'*8} {'─'*6} {'─'*4}")
    for cat in sorted(cat_diag.keys()):
        d = cat_diag[cat]
        print(f"  {cat:<16} {d['retrieval']:>8} {d['generation']:>8} {d['mixed']:>6} {d['ok']:>4}")
    
    # Actionable recommendations
    print(f"\n  ── RECOMMENDATIONS ──")
    if len(retrieval_miss) > len(generation_miss):
        print(f"  ⚡ PRIMARY BOTTLENECK: RETRIEVAL ({len(retrieval_miss)} questions)")
        print(f"     → Priority: Parent-child chunks (P2-2.3)")
        print(f"     → Consider: Increase max_docs_context from 6 to 8-10")
        print(f"     → Consider: Lower CE score threshold from 0.65")
    elif len(generation_miss) > len(retrieval_miss):
        print(f"  ⚡ PRIMARY BOTTLENECK: GENERATION ({len(generation_miss)} questions)")
        print(f"     → Priority: Improve generation prompt")
        print(f"     → Consider: Instruct LLM to extract ALL numbers from context")
        print(f"     → Consider: Larger context window / more chunks")
    else:
        print(f"  ⚡ MIXED BOTTLENECK: Both retrieval and generation need improvement")
    
    print(f"\n{'='*80}")
    
    # Save JSON report
    os.makedirs(output_dir, exist_ok=True)
    json_path = os.path.join(output_dir, f"retrieval_diagnostic_{ts}.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({
            "timestamp": ts,
            "summary": {
                "total": total,
                "retrieval_miss": len(retrieval_miss),
                "generation_miss": len(generation_miss),
                "mixed": len(mixed),
                "likely_ok": len(likely_ok),
            },
            "diagnostics": diagnostics,
        }, f, ensure_ascii=False, indent=2)
    
    print(f"\n  📄 JSON report: {json_path}")
    
    return {
        "retrieval_miss": retrieval_miss,
        "generation_miss": generation_miss,
        "mixed": mixed,
        "likely_ok": likely_ok,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="MACKIS Retrieval Diagnostic Tool")
    parser.add_argument(
        "--only-failing", action="store_true",
        help="Only diagnose questions that failed in the last test run"
    )
    parser.add_argument(
        "--ids", nargs="+",
        help="Diagnose specific question IDs (e.g., --ids Q1 Q4 Q19)"
    )
    parser.add_argument(
        "--all", action="store_true",
        help="Diagnose ALL questions (default: only failing)"
    )
    args = parser.parse_args()
    
    # Known failing questions from latest test (March 1, 2026)
    KNOWN_FAILING = {
        "Q1", "Q4", "Q9", "Q10", "Q11", "Q12", "Q15", "Q16", "Q17",
        "Q18", "Q19", "Q21", "Q23", "Q25", "Q26", "Q29",
        "Q39", "Q40", "Q41", "Q42", "Q44",
    }
    
    # Filter questions
    if args.ids:
        target_ids = set(args.ids)
        questions = [t for t in TEST_QUESTIONS if t["id"] in target_ids]
    elif args.all:
        questions = TEST_QUESTIONS
    else:
        # Default: only failing questions
        questions = [t for t in TEST_QUESTIONS if t["id"] in KNOWN_FAILING]
    
    if not questions:
        print("No questions matched the filter. Use --all or --ids Q1 Q2 ...")
        return
    
    print("=" * 80)
    print(f"  MACKIS RETRIEVAL DIAGNOSTIC")
    print(f"  Diagnosing {len(questions)} questions")
    print("=" * 80)
    
    # Load pipeline
    from services.pipeline.rag_pipeline import RAGPipeline
    from services.config.settings import RAGConfig
    
    config = RAGConfig.from_env()
    pipeline = RAGPipeline(config)
    
    print(f"  LLM: {config.ollama.chat_model}")
    print(f"  Chunks: {pipeline.vector_store.count}")
    print(f"  max_docs_context: {config.retrieval.max_docs_context}")
    print(f"  CE threshold: {config.reranking.score_threshold}")
    print(f"  max_candidates: {config.reranking.max_candidates}")
    
    # Run diagnostics
    diagnostics = []
    total_start = time.time()
    
    for i, test in enumerate(questions, 1):
        print(f"\n  [{i}/{len(questions)}] Running {test['id']}...")
        try:
            result = diagnose_question(pipeline, test)
            diagnostics.append(result)
        except Exception as e:
            print(f"  ❌ Error on {test['id']}: {e}")
            import traceback
            traceback.print_exc()
    
    total_elapsed = time.time() - total_start
    
    # Generate report
    output_dir = os.path.join(os.path.dirname(__file__), "test_results")
    summary = generate_report(diagnostics, output_dir)
    
    print(f"\n  ⏱️  Total time: {total_elapsed:.0f}s ({total_elapsed/60:.1f}m)")


if __name__ == "__main__":
    main()
