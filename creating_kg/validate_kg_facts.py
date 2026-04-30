"""
validate_kg_facts.py - LLM-ONLY validation of existing KG facts

Validates EVERY fact and triple using LLM to remove invalid/noisy data.
This is slower but produces cleaner results.

Usage:
    python validate_kg_facts.py

Expected time: 2-4 hours for ~3000 facts + ~5000 triples
"""

import os
import json
import time
import requests
from typing import Dict, List, Tuple
from dotenv import load_dotenv

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from services.config.settings import RAGConfig
from services.core.llm_service import LLMService

load_dotenv()

# =================== CONFIG ===================

settings = RAGConfig.from_env()
_llm_service = LLMService(settings.ollama)

PREPROCESSING_DIR = os.path.dirname(os.path.abspath(__file__))
KG_DIR = os.path.join(PREPROCESSING_DIR, "knowledge_graph")
KG_LLM_DIR = os.path.join(KG_DIR, "llm_validated")
os.makedirs(KG_LLM_DIR, exist_ok=True)

KG_FACTS_PATH = os.path.join(KG_DIR, "kg_facts.json")
KG_FACTS_VALIDATED_PATH = os.path.join(KG_LLM_DIR, "kg_facts_llm_validated.json")
KG_TRIPLES_PATH = os.path.join(KG_DIR, "kg_triples.json")
KG_TRIPLES_VALIDATED_PATH = os.path.join(KG_LLM_DIR, "kg_triples_llm_validated.json")

# Progress saving for resume capability
PROGRESS_FILE = os.path.join(KG_DIR, "validation_progress.json")

# Log file for debugging
LOG_FILE = os.path.join(KG_DIR, "validation_log.txt")

# Global log file handle
_log_file = None


def log_print(msg: str):
    """Print to console and log file simultaneously."""
    global _log_file
    print(msg)
    if _log_file:
        _log_file.write(msg + "\n")
        _log_file.flush()  # Ensure immediate write


def open_log_file():
    """Open the log file for writing."""
    global _log_file
    _log_file = open(LOG_FILE, "w", encoding="utf-8")
    log_print(f"[LOG] Logging to: {LOG_FILE}")
    log_print(f"[LOG] Started at: {time.strftime('%Y-%m-%d %H:%M:%S')}")


def close_log_file():
    """Close the log file."""
    global _log_file
    if _log_file:
        log_print(f"[LOG] Finished at: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        _log_file.close()
        _log_file = None


# =================== LLM VALIDATION ===================

def validate_fact_with_llm(topic: str, relation: str, value: str, source_text: str, retries: int = 2) -> Tuple[bool, str]:
    """
    Use LLM to validate if a fact is valid.
    Returns (is_valid, reason).
    """
    prompt = f"""You are validating extracted facts from a university knowledge base.

FACT TO VALIDATE:
- Topic: {topic}
- Relation: {relation}
- Value: {value}
- Source text: {source_text[:200] if source_text else 'N/A'}

VALIDATION RULES - Mark as INVALID if ANY of these apply:
1. GNO/GPA values must be between 0.0 and 4.0 (values like 3.17, 10.3, 9.3 are section numbers, NOT valid GPAs)
2. Value is just a word like "GNO" or "GPA" without an actual number
3. Value is a document or form code (e.g., "IIPAR-C710-02", "FPOP-S230-01-08")
4. Value is a date used as GNO (e.g., "23.01.2001", "21-06-2016")
5. Duration values that are years (e.g., "2024", "January 2020")
6. Value is placeholder text like "...", ".../../....."
7. The value is clearly truncated/chopped off mid-sentence (e.g. ends with "ve", "ile", "çalışanla")

CRITICAL: Do NOT use your own external knowledge to verify if a fact is true! Assume the source text is always true. 
You are ONLY checking if the FORMAT and CATEGORY make logical sense. For example, if relation is 'duration' and value is '15 gün', it is perfectly VALID. If relation is 'limit' and value is '10 gün', it is INVALID because '10 gün' is a duration. Always lean towards VALID if it makes basic logical sense.

EXAMPLES:
- Topic: "Erasmus", Relation: "minimum_gno", Value: "2.20 (lisans)" → VALID (proper GPA)
- Topic: "Erasmus", Relation: "minimum_gno", Value: "3.17" → INVALID (section number, GPA should be 2.20 or 2.5)
- Topic: "Erasmus", Relation: "minimum_gno", Value: "GNO" → INVALID (not a value, just the word GNO)
- Topic: "Staj", Relation: "duration", Value: "2 ay" → VALID (proper duration)
- Topic: "Disiplin", Relation: "has_duration", Value: ".../../....." → INVALID (placeholder)

Answer with EXACTLY one word: VALID or INVALID"""

    for attempt in range(retries + 1):
        try:
            content = _llm_service.chat(
                prompt=prompt,
                system_prompt="You are a data validation expert. Answer EXACTLY with either VALID or INVALID.",
                temperature=0.0
            ).strip().upper()
            
            # Parse response - be strict
            if "INVALID" in content:
                return False, "LLM: INVALID"
            elif "VALID" in content:
                return True, "LLM: VALID"
            else:
                # If unclear, default to INVALID (safer)
                return False, f"LLM unclear: {content[:30]}"
                
        except requests.exceptions.Timeout:
            if attempt < retries:
                log_print(f"    [Timeout, retry {attempt+1}/{retries}]")
                time.sleep(2)
            else:
                return True, "Timeout - keeping"
        except Exception as e:
            if attempt < retries:
                log_print(f"    [Error: {e}, retry {attempt+1}/{retries}]")
                time.sleep(2)
            else:
                return True, f"Error - keeping: {e}"
    
    return True, "Max retries - keeping"


def validate_triple_with_llm(head: str, relation: str, tail: str, source_text: str, retries: int = 2) -> Tuple[bool, str]:
    """
    Use LLM to validate if a triple is meaningful and relevant.
    """
    prompt = f"""You are validating knowledge graph triples from a university document system.

TRIPLE TO VALIDATE:
- Head Entity: {head}
- Relation: {relation}
- Tail Entity: {tail}
- Source: {source_text[:150] if source_text else 'N/A'}

VALIDATION RULES - Mark as INVALID if ANY of these apply:
1. Head or tail contains garbled text, JSON artifacts, or formatting errors
2. Head or tail is a generic placeholder (e.g., "...", "N/A", "TBD")
3. The triple is explicitly about internal university IT systems (SUTicket, Otomasyon, Mimari İnşaat)
4. Head or tail is just a document code without meaningful content (e.g. PIC-C840-0201)
5. Head or tail is clearly chopped off mid-sentence (e.g. ends with "ve", "ile")

CRITICAL: Do NOT use your own external knowledge to verify if the relationship is true in the real world! Assume the source text is always accurate. You are ONLY checking if the Head -> Relation -> Tail structure makes basic semantic sense. Assume true unless it is completely incomprehensible.

EXAMPLES:
- Head: "Erasmus Staj", Relation: "requires", Tail: "minimum 2.20 GNO" → VALID
- Head: "SUTicket sistemi", Relation: "used by", Tail: "Mimari ve İnşaat Bölümü" → INVALID (internal IT, not useful for students)
- Head: "değişiklikler", Relation: "results in", Tail: "SÜ hedeflerine..." → INVALID (too vague)
- Head: "Öğrenci", Relation: "applies to", Tail: "Staj Başvurusu" → VALID

Answer with EXACTLY one word: VALID or INVALID"""

    for attempt in range(retries + 1):
        try:
            content = _llm_service.chat(
                prompt=prompt,
                system_prompt="You are a data validation expert. Answer EXACTLY with either VALID or INVALID.",
                temperature=0.0
            ).strip().upper()
            
            if "INVALID" in content:
                return False, "LLM: INVALID"
            elif "VALID" in content:
                return True, "LLM: VALID"
            else:
                return False, f"LLM unclear: {content[:30]}"
                
        except requests.exceptions.Timeout:
            if attempt < retries:
                time.sleep(2)
            else:
                return True, "Timeout - keeping"
        except Exception as e:
            if attempt < retries:
                time.sleep(2)
            else:
                return True, f"Error: {e}"
    
    return True, "Max retries - keeping"


# =================== PROGRESS MANAGEMENT ===================

def save_progress(facts_done: int, triples_done: int, validated_facts: Dict, validated_triples: List):
    """Save progress for resume capability."""
    progress = {
        "facts_done": facts_done,
        "triples_done": triples_done,
        "timestamp": time.time()
    }
    with open(PROGRESS_FILE, "w", encoding="utf-8") as f:
        json.dump(progress, f)
    
    # Save partial results - ONLY if we have data (avoid overwriting with empty)
    if validated_facts:  # Only save facts if non-empty
        with open(KG_FACTS_VALIDATED_PATH, "w", encoding="utf-8") as f:
            json.dump(validated_facts, f, ensure_ascii=False, indent=2)
    
    if validated_triples:
        with open(KG_TRIPLES_VALIDATED_PATH, "w", encoding="utf-8") as f:
            json.dump({"triples": validated_triples}, f, ensure_ascii=False, indent=2)


def load_progress() -> Tuple[int, int]:
    """Load progress for resume."""
    if os.path.exists(PROGRESS_FILE):
        with open(PROGRESS_FILE, "r") as f:
            progress = json.load(f)
        return progress.get("facts_done", 0), progress.get("triples_done", 0)
    return 0, 0


# =================== MAIN VALIDATION ===================

def validate_facts():
    """Validate ALL facts using LLM."""
    
    log_print("=" * 60)
    log_print("[KG Validation] LLM-ONLY Mode - Validating all facts...")
    log_print("=" * 60)
    
    if not os.path.exists(KG_FACTS_PATH):
        log_print(f"[ERROR] Facts file not found: {KG_FACTS_PATH}")
        return {}
    
    with open(KG_FACTS_PATH, "r", encoding="utf-8") as f:
        facts_by_topic = json.load(f)
    
    # Flatten facts for processing
    all_facts = []
    for topic, facts in facts_by_topic.items():
        for fact in facts:
            all_facts.append((topic, fact))
    
    total_facts = len(all_facts)
    log_print(f"[OK] Loaded {len(facts_by_topic)} topics, {total_facts} facts")
    
    # Check for resume
    facts_done, _ = load_progress()
    validated_facts: Dict[str, List[Dict]] = {}
    removed_count = 0
    kept_count = 0

    if facts_done > 0:
        log_print(f"[RESUME] Continuing from fact {facts_done}/{total_facts}")
        if os.path.exists(KG_FACTS_VALIDATED_PATH):
            with open(KG_FACTS_VALIDATED_PATH, "r", encoding="utf-8") as f:
                validated_facts = json.load(f)
            kept_count = sum(len(f) for f in validated_facts.values())
            removed_count = facts_done - kept_count
            log_print(f"  - Loaded {kept_count} previously validated facts")
    
    start_time = time.time()
    
    for idx, (topic, fact) in enumerate(all_facts):
        if idx < facts_done:
            continue
        
        relation = fact.get("relation", "")
        value = fact.get("value", "")
        source_text = fact.get("source_text", "")
        
        # Progress update every 50 facts
        if idx % 50 == 0:
            elapsed = time.time() - start_time
            rate = idx / elapsed if elapsed > 0 else 0
            eta = (total_facts - idx) / rate / 60 if rate > 0 else 0
            log_print(f"[Progress] {idx}/{total_facts} facts | {removed_count} removed | ETA: {eta:.1f} min")
        
        # LLM validation
        is_valid, reason = validate_fact_with_llm(topic, relation, value, source_text)
        
        if not is_valid:
            log_print(f"  [REMOVED] {topic[:30]} | {relation}: {value[:50]} | {reason}")
            removed_count += 1
        else:
            if topic not in validated_facts:
                validated_facts[topic] = []
            validated_facts[topic].append(fact)
            kept_count += 1
        
        # Save progress every 100 facts
        if idx % 100 == 0 and idx > 0:
            save_progress(idx, 0, validated_facts, [])
    
    # Remove empty topics
    validated_facts = {t: f for t, f in validated_facts.items() if f}
    
    # DEBUG: Show what we're about to save
    log_print(f"\n[DEBUG] Before final save:")
    log_print(f"  - Topics in dict: {len(validated_facts)}")
    log_print(f"  - Total facts in dict: {sum(len(f) for f in validated_facts.values())}")
    
    # Final save
    with open(KG_FACTS_VALIDATED_PATH, "w", encoding="utf-8") as f:
        json.dump(validated_facts, f, ensure_ascii=False, indent=2)
    
    # Verify save
    saved_size = os.path.getsize(KG_FACTS_VALIDATED_PATH)
    log_print(f"  - Saved file size: {saved_size} bytes")
    
    log_print(f"\n[DONE] Fact validation complete!")
    log_print(f"  - Original: {total_facts}")
    log_print(f"  - Kept: {kept_count}")
    log_print(f"  - Removed: {removed_count}")
    log_print(f"  - Saved to: {KG_FACTS_VALIDATED_PATH}")
    
    return validated_facts


def validate_triples():
    """Validate ALL triples using LLM."""
    
    log_print("\n" + "=" * 60)
    log_print("[KG Validation] LLM-ONLY Mode - Validating all triples...")
    log_print("=" * 60)
    
    if not os.path.exists(KG_TRIPLES_PATH):
        log_print(f"[SKIP] Triples file not found: {KG_TRIPLES_PATH}")
        return []
    
    with open(KG_TRIPLES_PATH, "r", encoding="utf-8") as f:
        triples_data = json.load(f)
    
    triples = triples_data.get("triples", [])
    total_triples = len(triples)
    log_print(f"[OK] Loaded {total_triples} triples")
    
    # Check for resume
    _, triples_done = load_progress()
    validated_triples = []
    removed_count = 0
    kept_count = 0

    if triples_done > 0:
        log_print(f"[RESUME] Continuing from triple {triples_done}/{total_triples}")
        if os.path.exists(KG_TRIPLES_VALIDATED_PATH):
            with open(KG_TRIPLES_VALIDATED_PATH, "r", encoding="utf-8") as f:
                saved = json.load(f)
                validated_triples = saved.get("triples", [])
            kept_count = len(validated_triples)
            removed_count = triples_done - kept_count
            log_print(f"  - Loaded {kept_count} previously validated triples")
    
    start_time = time.time()
    
    for idx, triple in enumerate(triples):
        if idx < triples_done:
            continue
        
        head = str(triple.get("head", ""))
        relation = str(triple.get("relation", ""))
        tail = str(triple.get("tail", ""))
        source_text = triple.get("source_text", "")
        
        # Progress update every 100 triples
        if idx % 100 == 0:
            elapsed = time.time() - start_time
            rate = idx / elapsed if elapsed > 0 else 0
            eta = (total_triples - idx) / rate / 60 if rate > 0 else 0
            log_print(f"[Progress] {idx}/{total_triples} triples | {removed_count} removed | ETA: {eta:.1f} min")
        
        # LLM validation
        is_valid, reason = validate_triple_with_llm(head, relation, tail, source_text)
        
        if not is_valid:
            log_print(f"  [REMOVED] {head[:25]} -> [{relation}] -> {tail[:25]} | {reason}")
            removed_count += 1
        else:
            validated_triples.append(triple)
        
        # Save progress every 200 triples
        if idx % 200 == 0 and idx > 0:
            save_progress(99999, idx, {}, validated_triples)
    
    # Save validated triples
    validated_data = {
        "triples": validated_triples,
        "entity_embeddings": triples_data.get("entity_embeddings", {})
    }
    
    with open(KG_TRIPLES_VALIDATED_PATH, "w", encoding="utf-8") as f:
        json.dump(validated_data, f, ensure_ascii=False, indent=2)
    
    log_print(f"\n[DONE] Triple validation complete!")
    log_print(f"  - Original: {total_triples}")
    log_print(f"  - Kept: {len(validated_triples)}")
    log_print(f"  - Removed: {removed_count}")
    log_print(f"  - Saved to: {KG_TRIPLES_VALIDATED_PATH}")
    
    return validated_triples


def main():
    # Open log file for this run
    open_log_file()
    
    try:
        log_print("=" * 60)
        log_print("[KG VALIDATION] LLM-ONLY MODE")
        log_print("This will validate EVERY fact and triple using LLM.")
        log_print("Expected time: 2-4 hours")
        log_print("=" * 60)
        log_print("")
        
        start_time = time.time()
        
        # Validate facts
        validate_facts()
        
        # Validate triples
        validate_triples()
        
        elapsed = time.time() - start_time
        hours = elapsed / 3600
        
        # Clean up progress file
        if os.path.exists(PROGRESS_FILE):
            os.remove(PROGRESS_FILE)
        
        log_print("\n" + "=" * 60)
        log_print(f"[COMPLETE] Total time: {hours:.2f} hours ({elapsed:.0f}s)")
        log_print("=" * 60)
        log_print(f"\nNext steps:")
        log_print(f"  1. Update kg_service.py to use new validated files:")
        log_print(f"     - kg_facts_llm_validated.json")
        log_print(f"     - kg_triples_llm_validated.json")
        log_print(f"  2. Run test again: python test_rag_detailed.py")
    
    finally:
        # Always close log file
        close_log_file()


if __name__ == "__main__":
    main()
