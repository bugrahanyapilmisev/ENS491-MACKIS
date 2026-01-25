"""
clean_kg_data.py - Clean and fix KG data for better RAG performance

This script addresses the following issues found in the KG:
1. Wrong field names (e.g., "Minimum Gno" used for durations)
2. Garbage/placeholder data (e.g., ".../../.....")
3. Conflicting multiple values for same field
4. Irrelevant generic triples

Usage:
    python clean_kg_data.py
"""

import os
import json
import re
from typing import Dict, List, Any
from collections import defaultdict

# =================== CONFIG ===================

PREPROCESSING_DIR = os.path.dirname(os.path.abspath(__file__))
KG_DIR = os.path.join(PREPROCESSING_DIR, "knowledge_graph")

# Input files (pattern-validated - best performing)
INPUT_FACTS_PATH = os.path.join(KG_DIR, "kg_facts_validated.json")
INPUT_TRIPLES_PATH = os.path.join(KG_DIR, "kg_triples_validated.json")

# Output files (cleaned)
OUTPUT_FACTS_PATH = os.path.join(KG_DIR, "kg_facts_cleaned.json")
OUTPUT_TRIPLES_PATH = os.path.join(KG_DIR, "kg_triples_cleaned.json")

# =================== GARBAGE PATTERNS ===================

# Patterns that indicate garbage/placeholder data
GARBAGE_PATTERNS = [
    r"^\.\.\.+$",                    # Just dots: "...", "...."
    r"^\.\./\.\./\.+",              # Date placeholders: ".../../....."
    r"^/\s*\d+\.?$",                # Garbage like "/ 5."
    r"^\s*$",                        # Empty or whitespace
    r"^not specified$",             # Explicit "not specified"
    r"^N/A$",                       # N/A
    r"^-+$",                        # Just dashes
    r"^\d{1,2}\.$",                 # Just numbers with dot like "1.", "2."
]

# Values that are too vague to be useful
VAGUE_VALUES = [
    "makul bir süre içinde",
    "süresi sonunda",
    "iade_süresi_dolmamış",
    "son tarihe kadar",
    "dolmuş",
]

# =================== FIELD NAME FIXES ===================

# When a relation is used with wrong value type, fix it
# Key: relation name in data, Value: conditions and new name
RELATION_FIXES = {
    "minimum_gno": {
        # If value looks like a duration (days, months, years), rename to "duration"
        "duration_patterns": [
            r"\d+\s*(ay|gün|hafta|yıl|year|month|day|week|saat|hour)",
            r"\d+\s*(days?|months?|weeks?|years?|hours?)",
            r"^\d+\s*(years?)$",
        ],
        "new_relation": "duration"
    },
    "limit": {
        # If value looks like a duration, rename
        "duration_patterns": [
            r"\d+\s*(ay|gün|hafta|yıl|year|month|day|week)",
        ],
        "new_relation": "duration"
    }
}

# Relations that are semantically wrong for their values (explicit mappings)
WRONG_RELATION_VALUES = {
    # (relation, value pattern) -> new_relation
    ("minimum_gno", r"^\d+\s*yıl"): "duration",  # "3 yıl" should be duration
    ("minimum_gno", r"^\d+\s*ay"): "duration",   # "2 ay" should be duration
    ("minimum_gno", r"^GNO$"): None,              # Vague, remove
    ("minimum_gno", r"^%\d+"): "percentage",      # "%50" is a percentage
}

# =================== IRRELEVANT TRIPLE PATTERNS ===================

# Relationships that are too generic to be useful
IRRELEVANT_RELATION_PATTERNS = [
    r"SUTicket sistemi",
    r"Elektrik İşleri Bölümü",
    r"Mimari ve İnşaat Bölümü",
    r"Otomasyon İşleri Bölümü",
    r"teknik konularda",
    r"iş_organizasyonu",
    r"periyodik_bakım",
]

# =================== CLEANING FUNCTIONS ===================

def is_garbage_value(value: str) -> bool:
    """Check if a value is garbage/placeholder data."""
    if not isinstance(value, str):
        return False
    
    value = value.strip()
    
    # Check against garbage patterns
    for pattern in GARBAGE_PATTERNS:
        if re.match(pattern, value, re.IGNORECASE):
            return True
    
    # Check against vague values
    if value.lower() in [v.lower() for v in VAGUE_VALUES]:
        return True
    
    return False


def fix_relation_name(relation: str, value: str) -> tuple:
    """
    Fix relation names that are incorrectly used.
    Returns: (new_relation, should_remove)
            - new_relation: fixed name or original
            - should_remove: True if this fact should be removed entirely
    """
    relation_lower = relation.lower()
    
    # Check explicit wrong relation-value mappings
    for (rel, pattern), new_rel in WRONG_RELATION_VALUES.items():
        if relation_lower == rel and re.search(pattern, value, re.IGNORECASE):
            if new_rel is None:
                return relation, True  # Mark for removal
            return new_rel, False
    
    # Check RELATION_FIXES for pattern-based fixes
    if relation_lower in RELATION_FIXES:
        fix_info = RELATION_FIXES[relation_lower]
        
        # Check if value matches duration patterns
        for pattern in fix_info.get("duration_patterns", []):
            if re.search(pattern, value, re.IGNORECASE):
                return fix_info["new_relation"], False
    
    return relation, False


def deduplicate_facts(facts_list: List[Dict]) -> List[Dict]:
    """Remove duplicate facts, keeping the most specific one."""
    seen = {}
    
    for fact in facts_list:
        relation = fact.get("relation", "")
        value = fact.get("value", "")
        
        # Skip garbage values
        if is_garbage_value(value):
            continue
        
        # For duplicate relations, prefer longer/more specific values
        if relation in seen:
            existing_value = seen[relation]["value"]
            # Keep the longer, more specific value
            if len(str(value)) > len(str(existing_value)) and not is_garbage_value(value):
                seen[relation] = fact
        else:
            seen[relation] = fact
    
    return list(seen.values())


def clean_facts(facts: Dict[str, List[Dict]]) -> Dict[str, List[Dict]]:
    """Clean the facts dictionary."""
    cleaned = {}
    stats = {
        "topics_before": len(facts),
        "topics_after": 0,
        "facts_removed_garbage": 0,
        "facts_removed_duplicate": 0,
        "facts_removed_vague": 0,
        "relation_names_fixed": 0,
    }
    
    for topic, fact_list in facts.items():
        cleaned_facts = []
        
        for fact in fact_list:
            relation = fact.get("relation", "")
            value = fact.get("value", "")
            
            # Skip garbage values
            if is_garbage_value(value):
                stats["facts_removed_garbage"] += 1
                continue
            
            # Fix relation names
            new_relation, should_remove = fix_relation_name(relation, value)
            
            if should_remove:
                stats["facts_removed_vague"] += 1
                continue
            
            if new_relation != relation:
                stats["relation_names_fixed"] += 1
                fact = {**fact, "relation": new_relation}
            
            cleaned_facts.append(fact)
        
        # Deduplicate facts for this topic
        original_count = len(cleaned_facts)
        cleaned_facts = deduplicate_facts(cleaned_facts)
        stats["facts_removed_duplicate"] += original_count - len(cleaned_facts)
        
        # Only keep topics with at least one fact
        if cleaned_facts:
            cleaned[topic] = cleaned_facts
    
    stats["topics_after"] = len(cleaned)
    return cleaned, stats




def is_irrelevant_triple(triple: Dict) -> bool:
    """Check if a triple is too generic/irrelevant."""
    head = triple.get("head", "")
    tail = triple.get("tail", "")
    relation = triple.get("relation", "")
    
    # Check if any part matches irrelevant patterns
    for pattern in IRRELEVANT_RELATION_PATTERNS:
        if re.search(pattern, head, re.IGNORECASE):
            return True
        if re.search(pattern, tail, re.IGNORECASE):
            return True
    
    # Check for garbage in any part
    if is_garbage_value(head) or is_garbage_value(tail) or is_garbage_value(relation):
        return True
    
    return False


def clean_triples(triples_data: Dict) -> Dict:
    """Clean the triples data."""
    triples = triples_data.get("triples", [])
    stats = {
        "triples_before": len(triples),
        "triples_after": 0,
        "triples_removed_irrelevant": 0,
        "triples_removed_garbage": 0,
    }
    
    cleaned_triples = []
    
    for triple in triples:
        if is_irrelevant_triple(triple):
            stats["triples_removed_irrelevant"] += 1
            continue
        
        # Check for garbage values in head/tail/relation
        head = triple.get("head", "")
        tail = triple.get("tail", "")
        relation = triple.get("relation", "")
        
        if is_garbage_value(head) or is_garbage_value(tail):
            stats["triples_removed_garbage"] += 1
            continue
        
        cleaned_triples.append(triple)
    
    stats["triples_after"] = len(cleaned_triples)
    
    return {"triples": cleaned_triples}, stats


# =================== MAIN ===================

def main():
    print("=" * 60)
    print("[CLEAN KG DATA] Starting KG cleanup")
    print("=" * 60)
    
    # Load facts
    if not os.path.exists(INPUT_FACTS_PATH):
        print(f"[ERROR] Facts file not found: {INPUT_FACTS_PATH}")
        return
    
    with open(INPUT_FACTS_PATH, "r", encoding="utf-8") as f:
        facts = json.load(f)
    
    print(f"[OK] Loaded {len(facts)} topics from facts file")
    
    # Load triples
    if not os.path.exists(INPUT_TRIPLES_PATH):
        print(f"[ERROR] Triples file not found: {INPUT_TRIPLES_PATH}")
        return
    
    with open(INPUT_TRIPLES_PATH, "r", encoding="utf-8") as f:
        triples_data = json.load(f)
    
    triples_count = len(triples_data.get("triples", []))
    print(f"[OK] Loaded {triples_count} triples from triples file")
    
    # Clean facts
    print("\n[Cleaning facts...]")
    cleaned_facts, facts_stats = clean_facts(facts)
    
    print(f"  Topics: {facts_stats['topics_before']} -> {facts_stats['topics_after']}")
    print(f"  Removed garbage values: {facts_stats['facts_removed_garbage']}")
    print(f"  Removed vague values: {facts_stats['facts_removed_vague']}")
    print(f"  Removed duplicates: {facts_stats['facts_removed_duplicate']}")
    print(f"  Fixed relation names: {facts_stats['relation_names_fixed']}")
    
    # Clean triples
    print("\n[Cleaning triples...]")
    cleaned_triples, triples_stats = clean_triples(triples_data)
    
    print(f"  Triples: {triples_stats['triples_before']} -> {triples_stats['triples_after']}")
    print(f"  Removed irrelevant: {triples_stats['triples_removed_irrelevant']}")
    print(f"  Removed garbage: {triples_stats['triples_removed_garbage']}")
    
    # Save cleaned facts
    with open(OUTPUT_FACTS_PATH, "w", encoding="utf-8") as f:
        json.dump(cleaned_facts, f, ensure_ascii=False, indent=2)
    print(f"\n[SAVED] Cleaned facts: {OUTPUT_FACTS_PATH}")
    
    # Save cleaned triples
    with open(OUTPUT_TRIPLES_PATH, "w", encoding="utf-8") as f:
        json.dump(cleaned_triples, f, ensure_ascii=False, indent=2)
    print(f"[SAVED] Cleaned triples: {OUTPUT_TRIPLES_PATH}")
    
    # Summary
    print("\n" + "=" * 60)
    print("[SUMMARY]")
    print("=" * 60)
    print(f"Facts:   {facts_stats['topics_before']:,} topics -> {facts_stats['topics_after']:,} topics")
    print(f"Triples: {triples_stats['triples_before']:,} -> {triples_stats['triples_after']:,}")
    print(f"\nNext steps:")
    print("1. Update kg_service.py to use cleaned files:")
    print("   KG_FACTS_PATH = 'kg_facts_cleaned.json'")
    print("   KG_TRIPLES_PATH = 'kg_triples_cleaned.json'")
    print("2. Run rebuild_kg_index_pattern.py to rebuild embeddings")
    print("3. Run the test again to measure improvement")


if __name__ == "__main__":
    main()
