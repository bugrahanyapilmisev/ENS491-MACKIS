"""
kg_service.py - GENERIC Knowledge Graph Query Service

GENERIC DESIGN - Uses embedding-based semantic search, no hardcoded mappings.

Key changes from original:
1. Removed all TOPIC_KEYWORDS hardcoded mappings
2. Uses embedding similarity to match queries to topics
3. Format facts for RAG context without domain-specific assumptions
4. Works for any domain automatically
5. Uses centralized configuration from services/config/settings.py
"""

import os
import json
import pickle
import re
import numpy as np
from typing import List, Dict, Optional, Tuple
from collections import defaultdict

import requests
from dotenv import load_dotenv

load_dotenv()

# =================== CONFIG ===================

# Import centralized configuration
from services.config.settings import RAGConfig

# Load configuration
_config = RAGConfig.from_env()

OLLAMA_HOST = _config.ollama.host
EMBED_MODEL = _config.ollama.embed_model

# KG paths from config based on active source
KG_OUTPUT_DIR = _config.kg.kg_output_dir

def _get_kg_paths() -> Tuple[str, str, str]:
    """Get the KG file paths based on active source configuration."""
    source = _config.kg.active_source.lower()

    if source == "llm":
        return (
            _config.kg.kg_facts_llm_validated,
            _config.kg.kg_triples_llm_validated,
            _config.kg.kg_index_llm_validated,
        )
    elif source == "pattern":
        return (
            _config.kg.kg_facts_pattern_validated,
            _config.kg.kg_triples_pattern_validated,
            _config.kg.kg_index_pattern_validated,
        )
    else:  # raw
        return (
            _config.kg.kg_facts,
            _config.kg.kg_triples,
            os.path.join(KG_OUTPUT_DIR, "kg_index.pkl"),  # No raw index, fallback
        )

KG_FACTS_PATH, KG_TRIPLES_PATH, KG_INDEX_PATH = _get_kg_paths()

# Service state - Facts (Topic -> Fact)
_kg_facts: Dict[str, List[Dict]] = {}
_topic_embeddings: Dict[str, np.ndarray] = {}
_topic_list: List[str] = []

# Service state - Triples (Entity -> Relation -> Entity)
_kg_triples: List[Dict] = []
_entity_embeddings: Dict[str, np.ndarray] = {}

_loaded = False


# =================== EMBEDDING HELPER ===================
from services.core.embedding_service import EmbeddingService

_embed_service = EmbeddingService(_config.ollama)

def embed_text(text: str) -> np.ndarray:
    """Embed text using configured provider."""
    try:
        vec = _embed_service.embed(text, is_query=True)
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec /= norm
        return vec
    except Exception as e:
        print(f"[KG Service embed error] {e}")
        return None


def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine similarity between two vectors."""
    if a is None or b is None:
        return 0.0
    return float(np.dot(a, b))


# =================== LOADING ===================

def load_kg_facts() -> bool:
    """Load knowledge graph facts, triples, and embeddings."""
    global _kg_facts, _topic_embeddings, _topic_list, _kg_triples, _entity_embeddings, _loaded
    
    if _loaded:
        return True
    
    # Load facts (Topic -> Fact)
    if os.path.exists(KG_FACTS_PATH):
        try:
            with open(KG_FACTS_PATH, "r", encoding="utf-8") as f:
                _kg_facts = json.load(f)
            print(f"[KG Service] Loaded facts: {len(_kg_facts)} topics")
        except Exception as e:
            print(f"[KG Service] Failed to load facts: {e}")
            return False
    else:
        print(f"[KG Service] Facts file not found: {KG_FACTS_PATH}")
        return False
    
    # Load triples (Entity -> Relation -> Entity) - NEW
    if os.path.exists(KG_TRIPLES_PATH):
        try:
            with open(KG_TRIPLES_PATH, "r", encoding="utf-8") as f:
                triples_data = json.load(f)
            _kg_triples = triples_data.get("triples", [])
            # Load entity embeddings
            for ent, emb in triples_data.get("entity_embeddings", {}).items():
                _entity_embeddings[ent] = np.array(emb, dtype=np.float32)
            print(f"[KG Service] Loaded triples: {len(_kg_triples)} relations")
        except Exception as e:
            print(f"[KG Service] Triples load warning: {e}")
            # Continue without triples
    
    # Load topic embeddings from index
    if os.path.exists(KG_INDEX_PATH):
        try:
            with open(KG_INDEX_PATH, "rb") as f:
                index_data = pickle.load(f)
            
            # Convert embeddings back to numpy arrays
            for topic, emb in index_data.get("topic_embeddings", {}).items():
                _topic_embeddings[topic] = np.array(emb, dtype=np.float32)
            
            # Load entity embeddings if present in index
            for ent, emb in index_data.get("entity_embeddings", {}).items():
                if ent not in _entity_embeddings:
                    _entity_embeddings[ent] = np.array(emb, dtype=np.float32)
            
            _topic_list = index_data.get("topic_list", list(_kg_facts.keys()))
            print(f"[KG Service] Loaded embeddings for {len(_topic_embeddings)} topics")
        except Exception as e:
            print(f"[KG Service] Failed to load index: {e}")
            _topic_list = list(_kg_facts.keys())
    else:
        _topic_list = list(_kg_facts.keys())
    
    # ── Runtime quality filter: clean noisy facts before serving ──
    _runtime_quality_filter()
    
    _loaded = True
    print(f"[KG Service] Hybrid KG loaded: {len(_kg_facts)} topics, {len(_kg_triples)} triples")
    return True


def _runtime_quality_filter():
    """Clean noisy facts at load time (runs once)."""
    global _kg_facts, _topic_list
    before = len(_kg_facts)
    
    # 1. Remove pure-metadata noise topics
    NOISE_TOPICS = {
        "effective_date", "update_date", "statutory_basis",
        "prosedur", "procedure", "yonerge", "instruction",
    }
    for noise in NOISE_TOPICS:
        _kg_facts.pop(noise, None)
    
    # 2. Filter facts with bad values
    for topic in list(_kg_facts.keys()):
        filtered = []
        for f in _kg_facts[topic]:
            value = str(f.get("value", "")).strip()
            relation = f.get("relation", "")
            
            # Skip empty, too short, or template placeholders
            if len(value) < 2 or "..." in value or "___" in value:
                continue
            
            # Skip form codes used as GNO/duration/limit values
            if relation in ("minimum_gno", "duration", "limit") and \
               re.match(r'^[A-Z]{2,5}-[A-Z0-9]', value):
                continue
            
            # Skip years as durations
            if relation == "duration" and re.match(r'^(19|20)\d{2}$', value):
                continue
            
            # Skip vague values
            if value.lower() in ("belirlenir", "ilgili birim", "none", "yapılır", "uygulanır"):
                continue
            
            # minimum_gno must contain a number or GPA keyword
            if relation == "minimum_gno":
                if not re.search(r'\d', value) and \
                   not any(kw in value.lower() for kw in ("gno", "gpa", "%")):
                    continue
            
            filtered.append(f)
        _kg_facts[topic] = filtered
    
    # 3. Remove empty topics
    _kg_facts = {t: f for t, f in _kg_facts.items() if f}
    
    # 4. Merge synonym topics
    MERGES = {
        "burs": "scholarship_requirements",
        "burslar": "scholarship_requirements",
        "burs_miktar": "scholarship_requirements",
        "burs_tutari": "scholarship_requirements",
        "burs_suresi": "scholarship_requirements",
        "burs_benefit": "scholarship_requirements",
        "burs_payment": "scholarship_requirements",
        "burs_form": "scholarship_requirements",
        "scholarship": "scholarship_requirements",
        "scholarship_payment": "scholarship_requirements",
        "staj_suresi": "internship_requirements",
        "staj_kurumu": "internship_requirements",
    }
    for old_name, new_name in MERGES.items():
        if old_name in _kg_facts and old_name != new_name:
            _kg_facts.setdefault(new_name, []).extend(_kg_facts.pop(old_name))
    
    # Deduplicate merged topics
    for topic in _kg_facts:
        seen = set()
        unique = []
        for f in _kg_facts[topic]:
            key = (f.get("relation", ""), str(f.get("value", ""))[:50])
            if key not in seen:
                seen.add(key)
                unique.append(f)
        _kg_facts[topic] = unique
    
    # Update topic list
    _topic_list = list(_kg_facts.keys())
    
    after = len(_kg_facts)
    total = sum(len(f) for f in _kg_facts.values())
    print(f"[KG Service] Quality filter: {before} → {after} topics, {total} facts")


# =================== SEMANTIC TOPIC MATCHING ===================

def find_relevant_topics(query: str, top_k: int = 5, threshold: float = 0.35) -> List[Tuple[str, float]]:
    """
    Find relevant KG topics using SEMANTIC SIMILARITY.
    No hardcoded keyword mappings.
    
    Returns: List of (topic, similarity_score) tuples
    """
    if not _loaded:
        load_kg_facts()
    
    if not _topic_embeddings:
        # Fallback to simple keyword matching if no embeddings
        return _fallback_topic_matching(query, top_k)
    
    # Embed the query
    query_embedding = embed_text(query)
    if query_embedding is None:
        return _fallback_topic_matching(query, top_k)
    
    # Compute similarities with all topics
    scores = []
    for topic, topic_emb in _topic_embeddings.items():
        sim = cosine_sim(query_embedding, topic_emb)
        if sim >= threshold:
            scores.append((topic, sim))
    
    # Sort by similarity
    scores.sort(key=lambda x: x[1], reverse=True)
    
    return scores[:top_k]


def _fallback_topic_matching(query: str, top_k: int = 5) -> List[Tuple[str, float]]:
    """
    Fallback topic matching using keyword overlap.
    Still GENERIC - no hardcoded mappings.
    """
    query_lower = query.lower()
    query_words = set(re.findall(r"\w+", query_lower))
    
    scores = []
    for topic in _topic_list:
        topic_words = set(topic.replace("_", " ").split())
        overlap = len(query_words & topic_words)
        if overlap > 0:
            score = overlap / max(len(topic_words), 1)
            scores.append((topic, score))
    
    scores.sort(key=lambda x: x[1], reverse=True)
    return scores[:top_k]


# =================== FACT RETRIEVAL ===================

# Query intent patterns
INTENT_PATTERNS = {
    "gno": ["gno", "gpa", "not ortalaması", "ortalama", "minimum", "en az", "grade point"],
    "duration": ["süre", "kaç gün", "kaç ay", "kaç hafta", "ne kadar", "duration", "how long"],
    "penalty": ["ceza", "disiplin", "penalty", "punishment", "uzaklaştırma", "kınama", "uyarı"],
    "quantity": ["kaç adet", "kaç kitap", "kaç tane", "limit", "sınır", "how many"],
    "credit": ["kredi", "ects", "akts", "credit"],
}

# Relation types that match each intent
INTENT_TO_RELATIONS = {
    "gno": ["minimum_gno", "minimum_value"],
    "duration": ["duration"],
    "penalty": ["penalty_type", "penalty_duration"],
    "quantity": ["quantity_limit", "limit"],
    "credit": ["credit", "credit_requirement"],
}


def _detect_query_intent(query: str) -> Optional[str]:
    """Detect what type of fact the query is asking for."""
    query_lower = query.lower()
    
    for intent, patterns in INTENT_PATTERNS.items():
        for pattern in patterns:
            if pattern in query_lower:
                return intent
    return None


def _is_valid_fact(fact: Dict) -> bool:
    """Filter out noisy/invalid facts."""
    value = fact.get("value", "")
    relation = fact.get("relation", "")
    
    # Filter out years misinterpreted as durations (e.g., "2024 Gün")
    if relation == "duration":
        # Check for year patterns (1990-2030)
        import re
        if re.search(r"\b(19|20)\d{2}\b", value):
            return False
    
    # Filter out very short values
    if len(str(value).strip()) < 1:
        return False
    
    # Filter out "uyarı" penalty type that's not actually a penalty (false positive)
    if relation == "penalty_type" and value.lower() == "uyarı":
        source = fact.get("source_text", "").lower()
        # Only keep if source text actually talks about discipline
        if "disiplin" not in source and "ceza" not in source:
            return False
    
    return True


def query_facts(query: str, lang: str = "tr", max_facts: int = 10) -> str:
    """
    Query knowledge graph for relevant facts.
    Uses SEMANTIC matching + INTENT FILTERING for better relevance.
    
    Returns formatted string for RAG context.
    """
    if not load_kg_facts():
        return ""
    
    if not _kg_facts:
        return ""
    
    # Detect query intent
    query_intent = _detect_query_intent(query)
    preferred_relations = INTENT_TO_RELATIONS.get(query_intent, []) if query_intent else []
    
    # Find relevant topics using semantic similarity (stricter threshold)
    relevant_topics = find_relevant_topics(query, top_k=3, threshold=0.50)
    
    # Skip KG if no high-confidence matches found
    if not relevant_topics:
        return ""
    
    # Additional check: if best score is below 0.55, skip KG entirely
    top_score = relevant_topics[0][1] if relevant_topics else 0
    if top_score < 0.55:
        return ""  # Don't inject low-confidence facts
    
    # Collect facts from relevant topics
    all_facts = []
    for topic, score in relevant_topics:
        facts = _kg_facts.get(topic, [])
        for fact in facts:
            # Skip invalid facts
            if not _is_valid_fact(fact):
                continue
            
            # Compute relevance score
            relation = fact.get("relation", "")
            relevance = score
            
            # Boost facts that match query intent
            if preferred_relations and relation in preferred_relations:
                relevance *= 1.5
            # Slight penalty for facts that don't match intent
            elif preferred_relations:
                relevance *= 0.5
            
            all_facts.append({
                "topic": topic,
                "topic_score": score,
                "relevance": relevance,
                **fact
            })
    
    # Sort by relevance
    all_facts.sort(key=lambda x: x.get("relevance", 0), reverse=True)
    
    # Deduplicate similar facts (same relation + similar value)
    seen = set()
    unique_facts = []
    for fact in all_facts:
        key = (fact.get("relation", ""), fact.get("value", "")[:20])
        if key not in seen:
            seen.add(key)
            unique_facts.append(fact)
    
    # Limit facts
    unique_facts = unique_facts[:max_facts]
    
    # Detect if query is about specific student type
    query_lower = query.lower()
    filter_context = None
    if any(word in query_lower for word in ["lisansüstü", "graduate", "master", "phd", "doktora"]):
        filter_context = "lisansustu"
    elif any(word in query_lower for word in ["lisans öğrenci", "undergrad", "bachelor"]):
        filter_context = "lisans"
    
    # Filter by context type if specified
    if filter_context:
        filtered = [f for f in unique_facts if f.get("context_type") in [filter_context, "general"]]
        if filtered:
            unique_facts = filtered
    
    # Format facts for RAG prompt
    return _format_facts_for_rag(unique_facts, lang)


def _format_facts_for_rag(facts: List[Dict], lang: str = "tr") -> str:
    """Format extracted facts for RAG context."""
    if not facts:
        return ""
    
    # Group by topic for cleaner output
    by_topic = defaultdict(list)
    for f in facts:
        by_topic[f["topic"]].append(f)
    
    # Format header based on language
    if lang == "tr":
        header = "📊 DOĞRULANMIŞ BİLGİLER (Bilgi Grafiğinden):"
    else:
        header = "📊 VERIFIED FACTS (From Knowledge Graph):"
    
    lines = [header, ""]
    
    for topic, topic_facts in by_topic.items():
        # Format topic name for display
        display_topic = topic.replace("_", " ").title()
        lines.append(f"[{display_topic}]")
        
        for fact in topic_facts:
            relation = fact.get("relation", "")
            value = fact.get("value", "")
            context = fact.get("context_type", "general")
            
            # Format relation for display
            relation_display = _format_relation(relation, lang)
            
            # Build fact line
            if context and context != "general":
                fact_line = f"  • {relation_display}: {value} ({context})"
            else:
                fact_line = f"  • {relation_display}: {value}"
            
            lines.append(fact_line)
        
        lines.append("")
    
    return "\n".join(lines)


def _format_relation(relation: str, lang: str = "tr") -> str:
    """Format relation type for display."""
    # Generic relation formatting - works for any domain
    relation_map = {
        "minimum_value": "Minimum" if lang == "en" else "Minimum Değer",
        "duration": "Duration" if lang == "en" else "Süre",
        "quantity_limit": "Limit" if lang == "en" else "Limit",
        "credit_requirement": "Credits" if lang == "en" else "Kredi Gereksinimi",
        "penalty_type": "Penalty" if lang == "en" else "Ceza Türü",
        "penalty_duration": "Penalty Duration" if lang == "en" else "Ceza Süresi",
        "deadline": "Deadline" if lang == "en" else "Son Tarih",
        "requirement": "Requirement" if lang == "en" else "Gereksinim",
        "process_step": "Step" if lang == "en" else "Adım",
    }
    
    return relation_map.get(relation, relation.replace("_", " ").title())


# =================== ADDITIONAL UTILITIES ===================

def get_all_topics() -> List[str]:
    """Get all available topics in the knowledge graph."""
    if not load_kg_facts():
        return []
    return list(_kg_facts.keys())


def get_facts_for_topic(topic: str) -> List[Dict]:
    """Get all facts for a specific topic."""
    if not load_kg_facts():
        return []
    return _kg_facts.get(topic, [])


def get_topic_stats() -> Dict:
    """Get statistics about the knowledge graph."""
    if not load_kg_facts():
        return {}
    
    total_facts = sum(len(facts) for facts in _kg_facts.values())
    return {
        "total_topics": len(_kg_facts),
        "total_facts": total_facts,
        "total_triples": len(_kg_triples),  # NEW
        "topics_with_embeddings": len(_topic_embeddings),
        "entities_with_embeddings": len(_entity_embeddings),  # NEW
        "average_facts_per_topic": total_facts / max(len(_kg_facts), 1),
        "top_topics": sorted(
            [(t, len(f)) for t, f in _kg_facts.items()],
            key=lambda x: x[1],
            reverse=True
        )[:10]
    }


# =================== TRIPLE QUERY (NEW) ===================

def query_triples(query: str, max_triples: int = 10) -> List[Dict]:
    """
    Query triples (Entity -> Relation -> Entity) using semantic similarity.
    Returns matching triples for the query.
    """
    if not load_kg_facts():
        return []
    
    if not _kg_triples:
        return []
    
    query_lower = query.lower()
    
    # Method 1: Direct keyword matching in triples
    matching_triples = []
    for triple in _kg_triples:
        head = triple.get("head", "").lower()
        tail = triple.get("tail", "").lower()
        relation = triple.get("relation", "").lower()
        
        # Score based on query word overlap
        query_words = set(re.findall(r'\w+', query_lower))
        triple_words = set(re.findall(r'\w+', f"{head} {tail} {relation}"))
        overlap = len(query_words & triple_words)
        
        if overlap > 0:
            matching_triples.append({
                **triple,
                "relevance": overlap
            })
    
    # Sort by relevance
    matching_triples.sort(key=lambda x: x.get("relevance", 0), reverse=True)
    
    return matching_triples[:max_triples]


def _format_triples_for_rag(triples: List[Dict], lang: str = "tr") -> str:
    """Format triples for RAG context."""
    if not triples:
        return ""
    
    if lang == "tr":
        header = "🔗 İLİŞKİLER (Bilgi Grafiğinden):"
    else:
        header = "🔗 RELATIONS (From Knowledge Graph):"
    
    lines = [header, ""]
    for t in triples:
        head = t.get("head", "")
        relation = t.get("relation", "").replace("_", " ")
        tail = t.get("tail", "")
        lines.append(f"  • {head} → [{relation}] → {tail}")
    
    return "\n".join(lines)


def query_hybrid(query: str, lang: str = "tr", max_facts: int = 5, max_triples: int = 3) -> str:
    """
    HYBRID query: combines Topic->Fact AND Entity->Relation->Entity results.
    Returns formatted string for RAG context. Returns empty string if no high-confidence facts found.
    """
    # Get facts (Topic -> Fact)
    facts_str = query_facts(query, lang, max_facts)
    
    # Get triples (Entity -> Relation -> Entity)
    triples = query_triples(query, max_triples)
    triples_str = _format_triples_for_rag(triples, lang)
    
    # Combine
    result = []
    if facts_str:
        result.append(facts_str)
    if triples_str:
        result.append(triples_str)
    
    return "\n".join(result)


# =================== TEST ===================

if __name__ == "__main__":
    print("Testing GENERIC KG Service...")
    print("=" * 50)
    
    if load_kg_facts():
        stats = get_topic_stats()
        print(f"Topics: {stats['total_topics']}")
        print(f"Facts: {stats['total_facts']}")
        print(f"Embeddings: {stats['topics_with_embeddings']}")
        print(f"\nTop topics:")
        for topic, count in stats.get("top_topics", [])[:5]:
            print(f"  {topic}: {count} facts")
        
        # Test queries
        test_queries = [
            "Erasmus staj için minimum GNO nedir?",
            "Kütüphaneden kaç kitap alabilirim?",
            "Disiplin cezaları nelerdir?",
        ]
        
        print(f"\n{'='*50}")
        print("Testing semantic search...")
        
        for q in test_queries:
            print(f"\n❓ Query: {q}")
            topics = find_relevant_topics(q, top_k=3)
            print(f"   Relevant topics:")
            for topic, score in topics:
                print(f"     [{score:.3f}] {topic}")
            
            facts = query_facts(q)
            if facts:
                print(f"   Facts found: {len(facts.split(chr(10)))} lines")
            else:
                print("   No facts found")
    else:
        print("❌ Failed to load KG")
