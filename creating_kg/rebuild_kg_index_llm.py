"""
rebuild_kg_index.py - Rebuild the KG embedding index from validated files.

This script rebuilds kg_index.pkl to match the LLM-validated facts/triples.
Run this after running validate_kg_facts.py.

Usage:
    python rebuild_kg_index.py
"""

import os
import json
import pickle
import requests
import numpy as np
from typing import Dict, List
from dotenv import load_dotenv

load_dotenv()

# =================== CONFIG ===================

OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
EMBED_MODEL = os.getenv("EMBED_MODEL", "nomic-embed-text:latest")

PREPROCESSING_DIR = os.path.dirname(os.path.abspath(__file__))
KG_DIR = os.path.join(PREPROCESSING_DIR, "knowledge_graph")
KG_LLM_DIR = os.path.join(KG_DIR, "llm_validated")
os.makedirs(KG_LLM_DIR, exist_ok=True)

# Use LLM-validated files
KG_FACTS_PATH = os.path.join(KG_LLM_DIR, "kg_facts_llm_validated.json")
KG_TRIPLES_PATH = os.path.join(KG_LLM_DIR, "kg_triples_llm_validated.json")
KG_INDEX_PATH = os.path.join(KG_LLM_DIR, "kg_index_llm_validated.pkl")


# =================== EMBEDDING ===================

def embed_text(text: str) -> np.ndarray:
    """Embed text using Ollama."""
    url = f"{OLLAMA_HOST}/api/embeddings"
    try:
        r = requests.post(
            url,
            json={"model": EMBED_MODEL, "prompt": text},
            timeout=60
        )
        r.raise_for_status()
        vec = np.array(r.json()["embedding"], dtype=np.float32)
        vec /= (np.linalg.norm(vec) + 1e-12)  # Normalize
        return vec
    except Exception as e:
        print(f"[Embed error] {e}")
        return None


# =================== MAIN ===================

def rebuild_index():
    """Rebuild the KG index from validated files."""
    
    print("=" * 60)
    print("[REBUILD KG INDEX] From LLM-validated files")
    print("=" * 60)
    
    # Load validated facts
    if not os.path.exists(KG_FACTS_PATH):
        print(f"[ERROR] Facts file not found: {KG_FACTS_PATH}")
        return False
    
    with open(KG_FACTS_PATH, "r", encoding="utf-8") as f:
        facts = json.load(f)
    
    topics = list(facts.keys())
    print(f"[OK] Loaded {len(topics)} topics from validated facts")
    
    # Load validated triples for entity embeddings
    entities = set()
    if os.path.exists(KG_TRIPLES_PATH):
        with open(KG_TRIPLES_PATH, "r", encoding="utf-8") as f:
            triples_data = json.load(f)
        for triple in triples_data.get("triples", []):
            entities.add(triple.get("head", ""))
            entities.add(triple.get("tail", ""))
        entities.discard("")
        print(f"[OK] Found {len(entities)} unique entities from triples")
    
    # Build topic embeddings
    print(f"\n[Building topic embeddings...]")
    topic_embeddings: Dict[str, List[float]] = {}
    
    for i, topic in enumerate(topics):
        if i % 100 == 0:
            print(f"  Progress: {i}/{len(topics)} topics...")
        
        emb = embed_text(topic)
        if emb is not None:
            topic_embeddings[topic] = emb.tolist()
    
    print(f"[OK] Created embeddings for {len(topic_embeddings)} topics")
    
    # Build entity embeddings (sample - top 500 most common)
    print(f"\n[Building entity embeddings (top 500)...]")
    entity_embeddings: Dict[str, List[float]] = {}
    
    entity_list = list(entities)[:500]  # Limit to 500 for speed
    for i, entity in enumerate(entity_list):
        if i % 100 == 0:
            print(f"  Progress: {i}/{len(entity_list)} entities...")
        
        emb = embed_text(entity)
        if emb is not None:
            entity_embeddings[entity] = emb.tolist()
    
    print(f"[OK] Created embeddings for {len(entity_embeddings)} entities")
    
    # Save index
    index_data = {
        "topic_embeddings": topic_embeddings,
        "entity_embeddings": entity_embeddings,
        "topic_list": topics
    }
    
    with open(KG_INDEX_PATH, "wb") as f:
        pickle.dump(index_data, f)
    
    print(f"\n[DONE] Index saved to: {KG_INDEX_PATH}")
    print(f"  - Topics: {len(topic_embeddings)}")
    print(f"  - Entities: {len(entity_embeddings)}")
    
    return True


if __name__ == "__main__":
    rebuild_index()
