"""
rebuild_kg_index.py - Rebuild the KG embedding index from validated files.

This script rebuilds kg_index.pkl to match the LLM-validated facts/triples.
Run this after running validate_kg_facts.py.

Usage:
    python rebuild_kg_index.py
"""

import os
import sys
import json
import pickle
import time
import numpy as np
from typing import Dict, List
from dotenv import load_dotenv

# Add parent directory to path so we can import services
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

load_dotenv(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env"))

# =================== CONFIG ===================

from services.config.settings import RAGConfig
from services.core.embedding_service import EmbeddingService

_config = RAGConfig.from_env()
_embed_service = EmbeddingService(_config.ollama)

PREPROCESSING_DIR = os.path.dirname(os.path.abspath(__file__))
KG_DIR = os.path.join(PREPROCESSING_DIR, "knowledge_graph")
KG_LLM_DIR = os.path.join(KG_DIR, "llm_validated")
os.makedirs(KG_LLM_DIR, exist_ok=True)

# Use LLM-validated files
KG_FACTS_PATH = os.path.join(KG_LLM_DIR, "kg_facts_llm_validated.json")
KG_TRIPLES_PATH = os.path.join(KG_LLM_DIR, "kg_triples_llm_validated.json")
KG_INDEX_PATH = os.path.join(KG_LLM_DIR, "kg_index_llm_validated.pkl")


# =================== EMBEDDING ===================

def embed_text(text: str, max_retries: int = 3) -> np.ndarray:
    """Embed text using the configured provider (OpenRouter/Ollama)."""
    for attempt in range(max_retries):
        try:
            vec = _embed_service.embed(text, is_query=True)
            norm = np.linalg.norm(vec)
            if norm > 0:
                vec /= norm
            return vec
        except Exception as e:
            if attempt < max_retries - 1:
                wait = 2 ** (attempt + 1)  # Exponential backoff: 2s, 4s
                print(f"  [Embed retry {attempt+1}/{max_retries}] {e} — waiting {wait}s")
                time.sleep(wait)
            else:
                print(f"  [Embed error] {e}")
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
    # Load validated triples just for verification (optional)
    if os.path.exists(KG_TRIPLES_PATH):
        print(f"[OK] Triples file found, but skipping entity embeddings (unused in RAG pipeline).")
    
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
    
    # Save index (entity_embeddings is empty because it's not used)
    index_data = {
        "topic_embeddings": topic_embeddings,
        "entity_embeddings": {},
        "topic_list": topics
    }
    
    with open(KG_INDEX_PATH, "wb") as f:
        pickle.dump(index_data, f)
    
    print(f"\n[DONE] Index saved to: {KG_INDEX_PATH}")
    print(f"  - Topics: {len(topic_embeddings)}")
    print(f"  - Entities: 0 (Disabled)")
    
    return True


if __name__ == "__main__":
    rebuild_index()
