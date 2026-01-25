"""
build_hybrid_kg.py - Hybrid Knowledge Graph Builder with Full LLM Extraction

Builds a hybrid KG combining:
1. Topic -> Fact (structured facts)
2. Entity -> Relation -> Entity (knowledge triples)

Uses LLM for extraction - slower but more accurate.
Processes only selected documents (from selected_docs.json).

Usage:
    python build_hybrid_kg.py
"""

import os
import re
import json
import pickle
import numpy as np
from typing import List, Dict, Optional, Set, Tuple
from collections import defaultdict
from dataclasses import dataclass, asdict
import time

import pandas as pd
import requests
from dotenv import load_dotenv

load_dotenv()

# =================== CONFIG ===================

OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
CHAT_MODEL = os.getenv("CHAT_MODEL", "llama3.1:latest")
EMBED_MODEL = os.getenv("EMBED_MODEL", "bge-m3")

PREPROCESSING_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(PREPROCESSING_DIR)
CHECKPOINT_DIR = os.path.join(ROOT_DIR, "creating_database", "checkpoints_v2")
CHUNK_PARQUET = os.path.join(CHECKPOINT_DIR, "chunks_v2.parquet")
SELECTED_DOCS_PATH = os.path.join(PREPROCESSING_DIR, "selected_docs.json")

# Output paths
KG_OUTPUT_DIR = os.path.join(PREPROCESSING_DIR, "knowledge_graph")
KG_FACTS_PATH = os.path.join(KG_OUTPUT_DIR, "kg_facts.json")
KG_TRIPLES_PATH = os.path.join(KG_OUTPUT_DIR, "kg_triples.json")
KG_INDEX_PATH = os.path.join(KG_OUTPUT_DIR, "kg_index.pkl")

os.makedirs(KG_OUTPUT_DIR, exist_ok=True)


# =================== DATA CLASSES ===================

@dataclass
class ExtractedFact:
    """Topic -> Fact structure."""
    topic: str
    relation: str
    value: str
    context_type: str  # lisans, lisansustu, general
    source_chunk_id: str
    source_title: str
    confidence: float = 1.0


@dataclass
class Triple:
    """Entity -> Relation -> Entity structure."""
    head: str
    head_type: str  # program, requirement, duration, etc.
    relation: str
    tail: str
    tail_type: str
    source_chunk_id: str
    source_title: str
    confidence: float = 1.0


# =================== LLM EXTRACTION ===================

class LLMHybridExtractor:
    """Uses LLM to extract both Facts and Triples."""
    
    def __init__(self):
        self.url = f"{OLLAMA_HOST}/api/chat"
        self.model = CHAT_MODEL
        
    def extract(self, chunk_text: str, title: str, chunk_id: str) -> Tuple[List[ExtractedFact], List[Triple]]:
        """Extract both facts and triples from chunk using LLM."""
        
        prompt = f"""You are a Knowledge Graph extraction expert. Extract BOTH structured facts AND entity-relation-entity triples from the given text.

TEXT:
Title: {title}
Content: {chunk_text[:2000]}

---

Return JSON with two sections:

1. **facts**: Specific values/requirements (numbers, durations, limits)
   - topic: main topic (e.g., "erasmus_internship", "library_borrowing")
   - relation: type of fact (minimum_gno, duration, limit, penalty)
   - value: the specific value
   - context: who it applies to (lisans, lisansustu, general)

2. **triples**: Entity relationships
   - head: source entity
   - head_type: entity type (program, requirement, process, penalty)
   - relation: relationship (requires, has_duration, applies_to, results_in)
   - tail: target entity
   - tail_type: entity type

JSON OUTPUT SCHEMA:
{{
  "facts": [
    {{"topic": "...", "relation": "...", "value": "...", "context": "general/lisans/lisansustu"}}
  ],
  "triples": [
    {{"head": "...", "head_type": "...", "relation": "...", "tail": "...", "tail_type": "..."}}
  ]
}}

RULES:
1. Extract ONLY explicit information from the text
2. Do NOT make up values - use exact text
3. Keep entity names concise but descriptive
4. Return empty lists if nothing extractable

JSON:"""

        max_retries = 3
        for attempt in range(max_retries):
            try:
                resp = requests.post(
                    self.url,
                    json={
                        "model": self.model,
                        "messages": [{"role": "user", "content": prompt}],
                        "stream": False,
                        "format": "json",
                        "options": {"temperature": 0.0}
                    },
                    timeout=180
                )
                
                data = resp.json()
                content = data.get("message", {}).get("content", "{}")
                
                # Parse JSON
                try:
                    parsed = json.loads(content)
                except json.JSONDecodeError:
                    # Try to extract JSON
                    m = re.search(r"\{.*\}", content, flags=re.DOTALL)
                    if m:
                        parsed = json.loads(m.group(0))
                    else:
                        return [], []
                
                # Extract facts
                facts = []
                for f in parsed.get("facts", []):
                    if isinstance(f, dict) and f.get("value"):
                        facts.append(ExtractedFact(
                            topic=f.get("topic", "general"),
                            relation=f.get("relation", "general"),
                            value=str(f.get("value", "")),
                            context_type=f.get("context", "general"),
                            source_chunk_id=chunk_id,
                            source_title=title,
                            confidence=0.9
                        ))
                
                # Extract triples
                triples = []
                for t in parsed.get("triples", []):
                    if isinstance(t, dict) and t.get("head") and t.get("tail"):
                        triples.append(Triple(
                            head=str(t.get("head", "")),
                            head_type=t.get("head_type", "entity"),
                            relation=t.get("relation", "related_to"),
                            tail=str(t.get("tail", "")),
                            tail_type=t.get("tail_type", "entity"),
                            source_chunk_id=chunk_id,
                            source_title=title,
                            confidence=0.9
                        ))
                
                return facts, triples
                
            except requests.exceptions.Timeout:
                if attempt < max_retries - 1:
                    print(f"[LLM timeout, retry {attempt+1}/{max_retries}]")
                    time.sleep(30)
                else:
                    print(f"[LLM extraction failed: timeout]")
                    return [], []
            except Exception as e:
                print(f"[LLM extraction error] {e}")
                return [], []
        
        return [], []


# =================== EMBEDDING HELPER ===================

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
        vec /= (np.linalg.norm(vec) + 1e-12)
        return vec
    except Exception as e:
        print(f"[embed error] {e}")
        return np.zeros(1024, dtype=np.float32)


# =================== HYBRID KG BUILDER ===================

class HybridKGBuilder:
    """Builds hybrid KG with both Facts and Triples."""
    
    def __init__(self):
        self.extractor = LLMHybridExtractor()
        
        # Storage
        self.facts_by_topic: Dict[str, List[Dict]] = defaultdict(list)
        self.all_triples: List[Dict] = []
        self.topic_embeddings: Dict[str, List[float]] = {}
        self.entity_embeddings: Dict[str, List[float]] = {}
    
    def build_from_selected_chunks(self, chunks_df: pd.DataFrame, selected_paths: List[str]):
        """Build KG from selected documents only."""
        
        # Filter to selected documents
        filtered_df = chunks_df[chunks_df['source_path'].isin(selected_paths)]
        total = len(filtered_df)
        print(f"[KG] Processing {total} chunks from {len(selected_paths)} documents...")
        
        unique_topics: Set[str] = set()
        unique_entities: Set[str] = set()
        
        for idx, (_, row) in enumerate(filtered_df.iterrows()):
            if idx % 20 == 0:
                print(f"[KG] Progress: {idx}/{total} ({100*idx/total:.1f}%)")
            
            chunk_id = str(row.get("chunk_id", f"chunk_{idx}"))
            content = str(row.get("content", ""))
            title = str(row.get("title", ""))
            
            if len(content) < 50:
                continue
            
            # LLM extraction
            facts, triples = self.extractor.extract(content, title, chunk_id)
            
            # Store facts
            for fact in facts:
                unique_topics.add(fact.topic)
                self.facts_by_topic[fact.topic].append({
                    "relation": fact.relation,
                    "value": fact.value,
                    "context_type": fact.context_type,
                    "source_chunk_id": fact.source_chunk_id,
                    "source_title": fact.source_title,
                    "confidence": fact.confidence
                })
            
            # Store triples
            for triple in triples:
                unique_entities.add(triple.head)
                unique_entities.add(triple.tail)
                self.all_triples.append(asdict(triple))
        
        # Deduplicate
        self._deduplicate()
        
        # Build embeddings for semantic search
        print(f"[KG] Building embeddings for {len(unique_topics)} topics...")
        for topic in unique_topics:
            self.topic_embeddings[topic] = embed_text(topic.replace("_", " ")).tolist()
        
        print(f"[KG] Building embeddings for {min(len(unique_entities), 100)} entities...")
        for entity in list(unique_entities)[:100]:  # Limit to 100 for speed
            self.entity_embeddings[entity] = embed_text(entity).tolist()
        
        print(f"[KG] Done: {len(self.facts_by_topic)} topics, {len(self.all_triples)} triples")
    
    def _deduplicate(self):
        """Remove duplicates."""
        # Deduplicate facts
        for topic in self.facts_by_topic:
            facts = self.facts_by_topic[topic]
            seen = set()
            unique = []
            for f in facts:
                key = (f["relation"], f["value"][:50], f["context_type"])
                if key not in seen:
                    seen.add(key)
                    unique.append(f)
            self.facts_by_topic[topic] = unique
        
        # Deduplicate triples
        seen = set()
        unique_triples = []
        for t in self.all_triples:
            key = (t["head"], t["relation"], t["tail"])
            if key not in seen:
                seen.add(key)
                unique_triples.append(t)
        self.all_triples = unique_triples
    
    def save(self):
        """Save hybrid KG to files."""
        
        # Save facts
        with open(KG_FACTS_PATH, "w", encoding="utf-8") as f:
            json.dump(dict(self.facts_by_topic), f, ensure_ascii=False, indent=2)
        print(f"[OK] Facts saved to {KG_FACTS_PATH}")
        
        # Save triples
        triples_data = {
            "triples": self.all_triples,
            "entity_embeddings": self.entity_embeddings
        }
        with open(KG_TRIPLES_PATH, "w", encoding="utf-8") as f:
            json.dump(triples_data, f, ensure_ascii=False, indent=2)
        print(f"[OK] Triples saved to {KG_TRIPLES_PATH}")
        
        # Save index
        index_data = {
            "topic_embeddings": self.topic_embeddings,
            "entity_embeddings": self.entity_embeddings,
            "topic_list": list(self.facts_by_topic.keys()),
            "total_facts": sum(len(f) for f in self.facts_by_topic.values()),
            "total_triples": len(self.all_triples)
        }
        with open(KG_INDEX_PATH, "wb") as f:
            pickle.dump(index_data, f)
        print(f"[OK] Index saved to {KG_INDEX_PATH}")


# =================== MAIN ===================

def main():
    print("=" * 60)
    print("[KG] HYBRID Knowledge Graph Builder (LLM)")
    print("    Topic->Fact + Entity->Relation->Entity")
    print("=" * 60)
    
    # Load selected documents
    print(f"\n[1/5] Loading selected documents...")
    if not os.path.exists(SELECTED_DOCS_PATH):
        print(f"[ERROR] Run select_focused_docs.py first!")
        return
    
    with open(SELECTED_DOCS_PATH, "r", encoding="utf-8") as f:
        selected_data = json.load(f)
    
    # Get all selected paths
    selected_paths = []
    for category, docs in selected_data.get("documents", {}).items():
        for doc in docs:
            selected_paths.append(doc["source_path"])
    
    print(f"[OK] {len(selected_paths)} documents selected")
    
    # Load chunks
    print(f"\n[2/5] Loading chunks...")
    if not os.path.exists(CHUNK_PARQUET):
        print(f"[ERROR] Chunk file not found: {CHUNK_PARQUET}")
        return
    
    chunks_df = pd.read_parquet(CHUNK_PARQUET)
    print(f"[OK] Loaded {len(chunks_df)} total chunks")
    
    # Build KG
    print(f"\n[3/5] Building hybrid knowledge graph (LLM extraction)...")
    print(f"      This will take ~1-2 hours for {selected_data.get('total_chunks', '?')} chunks")
    
    builder = HybridKGBuilder()
    builder.build_from_selected_chunks(chunks_df, selected_paths)
    
    # Save
    print(f"\n[4/5] Saving knowledge graph...")
    builder.save()
    
    # Summary
    print(f"\n[5/5] Summary:")
    print(f"  - Topics (facts): {len(builder.facts_by_topic)}")
    print(f"  - Total facts: {sum(len(f) for f in builder.facts_by_topic.values())}")
    print(f"  - Total triples: {len(builder.all_triples)}")
    print(f"  - Topic embeddings: {len(builder.topic_embeddings)}")
    print(f"  - Entity embeddings: {len(builder.entity_embeddings)}")
    
    print(f"\n[DONE] Hybrid KG built successfully!")
    print(f"       Output: {KG_OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
