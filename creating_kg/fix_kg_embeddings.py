import os
import sys
import json
import pickle
import numpy as np

# Add parent directory to Python path to find 'services' module
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from services.config.settings import RAGConfig
from services.core.embedding_service import EmbeddingService
from services.kg_service import KG_FACTS_PATH, KG_INDEX_PATH, KG_TRIPLES_PATH

def main():
    print("=" * 60)
    print(f"[FIX KG EMBEDDINGS] Generating proper embeddings via API")
    print("=" * 60)
    
    # 1. Load Topics
    if not os.path.exists(KG_FACTS_PATH):
        print(f"[ERROR] Facts file not found: {KG_FACTS_PATH}")
        return
    with open(KG_FACTS_PATH, 'r', encoding='utf-8') as f:
        topics = list(json.load(f).keys())
    print(f'[OK] Found {len(topics)} topics')

    # 2. Load Entities from Triples
    entities = set()
    triples_data = {}
    if os.path.exists(KG_TRIPLES_PATH):
        with open(KG_TRIPLES_PATH, 'r', encoding='utf-8') as f:
            triples_data = json.load(f)
        for t in triples_data.get("triples", []):
            entities.add(t.get("head", ""))
            entities.add(t.get("tail", ""))
        entities.discard("")
        print(f'[OK] Found {len(entities)} unique entities')

    # 3. Setup Embedding Service
    config = RAGConfig.from_env()
    embed_service = EmbeddingService(config.ollama)

    # 4. Embed Topics 
    print('\n[1/2] Batch embedding topics...')
    topic_vectors = embed_service.embed_batch(topics, is_query=False)
    topic_embs = {}
    for topic, v in zip(topics, topic_vectors):
        vec = np.array(v, dtype=np.float32)
        norm = np.linalg.norm(vec)
        if norm > 0: vec /= norm
        topic_embs[topic] = vec.tolist()

    # 5. Embed All Entities
    entity_list = list(entities)
    print(f'\n[2/2] Batch embedding {len(entity_list)} entities...')
    entity_vectors = embed_service.embed_batch(entity_list, is_query=False)
    entity_embs = {}
    for entity, v in zip(entity_list, entity_vectors):
        vec = np.array(v, dtype=np.float32)
        norm = np.linalg.norm(vec)
        if norm > 0: vec /= norm
        entity_embs[entity] = vec.tolist()

    # 6. Save Index (pkl)
    index_data = {
        'topic_embeddings': topic_embs, 
        'entity_embeddings': entity_embs,
        'topic_list': topics
    }
    with open(KG_INDEX_PATH, 'wb') as f:
        pickle.dump(index_data, f)
    
    # 7. Save Triples JSON (update embeddings in json)
    if os.path.exists(KG_TRIPLES_PATH) and triples_data:
        triples_data["entity_embeddings"] = entity_embs
        with open(KG_TRIPLES_PATH, 'w', encoding='utf-8') as f:
            json.dump(triples_data, f, ensure_ascii=False, indent=2)

    dim = len(topic_embs[topics[0]])
    print(f'\n✅ Successfully saved KG embeddings! New vector dimension = {dim}')

if __name__ == "__main__":
    main()
