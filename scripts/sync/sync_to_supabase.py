"""
sync_to_supabase.py - Comprehensive data sync from local files to Supabase

This script synchronizes:
1. Documents (from unique source_paths in chunks parquet)
2. Chunks (from chunks_v2.parquet)
3. Chunk Embeddings (from ChromaDB)
4. KG Nodes and Edges (from KG JSON files)

Usage:
    python scripts/sync/sync_to_supabase.py --all          # Sync everything
    python scripts/sync/sync_to_supabase.py --documents    # Sync documents only
    python scripts/sync/sync_to_supabase.py --chunks       # Sync chunks only
    python scripts/sync/sync_to_supabase.py --embeddings   # Sync embeddings only
    python scripts/sync/sync_to_supabase.py --kg           # Sync KG only
"""

import os
import sys
import json
import hashlib
import argparse
from typing import Dict, List, Optional, Tuple
from datetime import datetime

import pandas as pd
import numpy as np
from sqlalchemy import text
from sqlalchemy.dialects.postgresql import insert
from tqdm import tqdm

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from dotenv import load_dotenv
load_dotenv()

from database import engine, SessionLocal
import models


# =====================================================================
# Configuration
# =====================================================================

CHECKPOINT_DIR = os.getenv("CHECKPOINT_DIR_V2", "creating_database/checkpoints_v2")
CHROMA_DIR = os.getenv("CHROMA_DIR_V2", "creating_database/chroma_db_v2")
CHROMA_COLLECTION = os.getenv("CHROMA_COLLECTION_NAME_V2", "mysu_v2_bge_m3")

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CHUNKS_PARQUET = os.path.join(ROOT_DIR, CHECKPOINT_DIR, "chunks_v2.parquet")
KG_DIR = os.path.join(ROOT_DIR, "creating_kg", "knowledge_graph", "llm_validated")
KG_FACTS_PATH = os.path.join(KG_DIR, "kg_facts_llm_validated.json")
KG_TRIPLES_PATH = os.path.join(KG_DIR, "kg_triples_llm_validated.json")

BATCH_SIZE = 500


# =====================================================================
# Helper Functions
# =====================================================================

def compute_hash(content: str) -> str:
    """Compute SHA256 hash of content."""
    return hashlib.sha256(content.encode('utf-8')).hexdigest()[:32]


def clean_text(text: str) -> str:
    """
    Remove NULL characters and other problematic bytes from text.
    PostgreSQL cannot store NUL (0x00) characters in text fields.
    """
    if not text:
        return ""
    # Remove NULL bytes
    text = text.replace('\x00', '')
    # Remove other control characters except newline, tab, carriage return
    text = ''.join(char for char in text if ord(char) >= 32 or char in '\n\t\r')
    return text


def get_source_type(source_path: str) -> str:
    """Determine source type from file extension."""
    ext = source_path.lower().split('.')[-1] if '.' in source_path else ''
    type_map = {
        'pdf': 'pdf',
        'html': 'html',
        'htm': 'html',
        'md': 'md',
        'txt': 'other',
    }
    return type_map.get(ext, 'html')


def extract_department(source_path: str, tags: str) -> Optional[str]:
    """Extract department from path or tags."""
    # Try to extract from common patterns
    parts = source_path.lower().split('/')

    # Common department indicators
    dept_keywords = ['ssbf', 'mdbf', 'do', 'ybf', 'gs', 'ik', 'bt', 'bm']
    for part in parts:
        for kw in dept_keywords:
            if kw in part:
                return kw.upper()

    return None


# =====================================================================
# 1. Sync Documents
# =====================================================================

def sync_documents(db_session, chunks_df: pd.DataFrame) -> Dict[str, int]:
    """
    Sync unique documents from chunks dataframe to documents table.

    Returns:
        Dict mapping source_path -> document_id
    """
    print("\n" + "="*60)
    print("📄 SYNCING DOCUMENTS")
    print("="*60)

    # Get unique documents
    unique_docs = chunks_df.groupby('source_path').first().reset_index()
    print(f"Found {len(unique_docs)} unique documents")

    # Map source_path -> document_id
    doc_id_map = {}

    # Check existing documents
    existing = db_session.execute(
        text("SELECT document_id, source_uri FROM documents")
    ).fetchall()
    existing_map = {row[1]: row[0] for row in existing}
    print(f"Existing documents in DB: {len(existing_map)}")

    new_docs = []
    for _, row in tqdm(unique_docs.iterrows(), total=len(unique_docs), desc="Processing documents"):
        source_path = row['source_path']

        if source_path in existing_map:
            doc_id_map[source_path] = existing_map[source_path]
            continue

        # Create new document record
        doc = models.Document(
            source_type=get_source_type(source_path),
            source_uri=source_path,
            title=clean_text(row.get('title', source_path.split('/')[-1])),
            lang=row.get('doc_lang', 'tr'),
            department=extract_department(source_path, row.get('tags', '')),
            hash=compute_hash(source_path),
        )
        new_docs.append(doc)

    # Batch insert new documents
    if new_docs:
        print(f"Inserting {len(new_docs)} new documents...")
        for i in range(0, len(new_docs), BATCH_SIZE):
            batch = new_docs[i:i+BATCH_SIZE]
            db_session.add_all(batch)
            db_session.commit()

        # Refresh to get IDs
        for doc in new_docs:
            db_session.refresh(doc)
            doc_id_map[doc.source_uri] = doc.document_id

    # Merge with existing
    doc_id_map.update({k: v for k, v in existing_map.items() if k not in doc_id_map})

    print(f"✅ Documents synced: {len(doc_id_map)} total")
    return doc_id_map


# =====================================================================
# 2. Sync Chunks
# =====================================================================

def sync_chunks(db_session, chunks_df: pd.DataFrame, doc_id_map: Dict[str, int]) -> Dict[str, int]:
    """
    Sync chunks from parquet to chunks table.

    Returns:
        Dict mapping chunk_id (hash) -> database chunk_id (int)
    """
    print("\n" + "="*60)
    print("📦 SYNCING CHUNKS")
    print("="*60)

    print(f"Total chunks to sync: {len(chunks_df)}")

    # Check existing chunks (by content hash or we can use a custom approach)
    # For simplicity, we'll check by document_id + ordinal
    existing = db_session.execute(
        text("SELECT chunk_id, document_id, ordinal FROM chunks")
    ).fetchall()
    existing_set = {(row[1], row[2]): row[0] for row in existing}
    print(f"Existing chunks in DB: {len(existing_set)}")

    chunk_id_map = {}
    new_chunks = []

    for _, row in tqdm(chunks_df.iterrows(), total=len(chunks_df), desc="Processing chunks"):
        source_path = row['source_path']
        doc_id = doc_id_map.get(source_path)

        if not doc_id:
            continue

        ordinal = row.get('chunk_index', 0)
        key = (doc_id, ordinal)

        if key in existing_set:
            chunk_id_map[row['chunk_id']] = existing_set[key]
            continue

        # Create new chunk
        content = clean_text(row.get('content', ''))
        section = clean_text(row.get('section_header', ''))

        chunk = models.Chunk(
            document_id=doc_id,
            ordinal=ordinal,
            section=section,
            page_num=None,  # Not available in our data
            content=content,
            content_html=None,  # Could store HTML if needed
            tokens=len(content.split()),  # Rough estimate
        )
        new_chunks.append((row['chunk_id'], chunk))

    # Batch insert
    if new_chunks:
        print(f"Inserting {len(new_chunks)} new chunks...")
        for i in range(0, len(new_chunks), BATCH_SIZE):
            batch = new_chunks[i:i+BATCH_SIZE]
            chunks_only = [c[1] for c in batch]
            db_session.add_all(chunks_only)
            db_session.commit()

            # Map original chunk_id to new DB chunk_id
            for orig_id, chunk in batch:
                db_session.refresh(chunk)
                chunk_id_map[orig_id] = chunk.chunk_id

    # Merge existing
    for key, db_id in existing_set.items():
        # Find original chunk_id for this doc_id + ordinal
        matches = chunks_df[
            (chunks_df['source_path'].map(doc_id_map) == key[0]) &
            (chunks_df['chunk_index'] == key[1])
        ]
        if len(matches) > 0:
            chunk_id_map[matches.iloc[0]['chunk_id']] = db_id

    print(f"✅ Chunks synced: {len(chunk_id_map)} mapped")
    return chunk_id_map


# =====================================================================
# 3. Sync Embeddings
# =====================================================================

def sync_embeddings(db_session, chunk_id_map: Dict[str, int]) -> int:
    """
    Sync embeddings from ChromaDB to chunk_embeddings table.

    Returns:
        Number of embeddings synced
    """
    print("\n" + "="*60)
    print("🧠 SYNCING EMBEDDINGS")
    print("="*60)

    import chromadb

    # Connect to ChromaDB
    chroma_path = os.path.join(ROOT_DIR, CHROMA_DIR)
    client = chromadb.PersistentClient(path=chroma_path)
    collection = client.get_collection(CHROMA_COLLECTION)

    print(f"ChromaDB collection: {collection.count()} embeddings")

    # Check existing embeddings
    existing = db_session.execute(
        text("SELECT chunk_id FROM chunk_embeddings")
    ).fetchall()
    existing_ids = {row[0] for row in existing}
    print(f"Existing embeddings in DB: {len(existing_ids)}")

    # Get all embeddings from ChromaDB in batches
    total = collection.count()
    synced = 0

    for offset in tqdm(range(0, total, BATCH_SIZE), desc="Syncing embeddings"):
        # Get batch from ChromaDB
        result = collection.get(
            limit=BATCH_SIZE,
            offset=offset,
            include=['embeddings']
        )

        if not result['ids']:
            break

        # Prepare batch for insertion
        values = []
        for i, chroma_id in enumerate(result['ids']):
            db_chunk_id = chunk_id_map.get(chroma_id)

            if not db_chunk_id or db_chunk_id in existing_ids:
                continue

            embedding = result['embeddings'][i]
            # Format as PostgreSQL array literal
            emb_str = '[' + ','.join(map(str, embedding)) + ']'
            values.append({
                'chunk_id': db_chunk_id,
                'embedding': emb_str,
                'model': 'bge-m3'
            })

        # Batch insert using raw SQL for vector type
        if values:
            batch_synced = 0
            for val in values:
                try:
                    # Use literal embedding in query to avoid SQLAlchemy parameter parsing issues
                    # Embedding is already a safe format: [0.1,0.2,...] containing only numbers
                    query = text(f"""
                        INSERT INTO chunk_embeddings (chunk_id, embedding, model)
                        VALUES (:chunk_id, '{val['embedding']}'::vector, :model)
                        ON CONFLICT (chunk_id) DO NOTHING
                    """)
                    db_session.execute(query, {'chunk_id': val['chunk_id'], 'model': val['model']})
                    batch_synced += 1
                    existing_ids.add(val['chunk_id'])  # Track inserted IDs
                except Exception as e:
                    # Rollback on error and continue with next batch
                    db_session.rollback()
                    # Skip remaining items in this batch
                    break

            if batch_synced > 0:
                try:
                    db_session.commit()
                    synced += batch_synced
                except Exception as e:
                    db_session.rollback()
                    print(f"Error committing batch: {e}")

    print(f"✅ Embeddings synced: {synced}")
    return synced


# =====================================================================
# 4. Sync Knowledge Graph
# =====================================================================

def sync_kg(db_session, chunk_id_map: Dict[str, int]) -> Tuple[int, int]:
    """
    Sync KG nodes and edges from JSON files.

    Returns:
        Tuple of (nodes_synced, edges_synced)
    """
    print("\n" + "="*60)
    print("🔗 SYNCING KNOWLEDGE GRAPH")
    print("="*60)

    # Load KG facts (topics -> nodes)
    with open(KG_FACTS_PATH, 'r', encoding='utf-8') as f:
        kg_facts = json.load(f)

    # Load KG triples (relations -> edges)
    with open(KG_TRIPLES_PATH, 'r', encoding='utf-8') as f:
        kg_triples_data = json.load(f)

    triples = kg_triples_data.get('triples', [])

    print(f"KG Facts: {len(kg_facts)} topics")
    print(f"KG Triples: {len(triples)} relations")

    # Check existing nodes
    existing_nodes = db_session.execute(
        text("SELECT node_id, name FROM kg_nodes")
    ).fetchall()
    node_name_map = {row[1]: row[0] for row in existing_nodes}
    print(f"Existing nodes in DB: {len(node_name_map)}")

    # Create nodes from topics
    nodes_synced = 0

    for topic, facts in tqdm(kg_facts.items(), desc="Syncing KG nodes"):
        clean_topic = clean_text(topic)
        if clean_topic in node_name_map:
            continue

        # Create node with facts stored in data JSONB
        # Facts are dicts with 'value' field - clean the value
        clean_facts = []
        for f in facts:
            if isinstance(f, dict):
                cleaned_fact = f.copy()
                if 'value' in cleaned_fact:
                    cleaned_fact['value'] = clean_text(str(cleaned_fact['value']))
                clean_facts.append(cleaned_fact)
            else:
                clean_facts.append(clean_text(str(f)))

        node = models.KGNode(
            label='topic',
            name=clean_topic,
            doc_id=None,  # Could link to document if needed
            data={
                'facts': clean_facts,
                'fact_count': len(clean_facts)
            }
        )
        db_session.add(node)
        nodes_synced += 1

        if nodes_synced % BATCH_SIZE == 0:
            db_session.commit()

    db_session.commit()

    # Refresh node map
    all_nodes = db_session.execute(
        text("SELECT node_id, name FROM kg_nodes")
    ).fetchall()
    node_name_map = {row[1]: row[0] for row in all_nodes}

    # Also create nodes for unique entities in triples
    entity_nodes = set()
    for triple in triples:
        entity_nodes.add((clean_text(triple['head']), clean_text(triple.get('head_type', 'entity'))))
        entity_nodes.add((clean_text(triple['tail']), clean_text(triple.get('tail_type', 'entity'))))

    for entity_name, entity_type in tqdm(entity_nodes, desc="Creating entity nodes"):
        clean_entity = clean_text(entity_name)
        if clean_entity in node_name_map:
            continue

        node = models.KGNode(
            label=clean_text(entity_type),
            name=clean_entity,
            data={}
        )
        db_session.add(node)
        nodes_synced += 1

    db_session.commit()

    # Refresh node map again
    all_nodes = db_session.execute(
        text("SELECT node_id, name FROM kg_nodes")
    ).fetchall()
    node_name_map = {row[1]: row[0] for row in all_nodes}

    # Check existing edges
    existing_edges = db_session.execute(
        text("SELECT src, dst, type FROM kg_edges")
    ).fetchall()
    existing_edge_set = {(row[0], row[1], row[2]) for row in existing_edges}
    print(f"Existing edges in DB: {len(existing_edge_set)}")

    # Create edges from triples
    edges_synced = 0

    for triple in tqdm(triples, desc="Syncing KG edges"):
        src_id = node_name_map.get(clean_text(triple['head']))
        dst_id = node_name_map.get(clean_text(triple['tail']))
        rel_type = clean_text(triple['relation'])

        if not src_id or not dst_id:
            continue

        if (src_id, dst_id, rel_type) in existing_edge_set:
            continue

        edge = models.KGEdge(
            src=src_id,
            dst=dst_id,
            type=rel_type,
            weight=triple.get('confidence', 1.0)
        )
        db_session.add(edge)
        edges_synced += 1

        if edges_synced % BATCH_SIZE == 0:
            db_session.commit()

    db_session.commit()

    print(f"✅ KG synced: {nodes_synced} nodes, {edges_synced} edges")
    return nodes_synced, edges_synced


# =====================================================================
# Main Execution
# =====================================================================

def main():
    parser = argparse.ArgumentParser(description="Sync data to Supabase")
    parser.add_argument('--all', action='store_true', help='Sync everything')
    parser.add_argument('--documents', action='store_true', help='Sync documents only')
    parser.add_argument('--chunks', action='store_true', help='Sync chunks only')
    parser.add_argument('--embeddings', action='store_true', help='Sync embeddings only')
    parser.add_argument('--kg', action='store_true', help='Sync KG only')
    parser.add_argument('--dry-run', action='store_true', help='Show what would be synced')

    args = parser.parse_args()

    # Default to --all if no specific option
    if not any([args.all, args.documents, args.chunks, args.embeddings, args.kg]):
        args.all = True

    print("="*60)
    print("🚀 SUPABASE DATA SYNC")
    print(f"   Started: {datetime.now().isoformat()}")
    print("="*60)

    # Load chunks data
    print(f"\nLoading chunks from: {CHUNKS_PARQUET}")
    chunks_df = pd.read_parquet(CHUNKS_PARQUET)
    print(f"Loaded {len(chunks_df)} chunks")

    if args.dry_run:
        print("\n⚠️  DRY RUN - No changes will be made")
        print(f"Documents to sync: {chunks_df['source_path'].nunique()}")
        print(f"Chunks to sync: {len(chunks_df)}")
        return

    # Create database session
    db = SessionLocal()

    try:
        doc_id_map = {}
        chunk_id_map = {}

        # 1. Sync Documents
        if args.all or args.documents:
            doc_id_map = sync_documents(db, chunks_df)
        else:
            # Load existing mapping
            existing = db.execute(
                text("SELECT document_id, source_uri FROM documents")
            ).fetchall()
            doc_id_map = {row[1]: row[0] for row in existing}

        # 2. Sync Chunks
        if args.all or args.chunks:
            chunk_id_map = sync_chunks(db, chunks_df, doc_id_map)
        else:
            # We need chunk mapping for embeddings
            # Build it from existing data
            existing_chunks = db.execute(
                text("""
                    SELECT c.chunk_id, d.source_uri, c.ordinal
                    FROM chunks c
                    JOIN documents d ON c.document_id = d.document_id
                """)
            ).fetchall()
            for row in existing_chunks:
                matches = chunks_df[
                    (chunks_df['source_path'] == row[1]) &
                    (chunks_df['chunk_index'] == row[2])
                ]
                if len(matches) > 0:
                    chunk_id_map[matches.iloc[0]['chunk_id']] = row[0]

        # 3. Sync Embeddings
        if args.all or args.embeddings:
            if not chunk_id_map:
                print("⚠️  No chunk mapping available. Run --chunks first.")
            else:
                sync_embeddings(db, chunk_id_map)

        # 4. Sync KG
        if args.all or args.kg:
            sync_kg(db, chunk_id_map)

        print("\n" + "="*60)
        print("✅ SYNC COMPLETE")
        print(f"   Finished: {datetime.now().isoformat()}")
        print("="*60)

    except Exception as e:
        print(f"\n❌ Error during sync: {e}")
        import traceback
        traceback.print_exc()
        db.rollback()
    finally:
        db.close()


if __name__ == "__main__":
    main()
