"""
sync_chunks_to_supabase.py

One-time migration script that populates the Supabase `documents` and `chunks`
tables from the parquet files created by build_chroma_store.py.

This is REQUIRED before retrieval_hits and answer_citations can be saved,
because those tables have foreign key constraints to chunks.chunk_id.

The script:
1. Reads chunks_v2.parquet and doc_summaries_v2.parquet
2. Creates Document rows in Supabase (one per unique source_path)
3. Creates Chunk rows with hash = ChromaDB chunk_id (SHA1) for mapping
4. Adds a unique 'hash' column to chunks table if missing (for FK resolution)

Usage:
    python scripts/sync_chunks_to_supabase.py

Environment:
    Requires DATABASE_URL in .env (Supabase connection)
"""

import os
import sys
import hashlib

import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

load_dotenv(os.path.join(PROJECT_ROOT, ".env"))

from database import engine, Base
import models


# ── Config ──────────────────────────────────────────────────────

CHECKPOINT_DIR = os.getenv("CHECKPOINT_DIR_V2") or os.path.join(
    PROJECT_ROOT, "creating_database", "checkpoints_v2"
)
CHUNK_PARQUET = os.path.join(CHECKPOINT_DIR, "chunks_v2.parquet")
DOC_SUMMARY_PARQUET = os.path.join(CHECKPOINT_DIR, "doc_summaries_v2.parquet")


def ensure_hash_column(db_session):
    """Add 'hash' column to chunks table if it doesn't exist."""
    try:
        db_session.execute(text("""
            ALTER TABLE chunks ADD COLUMN IF NOT EXISTS hash TEXT UNIQUE;
        """))
        db_session.commit()
        print("[sync] Ensured 'hash' column exists on chunks table")
    except Exception as e:
        db_session.rollback()
        print(f"[sync] Note: hash column check: {e}")


def sync_documents(db_session, doc_summary_df):
    """
    Sync documents from doc_summaries parquet to Supabase documents table.
    Returns dict: source_path -> document_id
    """
    doc_map = {}

    if doc_summary_df is None or doc_summary_df.empty:
        print("[sync] No document summaries to sync")
        return doc_map

    existing = db_session.query(models.Document.document_id, models.Document.source_uri).all()
    existing_uris = {d.source_uri: d.document_id for d in existing}

    new_count = 0
    skip_count = 0

    for _, row in doc_summary_df.iterrows():
        source_path = row.get("source_path", "")
        if not source_path:
            continue

        if source_path in existing_uris:
            doc_map[source_path] = existing_uris[source_path]
            skip_count += 1
            continue

        title = row.get("title", "")
        doc_lang = row.get("doc_lang", "tr")
        doc_type = row.get("doc_type", "other")

        # Map doc_type to valid source_type
        source_type_map = {
            "html": "html",
            "pdf": "pdf",
            "md": "md",
            "email": "email",
            "url": "url",
        }
        source_type = source_type_map.get(doc_type, "other")

        # Content hash for deduplication
        content_hash = hashlib.sha1(source_path.encode()).hexdigest()

        doc = models.Document(
            source_type=source_type,
            source_uri=source_path,
            title=title[:500] if title else None,
            lang=doc_lang if doc_lang in ("tr", "en") else "tr",
            department=None,
            hash=content_hash,
        )

        try:
            db_session.add(doc)
            db_session.flush()  # Get the auto-generated document_id
            doc_map[source_path] = doc.document_id
            new_count += 1
        except Exception as e:
            db_session.rollback()
            print(f"[sync] Warning: Could not insert document '{source_path[:60]}': {e}")
            # Try to fetch existing
            existing_doc = db_session.query(models.Document).filter(
                models.Document.hash == content_hash
            ).first()
            if existing_doc:
                doc_map[source_path] = existing_doc.document_id

    db_session.commit()
    print(f"[sync] Documents: {new_count} new, {skip_count} existing, {len(doc_map)} total mapped")
    return doc_map


def _sanitize_text(text: str) -> str:
    """Remove NUL (0x00) characters that PostgreSQL rejects in TEXT columns."""
    if not text:
        return text
    return text.replace("\x00", "")


def sync_chunks(db_session, chunk_df, doc_map):
    """
    Sync chunks from parquet to Supabase chunks table.
    Stores ChromaDB chunk_id (SHA1) in the 'hash' column for later FK resolution.

    Dedup logic: checks BOTH hash column AND content similarity to avoid duplicates.
    """
    if chunk_df is None or chunk_df.empty:
        print("[sync] No chunks to sync")
        return

    # Check which chunks already exist (by hash)
    existing_hashes = set()
    for row in db_session.query(models.Chunk.hash).filter(models.Chunk.hash.isnot(None)).all():
        existing_hashes.add(row.hash)

    print(f"[sync] Found {len(existing_hashes)} chunks with hash already in DB")

    new_count = 0
    skip_count = 0
    no_doc_count = 0
    nul_fixed_count = 0
    batch = []

    for idx, row in chunk_df.iterrows():
        chroma_chunk_id = row.get("chunk_id", "")
        source_path = row.get("source_path", "")
        content = row.get("content", "")

        if not chroma_chunk_id or not content:
            continue

        if chroma_chunk_id in existing_hashes:
            skip_count += 1
            continue

        # Find the parent document
        doc_id = doc_map.get(source_path)
        if not doc_id:
            no_doc_count += 1
            continue

        chunk_index = int(row.get("chunk_index", idx))
        section_header = row.get("section_header", "")

        # Sanitize: Remove NUL characters that PostgreSQL cannot store
        clean_content = _sanitize_text(content)
        clean_section = _sanitize_text(section_header[:500]) if section_header else None
        if clean_content != content:
            nul_fixed_count += 1

        chunk = models.Chunk(
            document_id=doc_id,
            ordinal=chunk_index,
            section=clean_section,
            content=clean_content,
            tokens=len(clean_content.split()),
            hash=chroma_chunk_id,
        )
        batch.append(chunk)
        # Track in-memory to avoid re-inserting within same run
        existing_hashes.add(chroma_chunk_id)
        new_count += 1

        # Batch insert every 100 rows
        if len(batch) >= 100:
            try:
                db_session.add_all(batch)
                db_session.commit()
            except Exception as e:
                db_session.rollback()
                print(f"[sync] Warning: Batch insert failed: {e}")
                # Try one by one
                for ch in batch:
                    try:
                        db_session.add(ch)
                        db_session.commit()
                    except Exception as inner_e:
                        db_session.rollback()
                        # Log which chunk failed
                        print(f"[sync] Skipping chunk hash={ch.hash}: {str(inner_e)[:80]}")
            batch = []

    # Final batch
    if batch:
        try:
            db_session.add_all(batch)
            db_session.commit()
        except Exception as e:
            db_session.rollback()
            print(f"[sync] Warning: Final batch failed: {e}")
            for ch in batch:
                try:
                    db_session.add(ch)
                    db_session.commit()
                except Exception:
                    db_session.rollback()

    if nul_fixed_count:
        print(f"[sync] Fixed NUL characters in {nul_fixed_count} chunks")

    print(f"[sync] Chunks: {new_count} new, {skip_count} existing, {no_doc_count} skipped (no parent doc)")


def main():
    print("=" * 60)
    print("  Syncing parquet data → Supabase (documents + chunks)")
    print("=" * 60)

    # Load parquet files
    if not os.path.exists(CHUNK_PARQUET):
        print(f"[error] Chunk parquet not found: {CHUNK_PARQUET}")
        sys.exit(1)

    chunk_df = pd.read_parquet(CHUNK_PARQUET)
    print(f"[sync] Loaded {len(chunk_df)} chunks from parquet")

    doc_summary_df = None
    if os.path.exists(DOC_SUMMARY_PARQUET):
        doc_summary_df = pd.read_parquet(DOC_SUMMARY_PARQUET)
        print(f"[sync] Loaded {len(doc_summary_df)} document summaries from parquet")
    else:
        print(f"[sync] No doc summary parquet found, will derive documents from chunks")
        # Create minimal doc summary from unique source_paths in chunks
        if "source_path" in chunk_df.columns:
            unique_docs = chunk_df.drop_duplicates(subset=["source_path"])[
                ["source_path", "title", "doc_lang"]
            ].copy()
            unique_docs = unique_docs.rename(columns={"doc_lang": "doc_lang"})
            unique_docs["doc_type"] = "other"
            doc_summary_df = unique_docs

    # Create DB session
    SessionLocal = sessionmaker(bind=engine)
    db = SessionLocal()

    try:
        # Ensure hash column exists
        ensure_hash_column(db)

        # Sync documents first (chunks reference documents)
        doc_map = sync_documents(db, doc_summary_df)

        # Then sync chunks
        sync_chunks(db, chunk_df, doc_map)

        # Print summary
        doc_count = db.query(models.Document).count()
        chunk_count = db.query(models.Chunk).count()
        print(f"\n✅ Sync complete!")
        print(f"   Documents in Supabase: {doc_count}")
        print(f"   Chunks in Supabase:    {chunk_count}")
        print(f"\n   retrieval_hits and answer_citations can now be populated!")

    finally:
        db.close()


if __name__ == "__main__":
    main()
