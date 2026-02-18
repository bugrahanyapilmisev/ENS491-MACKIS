"""
clear_tables.py - Clear Supabase tables for fresh sync

This script clears data from:
- chunk_embeddings (FK dependent)
- chunks (FK dependent)
- documents
- kg_edges (FK dependent)
- kg_nodes

Usage:
    python scripts/sync/clear_tables.py
    python scripts/sync/clear_tables.py --confirm  # Skip confirmation prompt
"""

import os
import sys
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from dotenv import load_dotenv
load_dotenv()

from database import SessionLocal
from sqlalchemy import text


def clear_tables(confirm: bool = False):
    """Clear all content-related tables in the correct order."""

    if not confirm:
        print("⚠️  WARNING: This will DELETE all data from:")
        print("   - chunk_embeddings")
        print("   - retrieval_hits")
        print("   - answer_citations")
        print("   - chunks")
        print("   - documents")
        print("   - kg_edges")
        print("   - kg_nodes")
        print()
        response = input("Type 'yes' to confirm: ")
        if response.lower() != 'yes':
            print("Aborted.")
            return False

    db = SessionLocal()

    try:
        # Order matters due to foreign key constraints
        tables = [
            "answer_citations",   # FK to chunks
            "retrieval_hits",     # FK to chunks
            "chunk_embeddings",   # FK to chunks
            "chunks",             # FK to documents
            "ingest_artifacts",   # FK to documents
            "documents",
            "kg_edges",           # FK to kg_nodes
            "kg_nodes",
        ]

        print("\n🗑️  Clearing tables...")

        for table in tables:
            try:
                result = db.execute(text(f"DELETE FROM {table}"))
                db.commit()
                print(f"   ✓ {table}: {result.rowcount} rows deleted")
            except Exception as e:
                print(f"   ✗ {table}: {e}")
                db.rollback()

        print("\n✅ Tables cleared successfully!")
        return True

    except Exception as e:
        print(f"\n❌ Error: {e}")
        db.rollback()
        return False
    finally:
        db.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Clear Supabase tables")
    parser.add_argument('--confirm', action='store_true', help='Skip confirmation prompt')
    args = parser.parse_args()

    clear_tables(confirm=args.confirm)
