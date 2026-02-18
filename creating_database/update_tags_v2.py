"""
update_tags.py - Update document tags in ChromaDB without re-embedding

This script updates only the tags metadata for documents in ChromaDB.
It does NOT re-compute embeddings, making it much faster than a full rebuild.

Features:
- Improved tagging prompt for better semantic tags
- Parallel processing with multiple workers
- Checkpoint/resume support
- Batch updates to ChromaDB

Usage:
    python update_tags.py
    
Environment variables:
    OLLAMA_HOST: Ollama API host (default: http://localhost:11434)
    CHAT_MODEL: LLM model for tagging (default: llama3.2)
    TAG_WORKERS: Number of parallel workers (default: 3)
"""

import os
import re
import json
import time
import textwrap
from typing import List, Dict, Any, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import defaultdict

import requests
import chromadb
from dotenv import load_dotenv

load_dotenv()

# ================= CONFIG =================

OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
CHAT_MODEL = os.getenv("CHAT_MODEL", "llama3.2")

BASE_DIR = os.getenv("PROJECT_ROOT") or os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PREPROCESSING_DIR = os.getenv("PREPROCESSING_PATH") or os.path.join(BASE_DIR, "preprocessing")

CREATING_DB_DIR = os.path.dirname(os.path.abspath(__file__))
CHROMA_DIR_V2 = os.getenv("CHROMA_DIR_V2") or os.path.join(CREATING_DB_DIR, "chroma_db_v2")
CHECKPOINT_DIR_V2 = os.getenv("CHECKPOINT_DIR_V2") or os.path.join(CREATING_DB_DIR, "checkpoints_v2")

COLL_NAME = os.getenv("CHROMA_COLLECTION_NAME_V2", "mysu_v2_bge_m3")

# Checkpoint file for resume
TAG_UPDATE_CHECKPOINT = os.path.join(CHECKPOINT_DIR_V2, "tag_update_checkpoint.json")

# Performance settings
TAG_WORKERS = int(os.getenv("TAG_WORKERS", "4"))  # Parallel LLM calls
BATCH_UPDATE_SIZE = int(os.getenv("BATCH_UPDATE_SIZE", "50"))  # ChromaDB batch size
REQUEST_TIMEOUT = int(os.getenv("REQUEST_TIMEOUT", "120"))

os.makedirs(CHECKPOINT_DIR_V2, exist_ok=True)

# ================= LLM TAGGING =================

def call_ollama_json(prompt: str, system_prompt: str = "") -> dict:
    """Call Ollama and expect JSON response."""
    url = f"{OLLAMA_HOST}/api/chat"
    
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})
    
    payload = {
        "model": CHAT_MODEL,
        "messages": messages,
        "stream": False,
        "options": {"temperature": 0.0},
    }
    
    try:
        resp = requests.post(url, json=payload, timeout=REQUEST_TIMEOUT)
        resp.raise_for_status()
        data = resp.json()
        text = (data.get("message", {}) or {}).get("content", "").strip()
        if not text:
            return {}
        
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            m = re.search(r"\{.*\}", text, flags=re.DOTALL)
            if not m:
                return {}
            return json.loads(m.group(0))
    except Exception as e:
        print(f"    [llm-error] {e}")
        return {}


# Banned/generic tags to filter out
BANNED_TAG_PARTS = {
    "yonerge", "yönergesi", "prosedur", "prosedürü", "procedure", "directive", "policy", 
    "document", "belge", "doküman",
    "surec_sahibi", "süreç_sahibi", "surec_sorumlulari", "süreç_sorumluları",
    "process_owner", "process_responsible", "process_supplier",
    "ilgili_birim", "related_unit", "department", "birim",
    "guncelleme", "güncelleme", "yurutluk", "yürürlük", "tarihi", "date",
    "page_not_found", "404_error", "unavailable", "not_found", "sayfa_bulunamadi",
    "turkish", "turkce", "türkçe", "english", "ingilizce",
    "sabanci", "sabancı", "university", "universitesi", "üniversitesi",
    "mütevelli", "mutevelli", "heyeti", "board",
    "tedarikci", "tedarikçi", "supplier", "müşteri", "musteri", "customer",
    "array", "string", "object",  # JSON artifacts
}

def clean_and_filter_tags(tags: List[Any]) -> List[str]:
    """Clean and filter tags, removing generic/banned ones."""
    if not isinstance(tags, list):
        return []
    
    clean_tags = []
    seen = set()
    
    for t in tags:
        if not isinstance(t, str):
            continue
        
        # Normalize
        tt = t.strip().lower()
        tt = re.sub(r"[^a-z0-9çğıöşü_]", "_", tt)
        tt = re.sub(r"_+", "_", tt).strip("_")
        
        # Skip if empty or too short
        if len(tt) < 4:
            continue
        
        # Skip if contains banned parts
        if any(banned in tt for banned in BANNED_TAG_PARTS):
            continue
        
        # Skip document codes (e.g., ihr-s420-05, iic-c820)
        if re.match(r"^[a-z]{1,5}_?[a-z]?\d", tt):
            continue
        
        # Skip if already seen
        if tt in seen:
            continue
        
        seen.add(tt)
        clean_tags.append(tt)
    
    return clean_tags[:8]  # Max 8 tags


def generate_tags_improved(title: str, doc_text: str, source_path: str) -> List[str]:
    """Generate improved semantic tags for a document."""
    
    # Take a sample of the document
    text_sample = (doc_text or "")[:1800]
    
    # Improved system prompt with concrete examples
    sys_prompt = textwrap.dedent("""
        You are a document classifier for Sabancı University's administrative documents.
        Extract 5-8 SEMANTIC TOPIC tags describing what the document is ABOUT.
        
        GOOD tags (specific, searchable topics):
        - academic_leave, sabbatical_leave, research_leave
        - ethics_committee, research_ethics, academic_integrity  
        - student_discipline, academic_misconduct, plagiarism_policy
        - budget_planning, expense_reporting, financial_audit
        - hiring_process, job_recruitment, staff_onboarding
        - graduation_ceremony, diploma_requirements, commencement
        - library_lending, book_reservation, database_access
        - travel_reimbursement, conference_funding, expense_advance
        - performance_review, promotion_criteria, annual_evaluation
        - student_exchange, erasmus_program, international_mobility
        - health_insurance, retirement_benefits, employee_benefits
        - parking_permit, campus_facilities, room_reservation
        - thesis_submission, dissertation_defense, graduate_requirements
        - course_registration, add_drop_period, transcript_request
        - scholarship_application, financial_aid, tuition_waiver
        
        BAD tags (DO NOT USE - too generic):
        - yonerge, prosedur, policy, directive, regulation, document
        - process_owner, responsible_unit, related_department
        - update_date, effective_date, version
        - sabanci, university, turkish, english
        - any codes like "ihr-s420" or "iic-c820"
        
        Return ONLY valid JSON: {"tags": ["tag1", "tag2", ...]}
        Use English. Use snake_case. 5-8 tags maximum.
    """).strip()
    
    user_prompt = f"Title: {title}\nPath: {source_path}\n\nContent:\n{text_sample}"
    
    result = call_ollama_json(user_prompt, sys_prompt)
    tags = result.get("tags", [])
    
    return clean_and_filter_tags(tags)


def process_one_document(doc_info: Dict) -> Tuple[str, List[str], bool]:
    """
    Process a single document and return new tags.
    Returns: (doc_id, new_tags, success)
    """
    doc_id = doc_info["id"]
    title = doc_info.get("title", "")
    text = doc_info.get("text", "")
    source_path = doc_info.get("source_path", "")
    
    try:
        new_tags = generate_tags_improved(title, text, source_path)
        return (doc_id, new_tags, True)
    except Exception as e:
        print(f"    [error] {doc_id}: {e}")
        return (doc_id, [], False)


# ================= CHECKPOINT =================

def load_checkpoint() -> Dict:
    """Load checkpoint for resume."""
    if os.path.exists(TAG_UPDATE_CHECKPOINT):
        try:
            with open(TAG_UPDATE_CHECKPOINT, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            print(f"[warn] Could not load checkpoint: {e}")
    return {"processed_ids": [], "stats": {"success": 0, "failed": 0}}


def save_checkpoint(processed_ids: List[str], stats: Dict):
    """Save checkpoint for resume."""
    try:
        with open(TAG_UPDATE_CHECKPOINT, "w", encoding="utf-8") as f:
            json.dump({
                "processed_ids": processed_ids,
                "stats": stats,
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            }, f, indent=2)
    except Exception as e:
        print(f"[warn] Could not save checkpoint: {e}")


# ================= MAIN =================

def get_unique_documents(coll) -> List[Dict]:
    """
    Get unique documents from ChromaDB collection.
    Groups chunks by source_path and returns one entry per document.
    """
    print("[info] Fetching all documents from ChromaDB...")
    
    # Get all items
    result = coll.get(include=["documents", "metadatas"])
    
    ids = result.get("ids", [])
    docs = result.get("documents", [])
    metas = result.get("metadatas", [])
    
    print(f"[info] Found {len(ids)} total chunks")
    
    # Group by source_path to get unique documents
    doc_map = {}  # source_path -> {id, title, text, source_path}
    
    for i in range(len(ids)):
        chunk_id = ids[i]
        doc_text = docs[i] if docs else ""
        meta = metas[i] if metas else {}
        
        source_path = meta.get("source_path", "")
        if not source_path:
            continue
        
        # Keep the first chunk as representative (usually has best content)
        if source_path not in doc_map:
            doc_map[source_path] = {
                "id": chunk_id,  # Use first chunk's ID
                "source_path": source_path,
                "title": meta.get("title", ""),
                "text": doc_text,
                "all_chunk_ids": [chunk_id],
            }
        else:
            # Add to list of chunk IDs for this document
            doc_map[source_path]["all_chunk_ids"].append(chunk_id)
            # Accumulate text for better tagging (up to a limit)
            if len(doc_map[source_path]["text"]) < 3000:
                doc_map[source_path]["text"] += " " + doc_text
    
    documents = list(doc_map.values())
    print(f"[info] Found {len(documents)} unique documents")
    
    return documents


def update_chunk_tags(coll, chunk_ids: List[str], new_tags: List[str]):
    """Update tags for all chunks of a document."""
    tags_str = ",".join(new_tags)
    
    for chunk_id in chunk_ids:
        try:
            # Get current metadata
            result = coll.get(ids=[chunk_id], include=["metadatas"])
            if not result["metadatas"]:
                continue
            
            meta = result["metadatas"][0]
            meta["tags"] = tags_str
            
            # Update metadata
            coll.update(ids=[chunk_id], metadatas=[meta])
        except Exception as e:
            print(f"    [update-error] {chunk_id}: {e}")


def main():
    print("=" * 60)
    print("TAG UPDATE SCRIPT v2")
    print("=" * 60)
    print(f"ChromaDB: {CHROMA_DIR_V2}")
    print(f"Collection: {COLL_NAME}")
    print(f"Workers: {TAG_WORKERS}")
    print(f"Model: {CHAT_MODEL}")
    print("=" * 60)
    
    # Connect to ChromaDB
    client = chromadb.PersistentClient(path=CHROMA_DIR_V2)
    
    try:
        coll = client.get_collection(name=COLL_NAME)
    except Exception as e:
        print(f"[error] Could not get collection '{COLL_NAME}': {e}")
        print("[hint] Make sure build_chroma_store.py has finished running first.")
        return
    
    # Get unique documents
    documents = get_unique_documents(coll)
    
    if not documents:
        print("[warn] No documents found in collection.")
        return
    
    # Load checkpoint
    checkpoint = load_checkpoint()
    processed_set = set(checkpoint.get("processed_ids", []))
    stats = checkpoint.get("stats", {"success": 0, "failed": 0})
    
    # Filter out already processed
    remaining = [d for d in documents if d["source_path"] not in processed_set]
    
    print(f"\n[info] Total documents: {len(documents)}")
    print(f"[info] Already processed: {len(processed_set)}")
    print(f"[info] Remaining: {len(remaining)}")
    
    if not remaining:
        print("\n✅ All documents already processed!")
        return
    
    # Process with parallel workers
    start_time = time.time()
    processed_paths = list(processed_set)
    
    print(f"\n[info] Starting tag update with {TAG_WORKERS} workers...\n")
    
    batch_num = 0
    for batch_start in range(0, len(remaining), TAG_WORKERS * 5):
        batch = remaining[batch_start:batch_start + TAG_WORKERS * 5]
        batch_num += 1
        
        # Process batch in parallel
        with ThreadPoolExecutor(max_workers=TAG_WORKERS) as executor:
            futures = {executor.submit(process_one_document, doc): doc for doc in batch}
            
            for future in as_completed(futures):
                doc = futures[future]
                doc_id, new_tags, success = future.result()
                
                elapsed = time.time() - start_time
                total_done = stats["success"] + stats["failed"]
                rate = total_done / elapsed if elapsed > 0 else 0
                eta = (len(remaining) - total_done) / rate / 60 if rate > 0 else 0
                
                idx = len(processed_paths) + 1
                source = doc["source_path"]
                short_name = os.path.basename(source)[:40]
                
                if success and new_tags:
                    # Update all chunks for this document
                    update_chunk_tags(coll, doc["all_chunk_ids"], new_tags)
                    stats["success"] += 1
                    tags_preview = ", ".join(new_tags[:4])
                    if len(new_tags) > 4:
                        tags_preview += f" (+{len(new_tags)-4})"
                    print(f"[{idx}/{len(documents)}] ✓ {short_name}")
                    print(f"    Tags: {tags_preview}")
                else:
                    stats["failed"] += 1
                    print(f"[{idx}/{len(documents)}] ✗ {short_name} (no tags)")
                
                processed_paths.append(source)
        
        # Save checkpoint after each batch
        save_checkpoint(processed_paths, stats)
        
        elapsed = time.time() - start_time
        total_done = stats["success"] + stats["failed"]
        print(f"\n  [checkpoint] Batch {batch_num} done. "
              f"Progress: {total_done}/{len(remaining)} "
              f"({elapsed/60:.1f} min elapsed)\n")
    
    # Final stats
    elapsed = time.time() - start_time
    print("\n" + "=" * 60)
    print("TAG UPDATE COMPLETE")
    print("=" * 60)
    print(f"Total time: {elapsed/60:.1f} minutes")
    print(f"Documents updated: {stats['success']}")
    print(f"Documents failed: {stats['failed']}")
    print(f"Rate: {(stats['success'] + stats['failed']) / elapsed * 60:.1f} docs/min")
    print("=" * 60)
    
    # Remove checkpoint on success
    if os.path.exists(TAG_UPDATE_CHECKPOINT):
        os.remove(TAG_UPDATE_CHECKPOINT)
        print("✅ Checkpoint file removed (update complete)")


if __name__ == "__main__":
    main()
