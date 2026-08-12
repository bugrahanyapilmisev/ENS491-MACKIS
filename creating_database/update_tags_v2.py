"""
update_tags_v2.py — Retag all ChromaDB chunks with a fixed taxonomy.

Uses OpenRouter LLM (Qwen3-32B or Gemini Flash) for vastly better tag quality
than local Ollama llama3.2. Tags are restricted to a FIXED DOMAIN TAXONOMY so
retrieval tag-filtering is consistent and meaningful.

Usage:
    python update_tags_v2.py

Environment variables:
    OPENROUTER_API_KEY : required for OpenRouter LLM
    CHAT_MODEL         : model name on OpenRouter (default: qwen/qwen3-32b)
    TAG_WORKERS        : parallel workers (default: 4)
    COLL_NAME_V3       : target collection (default: mysu_v3_qwen3)
                         set to COLL_NAME_V2=mysu_v2_bge_m3 to retag old DB

To retag the OLD v2 collection instead of the new v3:
    set COLL_NAME_V3=mysu_v2_bge_m3 && python update_tags_v2.py
"""

import os
import re
import json
import time
import textwrap
from typing import List, Dict, Any, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests
import chromadb
from dotenv import load_dotenv

load_dotenv()

# ================= CONFIG =================

OLLAMA_HOST   = os.getenv("OLLAMA_HOST", "http://localhost:11434")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")

# Use OpenRouter by default for much better tag quality; fall back to Ollama
CHAT_MODEL_OR  = os.getenv("CHAT_MODEL_OR",  "qwen/qwen3-32b")     # OpenRouter
CHAT_MODEL_OL  = os.getenv("CHAT_MODEL",     "llama3.2")             # Ollama fallback
USE_OPENROUTER = bool(OPENROUTER_API_KEY)

BASE_DIR         = os.getenv("PROJECT_ROOT") or os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CREATING_DB_DIR  = os.path.dirname(os.path.abspath(__file__))
CHROMA_DIR_V2    = os.getenv("CHROMA_DIR_V2") or os.path.join(CREATING_DB_DIR, "chroma_db_v2")
CHECKPOINT_DIR_V2 = os.getenv("CHECKPOINT_DIR_V2") or os.path.join(CREATING_DB_DIR, "checkpoints_v2")

# Default: retag the new v3 collection; override via env to retag v2
COLL_NAME = os.getenv("COLL_NAME_V3", "mysu_v3_qwen3")

TAG_UPDATE_CHECKPOINT = os.path.join(CHECKPOINT_DIR_V2, f"tag_update_checkpoint_{COLL_NAME}.json")

TAG_WORKERS       = int(os.getenv("TAG_WORKERS",        "2"))  # Reduced from 4 to avoid rate limits
BATCH_UPDATE_SIZE = int(os.getenv("BATCH_UPDATE_SIZE",  "50"))
REQUEST_TIMEOUT   = int(os.getenv("REQUEST_TIMEOUT",    "120"))

os.makedirs(CHECKPOINT_DIR_V2, exist_ok=True)

print(f"[Config] Collection    : {COLL_NAME}")
print(f"[Config] LLM backend   : {'OpenRouter (' + CHAT_MODEL_OR + ')' if USE_OPENROUTER else 'Ollama (' + CHAT_MODEL_OL + ')'}")
print(f"[Config] Workers       : {TAG_WORKERS}")

# ================= FIXED TAXONOMY =================
# All tags must come from this taxonomy. This ensures consistency across runs
# and makes tag-based retrieval filtering reliable.
DOMAIN_TAXONOMY = textwrap.dedent("""
ACADEMIC:
  course_registration, add_drop, course_drop, course_exemption, course_repeat,
  transcript, grading, grade_change, academic_probation, academic_standing,
  graduation_requirements, diploma, commencement, course_catalog, course_schedule,
  midterm_schedule, final_exam_schedule, academic_calendar, credit_transfer,
  double_major, minor_program, ects_credits, advisor_assignment, orientation,
  english_proficiency_exam, student_information_system, part_time_student

STUDENT_SERVICES:
  student_discipline, academic_integrity, plagiarism, cheating_penalty, exam_rules,
  student_rights, student_complaints, student_certificate, student_id_card,
  student_pass_card, disability_support, peer_tutoring, writing_center,
  psychological_counseling, academic_counseling, student_satisfaction

EXCHANGE_MOBILITY:
  erasmus_program, erasmus_internship, erasmus_teaching_mobility,
  erasmus_staff_training, erasmus_short_term_doctoral, student_exchange,
  bilateral_agreements, international_mobility, study_abroad,
  incoming_exchange, outgoing_exchange, iaeste_internship,
  exchange_gpa_requirement, grade_conversion, learning_agreement,
  study_visa, residence_permit

FINANCIAL_AID:
  tuition_fees, tuition_fee_management, scholarship_application, financial_aid,
  tuition_waiver, merit_scholarship, need_based_aid, payment_schedule,
  refund_policy, kyk_scholarship, su_scholarship, burs_ana_varlik_fonu,
  scholarship_fund_development, donation_acceptance, part_time_work_scholarship

RESEARCH_GRANTS:
  grant_calculation, grant_payment, erasmus_grant, research_fund,
  personal_research_fund, project_research_fund, rectorate_research_fund,
  tubitak_project, horizon_europe, erc_project, international_grant,
  grant_proposal, grant_management, research_reporting, postdoctoral_research,
  center_of_excellence, sop_fund, integration_project, contracted_project,
  technology_income_sharing, scientific_publication_incentive, teaching_award,
  su_science_art_award

LIBRARY:
  book_borrowing, ill_service, interlibrary_loan, database_access,
  multimedia_lending, overdue_fines, library_membership, reserve_materials,
  e_resources, collection_building, collection_organization, bibliographic_records,
  authority_index, document_supply, binding, labeling, inventory_weeding,
  reserve_collection, library_building_operation, library_marketing,
  university_archive, university_history_collection, isbn_management

REGISTRATION_STATUS:
  enrollment, new_enrollment, registration_freeze, leave_of_absence,
  enrollment_cancellation, withdrawal, lateral_transfer, readmission,
  special_student_admission, kayit_dondurma, semester_leave, study_status

ADMISSIONS:
  undergraduate_admission, graduate_admission, transfer_admission,
  international_admission, program_opening_closing, faculty_institute_opening,
  course_catalog_preparation, academic_program_management, student_recruitment

GRADUATE_STUDIES:
  thesis_submission, dissertation_defense, qualifying_exam,
  graduate_advisor, ta_ra_positions, graduate_requirements,
  phd_program, masters_program, thesis_committee, graduate_mobility,
  graduate_short_term_mobility, graduate_program_changes

FACULTY_HR:
  faculty_recruitment, faculty_appointment, faculty_promotion,
  associate_professor_promotion, professor_appointment,
  emeritus_faculty, chair_position, researcher_staff,
  periodic_review, sabbatical, academic_leave, work_permit,
  employment_exit, hybrid_work, honorary_award, honorary_member,
  performance_development, training_and_development, talent_management

STAFF_HR:
  hiring_process, staff_planning, staff_performance_review,
  administrative_staff_orientation, annual_leave, sick_leave,
  maternity_leave, marriage_leave, military_leave, death_disability_leave,
  excuse_leave, payroll, salary_advance, travel_reimbursement,
  employee_benefits, private_health_insurance, retirement_insurance,
  retirement_process, vehicle_allocation, mobile_device_allocation,
  laptop_allocation, compensation_policy, recognition_award,
  performance_reward, emergency_advance

HEALTH_CENTER:
  medical_examination, vaccination, infectious_disease_control,
  epidemic_procedure, health_insurance_processing, referral_to_hospital,
  athlete_examination, occupational_health_exam, new_employee_health,
  dormitory_health, ambulance_emergency, lab_services, first_aid,
  medication_supply, health_announcement, patient_records, rpt_procedure

OCCUPATIONAL_SAFETY:
  ohs_risk_assessment, ohs_incident_reporting, ohs_training,
  ohs_field_inspection, ohs_legal_compliance, fire_safety,
  fire_extinguishing, fire_protection_system, earthquake_procedure,
  emergency_procedure, disaster_container, personal_protective_equipment,
  periodic_maintenance, greenhouse_gas, subcontractor_management,
  work_permit_ohs, isg, sustainability, environmental_management,
  waste_management, chemical_safety, lab_safety

CAMPUS_SECURITY:
  security_services, visitor_admission, gate_access_control,
  vehicular_traffic, key_cabinet, lost_and_found, riot_response,
  bomb_threat, campus_safety

CAMPUS_FACILITIES:
  space_allocation, room_reservation, venue_allocation,
  transportation_services, campus_vehicle, general_service_vehicle,
  bus_service, cleaning_services, dormitory, student_housing,
  dormitory_application, dormitory_checkout, food_services,
  catering_services, hygiene_inspection, communication_services,
  audiovisual_services, electrical_works, mechanical_works, construction_works,
  warehouse_management, facilities_management

TECHNOLOGY_IT:
  it_planning, it_operations, it_service_management, service_request,
  user_accounts_authorization, hardware_software_provision, erp_sap_systems,
  crm_systems, web_systems, document_management_system, call_management,
  data_management, information_security, outsourced_it_services,
  software_development, budget_preparation_it, it_review

LEGAL_CONTRACTS:
  contract_creation, contract_review, contract_signature, stamp_duty,
  legal_services, industrial_property_rights, technology_transfer,
  intellectual_property, commercialization, right_to_information,
  official_correspondence, authority_delegation, governance

FINANCIAL_OPERATIONS:
  budget_preparation, budget_management, revenue_accounting,
  expense_accounting, general_ledger, payables_management,
  receivables_management, cost_accounting, financial_audit,
  external_audit, asset_management, asset_depreciation, fixed_assets,
  investment_accounting, expense_advance, travel_expense,
  domestic_travel, international_travel, tax_management, treasury

MARKETING_COMMUNICATIONS:
  institutional_identity, press_release, press_conference,
  social_media, digital_marketing, website_management,
  marketing_materials, media_planning, market_research,
  undergraduate_promotion, graduate_promotion, international_marketing,
  internal_communication, external_communication, corporate_communications,
  gazeteSU, student_activities

CAREER_ALUMNI:
  career_development, career_events, internship_undergraduate,
  internship_international, iaeste_internship, alumni_relations,
  commencement_ceremony, extracurricular_achievements, student_council,
  sports_management, sports_facilities, civic_involvement_projects,
  summer_schools, high_school_summer_school, ayvalik_workshop

MUSEUM_SSM:
  ssm_artwork_loan, ssm_artwork_donation, ssm_gift_shop,
  ssm_event_management, ssm_collection, ssm_operations

ENTREPRENEURSHIP:
  entrepreneurship, startup, technology_transfer_office, suatt, sunum,
  inovent, spin_off, academic_entrepreneurship, innovation

QUALITY_MANAGEMENT:
  quality_system, corrective_action, change_management, risk_management,
  risk_analysis, internal_audit, external_audit_quality, management_review,
  continuous_improvement, document_standards, records_control,
  student_satisfaction_monitoring, strategic_planning

PROCUREMENT:
  procurement, goods_services_purchase, supply_management, vendor_management,
  purchasing_principles, procurement_instruction
""").strip()


# ================= LLM CALL =================

def call_llm_json(prompt: str, system_prompt: str = "") -> dict:
    """Call OpenRouter or Ollama and expect a JSON response."""
    if USE_OPENROUTER:
        return _call_openrouter_json(prompt, system_prompt)
    else:
        return _call_ollama_json(prompt, system_prompt)


def _call_openrouter_json(prompt: str, system_prompt: str) -> dict:
    """Call OpenRouter chat completion endpoint."""
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})

    try:
        resp = requests.post(
            "https://openrouter.ai/api/v1/chat/completions",
            json={
                "model": CHAT_MODEL_OR,
                "messages": messages,
                "temperature": 0.0,
                "response_format": {"type": "json_object"},
                "max_tokens": 300,
            },
            headers={
                "Authorization": f"Bearer {OPENROUTER_API_KEY}",
                "HTTP-Referer": "http://localhost:3000",
                "X-Title": "MACKIS Tag Update",
                "Content-Type": "application/json",
            },
            timeout=REQUEST_TIMEOUT,
        )
        resp.raise_for_status()
        data = resp.json()
        # Guard: message or content can be None when model returns empty response
        choice = data.get("choices", [{}])[0]
        message = choice.get("message") or {}
        text = (message.get("content") or "").strip()
        if not text:
            return {}
        # Strip markdown fences if present
        text = re.sub(r"```(?:json)?|```", "", text).strip()
        return json.loads(text)
    except Exception as e:
        print(f"    [openrouter-error] {e}")
        return {}


def _call_ollama_json(prompt: str, system_prompt: str) -> dict:
    """Call local Ollama server."""
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})

    try:
        resp = requests.post(
            f"{OLLAMA_HOST}/api/chat",
            json={
                "model": CHAT_MODEL_OL,
                "messages": messages,
                "stream": False,
                "options": {"temperature": 0.0},
            },
            timeout=REQUEST_TIMEOUT,
        )
        resp.raise_for_status()
        data = resp.json()
        text = (data.get("message", {}) or {}).get("content", "").strip()
        if not text:
            return {}
        m = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if not m:
            return {}
        return json.loads(m.group(0))
    except Exception as e:
        print(f"    [ollama-error] {e}")
        return {}


# ================= TAGGING =================

SYSTEM_PROMPT = textwrap.dedent(f"""
You are a document classifier for Sabancı University's administrative document system.
Your job is to assign SEMANTIC TAGS from a FIXED TAXONOMY to university policy documents.

FIXED TAXONOMY (you MUST only use tags from this list):
{DOMAIN_TAXONOMY}

RULES:
1. Select 3-8 tags that best describe what the document is ABOUT.
2. ONLY use tags from the taxonomy above — do not invent new tags.
3. Also identify the PRIMARY AUDIENCE: "student", "staff", "faculty", or "all".
4. Use English tags only.
5. Focus on TOPICS, not document type (no "policy", "directive", "procedure" as tags).

Output ONLY valid JSON, no explanation:
{{"tags": ["tag1", "tag2", ...], "primary_audience": "student|staff|faculty|all"}}
""").strip()


def generate_tags(title: str, doc_text: str, source_path: str) -> Tuple[List[str], str]:
    """
    Generate tags and audience from a document using the fixed taxonomy.
    Retries up to 2 times on empty response (rate-limit / null content).

    Returns:
        (tags_list, primary_audience)
    """
    text_sample = (doc_text or "")[:2000]
    user_prompt = f"Title: {title}\nSource: {os.path.basename(source_path)}\n\nContent:\n{text_sample}"

    for attempt in range(3):  # up to 3 attempts
        if attempt > 0:
            time.sleep(3)  # brief back-off before retry

        result = call_llm_json(user_prompt, SYSTEM_PROMPT)

        # Guard: sometimes the model returns a bare JSON array instead of an object
        if isinstance(result, list):
            result = {"tags": result}

        raw_tags = result.get("tags", [])
        audience = result.get("primary_audience", "all")

        # Validate against taxonomy — only keep known tags
        all_valid = set(re.findall(r"[a-z_]+", DOMAIN_TAXONOMY))
        clean_tags: List[str] = []
        seen: set = set()
        for t in raw_tags:
            if not isinstance(t, str):
                continue
            tt = t.strip().lower().replace(" ", "_")
            tt = re.sub(r"[^a-z0-9_]", "_", tt)
            tt = re.sub(r"_+", "_", tt).strip("_")
            if tt in all_valid and tt not in seen:
                seen.add(tt)
                clean_tags.append(tt)

        if audience not in ("student", "staff", "faculty", "all"):
            audience = "all"

        if clean_tags:  # success
            return clean_tags[:8], audience

        # Empty result — retry
        if attempt < 2:
            print(f"    [retry {attempt+1}] {os.path.basename(source_path)} — empty response, retrying...")

    return [], "all"  # all retries exhausted


def process_one_document(doc_info: Dict) -> Tuple[str, List[str], str, bool]:
    """Process one document and return (source_path, tags, audience, success)."""
    try:
        tags, audience = generate_tags(
            doc_info.get("title", ""),
            doc_info.get("text", ""),
            doc_info.get("source_path", ""),
        )
        return (doc_info["source_path"], tags, audience, True)
    except Exception as e:
        print(f"    [error] {doc_info.get('source_path', '?')}: {e}")
        return (doc_info["source_path"], [], "all", False)


# ================= CHECKPOINT =================

def load_checkpoint() -> Dict:
    if os.path.exists(TAG_UPDATE_CHECKPOINT):
        try:
            with open(TAG_UPDATE_CHECKPOINT, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            print(f"[warn] Could not load checkpoint: {e}")
    return {"processed_ids": [], "stats": {"success": 0, "failed": 0}}


def save_checkpoint(processed_ids: List[str], stats: Dict):
    try:
        with open(TAG_UPDATE_CHECKPOINT, "w", encoding="utf-8") as f:
            json.dump({
                "processed_ids": processed_ids,
                "stats": stats,
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            }, f, indent=2)
    except Exception as e:
        print(f"[warn] Could not save checkpoint: {e}")


# ================= CHROMA OPS =================

def get_unique_documents(coll) -> List[Dict]:
    """Fetch unique documents from ChromaDB, accumulating text per document."""
    print("[info] Fetching all chunks from ChromaDB...")
    result = coll.get(include=["documents", "metadatas"])
    ids   = result.get("ids", [])
    docs  = result.get("documents", [])
    metas = result.get("metadatas", [])
    print(f"[info] Total chunks: {len(ids)}")

    doc_map: Dict[str, Dict] = {}
    for i in range(len(ids)):
        chunk_id  = ids[i]
        doc_text  = docs[i] if docs else ""
        meta      = metas[i] if metas else {}
        src       = meta.get("source_path", "")
        if not src:
            continue
        if src not in doc_map:
            doc_map[src] = {
                "source_path": src,
                "title": meta.get("title", ""),
                "text": doc_text,
                "all_chunk_ids": [chunk_id],
            }
        else:
            doc_map[src]["all_chunk_ids"].append(chunk_id)
            if len(doc_map[src]["text"]) < 3000:
                doc_map[src]["text"] += " " + doc_text

    documents = list(doc_map.values())
    print(f"[info] Unique documents: {len(documents)}")
    return documents


def update_chunk_tags(coll, chunk_ids: List[str], tags: List[str], audience: str):
    """Update tags and primary_audience metadata for all chunks of a document."""
    tags_str = ",".join(tags)
    for chunk_id in chunk_ids:
        try:
            result = coll.get(ids=[chunk_id], include=["metadatas"])
            if not result["metadatas"]:
                continue
            meta = result["metadatas"][0]
            meta["tags"] = tags_str
            meta["primary_audience"] = audience
            coll.update(ids=[chunk_id], metadatas=[meta])
        except Exception as e:
            print(f"    [update-error] {chunk_id}: {e}")


# ================= MAIN =================

def main():
    print("=" * 60)
    print("TAG UPDATE SCRIPT v3 — Fixed Taxonomy + OpenRouter LLM")
    print("=" * 60)
    print(f"ChromaDB path: {CHROMA_DIR_V2}")
    print(f"Collection:    {COLL_NAME}")
    print("=" * 60)

    client = chromadb.PersistentClient(path=CHROMA_DIR_V2)
    try:
        coll = client.get_collection(name=COLL_NAME)
    except Exception as e:
        print(f"[error] Could not open collection '{COLL_NAME}': {e}")
        print("[hint] Run build_chroma_store.py first, or set COLL_NAME_V3 env var.")
        return

    documents = get_unique_documents(coll)
    if not documents:
        print("[warn] No documents found.")
        return

    checkpoint   = load_checkpoint()
    processed_set = set(checkpoint.get("processed_ids", []))
    stats        = checkpoint.get("stats", {"success": 0, "failed": 0})
    remaining    = [d for d in documents if d["source_path"] not in processed_set]

    print(f"\nTotal: {len(documents)} | Done: {len(processed_set)} | Remaining: {len(remaining)}")

    if not remaining:
        print("\nAll documents already tagged!")
        return

    start_time       = time.time()
    processed_paths  = list(processed_set)
    batch_num        = 0

    print(f"\nStarting with {TAG_WORKERS} parallel workers...\n")

    for batch_start in range(0, len(remaining), TAG_WORKERS * 5):
        batch     = remaining[batch_start: batch_start + TAG_WORKERS * 5]
        batch_num += 1

        with ThreadPoolExecutor(max_workers=TAG_WORKERS) as executor:
            futures = {executor.submit(process_one_document, doc): doc for doc in batch}

            for future in as_completed(futures):
                source, tags, audience, success = future.result()
                doc = futures[future]
                elapsed   = time.time() - start_time
                total_done = stats["success"] + stats["failed"]
                rate = total_done / elapsed if elapsed > 0 else 0
                eta  = (len(remaining) - total_done) / rate / 60 if rate > 0 else 0

                short_name = os.path.basename(source)[:45]
                idx = len(processed_paths) + 1

                if success and tags:
                    update_chunk_tags(coll, doc["all_chunk_ids"], tags, audience)
                    stats["success"] += 1
                    preview = ", ".join(tags[:4]) + (f" (+{len(tags)-4})" if len(tags) > 4 else "")
                    print(f"[{idx}/{len(documents)}] OK  {short_name}")
                    print(f"    Tags: {preview}  |  Audience: {audience}")
                else:
                    stats["failed"] += 1
                    print(f"[{idx}/{len(documents)}] ERR {short_name} (no tags generated)")

                processed_paths.append(source)

        save_checkpoint(processed_paths, stats)
        elapsed = time.time() - start_time
        print(f"\n  [batch {batch_num}] {stats['success']+stats['failed']}/{len(remaining)} done "
              f"({elapsed/60:.1f} min elapsed)\n")

    elapsed = time.time() - start_time
    print("\n" + "=" * 60)
    print("TAG UPDATE COMPLETE")
    print("=" * 60)
    print(f"Time        : {elapsed/60:.1f} minutes")
    print(f"Success     : {stats['success']}")
    print(f"Failed      : {stats['failed']}")
    if elapsed > 0:
        print(f"Rate        : {(stats['success']+stats['failed'])/elapsed*60:.1f} docs/min")
    print("=" * 60)

    if os.path.exists(TAG_UPDATE_CHECKPOINT):
        os.remove(TAG_UPDATE_CHECKPOINT)
        print("Checkpoint file cleaned up.")


if __name__ == "__main__":
    main()
