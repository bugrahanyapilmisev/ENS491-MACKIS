"""test_rag_detailed.py - Multi-Metric RAG Evaluation

Integrates the Evaluator class for comprehensive quality assessment.
Run: python test_rag_detailed.py
All terminal output is also recorded to rag_test_output.txt

Flags:
  --eval         Use multi-metric Evaluator (default: True)
  --no-eval      Legacy keyword-coverage-only mode
  -q QUESTION    Ask a single question interactively
"""

import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ========== OUTPUT REDIRECTION ========== #
class Tee:
    def __init__(self, console, file_handle):
        self.console = console
        self.file_handle = file_handle
    def write(self, obj):
        try:
            self.console.write(obj)
        except (UnicodeEncodeError, UnicodeDecodeError):
            self.console.write(obj.encode('ascii', 'replace').decode('ascii'))
        self.file_handle.write(obj)
    def flush(self):
        self.console.flush()
        self.file_handle.flush()

output_file_path = os.path.join(os.path.dirname(__file__), "rag_test_output.txt")
output_file = open(output_file_path, "w", encoding="utf-8")
original_stdout = sys.stdout
original_stderr = sys.stderr
sys.stdout = Tee(original_stdout, output_file)
sys.stderr = Tee(original_stderr, output_file)

from dotenv import load_dotenv
load_dotenv()

# Test questions with expected answers from documents
# Organized by category for comprehensive coverage
TEST_QUESTIONS = [
    # =================== ERASMUS & INTERNATIONAL ===================
    {
        "id": "Q1",
        "category": "Erasmus",
        "question": "Erasmus staj programına başvurmak için minimum GNO ne olmalı?",
        "expected_answer": "Lisans için en az 2.20, Lisansüstü için en az 2.5 GNO gereklidir.",
        "source": "IIPAR-C710-02 Uluslararası Staj Yönergesi"
    },
    {
        "id": "Q2",
        "category": "Erasmus", 
        "question": "Erasmus staj hareketliliği minimum süresi ne kadar?",
        "expected_answer": "Staj hareketliliği süresi minimum 2 aydır.",
        "source": "IIPAR-C710-02"
    },
    {
        "id": "Q3",
        "category": "Erasmus",
        "question": "How can I apply for Erasmus internship program?",
        "expected_answer": "Application is made through Career Development and Internship Office. Requirements: acceptance letter from host organization, minimum GPA 2.20 for undergrad / 2.5 for graduate.",
        "source": "PIPAR-C710-0201"
    },
    {
        "id": "Q4",
        "category": "Erasmus",
        "question": "Erasmus öğrenim hareketliliğinde hibe nasıl hesaplanır?",
        "expected_answer": "Erasmus+ hibesi Ulusal Ajans (UA) tarafından her akademik yıl belirlenen miktarlar üzerinden hesaplanır. Hibe ödemesi iki taksit olarak yapılır. Öğrenciler hibe almadan önce üniversiteyle bir Öğrenci Sözleşmesi imzalar. Nihai ödeme öğrencinin karşı kurumda kaldığı gün sayısına göre yapılır.",
        "source": "IIRO-C420-01 Değişim Programları Kapsamında Giden Öğrenci Yönergesi"
    },
    
    # =================== LIBRARY ===================
    {
        "id": "Q5",
        "category": "Library",
        "question": "Kütüphaneden kaç kitap ödünç alabilirim ve süresi ne kadar?",
        "expected_answer": "Öğrenciler (lisans, lisansüstü, değişim) 60 gün süre ile 60 adet kitap ödünç alabilir. Ayrıca 5 multimedya kaynağı 7 gün, 2 ciltli süreli yayın 7 gün ve 5 popüler dergi 3 gün süreliğine ödünç alınabilir.",
        "source": "IIC-C840-02 Ödünç Verme ve Yararlanma Yönergesi"
    },
    {
        "id": "Q6",
        "category": "Library",
        "question": "Kütüphanelerarası ödünç alma (ILL) hizmeti nasıl çalışır?",
        "expected_answer": "Kütüphanelerarası ödünç alma (ILL) hizmeti ile Bilgi Merkezi koleksiyonunda bulunmayan kitap ve makaleler diğer kütüphanelerden temin edilebilir. Başvuru Bilgi Merkezi üzerinden yapılır.",
        "source": "IIC-C820-01 Kütüphanelerarası Ödünç Yönergesi"
    },
    
    # =================== DISCIPLINE ===================
    {
        "id": "Q7",
        "category": "Discipline",
        "question": "Öğrenci disiplin cezaları nelerdir?",
        "expected_answer": "Uyarma, kınama, 1 haftadan 1 aya kadar uzaklaştırma, 1-2 yarıyıl uzaklaştırma, yükseköğretim kurumundan çıkarma.",
        "source": "ISR-C210-01 Öğrenci Disiplin Yönergesi"
    },
    {
        "id": "Q8",
        "category": "Discipline",
        "question": "Disiplin soruşturması ne kadar sürede sonuçlanmalı?",
        "expected_answer": "Disiplin soruşturmasında karar en geç 10 gün içinde bildirilmelidir. Disiplin cezası gerektiren fiillerin işlendiği tarihten itibaren 2 yıl geçmesi halinde zamanaşımı oluşur.",
        "source": "PSR-C210-0101"
    },
    {
        "id": "Q9",
        "category": "Discipline",
        "question": "Kopya çekmek hangi disiplin cezasını gerektirir?",
        "expected_answer": "Sınavlarda kopyaya teşebbüs etmek kınama cezası gerektirir. Kopya çekmek veya çektirmek bir yarıyıl uzaklaştırma cezası gerektirir. Tehditle kopya çekmek, kopya çeken öğrencilerin sınav salonundan çıkarılmasına engel olmak veya başkasının yerine sınava girmek iki yarıyıl uzaklaştırma cezası gerektirir.",
        "source": "ISR-C210-01 Öğrenci Disiplin Yönergesi Madde 3"
    },
    
    # =================== SCHOLARSHIPS & FINANCIAL ===================
    {
        "id": "Q10",
        "category": "Scholarship",
        "question": "Burs başvurusu nasıl yapılır?",
        "expected_answer": "Burs başvurusu ilan edilen tarihlerde ÖBS (Öğrenci Bilgi Sistemi) üzerinden online olarak yapılır.",
        "source": "PSR-C160-0101"
    },
    {
        "id": "Q11",
        "category": "Scholarship",
        "question": "Burs devam şartları nelerdir?",
        "expected_answer": "Burs devam şartları bursun türüne göre değişir. Üstün Akademik Başarı Bursu için GNO en az 3.00, ilk yıl 34 SÜ / sonraki yıllar 30 SÜ kredi gerekir. Akademik Başarı ve İhtiyaç Bursu için GNO en az 2.50 ve 34 SÜ kredi gerekir. İlk giriş bursları akademik başarı durumuna bakılmaksızın normal öğrenim süresince devam eder. Disiplin cezalarında: kınama ve kısa süreli uzaklaştırmada burs devam eder, 1-2 yarıyıl uzaklaştırmada o dönem kesilir ama sonra yeniden bağlanır, çıkarmada tamamen kesilir.",
        "source": "ISR-C160-01 Burs ve Mali Destek Yönergesi"
    },
    
    # =================== GRADUATE PROGRAMS ===================
    {
        "id": "Q12",
        "category": "Graduate",
        "question": "Yüksek lisans programı kaç yarıyıl sürer?",
        "expected_answer": "Tezli yüksek lisans en fazla 6 yarıyıl, tezsiz yüksek lisans en fazla 3 yarıyıldır.",
        "source": "Lisansüstü Yönetmeliği"
    },
    {
        "id": "Q13",
        "category": "Graduate",
        "question": "Doktora yeterlik sınavı ne zaman yapılır?",
        "expected_answer": "Ders dönemini tamamladıktan sonra doktora yeterlik sınavına girilir.",
        "source": "Lisansüstü Yönetmeliği"
    },
    {
        "id": "Q14",
        "category": "Graduate",
        "question": "Tez savunması için jüri kaç kişiden oluşur?",
        "expected_answer": "Yüksek lisans tez savunma jürisi, biri tez danışmanı ve en az biri üniversite dışından olmak üzere üç veya beş öğretim üyesinden oluşur. Doktora tez savunma jürisi, danışman dahil beş öğretim üyesinden oluşur ve en az ikisi başka bir yükseköğretim kurumunun öğretim üyesi olmalıdır.",
        "source": "Lisansüstü Yönetmeliği Madde 33 ve Madde 38"
    },
    
    # =================== UNDERGRADUATE ===================
    {
        "id": "Q15",
        "category": "Undergraduate",
        "question": "Yatay geçiş başvurusu için GNO şartı nedir?",
        "expected_answer": "Yatay geçiş başvurusu için başvuru sırasında bir yükseköğretim kurumunda öğrenci statüsünde kayıtlı olmak, ilişiği kesilmemiş olmak ve İngilizce dil yeterliliğini sağlamak gerekir. Ayrıca ÖSYM puanının taban puanına eşit veya yüksek olması şartı aranır.",
        "source": "Lisans Yönetmeliği Madde 9"
    },
    {
        "id": "Q16",
        "category": "Undergraduate",
        "question": "Çift anadal programına nasıl başvurulur?",
        "expected_answer": "Çift anadal programına başvuru için GNO en az 3.20 olmalı ve öğrenci sınıfının ilk %20'sinde yer almalıdır. Tüm dersleri geçmiş olmalıdır. Başvuru en erken 2. dönem, en geç 4. dönemde yapılabilir. En fazla bir diploma programına daha kayıt yaptırılabilir.",
        "source": "ISR-C290-02 Çift Anadal Yönergesi ve Lisans Yönetmeliği Madde 34"
    },
    {
        "id": "Q17",
        "category": "Undergraduate",
        "question": "Ders ekleme-bırakma süresi ne kadar?",
        "expected_answer": "Ders ekleme-bırakma işlemi, sonbahar ve ilkbahar dönemlerinde derslerin başladığı haftayı takip eden ikinci hafta içinde, akademik takvimde belirtilen tarihlerde yapılır.",
        "source": "Akademik Takvim"
    },
    
    # =================== REGISTRATION & GRADUATION ===================
    {
        "id": "Q18",
        "category": "Registration",
        "question": "Mezuniyet başvurusu nasıl yapılır?",
        "expected_answer": "Mezuniyet başvurusu ÖBS üzerinden yapılır.",
        "source": "PSR-C240 Mezuniyet Prosedürü"
    },
    {
        "id": "Q19",
        "category": "Registration",
        "question": "Kayıt dondurma şartları nelerdir?",
        "expected_answer": "Dönem izni (kayıt dondurma); sağlık, maddi, aile, kişisel, akademik ve beklenmedik zorunlu olaylar gibi nedenlerle, ayrıca askerlik, gözaltı, tutukluluk veya mahkûmiyet durumlarında verilebilir. İzin gerekçesine ilişkin belgeler eklenerek dilekçe ile derslerin başlamasını takip eden 4. haftanın son iş gününe kadar ilgili fakülte dekanlığına başvurulur. Bir defada en çok 2 dönem, toplam 4 dönem izin verilebilir.",
        "source": "Lisans Yönetmeliği Madde 39-41, ISR-C210-02"
    },
    
    # =================== ENGLISH QUESTIONS ===================
    {
        "id": "Q20",
        "category": "English",
        "question": "What is the minimum GPA requirement for Erasmus?",
        "expected_answer": "Minimum GPA is 2.20 for undergrad and 2.5 for graduate students.",
        "source": "International Internship Instruction"
    },
    {
        "id": "Q21",
        "category": "English",
        "question": "How many books can I borrow from the library?",
        "expected_answer": "Undergraduate, graduate, and exchange students can borrow 60 books for 60 days, 5 multimedia items for 7 days, and 2 bound periodicals for 7 days.",
        "source": "Library Lending Policy"
    },
    {
        "id": "Q22",
        "category": "English",
        "question": "What are the disciplinary penalties for students?",
        "expected_answer": "Warning, reprimand, suspension, and expulsion from university.",
        "source": "Student Discipline Instruction"
    },
    
    # =================== SPECIFIC NUMERIC QUESTIONS ===================
    {
        "id": "Q23",
        "category": "Numeric",
        "question": "Lisans mezuniyeti için kaç kredi gerekiyor?",
        "expected_answer": "Lisans mezuniyeti için kayıtlı olunan diploma programının gerektirdiği tüm mezuniyet yükümlülüklerinin tamamlanması ve SÜ kredilerine göre hesaplanan genel not ortalamasının en az 2.00 olması gerekir.",
        "source": "Lisans Yönetmeliği Madde 35"
    },
    {
        "id": "Q24",
        "category": "Numeric",
        "question": "Bir dersin kaç kez tekrar edilebilir?",
        "expected_answer": "Geçer not alınan dersler, notun alındığı dönemi izleyen en çok 3 dönem içinde tekrar edilebilir (izinli dönemler ve yaz dönemleri hariç). Bu bir zaman sınırıdır, tekrar sayısı sınırı değildir. Başarısız olunan zorunlu dersler ise mezuniyete kadar tekrar edilerek başarılmalıdır.",
        "source": "Lisans Yönetmeliği Madde 30"
    },
    
    # =================== PROCEDURAL QUESTIONS ===================
    {
        "id": "Q25",
        "category": "Procedure",
        "question": "Staj başvurusu nasıl yapılır?",
        "expected_answer": "Staj başvurusu Kariyer Geliştirme Merkezi üzerinden yapılır.",
        "source": "Staj Yönergesi"
    },
    {
        "id": "Q26",
        "category": "Procedure",
        "question": "Transkript nasıl alınır?",
        "expected_answer": "Transkript, MySU'daki online belge talep formu doldurularak Öğrenci Kaynakları'ndan (ÖK) talep edilir. ÖK raporlama yazılımı ile hazırlar. Basılı kopya ücretlidir, e-imzalı transkript ücretsizdir. Transkriptte öğrencinin tüm dersleri, kodları, SÜ ve AKTS kredileri, notları, DNO ve GNO bilgileri yer alır.",
        "source": "PSR-C230-0103 Transkript Düzenleme Prosedürü"
    },
    {
        "id": "Q27",
        "category": "Procedure",
        "question": "Öğrenci belgesi nereden alınır?",
        "expected_answer": "Öğrenci belgesi, Öğrenci Kaynakları (ÖK) birimi tarafından hazırlanır. MySU'daki online Belge Talep Formu doldurularak başvuru yapılır. Belge 2 iş günü içinde hazırlanır ve en fazla 2 hafta muhafaza edilir; bu sürede teslim alınmazsa imha edilir.",
        "source": "PSR-C210-0402 Öğrenci Belgesi Düzenleme Prosedürü"
    },
    
    # =================== EDGE CASES ===================
    {
        "id": "Q28",
        "category": "Edge",
        "question": "Cinsel taciz şikayeti nasıl yapılır?",
        "expected_answer": "Cinsel taciz şikayeti Cinsel Tacize Karşı Önlem ve Destek Komitesi'ne telefon, e-posta, yüz yüze görüşme veya yazılı olarak yapılabilir. Başvuranın ispatla yükümlülüğü yoktur.",
        "source": "Cinsel Taciz Yönergesi"
    },
    {
        "id": "Q29",
        "category": "Edge",
        "question": "İtiraz süresi ne kadar?",
        "expected_answer": "Disiplin kararlarına itiraz süresi, cezanın tebliğ tarihinden itibaren 15 gündür. İdari yargı yoluna başvuru süresi 60 gündür.",
        "source": "Disiplin Prosedürü"
    },
    {
        "id": "Q30",
        "category": "Edge",
        "question": "Sabancı Üniversitesi nerede?",
        "expected_answer": "Sabancı Üniversitesi İstanbul Tuzla'da bulunmaktadır.",
        "source": "Genel Bilgi"
    },
]


# ─────────────────────────────────────────────────────────────────────────────
# Helper: wraps text at width chars with a leading indent
# ─────────────────────────────────────────────────────────────────────────────

def _wrap(text: str, width: int = 74, indent: str = "   ") -> str:
    """Word-wrap *text* and prefix each line with *indent*."""
    import textwrap
    return "\n".join(
        indent + line
        for line in textwrap.wrap(text, width=width) or [""]
    )


def _build_context_from_chunks(chunks):
    """Convert a list of chunk dicts (from search_only) to a context string."""
    parts = []
    for i, c in enumerate(chunks, 1):
        meta = c.get("meta", {}) or {}
        title   = meta.get("title", "")
        section = meta.get("section_header", "")
        text    = c.get("text", "")
        header  = f"[{i}] {title}" + (f" | {section}" if section else "")
        parts.append(header + "\n" + text)
    return "\n\n-----\n\n".join(parts)


# ─────────────────────────────────────────────────────────────────────────────
# MAIN TEST RUNNER  (multi-metric)
# ─────────────────────────────────────────────────────────────────────────────

def run_detailed_test(use_evaluator: bool = True):
    """
    Run all TEST_QUESTIONS through the RAG pipeline.

    When *use_evaluator* is True (default), the Evaluator class is used to
    compute Answer Similarity, Faithfulness, Relevance, Factual Accuracy, and
    a composite score.  Keyword coverage is also computed for side-by-side
    comparison with the legacy metric.

    When *use_evaluator* is False, only legacy keyword coverage is computed
    (original behaviour).
    """

    print("=" * 80)
    mode_label = "MULTI-METRIC" if use_evaluator else "LEGACY KEYWORD-COVERAGE"
    print(f"🔬 RAG EVALUATION — {mode_label} MODE")
    print(f"📊 Total Questions: {len(TEST_QUESTIONS)}")
    print("=" * 80)

    # ── Load RAG Pipeline ────────────────────────────────────────────────────
    try:
        from services.pipeline.rag_pipeline import RAGPipeline
        from services.config.settings import RAGConfig
        config = RAGConfig.from_env()
        pipeline = RAGPipeline(config)
        print(f"🤖 LLM model  : {config.ollama.chat_model}")
        print(f"🔢 Embed model: {config.ollama.embed_model}")
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return

    print(f"✅ ChromaDB connected : {pipeline.vector_store.count} chunks")
    print(f"✅ BM25 loaded        : {pipeline.bm25_service.document_count} docs\n")

    # ── Load Evaluator ───────────────────────────────────────────────────────
    evaluator = None
    if use_evaluator:
        try:
            from testing.evaluator import Evaluator
        except ImportError:
            try:
                from evaluator import Evaluator
            except ImportError:
                print("⚠️  evaluator.py not found – falling back to legacy mode")
                use_evaluator = False

        if use_evaluator:
            evaluator = Evaluator(
                ollama_host=config.ollama.host,
                embed_model=config.ollama.embed_model,
                chat_model=config.ollama.chat_model,
                pass_threshold=0.70,
            )
            print(f"✅ Evaluator ready (ollama_embed={evaluator._ollama_embed_ok}, "
                  f"ollama_llm={evaluator._ollama_llm_ok})\n")

    # ── Result containers ────────────────────────────────────────────────────
    eval_results   = []   # EvaluationResult objects (new path)
    legacy_results = []   # plain dicts             (old path / comparison)

    total_start = time.time()

    for i, test in enumerate(TEST_QUESTIONS, 1):
        qid      = test["id"]
        category = test.get("category", "Other")
        question = test["question"]
        expected = test["expected_answer"]

        print("=" * 80)
        print(f"📝 TEST {i}/{len(TEST_QUESTIONS)}: {qid} [{category}]")
        print("=" * 80)
        print(f"\n❓ QUESTION:")
        print(_wrap(question))
        print(f"\n📚 SOURCE   : {test['source']}")
        print(f"\n✅ EXPECTED :")
        print(_wrap(expected))

        # ── Run pipeline ─────────────────────────────────────────────────────
        t0 = time.time()
        try:
            answer  = pipeline.answer(question, history=[])
            latency = time.time() - t0
        except Exception as e:
            latency = time.time() - t0
            print(f"\n❌ PIPELINE ERROR: {e}")
            legacy_results.append({"id": qid, "category": category, "error": str(e)})
            print()
            continue

        print(f"\n🤖 RAG ANSWER ({latency:.1f}s):")
        print("-" * 60)
        print(_wrap(answer, width=74))
        print("-" * 60)

        # ── Legacy keyword coverage (always computed for comparison) ──────────
        kws      = expected.lower().split()
        ans_low  = answer.lower()
        kw_found = sum(1 for kw in kws if kw in ans_low)
        kw_cov   = kw_found / len(kws) * 100 if kws else 0

        print(f"\n📊 Keyword coverage (legacy): {kw_cov:.0f}%")

        legacy_entry = {
            "id": qid,
            "category": category,
            "question": question,
            "expected": expected,
            "actual": answer,
            "latency": latency,
            "coverage": kw_cov,
        }
        legacy_results.append(legacy_entry)

        # ── Multi-metric evaluation ───────────────────────────────────────────
        if evaluator is not None:
            # Get context chunks for faithfulness scoring
            try:
                chunks  = pipeline.search_only(question, top_k=10)
                context = _build_context_from_chunks(chunks)
            except Exception:
                context = ""

            result = evaluator.evaluate(
                question_id=qid,
                category=category,
                question=question,
                answer=answer,
                expected=expected,
                context=context,
                latency=latency,
            )
            eval_results.append(result)

            s = result.scores
            print(f"\n📐 Multi-metric scores:")
            print(f"   Answer Similarity : {s.answer_similarity:.3f}")
            print(f"   Faithfulness      : {s.faithfulness:.3f}")
            print(f"   Answer Relevance  : {s.answer_relevance:.3f}")
            print(f"   Factual Accuracy  : {s.factual_accuracy:.3f}")
            print(f"   ─────────────────────────────────────────")
            comp_icon = "✅" if s.composite_score >= 0.70 else "❌"
            print(f"   Composite Score   : {s.composite_score:.3f}  {comp_icon}")
            kw_icon   = "✅" if kw_cov >= 50 else "❌"
            print(f"   Keyword Coverage  : {kw_cov:.0f}%  {kw_icon}  (legacy)")

        print()

    total_elapsed = time.time() - total_start

    # ─────────────────────────────────────────────────────────────────────────
    # LEGACY SUMMARY  (always shown for comparison)
    # ─────────────────────────────────────────────────────────────────────────
    valid_legacy = [r for r in legacy_results if "coverage" in r]

    print("=" * 80)
    print("📊 LEGACY SUMMARY (keyword coverage)")
    print("=" * 80)
    cat_kw: dict = {}
    for r in valid_legacy:
        c = r["category"]
        cat_kw.setdefault(c, []).append(r["coverage"])
        status = "✅" if r["coverage"] >= 50 else "⚠️"
        print(f"  {status} {r['id']:<6} [{r['category']:<14}]  "
              f"{r['coverage']:>5.0f}%  ({r['latency']:.1f}s)")

    if valid_legacy:
        avg_kw  = sum(r["coverage"] for r in valid_legacy) / len(valid_legacy)
        old_pass = sum(1 for r in valid_legacy if r["coverage"] >= 50)
        print(f"\n  Average: {avg_kw:.0f}%  |  Passed (≥50%): {old_pass}/{len(valid_legacy)}")

    # ─────────────────────────────────────────────────────────────────────────
    # MULTI-METRIC REPORT
    # ─────────────────────────────────────────────────────────────────────────
    if evaluator is not None and eval_results:
        test_results_dir = os.path.join(os.path.dirname(__file__), "test_results")
        evaluator.generate_report(eval_results, output_dir=test_results_dir)

    print(f"\n⏱️  Total wall-clock time: {total_elapsed:.0f}s")

    # ── Close output file ─────────────────────────────────────────────────────
    output_file.close()
    sys.stdout = original_stdout
    sys.stderr = original_stderr

    return eval_results if evaluator is not None else legacy_results


# ─────────────────────────────────────────────────────────────────────────────
# SINGLE-QUESTION MODE
# ─────────────────────────────────────────────────────────────────────────────

def run_single_question(question: str, use_evaluator: bool = True):
    """Run a single question; optionally score with the Evaluator."""

    print("=" * 80)
    print(f"❓ Question: {question}")
    print("=" * 80)

    from services.pipeline.rag_pipeline import RAGPipeline
    from services.config.settings import RAGConfig

    config   = RAGConfig.from_env()
    pipeline = RAGPipeline(config)

    t0      = time.time()
    answer  = pipeline.answer(question, history=[])
    latency = time.time() - t0

    print(f"\n🤖 ANSWER ({latency:.1f}s):")
    print("-" * 60)
    print(answer)
    print("-" * 60)

    if use_evaluator:
        try:
            from testing.evaluator import Evaluator
        except ImportError:
            from evaluator import Evaluator

        evaluator = Evaluator(
            ollama_host=config.ollama.host,
            embed_model=config.ollama.embed_model,
            chat_model=config.ollama.chat_model,
        )
        chunks  = pipeline.search_only(question, top_k=10)
        context = _build_context_from_chunks(chunks)

        result = evaluator.evaluate(
            question_id="Q?",
            category="adhoc",
            question=question,
            answer=answer,
            expected="",   # no ground-truth in ad-hoc mode
            context=context,
            latency=latency,
        )
        s = result.scores
        print(f"\n📐 Scores:")
        print(f"   Faithfulness   : {s.faithfulness:.3f}")
        print(f"   Answer Relevance: {s.answer_relevance:.3f}")
        print(f"   Factual Accuracy: {s.factual_accuracy:.3f}  (no ground-truth)")
        print(f"   Composite       : {s.composite_score:.3f}")

    output_file.close()
    sys.stdout = original_stdout
    sys.stderr = original_stderr


# ─────────────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="MACKIS RAG Evaluation Suite"
    )
    parser.add_argument("-q", "--question", type=str, help="Ask a single question")
    parser.add_argument(
        "--no-eval", dest="no_eval", action="store_true",
        help="Disable multi-metric evaluator; use legacy keyword coverage only",
    )
    args = parser.parse_args()

    use_eval = not args.no_eval

    if args.question:
        run_single_question(args.question, use_evaluator=use_eval)
    else:
        run_detailed_test(use_evaluator=use_eval)
