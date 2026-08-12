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

output_file_path = os.path.join(os.path.dirname(__file__), "rag_test_output35_qwen3_80b_moe_judge_gpt4mini.txt")
# Check for --append flag early to decide file open mode
_append_mode = "--append" in sys.argv
output_file = open(output_file_path, "a" if _append_mode else "w", encoding="utf-8")
original_stdout = sys.stdout
original_stderr = sys.stderr
sys.stdout = Tee(original_stdout, output_file)
sys.stderr = Tee(original_stderr, output_file)

from dotenv import load_dotenv
load_dotenv()

import math
import re

def compute_retrieval_metrics(expected_source: str, retrieved_chunks: list, top_k: int = 10, expected_answer: str = "") -> dict:
    """Computes retrieval metrics using fuzzy document-title matching.
    
    Instead of regex code extraction (which fails for sources like 
    'Lisansüstü Yönetmeliği Madde 32'), this uses keyword overlap 
    between the expected source description and each chunk's title/path.
    
    Also computes Context Recall: what fraction of expected answer 
    keywords appear in the retrieved chunks' text (CS 455/555 §8.6).
    """
    if not retrieved_chunks:
        return {"Hit@K": 0.0, "MRR@K": 0.0, "MAP@K": 0.0, "nDCG@K": 0.0, "ContextRecall": 0.0}
    
    # -- Build matching keywords from expected source --
    # Extract document codes if present (e.g., "IID-C710-02", "PSR-C210-0101")
    codes = re.findall(r'[A-Z]{2,6}-[A-Z]?\d+-\d+', expected_source)
    # Extract meaningful words (4+ chars, Turkish-aware)
    source_words = set(re.findall(r'[a-zA-ZçğıöşüÇĞİÖŞÜ]{4,}', expected_source.lower()))
    # Remove very generic words that would match too broadly
    generic = {'için', 'olan', 'veya', 'madde', 'from', 'with', 'that', 'this'}
    source_words -= generic
    
    rel_array = []
    for i, c in enumerate(retrieved_chunks[:top_k]):
        meta = c.get("meta", {}) or {}
        source_path = (meta.get("source_path", "") or "").lower()
        title = (meta.get("title", "") or "").lower()
        search_text = source_path + " " + title
        
        is_relevant = False
        
        # Method 1: Direct code matching (high confidence)
        if codes:
            for code in codes:
                if code.lower() in search_text:
                    is_relevant = True
                    break
        
        # Method 2: Keyword overlap (fuzzy matching)
        if not is_relevant and source_words:
            matched_words = sum(1 for w in source_words if w in search_text)
            # Require at least 40% of source words to match
            if len(source_words) > 0 and matched_words / len(source_words) >= 0.40:
                is_relevant = True
            # Also check if at least 2 specific words match for short sources
            elif matched_words >= 2:
                is_relevant = True
        
        rel_array.append(1 if is_relevant else 0)
    
    # -- Standard IR metrics --
    hit_at_k = 1.0 if any(rel_array) else 0.0
    
    mrr_at_k = 0.0
    for i, rel in enumerate(rel_array):
        if rel == 1:
            mrr_at_k = 1.0 / (i + 1)
            break
            
    num_relevant = sum(rel_array)
    map_at_k = 0.0
    if num_relevant > 0:
        precisions = []
        hits = 0
        for i, rel in enumerate(rel_array):
            if rel == 1:
                hits += 1
                precisions.append(hits / (i + 1))
        map_at_k = sum(precisions) / num_relevant
        
    dcg = 0.0
    for i, rel in enumerate(rel_array):
        if rel == 1:
            dcg += 1.0 / math.log2(i + 2)
            
    idcg = 0.0
    for i in range(num_relevant):
        idcg += 1.0 / math.log2(i + 2)
        
    ndcg_at_k = dcg / idcg if idcg > 0 else 0.0
    
    # -- Context Recall (CS 455/555 §8.6) --
    # Measures what fraction of expected answer facts appear in retrieved text
    context_recall = 0.0
    if expected_answer:
        answer_keywords = set(re.findall(
            r'[a-zA-ZçğıöşüÇĞİÖŞÜ]{4,}',
            expected_answer.lower()
        ))
        # Remove stopwords
        stopwords = {'için', 'olan', 'veya', 'daha', 'kadar', 'bile', 'olan',
                     'with', 'from', 'that', 'this', 'they', 'their', 'also',
                     'have', 'does', 'olarak', 'ancak', 'ayrıca'}
        answer_keywords -= stopwords
        
        if answer_keywords:
            # Combine all retrieved chunk text
            all_chunk_text = " ".join(
                (c.get("text", "") or "").lower()
                for c in retrieved_chunks[:top_k]
            )
            found_keywords = sum(1 for kw in answer_keywords if kw in all_chunk_text)
            context_recall = found_keywords / len(answer_keywords)
    
    return {
        "Hit@K": hit_at_k,
        "MRR@K": mrr_at_k,
        "MAP@K": map_at_k,
        "nDCG@K": ndcg_at_k,
        "ContextRecall": round(context_recall, 3),
    }

# Test questions with expected answers from documents
# Organized by category for comprehensive coverage
# 90 questions across 18 categories
TEST_QUESTIONS = [
    # =================== ERASMUS & INTERNATIONAL (6) ===================
    {
        "id": "Q1",
        "category": "Erasmus",
        "question": "Erasmus staj programına başvurmak için minimum GNO ne olmalı?",
        "expected_answer": "Lisans için en az 2.20, Lisansüstü için en az 2.5 GNO gereklidir.",
        "source": "IID-C710-02 Uluslararası Staj Yönergesi §3.3"
    },
    {
        "id": "Q2",
        "category": "Erasmus", 
        "question": "Erasmus staj hareketliliği minimum süresi ne kadar?",
        "expected_answer": "Staj hareketliliği süresi minimum 2 aydır.",
        "source": "IID-C710-02 Uluslararası Staj Yönergesi §3.9"
    },
    {
        "id": "Q3",
        "category": "Erasmus",
        "question": "How can I apply for Erasmus internship program?",
        "expected_answer": "Students submit their application documents (CV, transcript) together with a minimum 2-month acceptance letter from the host organization to the Internship Office. Minimum GPA requirement is 2.20 for undergraduates and 2.5 for graduate students.",
        "source": "PIPAR-C710-0201 Erasmus Internship Mobility Procedure §3.2"
    },
    {
        "id": "Q4",
        "category": "Erasmus",
        "question": "Erasmus öğrenim hareketliliğinde hibe nasıl hesaplanır?",
        "expected_answer": "Erasmus+ hibesi Ulusal Ajans (UA) tarafından her akademik yıl belirlenen aylık hibe miktarı üzerinden hesaplanır. Hibe = ay sayısı × aylık birim masraf + (tamamlanmayan ayların gün sayısı × aylık birim masrafın 1/30'u). Öğrenciler hibe almadan önce Hibe Sözleşmesi imzalar ve ilk Öğrenim Anlaşması'nı tamamlar. Nihai ödeme öğrencinin karşı kurumda kaldığı gün sayısına göre yapılır.",
        "source": "IIRO-C420-01 Değişim Programları Giden Öğrenci Yönergesi §7"
    },
    {
        "id": "Q5",
        "category": "Erasmus",
        "question": "Erasmus Kısa Dönem Doktora Hareketliliği ne kadar sürer?",
        "expected_answer": "Erasmus Kısa Dönem Doktora Hareketliliği 5 ila 30 gün arasında sürer.",
        "source": "PIRO-C420-0105 Erasmus Kısa Dönem Doktora Hareketliliği"
    },
    {
        "id": "Q6",
        "category": "Erasmus",
        "question": "How does the Learning Agreement process work for exchange students?",
        "expected_answer": "Students create a Learning Agreement (LA) specifying courses to be taken abroad. The Exchange Academic Advisor verifies course equivalencies (credits, types, ECTS). For double major students, separate approval from the DDPO program coordinator is needed. Minor program equivalencies also require advisor approval. The signed LA is submitted to the International Relations Office (IRO), which forwards it to the partner university. Changes during the exchange period require completing a Change Request in the LA system.",
        "source": "PIRO-C420-0102 Procedure for Learning Agreement"
    },
    
    # =================== LIBRARY (6) ===================
    {
        "id": "Q7",
        "category": "Library",
        "question": "Kütüphaneden kaç kitap ödünç alabilirim ve süresi ne kadar?",
        "expected_answer": "Öğrenciler (lisans, lisansüstü, değişim) 60 gün süre ile 60 adet kitap ödünç alabilir. Ayrıca 5 multimedya kaynağı 7 gün, 2 ciltli süreli yayın 7 gün ve 5 popüler dergi 3 gün süreliğine ödünç alınabilir.",
        "source": "IIC-C840-02 Ödünç Verme ve Yararlanma Yönergesi §1.3"
    },
    {
        "id": "Q8",
        "category": "Library",
        "question": "Kütüphanelerarası ödünç alma (ILL) hizmeti nasıl çalışır?",
        "expected_answer": "Kütüphanelerarası ödünç alma (ILL) hizmeti ile Bilgi Merkezi koleksiyonunda bulunmayan kitap ve makaleler diğer kütüphanelerden temin edilebilir. Başvuru 'Ödünç Kitap İstek Formu' ile Bilgi Merkezi üzerinden yapılır. İstekler aynı anda lisans öğrencileri için en çok 3, lisansüstü öğrencileri için 5 bilgi kaynağı ile sınırlıdır. Süresinde iade edilmeyen kaynaklar için 2 kez uyarı yapılır, aksi halde 1 yıl hizmetten men edilir.",
        "source": "IIC-C820-01 Kütüphanelerarası Ödünç Yönergesi"
    },
    {
        "id": "Q9",
        "category": "Library",
        "question": "What happens if I return library books late?",
        "expected_answer": "The qualified membership right to benefit from the circulation service of Sabancı University Alumni, Retirees and Families who do not pay the fees will be revoked and their subsequent applications will not be accepted for 1 year. Reminders are sent for returned items. Students with overdue fines are blocked from borrowing services until fees are paid.",
        "source": "IIC-C840-02 Circulation and Utilization Instruction Letter §4"
    },
    {
        "id": "Q10",
        "category": "Library",
        "question": "Who can access the Faculty Reserve Collection?",
        "expected_answer": "Only academic and teaching Emeritus employees can access and borrow the Faculty Reserve Collection. Renewals and reservations can be made. Students cannot access the Faculty Reserve Collection.",
        "source": "IIC-C840-02 Circulation and Utilization Instruction Letter §1.1"
    },
    {
        "id": "Q11",
        "category": "Library",
        "question": "Doküman sağlama hizmetinden kimler faydalanabilir?",
        "expected_answer": "Doküman sağlama hizmetinden Sabancı Üniversitesi mensupları (akademik, idari çalışanlar, öğrenciler, mezunlar, emekliler) faydalanabilir. Üniversite lojmanlarında oturanların aileleri ve dış kullanıcılar da maliyet bedelini ödeyerek yararlanabilir. SuNet kullanıcı adı gerekir. Akademik ve emeritus çalışanlara yurtiçinden 50, yurtdışından 15 adet ücretsiz doküman kontenjanı verilir.",
        "source": "IIC-C820-05 Doküman Sağlama Yönergesi"
    },
    {
        "id": "Q12",
        "category": "Library",
        "question": "What are the borrowing limits for alumni at the library?",
        "expected_answer": "Alumni and retirees can borrow 2 books for 30 days, 1 multimedia item for 7 days, 1 bound periodical for 7 days, and 5 popular magazines for 3 days.",
        "source": "IIC-C840-02 Circulation and Utilization Instruction §1.5"
    },

    # =================== DISCIPLINE (6) ===================
    {
        "id": "Q13",
        "category": "Discipline",
        "question": "Öğrenci disiplin cezaları nelerdir?",
        "expected_answer": "Uyarma, kınama, 1 haftadan 1 aya kadar uzaklaştırma, 1-2 yarıyıl uzaklaştırma, yükseköğretim kurumundan çıkarma.",
        "source": "ISR-C210-01 Öğrenci Disiplin Yönergesi §3 Madde 1"
    },
    {
        "id": "Q14",
        "category": "Discipline",
        "question": "Disiplin soruşturması ne kadar sürede sonuçlanmalı?",
        "expected_answer": "Genel disiplin cezaları (uyarma, kınama, kısa süreli uzaklaştırma) için soruşturma tamamlandıktan sonra en geç 10 gün içinde karar verilir. Cinsel taciz fiillerinde soruşturma fiilin öğrenilmesinden itibaren en geç 3 ay içinde başlar ve eylemin tarihinden itibaren 2 yıl içinde tamamlanır. Can ve mal güvenliği tehdit eden durumlarda çıkarma cezası 24 saat içinde sonuçlanabilir.",
        "source": "PSR-C210-0101 Disiplin Prosedürü §1.6, §2.5"
    },
    {
        "id": "Q15",
        "category": "Discipline",
        "question": "Kopya çekmek hangi disiplin cezasını gerektirir?",
        "expected_answer": "Sınavlarda kopyaya teşebbüs etmek kınama cezası gerektirir. Kopya çekmek veya çektirmek bir yarıyıl uzaklaştırma cezası gerektirir. Tehditle kopya çekmek, kopya çeken öğrencilerin sınav salonundan çıkarılmasına engel olmak veya başkasının yerine sınava girmek iki yarıyıl uzaklaştırma cezası gerektirir.",
        "source": "ISR-C210-01 Öğrenci Disiplin Yönergesi Madde 3"
    },
    {
        "id": "Q16",
        "category": "Discipline",
        "question": "How does the disciplinary objection process work?",
        "expected_answer": "In the event of an objection, the University Administrative Board examines the decision and accepts or rejects the sanction within fifteen days from the notification of punishment. If the student chooses to make an appeal at the administrative jurisdiction and a stay of execution decision comes from the administrative board, SR notifies suspension of execution of the disciplinary punishment.",
        "source": "PSR-C210-0101 Disciplinary Procedure §2.7, §3.2"
    },
    {
        "id": "Q17",
        "category": "Discipline",
        "question": "Disiplin cezasına itiraz süresi ne kadardır?",
        "expected_answer": "Disiplin cezasına itiraz süresi, cezanın tebliğinden itibaren 15 gündür. İtiraz, Üniversite Yönetim Kuruluna yapılır ve 15 gün içinde kesin karara bağlanır. İdari yargı yolu olarak idare mahkemesine 60 gün içinde başvurulabilir. Yurt kurallarına özgü disiplin cezaları için ise 5 iş günü itiraz süresi geçerlidir.",
        "source": "PSR-C210-0101 Disiplin Prosedürü §2.7"
    },
    {
        "id": "Q18",
        "category": "Discipline",
        "question": "Disiplin suçlarında zamanaşımı süreleri ne kadardır?",
        "expected_answer": "Uyarma, kınama ve kısa süreli uzaklaştırma cezası gerektiren fiillerde soruşturma zamanaşımı 1 aydır. 1-2 yarıyıl uzaklaştırma ve çıkarma cezası gerektiren fiillerde zamanaşımı 3 aydır. Tüm disiplin suçlarında mutlak zamanaşımı süresi 2 yıldır.",
        "source": "PSR-C210-0101 Disiplin Prosedürü §2.5"
    },
    
    # =================== SCHOLARSHIPS & FINANCIAL (5) ===================
    {
        "id": "Q19",
        "category": "Scholarship",
        "question": "Burs başvurusu nasıl yapılır?",
        "expected_answer": "Burs başvurusu, ilan edilen tarihlerde ÖBS üzerinden online olarak yapılır. Lisans öğrencileri başvuru formunu doldurur ve maddi gereksinimi kanıtlayan belgeler sunar. Lisansüstü öğrencileri program başvurusu sırasında burs talebini iletir. Burs değerlendirmeleri Güz dönemi için Ağustos, Bahar dönemi için Ocak ayında yapılır. Üstün Akademik Başarı Bursu için ayrıca başvuru yapılmasına gerek yoktur.",
        "source": "ISR-C160-01 Burs ve Mali Destek Yönergesi §2"
    },
    {
        "id": "Q20",
        "category": "Scholarship",
        "question": "Burs devam şartları nelerdir?",
        "expected_answer": "Burs devam şartları program türüne göre farklılık gösterir. Lisans bursları normal öğrenim süresi boyunca devam eder; mezun olma koşullarını yerine getiremeyen öğrencilerin burs süresi en fazla iki dönem uzatılabilir. Lisansüstü burslarında yüksek lisans için aylık nakit burs en fazla 10 ay (Eylül-Haziran), doktora için 12 ay (Eylül-Ağustos) ödenir. Burs devamı her akademik dönem başında Enstitü Yönetim Kurulu kararı ile değerlendirilir. Disiplin cezalarında: kınama ve kısa süreli uzaklaştırmada burs devam eder, 1-2 yarıyıl uzaklaştırmada o dönem kesilir ama sonra yeniden bağlanır, çıkarmada tamamen kesilir.",
        "source": "ISR-C160-01 Burs ve Mali Destek Yönergesi §2-4"
    },
    {
        "id": "Q21",
        "category": "Scholarship",
        "question": "What is the Sabancı University Main Endowment Scholarship Fund?",
        "expected_answer": "The Sabancı University Main Endowment Scholarship Fund is a financial aid initiative supporting students who demonstrate academic success and financial need. It is expanded through donations from SU employees, alumni, students, and international contributors via the 'Friends of Sabancı University Fund' under the Turkish Philanthropic Fund in New York. Employee donations are deducted from salaries at a self-determined rate. The Board of Trustees oversees governance, and IPAR manages donation acceptance procedures.",
        "source": "IFA-S360-01 Burs Anavarlık Fonu / PSR-C160-0201"
    },
    {
        "id": "Q22",
        "category": "Scholarship",
        "question": "Çalışma burslu öğrenci nasıl işe alınır?",
        "expected_answer": "Çalışma burslu öğrenci işe alımı SUform üzerinden ilan yayınlanarak başlar. Başvuran öğrencilerle mülakat yapılır ve Çalışma Burslu Öğrenci İşe Alım Değerlendirme Formu doldurulur. Sözleşme bir dönem geçerlidir, ayda en fazla 40 saat çalışılır. Seçilen öğrencilere oryantasyon verilir.",
        "source": "PCIAD-A320-01-02 / PTGP-A230-0101 Çalışma Burslu Öğrenci Prosedürü"
    },
    {
        "id": "Q23",
        "category": "Scholarship",
        "question": "Disiplin cezası alan öğrencinin bursu ne olur?",
        "expected_answer": "Disiplin cezasının bursa etkisi cezanın türüne göre değişir. Kınama ve kısa süreli uzaklaştırmada (1 hafta-1 ay) burs devam eder. 1-2 yarıyıl uzaklaştırma cezasında burs o dönem kesilir ancak ceza bitiminde yeniden bağlanır. Yükseköğretim kurumundan çıkarma cezasında burs tamamen kesilir.",
        "source": "ISR-C160-01 Burs Yönergesi §4"
    },

    # =================== GRADUATE PROGRAMS (6) ===================
    {
        "id": "Q24",
        "category": "Graduate",
        "question": "Yüksek lisans programı kaç yarıyıl sürer?",
        "expected_answer": "Tezli yüksek lisans programı, bilimsel hazırlık süresi hariç, en az dört dönem ve azami altı dönem sürer. Tezsiz yüksek lisans programı azami üç dönemdir. Lisans derecesiyle kabul edilen tezli yüksek lisans öğrencileri için azami altı dönem, tezli yüksek lisans derecesiyle kabul edilenler için azami dört dönem geçerlidir.",
        "source": "Lisansüstü Yönetmeliği Madde 32(2), 34(2)"
    },
    {
        "id": "Q25",
        "category": "Graduate",
        "question": "Doktora yeterlik sınavı ne zaman yapılır?",
        "expected_answer": "Derslerini ve seminerini tamamlayan öğrenciler doktora yeterlik sınavına bir yılda en fazla iki kez girebilir. Yüksek lisans derecesiyle kabul edilenler en geç beşinci, lisans derecesiyle kabul edilenler en geç yedinci dönemlerinin sonuna kadar bu sınavı tamamlamalıdır. Sınav jürileri beş öğretim üyesinden oluşur, en az ikisi Üniversite dışından olmalıdır.",
        "source": "Lisansüstü Yönetmeliği Madde 36(1)"
    },
    {
        "id": "Q26",
        "category": "Graduate",
        "question": "Tez savunması için jüri kaç kişiden oluşur?",
        "expected_answer": "Yüksek lisans tez savunma jürisi, biri tez danışmanı ve en az biri üniversite dışından olmak üzere üç veya beş öğretim üyesinden oluşur. Doktora tez savunma jürisi, danışman dahil beş öğretim üyesinden oluşur ve en az ikisi başka bir yükseköğretim kurumunun öğretim üyesi olmalıdır.",
        "source": "Lisansüstü Yönetmeliği Madde 33 ve Madde 38"
    },
    {
        "id": "Q27",
        "category": "Graduate",
        "question": "Lisansüstü programlara nasıl başvurulur?",
        "expected_answer": "Lisansüstü programlara başvuru online başvuru sistemi üzerinden yapılır. Adaylar transkript, diploma, referans mektupları, niyet mektubu ve İngilizce yeterlilik belgesini yükler. Başvurular ilgili program tarafından değerlendirilir, mülakatlar yapılabilir ve kabul kararı Enstitü Yönetim Kurulu tarafından onaylanır.",
        "source": "PSR-C120-0104 Procedure For Application And Admission To Graduate Programs"
    },
    {
        "id": "Q28",
        "category": "Graduate",
        "question": "Lisansüstü programlarda tekrarlanan derslerde GNO nasıl hesaplanır?",
        "expected_answer": "Tekrarlanan derslerde, daha önce alınmış olan dersin toplam puanı GNO hesaplarından düşürülür ve en son alınan dersin toplam puanı GNO hesaplamalarına katılır. Bu işlem sistem tarafından otomatik yapılır. Lisansüstü öğrenciler C, C+ veya B- aldıkları dersleri tekrar alabilir. Öğrencinin kayıt yaptırdığı tüm dersler transkriptte gösterilir.",
        "source": "Lisansüstü Yönetmeliği"
    },
    {
        "id": "Q29",
        "category": "Graduate",
        "question": "How does the graduate program opening/closing process work?",
        "expected_answer": "To open a graduate program, the Graduate School Board provides required information in YOKSIS, including courses, total credits, advisor requirements, and admission criteria. The relevant Faculty Board and Academic Board approve the proposal, which is then sent to the Board of Trustees. Programs are closed by cancelling new admissions, and enrolled students must be transferred or allowed to complete. The Council of Higher Education (YÖK) is notified.",
        "source": "PSR-C510-0102 Procedure For Opening And Closing a Graduate Degree Program"
    },

    # =================== UNDERGRADUATE (5) ===================
    {
        "id": "Q30",
        "category": "Undergraduate",
        "question": "Yatay geçiş başvuru şartları nelerdir?",
        "expected_answer": "Yatay geçiş başvurusu için hazırlık sınıfı dışında 2 dönemi en az 60/100 genel not ortalaması ile tamamlamış olmak gerekir. Başvuru sırasında bir yükseköğretim kurumunda öğrenci statüsünde kayıtlı olmak, ilişiği kesilmemiş olmak ve İngilizce dil yeterliliğini sağlamak gerekir. Ayrıca merkezi yerleştirme puanının taban puanına eşit veya yüksek olması şartı aranır.",
        "source": "PSR-C120-0103 Yatay Geçiş Prosedürü, ISR-C120-01 Yönergesi"
    },
    {
        "id": "Q31",
        "category": "Undergraduate",
        "question": "Çift anadal programına nasıl başvurulur?",
        "expected_answer": "Çift anadal programına başvuru için GNO en az 3.20 olmalı ve öğrenci sınıfının ilk %20'sinde yer almalıdır. Tüm dersleri geçmiş olmalıdır. Başvuru en erken 2. dönem, en geç 4. dönemde yapılabilir. En fazla bir diploma programına daha kayıt yaptırılabilir.",
        "source": "ISR-C290-02 Çift Anadal Yönergesi ve Lisans Yönetmeliği Madde 34"
    },
    {
        "id": "Q32",
        "category": "Undergraduate",
        "question": "Ders ekleme-bırakma süresi ne kadar?",
        "expected_answer": "Ders ekleme-bırakma işlemi, sonbahar ve ilkbahar dönemlerinde derslerin başladığı haftayı takip eden ikinci hafta içinde, akademik takvimde belirtilen tarihlerde yapılır.",
        "source": "Akademik Takvim / Lisans Yönetmeliği Madde 22"
    },
    {
        "id": "Q33",
        "category": "Undergraduate",
        "question": "Yan dal programına başvuru şartları nelerdir?",
        "expected_answer": "Yan dal programına başvuru için öğrencinin GNO en az 3.00 olmalıdır. Başvuru en erken 3. dönemin başında, en geç 5. dönemin sonunda yapılabilir. Yan dal programı en az 5 dersi (minimum 30 SÜ / 60 AKTS) kapsar ve tüm derslerin en az C- notla geçilmesi gerekir. Çift anadal programına kayıtlı öğrenciler ayrıca bir yan dal programına da kayıt yaptırabilir.",
        "source": "ISR-C290-02 Çift Anadal Yönergesi / Lisans Yönetmeliği"
    },
    {
        "id": "Q34",
        "category": "Undergraduate",
        "question": "How is GPA calculated at Sabancı University?",
        "expected_answer": "GPA is calculated by multiplying the SU credit of each course by its grade coefficient, summing these total points, and dividing by the total SU credits taken. Averages are shown with two decimals. In academic ranking calculations, all digits after comma are taken into consideration. Courses taken at other universities during exchange and summer programs are included. For repeated courses, the last grade is used in GPA calculation.",
        "source": "Lisans Yönetmeliği / lisans_yon_eng"
    },

    # =================== REGISTRATION & GRADUATION (5) ===================
    {
        "id": "Q35",
        "category": "Registration",
        "question": "Mezuniyet başvurusu nasıl yapılır?",
        "expected_answer": "Mezuniyet başvurusu akademik takvimde belirtilen tarihlerde SIS üzerinden yapılır. Öğrencilerin programa ait tüm mezuniyet yükümlülüklerini (gerekli dersler, kredi sayıları) tamamlaması ve genel not ortalamasının en az 2.00 olması gerekir. Fakülte/Enstitü yönetimi gerekli kontrolleri yapar ve ÖK diplomayı düzenler.",
        "source": "PSR-C240-0101 Mezuniyet Denetimi ve Diploma Düzenleme Prosedürü §1.1-1.2"
    },
    {
        "id": "Q36",
        "category": "Registration",
        "question": "Kayıt dondurma şartları nelerdir?",
        "expected_answer": "Dönem izni (kayıt dondurma); sağlık, maddi, aile, kişisel, akademik ve beklenmedik zorunlu olaylar gibi nedenlerle, ayrıca askerlik, gözaltı, tutukluluk veya mahkûmiyet durumlarında verilebilir. İzin gerekçesine ilişkin belgeler eklenerek dilekçe ile derslerin başlamasını takip eden 4. haftanın son iş gününe kadar ilgili fakülte dekanlığına başvurulur. Bir defada en çok 2 dönem, toplam 4 dönem izin verilebilir.",
        "source": "Lisans Yönetmeliği Madde 39-41, ISR-C210-02"
    },
    {
        "id": "Q37",
        "category": "Registration",
        "question": "Transkript nasıl alınır?",
        "expected_answer": "Transkript, MySU'daki online belge talep formu doldurularak Öğrenci Kaynakları'ndan (ÖK) talep edilir. ÖK raporlama yazılımı ile hazırlar. Basılı kopya ücretlidir, e-imzalı transkript ücretsizdir. Transkriptte öğrencinin tüm dersleri, kodları, SÜ ve AKTS kredileri, notları, DNO ve GNO bilgileri yer alır.",
        "source": "PSR-C230-0103 Transkript Düzenleme Prosedürü"
    },
    {
        "id": "Q38",
        "category": "Registration",
        "question": "How does the Leave of Absence process work?",
        "expected_answer": "Leave of absence (dönem izni) is granted for health, financial, family, personal, academic, and emergency reasons. Students must apply with supporting documents to the Faculty Dean's Office by the end of the 4th week after classes start. At most 2 semesters at once and 4 total are allowed. Students must return laptops and residence permits are cancelled during leave. Tuition percentage depends on application timing (25% if applied by 4th week).",
        "source": "ISR-C210-02 / PSR-C210-0201 Procedure For Semester Leave of Absence"
    },
    {
        "id": "Q39",
        "category": "Registration",
        "question": "Diploma programı değişikliği nasıl yapılır?",
        "expected_answer": "Lisans diploma programı bildirimi, diploma programına yerleştirme ve diploma programı değiştirme (iç yatay geçiş) prosedürüne göre yapılır. Başvuru ilgili fakülte dekanlığına yapılır.",
        "source": "ISR-C240-02 / PSR-C240-0201 Lisans Diploma Programı Değiştirme Prosedürü"
    },

    # =================== INTERNSHIP (4) ===================
    {
        "id": "Q40",
        "category": "Internship",
        "question": "Zorunlu staj başvurusu nasıl yapılır?",
        "expected_answer": "Zorunlu staj başvurusu OMS (Online Management System) üzerinden staj talebi oluşturularak yapılır. Öğrenciler Yeni Staj Projesi Formu (FIPAR-C71001-02) doldurur ve kurumlarda yapacakları projenin tanımını girer. Staj Ofisi, kabul edildikleri kuruma teslim edilmek üzere Zorunlu Staj Belgesi verir.",
        "source": "PIPAR-C710-0101 Lisans Yaz Stajı Prosedürü"
    },
    {
        "id": "Q41",
        "category": "Internship",
        "question": "Staj derslerine kayıt nasıl yapılır?",
        "expected_answer": "Staj derslerine kayıt, bahar dönemi ders kayıtları veya ders ekleme-bırakma tarihlerinde Bilgi Sistemi üzerinden yapılır. Öğrencilerin PROJ 102 veya PROJ 201 derslerini tamamlamış olması gerekir. Staj kayıtları ÖK tarafından duyurulur ve akademik takvime göre gerçekleştirilir.",
        "source": "IIPAR-C710-01 Lisans Öğrencileri Yaz Stajı Yönergesi §2"
    },
    {
        "id": "Q42",
        "category": "Internship",
        "question": "IAESTE ile uluslararası staj nasıl yapılır?",
        "expected_answer": "IAESTE ile uluslararası staj başvurusu, Kariyer Geliştirme ve Staj Ofisi tarafından duyurulan tarihlerde yapılır. Öğrenciler başvuru belgelerini tamamlayarak IAESTE programı kapsamında yurtdışında staj yapar. Kabul edilen öğrenciler Zorunlu Staj Belgesi alır ve stajı tamamladıktan sonra değerlendirme formlarını sunar.",
        "source": "PID-C710-0202 IAESTE ile Uluslararası Staja Giden Öğrenci Prosedürü"
    },
    {
        "id": "Q43",
        "category": "Internship",
        "question": "Öğrenci belgesi nereden alınır?",
        "expected_answer": "Öğrenci belgesi, Öğrenci Kaynakları (ÖK) birimi tarafından hazırlanır. MySU'daki online Belge Talep Formu doldurularak başvuru yapılır. Belge 2 iş günü içinde hazırlanır ve en fazla 2 hafta muhafaza edilir; bu sürede teslim alınmazsa imha edilir.",
        "source": "PSR-C210-0402 Öğrenci Belgesi Düzenleme Prosedürü"
    },

    # =================== ENGLISH QUESTIONS (3) ===================
    {
        "id": "Q44",
        "category": "English",
        "question": "What is the minimum GPA requirement for Erasmus?",
        "expected_answer": "Minimum GPA is 2.20 for undergrad and 2.5 for graduate students.",
        "source": "IID-C710-02 International Internship Instruction §3.3"
    },
    {
        "id": "Q45",
        "category": "English",
        "question": "How many books can I borrow from the library?",
        "expected_answer": "Undergraduate, graduate, and exchange students can borrow 60 books for 60 days, 5 multimedia items for 7 days, and 2 bound periodicals for 7 days.",
        "source": "IIC-C840-02 Circulation and Utilization Instruction §1.3"
    },
    {
        "id": "Q46",
        "category": "English",
        "question": "What are the disciplinary penalties for students?",
        "expected_answer": "Warning, reprimand, suspension, and expulsion from university.",
        "source": "PSR-C210-0101 Student Disciplinary Procedure §1.3"
    },
    
    # =================== SPECIFIC NUMERIC QUESTIONS (4) ===================
    {
        "id": "Q47",
        "category": "Numeric",
        "question": "Lisans mezuniyeti için kaç kredi gerekiyor?",
        "expected_answer": "Lisans mezuniyeti için belirli bir sabit kredi sayısı yoktur. Kayıtlı olunan diploma programının gerektirdiği tüm ders ve kredi yükümlülüklerinin tamamlanması ve SÜ kredilerine göre hesaplanan genel not ortalamasının en az 2.00 olması gerekir.",
        "source": "Lisans Yönetmeliği Madde 35"
    },
    {
        "id": "Q48",
        "category": "Numeric",
        "question": "Bir dersin kaç kez tekrar edilebilir?",
        "expected_answer": "Lisans öğrencileri daha önce geçer not aldıkları dersi, aldıkları dönemi izleyen en çok 3 dönem içinde tekrarlayabilir (izinli dönemler ve yaz dönemleri hariç). Başarısız olunan zorunlu dersler mezuniyete kadar tekrar edilerek başarılmalıdır. Lisansüstü öğrenciler için belirli bir tekrar sayısı sınırlaması yoktur; C, C+ veya B- notu aldıkları dersleri programdaki dersleri tamamlama süresinin sonuna kadar tekrar alabilir.",
        "source": "Lisans Yönetmeliği Madde 30"
    },
    {
        "id": "Q49",
        "category": "Numeric",
        "question": "Çift anadal programı için minimum kaç ek kredi gerekir?",
        "expected_answer": "Çift anadal diploma programlarında anadal diploma programı yükümlülükleri dışındaki fark derslerle en az 30 SÜ ve 60 AKTS kredisi tamamlanması gerekir.",
        "source": "ISR-C290-02 Çift Anadal Yönergesi"
    },
    {
        "id": "Q50",
        "category": "Numeric",
        "question": "Rektörlük Araştırma Fonu bütçe limiti ne kadardır?",
        "expected_answer": "Rektörlük Araştırma Fonu bütçesi deneysel olmayan projeler için 300.000 TL, deneysel projeler için 600.000 TL'dir. Proje süresi en fazla 36 aydır. Laboratuvar ekipmanı, sarf malzemesi ve deneysel saha çalışmalarının maliyetleri toplam proje bütçesinin %30'undan fazlasını oluşturması durumunda proje deneysel olarak değerlendirilir.",
        "source": "PSUATT-A610-01-03 Rektörlük Araştırma Fonu Prosedürü"
    },

    # =================== PROCEDURAL QUESTIONS (5) ===================
    {
        "id": "Q51",
        "category": "Procedure",
        "question": "Araç talep prosedürü nasıl işler?",
        "expected_answer": "Araç talep prosedürü, SUForm üzerinden Vehicle Request / Temporary Replacement Car Request Form doldurularak başlar. Talep eden kişi araç kullanım amacını, tarihini ve güzergahını belirtir. Talep en az 5 iş günü önceden yapılmalıdır. Onay süreci birim amiri ve İdari İşler koordinasyonuyla yürütülür. Shuttle güzergahındaki rotalar için araç talep edilmemelidir.",
        "source": "PSER-C930-0103 Genel Hizmet Aracı Prosedürü"
    },
    {
        "id": "Q52",
        "category": "Procedure",
        "question": "Yurtlardan ayrılma prosedürü nasıl işler?",
        "expected_answer": "Üniversiteden 1 ayı aşan süre için ayrılan öğrencilerin yurt kaydı Öğrenci Kaynakları Birimi tarafından bildirilir ve yurt yönetimi tarafından silinir. Öğrenciye bildirim sonrası 2 gün içinde yurtla ilişkilerini kesmesi gerekir. Öğrenci Yurt Odası Teslim Formu doldurulur. Depozito iadesi hasar yoksa kesintisiz yapılır.",
        "source": "PSER-C910-0102 Yurtlardan Ayrılma Prosedürü"
    },
    {
        "id": "Q53",
        "category": "Procedure",
        "question": "Acil durumlarda kampüste ambulans nasıl çağrılır?",
        "expected_answer": "Ambulans gereken durumlarda en yakın telefondan Sağlık Merkezi Acil Hattı (6666) aranmalıdır. Telefona cevap veren sağlık personeline 5N Kuralına uygun şekilde cevap verilir. Ambulans yalnızca SÜ kampüsünde yaşayanlar ve kampüsün çok yakın çevresindeki acil durumlar için görevlendirilir.",
        "source": "POP-C330-0204 Ambulans İşleyişi ve Acil Çağrı Prosedürü"
    },
    {
        "id": "Q54",
        "category": "Procedure",
        "question": "Donanım ve yazılım arızası durumunda ne yapılmalı?",
        "expected_answer": "Dizüstü bilgisayarlardaki donanım ve yazılım arızalarında teknik destek ofisine şahsen başvuru yapılır ve sorunu açıklayan Teknik Destek Talep Formu (FIT-S14004-02) doldurulur. Başvuru BT Çağrılarının Yönetimi prosedürüne göre kaynak planlaması yapılarak değerlendirilir.",
        "source": "PIT-S140-0401 Dizüstü Bilgisayar Hizmet Paketi Prosedürü"
    },
    {
        "id": "Q55",
        "category": "Procedure",
        "question": "How does the graduation audit process work?",
        "expected_answer": "Graduation audit begins with students submitting applications through SIS within academic calendar dates. SR collaborates with Faculties/Institutes to verify graduation requirements (courses, credits, GPA). For thesis programs, the thesis advisor and Institute oversee proposal, defense, and submission stages. GPA calculations are based on grades submitted up to 2 business days before the graduation ceremony. Diplomas are issued after all conditions are fulfilled.",
        "source": "PSR-C240-0101 Graduation Audit and Diploma Issuance Procedure"
    },

    # =================== EDGE CASES (4) ===================
    {
        "id": "Q56",
        "category": "Edge",
        "question": "Cinsel taciz şikayeti nasıl yapılır?",
        "expected_answer": "Cinsel taciz veya ayrımcılık şikayetleri, herhangi bir zamanda Komiteye veya Üniversite Yönetimine sunulabilir. Komite, şikayetçinin akademik/çalışma hayatının olumsuz etkilenmemesini güvence altına alır. Şikayetler; çalışanlar, öğrenciler veya üniversite topluluğunun herhangi bir üyesi tarafından yapılabilir. Özel bir komite, iddiaları belirlenen süreler içinde inceler.",
        "source": "IPO-A510-01 Cinsel Taciz Yönergesi §5"
    },
    {
        "id": "Q57",
        "category": "Edge",
        "question": "Sabancı Üniversitesi nerede?",
        "expected_answer": "Sabancı Üniversitesi İstanbul Tuzla'da bulunmaktadır.",
        "source": "Genel Bilgi / Kurumsal Web"
    },
    {
        "id": "Q58",
        "category": "Edge",
        "question": "Kampüs içi trafik cezası itirazı nasıl yapılır?",
        "expected_answer": "Trafik kurallarının ihlali halinde Güvenlik Müdürlüğünce e-mail yoluyla ilgili kişi uyarılır ve ceza puanı durumu bildirilir. Sürücünün itiraz hakkı vardır.",
        "source": "PSER-C940-0104 Kampüs Alanı İçindeki Taşıt Trafiğinin Kontrol Prosedürü"
    },
    {
        "id": "Q59",
        "category": "Edge",
        "question": "How does the lost and found procedure work on campus?",
        "expected_answer": "Lost property found on campus is turned over to the Security Directorate. The Security Directorate records the items and attempts to return them to their owners. Items that remain unclaimed within 30 days are processed in accordance with the relevant procedure.",
        "source": "PSER-C940-0106 Kayıp ve Bulunan Eşyalara Yapılacak İşlem Prosedürü"
    },

    # =================== NEGATION QUESTIONS (6) ===================
    {
        "id": "Q60",
        "category": "Negation",
        "question": "Can audience students borrow books from the library?",
        "expected_answer": "No, audience students cannot use borrowing services. They can only access the Information Center and use information resources on-site. They also cannot access electronic resources from off-campus.",
        "source": "IIC-C840-02 Circulation and Utilization Instruction §1.4"
    },
    {
        "id": "Q61",
        "category": "Negation",
        "question": "Üniversiteden uzaklaştırma cezası alan öğrenci yurtta kalabilir mi?",
        "expected_answer": "Üniversiteden uzaklaştırma cezası alan öğrenciler, ceza süresi boyunca yurtlarda konaklayamaz. Yurtlar Yönergesi'ne göre disiplin suçu nedeniyle uzaklaştırma durumunda yurt kaydı iptal edilir ve öğrenci bildirildikten sonra 2 gün içinde yurtla ilişkisini kesmek zorundadır.",
        "source": "IOP-C310-01 Yurt Yönergesi §6"
    },
    {
        "id": "Q62",
        "category": "Negation",
        "question": "Kafeteryadan paket yemek servisi yapılır mı?",
        "expected_answer": "Hayır, kafeteryada paket yemek servisi yapılmaz. Bu kural yalnızca pandemi veya doğal afet gibi olağanüstü durumlarda esnetilebilir.",
        "source": "IOP-C320-01 Yemek Hizmeti Yönergesi"
    },
    {
        "id": "Q63",
        "category": "Negation",
        "question": "Can students access the Faculty Reserve Collection?",
        "expected_answer": "No, students cannot benefit from the Faculty Reserve Collection. Only academic staff and emeritus faculty can access it.",
        "source": "IIC-C840-02 Circulation and Utilization Instruction §1.1, §1.3"
    },
    {
        "id": "Q64",
        "category": "Negation",
        "question": "Öğrenci Konseyi seçimlerine aday olmak için disiplin koşulu nedir?",
        "expected_answer": "Adayların uyarma cezası dışında disiplin cezası almamış olması gerekir. Ayrıca siyasi parti organlarında üye veya görevli olmaması zorunludur.",
        "source": "ISR-C620-02 Öğrenci Konseyi Yönergesi §5"
    },
    {
        "id": "Q65",
        "category": "Negation",
        "question": "Uzaklaştırma cezası alan öğrenci ücret iadesi alabilir mi?",
        "expected_answer": "Uzaklaştırma cezası alan öğrencilere herhangi bir ücret iadesi yapılmaz. Disiplin suçu nedeniyle uzaklaştırılan öğrenciler, ceza dönemindeki öğrenim ücretini ödemekle yükümlüdür. Burs alan öğrencilerin bursu uzaklaştırma süresince dondurulur.",
        "source": "PSR-C210-0101 Disiplin Prosedürü §1.5"
    },

    # =================== LIST / COUNT QUESTIONS (6) ===================
    {
        "id": "Q66",
        "category": "List",
        "question": "How many types of disciplinary penalties exist for students?",
        "expected_answer": "There are 5 types: (1) Warning, (2) Reprimand, (3) Suspension from university for 1 week to 1 month, (4) Suspension from university for 1 or 2 semesters, (5) Dismissal from the higher education institution.",
        "source": "PSR-C210-0101 Student Disciplinary Procedure §1.3"
    },
    {
        "id": "Q67",
        "category": "List",
        "question": "Sabancı Üniversitesi'nde kaç fakülte vardır?",
        "expected_answer": "Sabancı Üniversitesi'nde üç fakülte vardır: Mühendislik ve Doğa Bilimleri Fakültesi (FENS), Sanat ve Sosyal Bilimler Fakültesi (FASS) ve Yönetim Bilimleri Fakültesi (SOM).",
        "source": "Lisans Yönetmeliği / Lisansüstü Yönetmeliği"
    },
    {
        "id": "Q68",
        "category": "List",
        "question": "Araştırma projeleri yönetiminde hangi süreç adımları vardır?",
        "expected_answer": "Araştırma projeleri yönetiminde şu süreç adımları vardır: projelerin başlatılması, projelerin yönetilmesi, proje revizyonları, raporlama, proje kapanışı ve bütçe transferi, dış denetimler.",
        "source": "ISUATT-A610-01 Araştırma Projeleri Süreçlerinin Yönetilmesi Yönergesi"
    },
    {
        "id": "Q69",
        "category": "List",
        "question": "Kampüste hangi yemek servisi türleri sunulur?",
        "expected_answer": "Kampüste catering, çay ofisleri ve üniversite merkezinde yemek hizmetleri sunulur. Yemek servisi akşam yemeği seçmeli ve indirimli set menülerden oluşur. Kumanya içeriği her gün değişen kahvaltılık malzemelerden oluşur. Menü YİH tarafından onaylanır ve aylık olarak yayımlanır.",
        "source": "ISER-C960-01 Yiyecek ve İçecek Hizmetlerini Sağlama Yönergesi"
    },
    {
        "id": "Q70",
        "category": "List",
        "question": "What are the research project fund types at Sabancı University?",
        "expected_answer": "Research fund types include: (1) Integration Projects (EPD) — 1 year for newly hired faculty, (2) Rectorate Research Fund — up to 36 months, 300K-600K TL, (3) Internal Research Grant (İç Araştırma Desteği), (4) Internal Grant for Academic Activities, (5) ERC Support Package, (6) SU President's Office Research Fund, and (7) Project Research Fund (PAF).",
        "source": "ISUATT-A610-01 Araştırma Projeleri Yönetilmesi Yönergesi"
    },
    {
        "id": "Q71",
        "category": "List",
        "question": "What types of shuttle services are provided?",
        "expected_answer": "Personnel shuttles are provided for university employees, interns, and subcontractors. The service follows the Transportation Services Procedure and adheres to relevant traffic laws including Highway Traffic Law No. 2918 and Road Transport Law No. 4925.",
        "source": "ISER-C930-01 Transportation Services Instruction Letter"
    },

    # =================== MULTI-HOP QUESTIONS (6) ===================
    {
        "id": "Q72",
        "category": "MultiHop",
        "question": "Can a suspended student take courses at another university and transfer credits back to Sabancı?",
        "expected_answer": "No, a suspended student cannot take courses at another university and transfer credits back to Sabancı University. According to the Disciplinary Regulation, students under suspension lose the right to attend classes and use university services. Course equivalencies and credit transfers require active enrollment status.",
        "source": "PSR-C210-0101 Disciplinary Procedure §1.5"
    },
    {
        "id": "Q73",
        "category": "MultiHop",
        "question": "Kütüphane gecikme borcu mezuniyeti engeller mi?",
        "expected_answer": "Evet, gecikme borcu olan öğrencilerin öğrenci belgesi, transkript ve mezuniyet için gerekli diğer belgeleri alması engellenir. Gecikme ücreti ödenene kadar ödünç alma hizmetleri de askıya alınır.",
        "source": "IIC-C840-02 Ödünç Verme Yönergesi §4 + PSR-C240-0101 Mezuniyet Prosedürü"
    },
    {
        "id": "Q74",
        "category": "MultiHop",
        "question": "Hangi disiplin cezaları Öğrenci Konseyi adaylığına engel olur?",
        "expected_answer": "Uyarma dışındaki tüm disiplin cezaları Öğrenci Konseyi adaylığına engel olur. Yani kınama, uzaklaştırma (kısa ve uzun süreli) ve yükseköğretim kurumundan çıkarma cezası alan öğrenciler aday olamaz.",
        "source": "ISR-C620-02 Öğrenci Konseyi Yönergesi §5 + ISR-C210-01 Disiplin Yönergesi §3"
    },
    {
        "id": "Q75",
        "category": "MultiHop",
        "question": "Kayıt donduran öğrencinin yurt ve ikamet izni durumu ne olur?",
        "expected_answer": "Kayıt dondurma nedeniyle 1 ayı aşan süre için ayrılan öğrencilerin yurt kaydı sonlandırılır. Ayrıca kayıt dondurma durumunda ikamet izninin iptali işlemi başlatılır.",
        "source": "PSER-C910-0102 Yurtlardan Ayrılma + IIRO-C430-02 Vize ve İkamet İzni Yönergesi"
    },
    {
        "id": "Q76",
        "category": "MultiHop",
        "question": "Entegrasyon Projesi almış bir öğretim üyesi Rektörlük Araştırma Fonuna başvurabilir mi?",
        "expected_answer": "Evet, daha önce Entegrasyon Projesi almış olan öğretim üyeleri de Rektörlük Araştırma Fonundan yararlanabilir. Ancak öğretim üyeleri bu fona sadece bir kez başvurabilirler.",
        "source": "PSUATT-A610-01-03 Rektörlük Araştırma Fonu Prosedürü"
    },
    {
        "id": "Q77",
        "category": "MultiHop",
        "question": "Kayıt donduran öğrenci dizüstü bilgisayarını iade etmeli mi?",
        "expected_answer": "Evet, öğrenimlerine ara veren veya kayıt donduran öğrencilerin dizüstü bilgisayarlarını iade etmesi gerekir.",
        "source": "PIT-S140-0401 Laptop Service Package Procedure + ISR-C210-02 Leave of Absence"
    },

    # =================== NEW: HOUSING (5) ===================
    {
        "id": "Q78",
        "category": "Housing",
        "question": "Yurt başvuru ve kabul koşulları nelerdir?",
        "expected_answer": "Yurt başvurusu https://dormapp.sabanciuniv.edu sistemi üzerinden yapılır. Kabul öncelik sırasına göre değerlendirme yapılır. Burs tipi odalarda burs alan öğrenciler konaklar. Yaz dönemi için ayrıca başvuru alınır. Ödeme koşulları Student Housing Instruction Letter ile belirlenir.",
        "source": "ISER-C910-01 Student Housing Instruction Letter"
    },
    {
        "id": "Q79",
        "category": "Housing",
        "question": "What are the rules for students staying at the dormitory?",
        "expected_answer": "Students staying in dormitories must adhere to the following rules: Smoking is prohibited in all closed areas; violations result in notification to the Disciplinary Committee. Visitors must register and leave before curfew. Pets are not allowed. Room changes require written approval. Students must keep rooms clean and report maintenance issues.",
        "source": "ISER-C910-01 Student Housing Instruction Letter"
    },
    {
        "id": "Q80",
        "category": "Housing",
        "question": "Yurtta kalan öğrencilerin sağlık prosedürleri nelerdir?",
        "expected_answer": "Yurtta kalan öğrenciler yurda giriş tarihinden en fazla 6 ay önce çekilmiş akciğer film raporu teslim etmelidir. Rapor radyoloji veya göğüs hastalıkları uzmanı tarafından imzalanmış olmalıdır. Sağlık Merkezi (VSD) dışındaki kuruluşlardan alınan filmler için uzman onayı gerekir.",
        "source": "POP-C330-0207 Health Procedure For Students Staying At The Dormitory"
    },
    {
        "id": "Q81",
        "category": "Housing",
        "question": "Yurtlarda acil müdahale gerektiren durumlarda ne yapılır?",
        "expected_answer": "Elektrik kesintisi, su kesintisi ya da kaçağı gibi acil müdahale gerektiren çağrılara 7 gün 24 saat müdahale edilebilmektedir. Teknik ekibe çağrı açıldığı takdirde yurt odalarında kimse yokken de müdahale yapılabilir.",
        "source": "PPOP-S230-01-03 Mimari ve İnşaat İşleri Prosedürü"
    },
    {
        "id": "Q82",
        "category": "Housing",
        "question": "Mezuniyet veya kayıt dondurma durumunda yurt kaydı ne olur?",
        "expected_answer":  "Kayıt dondurma, mezuniyet, izin ya da disiplin cezası gibi sebeplerle 1 ayı aşan süre için ayrılan öğrencilerin yurt kaydı Öğrenci Kaynakları Birimi bilgisi doğrultusunda silinir. Yurt kaydı silinen öğrenci bildirildikten sonra 2 gün içinde yurtla ilişkisini keser. Depozito iadesi hasar yoksa yapılır.",
        "source": "PSER-C910-0102 Yurtlardan Ayrılma Prosedürü"
    },

    # =================== NEW: FOOD SERVICES (3) ===================
    {
        "id": "Q83",
        "category": "FoodServices",
        "question": "Kampüste yemek hizmetleri nerede sunulur?",
        "expected_answer": "Sabancı Üniversitesi çalışanları, öğrencileri ve ziyaretçileri için yemek ve içecek hizmetleri Üniversite Merkezi'nde ve Kiralama Sözleşmesi kapsamında hizmet veren işletmelerin mekanlarında sunulur.",
        "source": "ISER-C960-01 Food and Beverage Services Provision Instruction Letter"
    },
    {
        "id": "Q84",
        "category": "FoodServices",
        "question": "Üniversite Merkezinde uyulması gereken yemek kuralları nelerdir?",
        "expected_answer": "SÜ çalışanları ve öğrencileri kendi yemeğini almakla ve boş tepsisini tepsi toplama bandına bırakmakla yükümlüdür. Üniversite Merkezinden masalardaki tabak, çatal, bıçak, tepsi, fincan, tuzluk gibi malzemelerin dışarıya çıkarılması yasaktır. Paket yemek servisi yapılmaz.",
        "source": "ISER-C960-01 Yiyecek ve İçecek Hizmetlerini Sağlama Yönergesi §3"
    },
    {
        "id": "Q85",
        "category": "FoodServices",
        "question": "How are special event and catering services arranged?",
        "expected_answer": "Special event and catering services are arranged through the Special Event and Catering Services Procedure. Requests are coordinated with the Food and Beverage Services department.",
        "source": "PSER-C960-0101 Special Event and Catering Services Procedure"
    },

    # =================== NEW: TRANSPORTATION (3) ===================
    {
        "id": "Q86",
        "category": "Transportation",
        "question": "Kampüs servis hizmetleri nasıl düzenlenir?",
        "expected_answer": "İlgili yasa ve yönetmeliklere bağlı kalarak üniversite çalışanları, stajyerler ve üniversite tarafından belirlenen alt yüklenici firmaların çalışanlarının mesai için işyerine geliş ve gidiş ulaşımlarını sağlamak üzere hizmet verilmektedir. Hizmet Ulaşım Hizmetleri Prosedürüne göre sağlanır.",
        "source": "ISER-C930-01 Ulaşım Hizmetleri Yönergesi"
    },
    {
        "id": "Q87",
        "category": "Transportation",
        "question": "Kampüs içi araç kullanım kuralları nelerdir?",
        "expected_answer": "Kampüs içi araç kullanım kuralları: Alkollü araç kullanımı yasaktır. Araç içinde sigara içilmez. Trafik kurallarına uyulmak zorunludur; klakson çalma, konvoy oluşturma ve yaya trafiğini tehlikeye atmak yasaktır. Hasar meydana geldiğinde derhal SÜ yetkilisine bildirilir. Trafik cezası personelin bordrosuna yansıtılır.",
        "source": "PSER-S930-0102 Vehicle Usage Procedure"
    },
    {
        "id": "Q88",
        "category": "Transportation",
        "question": "How does the electric vehicle charging process work?",
        "expected_answer": "Electric vehicles must be used in ECO mode. When the charge indicator is at 45% or below, the vehicle must be plugged in for charging before returning the key. If the vehicle is charged off-campus at a non-contracted station, the expense must be declared with an invoice issued in the name of SU.",
        "source": "PSER-C930-0103 Genel Hizmet Aracı Prosedürü"
    },

    # =================== NEW: HEALTH & SAFETY (4) ===================
    {
        "id": "Q89",
        "category": "HealthSafety",
        "question": "Kampüsteki sağlık merkezinin acil hattı numarası nedir?",
        "expected_answer": "Kampüsteki Sağlık Merkezi Acil Hattı 6666'dır. Ambulans gereken durumlarda en yakın telefondan bu hat aranmalıdır.",
        "source": "POP-C330-0204 Ambulans İşleyişi ve Acil Çağrı Prosedürü"
    },
    {
        "id": "Q90",
        "category": "HealthSafety",
        "question": "Kampüs ambulansı hangi durumlarda görevlendirilir?",
        "expected_answer": "Ambulans yalnızca SÜ kampüsünde yaşayanlar ve kampüsün çok yakın çevresindeki acil durumlar için görevlendirilir. Ambulansın herhangi bir şekilde SÜ kampüsünden ayrılması durumunda, özel sağlık şirketi en yakında bulunan tam teşekküllü hastaneye sevk yapar.",
        "source": "POP-C330-0204 Ambulans İşleyişi ve Acil Çağrı Prosedürü"
    },
    {
        "id": "Q91",
        "category": "HealthSafety",
        "question": "What health services are available on campus?",
        "expected_answer": "The Health Center (Sağlık Merkezi) provides occupational physician consultations, IM/SC/IV injections, ambulance services, emergency response, and periodic health screenings. It coordinates with Food and Beverage Services for food safety inspections. The center also handles work-entry medical examinations for new employees and chest X-ray evaluations for dormitory residents.",
        "source": "ISER-C330-02 Health Services Instruction Letter"
    },
    {
        "id": "Q92",
        "category": "HealthSafety",
        "question": "Yeni çalışanlar için sağlık prosedürü nasıl işler?",
        "expected_answer":  "Yeni çalışanlar işe başladıkları hafta içinde Sağlık Merkezi'ne (9954) randevu alarak işe giriş muayenesine katılır. Muayene kapsamında akciğer film raporu gerekir. Rapor radyoloji veya göğüs hastalıkları uzmanı tarafından imzalanmış olmalıdır. İş sağlığı ve güvenliği kapsamında gerekli değerlendirmeler yapılır.",
        "source": "POP-C330-0205 Health Procedure For New Employees"
    },

    # =================== NEW: RESEARCH (3) ===================
    {
        "id": "Q93",
        "category": "Research",
        "question": "Entegrasyon Projelerine nasıl başvurulur?",
        "expected_answer": "EPD başvurusu tam zamanlı öğretim üyelerinin işe başlama tarihinden itibaren 1 yıl içinde yapılmalıdır. PY, EPD Teklif Formunu doldurur ve proje partnerine yönlendirir. Proje partneri teknik uygunluğu değerlendirir. Bütçe 50.000 TL'ye kadardır. Başvuru formu proje özeti, yönetim planı ve harcama kalemlerinin gerekçesini içerir.",
        "source": "PSUATT-A610-01-02 Entegrasyon Projeleri Prosedürü"
    },
    {
        "id": "Q94",
        "category": "Research",
        "question": "Araştırma Etik Kurulu başvurusu nasıl yapılır?",
        "expected_answer":  "Araştırma Etik Kurulu başvurusu, araştırma veya tez çalışmasına başlamadan önce yapılır. Başvuru PSUATT-A610-01-10 prosedürüne göre yürütülür. İnsan denekleri içeren araştırmalar için etik kurul onayı zorunludur. Başvuru formu ve araştırma protokolü sunularak kurul değerlendirmesine alınır. Onay olmadan araştırma başlatılamaz.",
        "source": "PSUATT-A610-01-10 Araştırma Etik Kurulu Prosedürü / ISUATT-A610-01"
    },
    {
        "id": "Q95",
        "category": "Research",
        "question": "What are the requirements for the Rectorate Research Fund?",
        "expected_answer": "Faculty members can apply once. Previous Integration Project recipients are also eligible. Project duration is up to 36 months. Requirements include at least two documented external grant project applications or at least one guaranteed external financing and high-impact scientific article publications. Application deadline is April 30 (once per year).",
        "source": "PSUATT-A610-01-03 Rectorate Research Fund Procedure"
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

def run_detailed_test(use_evaluator: bool = True, start_from: int = 1):
    """
    Run TEST_QUESTIONS through the RAG pipeline.

    When *use_evaluator* is True (default), the Evaluator class is used to
    compute Answer Similarity, Faithfulness, Relevance, Factual Accuracy, and
    a composite score.  Keyword coverage is also computed for side-by-side
    comparison with the legacy metric.

    When *use_evaluator* is False, only legacy keyword coverage is computed
    (original behaviour).

    When *start_from* > 1, skips questions before that index (1-based).
    Useful for resuming after a crash.
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
    retrieval_results = [] # list of retrieval metric dicts

    total_start = time.time()

    for i, test in enumerate(TEST_QUESTIONS, 1):
        if i < start_from:
            continue
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
            pipeline_result = pipeline.answer(question, history=[])
            answer = pipeline_result["answer"] if isinstance(pipeline_result, dict) else pipeline_result
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

        # ── Retrieval Metrics ────────────────────────────────────────────────
        retrieval_chunks = pipeline_result.get("retrieved_chunks", []) if isinstance(pipeline_result, dict) else []
        r_metrics = compute_retrieval_metrics(test["source"], retrieval_chunks, top_k=10, expected_answer=expected)
        retrieval_results.append(r_metrics)
        
        print(f"\n🔍 Retrieval Metrics (Top 10):")
        print(f"   Hit@10 : {r_metrics['Hit@K']:.3f} | MRR@10 : {r_metrics['MRR@K']:.3f} | MAP@10 : {r_metrics['MAP@K']:.3f} | nDCG@10 : {r_metrics['nDCG@K']:.3f}")
        print(f"   Context Recall : {r_metrics['ContextRecall']:.3f}")

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
            # Use the EXACT chunks that were used during generation.
            # pipeline.answer() returns retrieved_chunks already — using
            # search_only() would give different (smaller, no neighbor expansion)
            # context and cause false low-faithfulness scores.
            try:
                raw_chunks = (
                    pipeline_result.get("context_chunks", [])
                    if isinstance(pipeline_result, dict) else []
                )
                context = _build_context_from_chunks(raw_chunks) if raw_chunks else ""
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
            print(f"\n📐 RAGAS scores:")
            print(f"   Faithfulness       : {s.faithfulness:.3f}")
            print(f"   Answer Correctness : {s.answer_correctness:.3f}")
            print(f"   Context Recall     : {s.context_recall:.3f}")
            print(f"   Answer Relevancy   : {s.answer_relevancy:.3f}")
            print(f"   ─────────────────────────────────────────")
            comp_icon = "✅" if s.composite_score >= 0.70 else "❌"
            print(f"   Composite Score    : {s.composite_score:.3f}  {comp_icon}")

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
    # RETRIEVAL SUMMARY
    # ─────────────────────────────────────────────────────────────────────────
    print("=" * 80)
    print("📈 RETRIEVAL METRICS SUMMARY")
    print("=" * 80)
    if retrieval_results:
        avg_hit = sum(m["Hit@K"] for m in retrieval_results) / len(retrieval_results)
        avg_mrr = sum(m["MRR@K"] for m in retrieval_results) / len(retrieval_results)
        avg_map = sum(m["MAP@K"] for m in retrieval_results) / len(retrieval_results)
        avg_ndcg = sum(m["nDCG@K"] for m in retrieval_results) / len(retrieval_results)
        avg_ctx_recall = sum(m.get("ContextRecall", 0) for m in retrieval_results) / len(retrieval_results)
        print(f"  Hit@10          : {avg_hit:.3f}")
        print(f"  MRR@10          : {avg_mrr:.3f}")
        print(f"  MAP@10          : {avg_map:.3f}")
        print(f"  nDCG@10         : {avg_ndcg:.3f}")
        print(f"  Context Recall  : {avg_ctx_recall:.3f}")
        print()

    # ─────────────────────────────────────────────────────────────────────────
    # MULTI-METRIC REPORT
    # ─────────────────────────────────────────────────────────────────────────
    if evaluator is not None and eval_results:
        test_results_dir = os.path.join(os.path.dirname(__file__), "test_results")
        evaluator.generate_report(eval_results, output_dir=test_results_dir)

    print(f"\n⏱️  Total wall-clock time: {total_elapsed:.0f}s")

    # ── Close output file & save baseline report copy ─────────────────────────
    output_file.close()
    sys.stdout = original_stdout
    sys.stderr = original_stderr

    # Save a copy as baseline_report_v2.txt
    import shutil
    baseline_path = os.path.join(os.path.dirname(__file__), "baseline_report_v2.txt")
    shutil.copy2(output_file_path, baseline_path)
    print(f"📄 Baseline report saved: {baseline_path}")

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
    parser.add_argument(
        "--start-from", type=int, default=1,
        help="Start from question N (1-based). Skips earlier questions.",
    )
    parser.add_argument(
        "--append", action="store_true",
        help="Append to rag_test_output.txt instead of overwriting.",
    )
    args = parser.parse_args()

    use_eval = not args.no_eval

    if args.question:
        run_single_question(args.question, use_evaluator=use_eval)
    else:
        run_detailed_test(use_evaluator=use_eval, start_from=args.start_from)
