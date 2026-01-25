"""test_rag_detailed.py - Detailed RAG Test with Full Answers"""

"""
test_rag_detailed.py - Detailed RAG Test with Full Answers

Shows complete answers for manual evaluation.
Run: python test_rag_detailed.py
All terminal output is also recorded to rag_test_output.txt
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
        "expected_answer": "Hibe miktarı gidilen ülkeye göre belirlenir ve aylık olarak ödenir.",
        "source": "Erasmus Hibe Sözleşmesi"
    },
    
    # =================== LIBRARY ===================
    {
        "id": "Q5",
        "category": "Library",
        "question": "Kütüphaneden kaç kitap ödünç alabilirim ve süresi ne kadar?",
        "expected_answer": "Paket 2 kullanıcıları 30 gün süre ile 10 adet kitap ödünç alabilir.",
        "source": "IIC-C840-02 Ödünç Verme ve Yararlanma Yönergesi"
    },
    {
        "id": "Q6",
        "category": "Library",
        "question": "Kütüphanelerarası ödünç alma (ILL) hizmeti nasıl çalışır?",
        "expected_answer": "Diğer kütüphanelerden kitap ve makale temin edilebilir.",
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
        "expected_answer": "Soruşturmanın tamamlandığı günden itibaren en geç 10 gün içinde karar verilmelidir.",
        "source": "PSR-C210-0101"
    },
    {
        "id": "Q9",
        "category": "Discipline",
        "question": "Kopya çekmek hangi disiplin cezasını gerektirir?",
        "expected_answer": "Sınavlarda kopya çekmek disiplin suçudur.",
        "source": "ISR-C210-01"
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
        "expected_answer": "Akademik başarı ve disiplin durumu değerlendirilir.",
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
        "expected_answer": "Tez savunma jürisi en az 3 öğretim üyesinden oluşur.",
        "source": "Lisansüstü Yönetmeliği"
    },
    
    # =================== UNDERGRADUATE ===================
    {
        "id": "Q15",
        "category": "Undergraduate",
        "question": "Yatay geçiş başvurusu için GNO şartı nedir?",
        "expected_answer": "Yatay geçiş için minimum GNO şartı aranır.",
        "source": "Lisans Programlarına Yatay Geçiş"
    },
    {
        "id": "Q16",
        "category": "Undergraduate",
        "question": "Çift anadal programına nasıl başvurulur?",
        "expected_answer": "Çift anadal programı için belirli GNO şartı ve başvuru süreci vardır.",
        "source": "Diploma Programı Yönergesi"
    },
    {
        "id": "Q17",
        "category": "Undergraduate",
        "question": "Ders ekleme-bırakma süresi ne kadar?",
        "expected_answer": "Her yarıyıl başında belirlenen süre içinde ders ekleme-bırakma yapılabilir.",
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
        "expected_answer": "Geçerli mazeretler ile kayıt dondurulabilir.",
        "source": "Kayıt Yönergesi"
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
        "expected_answer": "Package 2 users can borrow up to 10 books for 30 days.",
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
        "expected_answer": "Lisans programı için belirli sayıda kredi tamamlanmalıdır.",
        "source": "Lisans Yönetmeliği"
    },
    {
        "id": "Q24",
        "category": "Numeric",
        "question": "Bir dersin kaç kez tekrar edilebilir?",
        "expected_answer": "Başarısız olunan dersler tekrar alınabilir.",
        "source": "Eğitim-Öğretim Yönetmeliği"
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
        "expected_answer": "Transkript ÖBS üzerinden veya Öğrenci İşleri'nden alınabilir.",
        "source": "Öğrenci İşleri Prosedürleri"
    },
    {
        "id": "Q27",
        "category": "Procedure",
        "question": "Öğrenci belgesi nereden alınır?",
        "expected_answer": "Öğrenci belgesi ÖBS üzerinden alınabilir.",
        "source": "Öğrenci İşleri"
    },
    
    # =================== EDGE CASES ===================
    {
        "id": "Q28",
        "category": "Edge",
        "question": "Cinsel taciz şikayeti nasıl yapılır?",
        "expected_answer": "Şikayet ilgili birime yazılı olarak yapılır.",
        "source": "Cinsel Taciz Yönergesi"
    },
    {
        "id": "Q29",
        "category": "Edge",
        "question": "İtiraz süresi ne kadar?",
        "expected_answer": "Kararlara itiraz belirli süre içinde yapılmalıdır.",
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


def run_detailed_test():
    """Run tests and show full answers for evaluation."""
    
    print("=" * 80)
    print("🔬 RAG DETAILED TEST - Full Answers for Evaluation")
    print(f"📊 Total Questions: {len(TEST_QUESTIONS)}")
    print("=" * 80)
    
    # Load RAG
    try:
        from services.rag_core import get_chroma_collection, answer_with_rag, CHAT_MODEL
        print(CHAT_MODEL)
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return
    
    coll = get_chroma_collection()
    print(f"✅ ChromaDB connected: {coll.count()} chunks")
    print(f"🤖 Using model: {CHAT_MODEL}\n")
    
    results = []
    category_scores = {}
    
    for i, test in enumerate(TEST_QUESTIONS, 1):
        category = test.get('category', 'Other')
        print("=" * 80)
        print(f"📝 TEST {i}/{len(TEST_QUESTIONS)}: {test['id']} [{category}]")
        print("=" * 80)
        print(f"\n❓ QUESTION:\n   {test['question']}\n")
        print(f"📚 EXPECTED SOURCE: {test['source']}")
        print(f"\n✅ EXPECTED ANSWER:\n   {test['expected_answer']}\n")
        
        start = time.time()
        try:
            answer = answer_with_rag(test['question'], coll, history=[])
            latency = time.time() - start
            
            print(f"\n🤖 RAG ANSWER ({latency:.1f}s):")
            print("-" * 60)
            # Print full answer with word wrap
            words = answer.split()
            line = "   "
            for word in words:
                if len(line) + len(word) > 75:
                    print(line)
                    line = "   " + word
                else:
                    line += " " + word if line.strip() else word
            if line.strip():
                print(line)
            print("-" * 60)
            
            # Simple evaluation
            expected_keywords = test['expected_answer'].lower().split()
            answer_lower = answer.lower()
            found = sum(1 for kw in expected_keywords if kw in answer_lower)
            coverage = found / len(expected_keywords) * 100
            
            print(f"\n📊 Keyword coverage: {coverage:.0f}%")
            
            results.append({
                "id": test['id'],
                "category": category,
                "question": test['question'],
                "expected": test['expected_answer'],
                "actual": answer,
                "latency": latency,
                "coverage": coverage
            })
            
            # Track category scores
            if category not in category_scores:
                category_scores[category] = []
            category_scores[category].append(coverage)
            
        except Exception as e:
            print(f"\n❌ ERROR: {e}")
            results.append({
                "id": test['id'],
                "category": category,
                "error": str(e)
            })
        
        print("\n")
    
    # Summary
    print("=" * 80)
    print("📊 SUMMARY BY QUESTION")
    print("=" * 80)
    
    for r in results:
        if "error" in r:
            print(f"❌ {r['id']} [{r['category']}]: ERROR")
        else:
            status = "✅" if r['coverage'] >= 50 else "⚠️"
            print(f"{status} {r['id']} [{r['category']}]: {r['coverage']:.0f}% coverage, {r['latency']:.1f}s")
    
    # Category summary
    print("\n" + "=" * 80)
    print("📊 SUMMARY BY CATEGORY")
    print("=" * 80)
    
    for cat, scores in sorted(category_scores.items()):
        avg = sum(scores) / len(scores) if scores else 0
        passed = sum(1 for s in scores if s >= 50)
        status = "✅" if avg >= 50 else "⚠️"
        print(f"{status} {cat}: {avg:.0f}% avg ({passed}/{len(scores)} passed)")
    
    # Overall stats
    valid_results = [r for r in results if 'coverage' in r]
    avg_coverage = sum(r['coverage'] for r in valid_results) / len(valid_results) if valid_results else 0
    avg_latency = sum(r['latency'] for r in valid_results) / len(valid_results) if valid_results else 0
    passed_count = sum(1 for r in valid_results if r['coverage'] >= 50)
    
    print("\n" + "=" * 80)
    print("📊 OVERALL RESULTS")
    print("=" * 80)
    print(f"✅ Passed (≥50%): {passed_count}/{len(valid_results)} ({100*passed_count/len(valid_results):.0f}%)")
    print(f"📈 Average coverage: {avg_coverage:.0f}%")
    print(f"⏱️  Average latency: {avg_latency:.1f}s")
    print(f"⏱️  Total time: {sum(r.get('latency', 0) for r in results):.0f}s")
    
    # Close output file properly
    output_file.close()
    sys.stdout = sys.__stdout__
    sys.stderr = sys.__stderr__
    
    return results


def run_single_question(question: str):
    """Run a single question and show detailed output."""
    
    print("=" * 80)
    print(f"❓ Question: {question}")
    print("=" * 80)
    
    from services.rag_core import get_chroma_collection, answer_with_rag
    
    coll = get_chroma_collection()
    
    start = time.time()
    answer = answer_with_rag(question, coll, history=[])
    latency = time.time() - start
    
    print(f"\n🤖 ANSWER ({latency:.1f}s):")
    print("-" * 60)
    print(answer)
    print("-" * 60)
    
    # Close output file properly
    output_file.close()
    sys.stdout = sys.__stdout__
    sys.stderr = sys.__stderr__


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("-q", "--question", type=str, help="Ask a single question")
    args = parser.parse_args()
    
    if args.question:
        run_single_question(args.question)
    else:
        run_detailed_test()
