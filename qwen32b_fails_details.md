# Qwen3 32B RAGAS Failures Details

Total RAGAS failures: 31

================================================================================
## TEST Q5: Erasmus Kısa Dönem Doktora Hareketliliği ne kadar sürer?
================================================================================

**Expected Answer:**
> Erasmus Kısa Dönem Doktora Hareketliliği 5 ila 30 gün arasında sürer.

**Qwen 32B Answer:**
```
Erasmus Kısa Dönem Doktora Hareketliliği için belgelerde doğrudan süre
   belirtilmemiştir. Ancak Erasmus+ hibe sözleşmeleri ve Uluslararası Staj
   Yönergesi'nde genel Erasmus hareketliliği süreleri şu şekilde
   tanımlanmıştır:  - Hareketlilik süresi **en az 2 ay** ve **2 aydan fazla
   olamaz** (stajlar için).  - Ders verme hareketliliğinde yükseköğretim
   personeli için **asgari 8 saat ders verme süresi** gereklidir.   Doktora
   düzeyi için özel bir süre belirtilmediği için, genel Erasmus hareketliliği
   kuralları geçerlidir.
```

**RAGAS Scores:**
- Faithfulness: 0.8
- Answer Correctness: 0.18
- Answer Relevancy: 0.0
- Composite Score: 0.294

================================================================================
## TEST Q6: How does the Learning Agreement process work for exchange students?
================================================================================

**Expected Answer:**
> Students create a Learning Agreement (LA) specifying courses to be taken
   abroad. The Exchange Academic Advisor verifies course equivalencies
   (credits, types, ECTS). For double major students, separate approval from
   the DDPO program coordinator is needed. Minor program equivalencies also
   require advisor approval. The signed LA is submitted to the International
   Relations Office (IRO), which forwards it to the partner university.
   Changes during the exchange period require completing a Change Request in
   the LA system.

**Qwen 32B Answer:**
```
The Learning Agreement (LA) is prepared, signed, and submitted by the
   selected student within the timeframe announced by the university, with
   the approval of the exchange academic advisor. The advisor determines the
   equivalent credits, course types, and ECTS values for the selected courses
   at the partner institution. The student must ensure the courses taken
   abroad fulfill SU graduation requirements and inform the advisor of any
   changes before completing the term. The signed LA must be approved by the
   partner institution and submitted to the International Relations Office
   (IRO). For graduate students, the LA is prepared online in consultation
   with the advisor and submitted to IRO within the specified timeline. After
   signatures are obtained, the IRO forwards the LA to the Faculty
   Administrative Officer for approval. Students must also submit the signed
   LA, transcript, and Certificate of Attendance upon returning from the
   exchange program.
```

**RAGAS Scores:**
- Faithfulness: 0.933
- Answer Correctness: 0.352
- Answer Relevancy: 0.87
- Composite Score: 0.6

================================================================================
## TEST Q9: What happens if I return library books late?
================================================================================

**Expected Answer:**
> The qualified membership right to benefit from the circulation service of
   Sabancı University Alumni, Retirees and Families who do not pay the fees
   will be revoked and their subsequent applications will not be accepted for
   1 year. Reminders are sent for returned items. Students with overdue fines
   are blocked from borrowing services until fees are paid.

**Qwen 32B Answer:**
```
If you fail to return, damage, or lose items obtained through interlibrary
   loan, your Information Center membership will be blocked, and your use of
   circulation services will be suspended. The context does not specify a
   duration for the suspension, but it emphasizes that such actions trigger
   immediate restrictions on borrowing privileges. No additional penalties
   for late returns are explicitly mentioned in the provided information.
```

**RAGAS Scores:**
- Faithfulness: 1.0
- Answer Correctness: 0.114
- Answer Relevancy: 0.0
- Composite Score: 0.501

================================================================================
## TEST Q12: What are the borrowing limits for alumni at the library?
================================================================================

**Expected Answer:**
> Alumni and retirees can borrow 2 books for 30 days, 1 multimedia item for
   7 days, 1 bound periodical for 7 days, and 5 popular magazines for 3 days.

**Qwen 32B Answer:**
```
The context does not provide specific borrowing limits for Sabancı
   University alumni. It only states that alumni and retirees who fail to pay
   fees will lose their "qualified membership right to benefit from the
   lending service" and cannot reapply for one year. No details about the
   number of items or durations for alumni with active membership are
   mentioned in the provided chunks.
```

**RAGAS Scores:**
- Faithfulness: 1.0
- Answer Correctness: 0.131
- Answer Relevancy: 0.0
- Composite Score: 0.339

================================================================================
## TEST Q15: Kopya çekmek hangi disiplin cezasını gerektirir?
================================================================================

**Expected Answer:**
> Sınavlarda kopyaya teşebbüs etmek kınama cezası gerektirir. Kopya çekmek
   veya çektirmek bir yarıyıl uzaklaştırma cezası gerektirir. Tehditle kopya
   çekmek, kopya çeken öğrencilerin sınav salonundan çıkarılmasına engel
   olmak veya başkasının yerine sınava girmek iki yarıyıl uzaklaştırma cezası
   gerektirir.

**Qwen 32B Answer:**
```
Kopya çekmek, Sabancı Üniversitesi Öğrenci Disiplin Yönergesi'ne göre iki
   yarıyıl uzaklaştırma cezasını gerektirir. Bu ceza, öğrencinin öğrencilik
   haklarından iki yarıyıl boyunca yararlanamayacağını belirten bir yazı ile
   bildirilir. Sınavlarda kopya çekme fiili, kurumun hizmetlerini engellemek
   gibi ciddi ihlaller arasında yer alır. Ayrıca, kopya çekme girişimleri
   (tehditle kopya çekme veya kopya çekenleri engellemek) bir haftadan bir
   aya kadar uzaklaştırma cezasına neden olabilir. Ancak, doğrudan kopya
   çekme eylemi kesinlikle iki yarıyıl uzaklaştırma ile cezalandırılır.
```

**RAGAS Scores:**
- Faithfulness: 0.6
- Answer Correctness: 0.712
- Answer Relevancy: 0.431
- Composite Score: 0.625

================================================================================
## TEST Q18: Disiplin suçlarında zamanaşımı süreleri ne kadardır?
================================================================================

**Expected Answer:**
> Uyarma, kınama ve kısa süreli uzaklaştırma cezası gerektiren fiillerde
   soruşturma zamanaşımı 1 aydır. 1-2 yarıyıl uzaklaştırma ve çıkarma cezası
   gerektiren fiillerde zamanaşımı 3 aydır. Tüm disiplin suçlarında mutlak
   zamanaşımı süresi 2 yıldır.

**Qwen 32B Answer:**
```
Disiplin suçlarında zamanaşımı süreleri, ceza türüne göre değişir. Uyarma,
   kınama ve bir haftadan bir aya kadar uzaklaştırma cezaları için, fiilin
   işlendiğinin soruşturmaya yetkili amirlerce öğrenildiği tarihten itibaren
   **1 ay** içinde disiplin soruşturmasına başlanmazsa zamanaşımına uğrar.
   Bir veya iki yarıyıl uzaklaştırma ile yükseköğretim kurumundan çıkarma
   cezaları için bu süre **3 ay**dır. Genel olarak, disiplin cezası verme
   yetkisi, fiilin işlendiği tarihten itibaren **en geç 2 yıl** içinde
   kullanılmazsa zaman aşımına uğrar. Ancak adli yargı hükmüne ihtiyaç
   duyulan hallerde, zaman aşımı süresi adli yargı kararının kesinleştiği
   günden itibaren başlar. Cinsel taciz/saldırı gibi suçlar için ise eylemin
   işlendiği tarihten itibaren **en geç 2 yıl** içinde soruşturma
   tamamlanmalı ve karara bağlanmalıdır.
```

**RAGAS Scores:**
- Faithfulness: 1.0
- Answer Correctness: 0.701
- Answer Relevancy: 0.53
- Composite Score: 0.673

================================================================================
## TEST Q20: Burs devam şartları nelerdir?
================================================================================

**Expected Answer:**
> Burs devam şartları program türüne göre farklılık gösterir. Lisans
   bursları normal öğrenim süresi boyunca devam eder; mezun olma koşullarını
   yerine getiremeyen öğrencilerin burs süresi en fazla iki dönem
   uzatılabilir. Lisansüstü burslarında yüksek lisans için aylık nakit burs
   en fazla 10 ay (Eylül-Haziran), doktora için 12 ay (Eylül-Ağustos) ödenir.
   Burs devamı her akademik dönem başında Enstitü Yönetim Kurulu kararı ile
   değerlendirilir. Disiplin cezalarında: kınama ve kısa süreli
   uzaklaştırmada burs devam eder, 1-2 yarıyıl uzaklaştırmada o dönem kesilir
   ama sonra yeniden bağlanır, çıkarmada tamamen kesilir.

**Qwen 32B Answer:**
```
Burs devam şartları, lisans ve lisansüstü öğrenciler için farklı
   kriterlere sahiptir:   1. **Lisans Bursları**:   - Üniversiteye ilk
   girişte verilen burslar, normal öğrenim süresi boyunca devam eder. Mezun
   olma koşullarını yerine getiremeyen öğrencilerin burs süresi en fazla iki
   dönem uzatılabilir.   - Aylık nakit burs, 9 ay (Ekim-Haziran) süreyle
   verilir. Öğrenci Güz döneminde izin alırsa burs kesilir, Bahar dönemi ders
   kaydıyla tekrar başlar.   2. **Lisansüstü Bursları**:   - Yüksek lisans
   öğrencileri için 10 ay (Eylül-Haziran), doktora öğrencileri için 12 ay
   (Eylül-Ağustos) ödeme yapılır.   - Burs devamlılığı her dönem başında
   "Lisansüstü Bursluluk Devam Koşulları Tablosu"na göre değerlendirilir. Güz
   dönemi için 1
```

**RAGAS Scores:**
- Faithfulness: 0.909
- Answer Correctness: 0.396
- Answer Relevancy: 0.314
- Composite Score: 0.689

================================================================================
## TEST Q33: Yan dal programına başvuru şartları nelerdir?
================================================================================

**Expected Answer:**
> Yan dal programına başvuru için öğrencinin GNO en az 3.00 olmalıdır.
   Başvuru en erken 3. dönemin başında, en geç 5. dönemin sonunda
   yapılabilir. Yan dal programı en az 5 dersi (minimum 30 SÜ / 60 AKTS)
   kapsar ve tüm derslerin en az C- notla geçilmesi gerekir. Çift anadal
   programına kayıtlı öğrenciler ayrıca bir yan dal programına da kayıt
   yaptırabilir.

**Qwen 32B Answer:**
```
Yan dal programına başvuru için öğrencilerin bir dönem itibarıyla en erken
   ikinci ve en geç beşinci yarıyılda olmaları gerekir. Başvuru yapılan
   dönemin not girişi sonrası oluşan GNO değeri, yan dal programının bağlı
   olduğu fakülte tarafından belirlenen minimum GNO şartını karşılamalıdır.
   Öğrenciler, Akademik Takvimde ilan edilen tarihlerde ilgili fakülte
   dekanlığına dilekçe ile başvurmalıdır. Yan dal programına kabul için kayıt
   olunan yarıyılın başı itibarıyla diploma programına yerleşmiş olmak ve tüm
   kredili dersleri başarıyla tamamlamış olmak zorunludur. Kontenjan dolu
   olan öğrenciler, başka bir yan dal programına tekrar başvuru yapabilir.
   Kurum içi veya kurumlar arası yatay geçiş başvurusu yapanlar, öğrenci
   statüsünde kayıtlı ve İngilizce dil yeterliliğini sağlamaları ek koşul
   olarak aranır.
```

**RAGAS Scores:**
- Faithfulness: 1.0
- Answer Correctness: 0.628
- Answer Relevancy: 0.418
- Composite Score: 0.614

================================================================================
## TEST Q35: Mezuniyet başvurusu nasıl yapılır?
================================================================================

**Expected Answer:**
> Mezuniyet başvurusu akademik takvimde belirtilen tarihlerde SIS üzerinden
   yapılır. Öğrencilerin programa ait tüm mezuniyet yükümlülüklerini (gerekli
   dersler, kredi sayıları) tamamlaması ve genel not ortalamasının en az 2.00
   olması gerekir. Fakülte/Enstitü yönetimi gerekli kontrolleri yapar ve ÖK
   diplomayı düzenler.

**Qwen 32B Answer:**
```
Mezuniyet başvurusu, akademik takvimde belirtilen süreler içinde Öğrenci
   Bilgi Sistemi (ÖBS) üzerinden yapılır. Öğrenciler, her dönem aldıkları
   dersler ve mezuniyet kriterlerini takip ederek, başvurabilecekleri
   dönemlerde elektronik form doldurarak Öğrenci Kaynakları Birimine (ÖK)
   iletir. Fakülteler/Enstitüler, başvuruları inceleyip onaylar ve sonuçları
   ÖK'ya gönderir. Mezuniyet hakkı elde edemeyenlere, başvuruları işleme
   alınamadığı bildirilir. Lisansüstü öğrenciler için ayrıca, programın
   mezuniyet yükümlülüklerini yerine getirip genel not ortalamasının en az
   3,00 olması gerekir. Gecikmiş başvurular için "Geç Mezuniyet Başvuru
   Formu" kullanılır.
```

**RAGAS Scores:**
- Faithfulness: 1.0
- Answer Correctness: 0.483
- Answer Relevancy: 0.45
- Composite Score: 0.596

================================================================================
## TEST Q36: Kayıt dondurma şartları nelerdir?
================================================================================

**Expected Answer:**
> Dönem izni (kayıt dondurma); sağlık, maddi, aile, kişisel, akademik ve
   beklenmedik zorunlu olaylar gibi nedenlerle, ayrıca askerlik, gözaltı,
   tutukluluk veya mahkûmiyet durumlarında verilebilir. İzin gerekçesine
   ilişkin belgeler eklenerek dilekçe ile derslerin başlamasını takip eden 4.
   haftanın son iş gününe kadar ilgili fakülte dekanlığına başvurulur. Bir
   defada en çok 2 dönem, toplam 4 dönem izin verilebilir.

**Qwen 32B Answer:**
```
Bu konuda bilgim yok. Verilen belgelerde kayıt dondurma işlemi için
   belirli şartlar veya koşullar açıkça belirtilmemiştir. Ancak, kayıt
   dondurulan öğrencilerin ikamet izninin iptal edildiği ve burs durumunun
   dondurulduğu bilgileri yer almaktadır.
```

**RAGAS Scores:**
- Faithfulness: 0.5
- Answer Correctness: 0.137
- Answer Relevancy: 0.0
- Composite Score: 0.358

================================================================================
## TEST Q38: How does the Leave of Absence process work?
================================================================================

**Expected Answer:**
> Leave of absence (dönem izni) is granted for health, financial, family,
   personal, academic, and emergency reasons. Students must apply with
   supporting documents to the Faculty Dean's Office by the end of the 4th
   week after classes start. At most 2 semesters at once and 4 total are
   allowed. Students must return laptops and residence permits are cancelled
   during leave. Tuition percentage depends on application timing (25% if
   applied by 4th week).

**Qwen 32B Answer:**
```
The Leave of Absence process requires students to apply within specific
   deadlines and pay a percentage of tuition fees based on the timing of the
   request. For undergraduate students, if the request is submitted by the
   end of the 4th week following the academic year's start, 25% of the
   semester fee is paid for a semester leave, and 25% for an academic year
   leave. Applications submitted after the 4th week but before the 7th week
   require 50% payment for a semester leave and 50% for an academic year
   leave. If the request is made after the 7th week up to the 10th week, full
   tuition fees are required. Students enrolled from 2020-2021 onward must
   pay the full fee for
```

**RAGAS Scores:**
- Faithfulness: 0.75
- Answer Correctness: 0.343
- Answer Relevancy: 0.791
- Composite Score: 0.497

================================================================================
## TEST Q40: Zorunlu staj başvurusu nasıl yapılır?
================================================================================

**Expected Answer:**
> Zorunlu staj başvurusu OMS (Online Management System) üzerinden staj
   talebi oluşturularak yapılır. Öğrenciler Yeni Staj Projesi Formu
   (FIPAR-C71001-02) doldurur ve kurumlarda yapacakları projenin tanımını
   girer. Staj Ofisi, kabul edildikleri kuruma teslim edilmek üzere Zorunlu
   Staj Belgesi verir.

**Qwen 32B Answer:**
```
Zorunlu staj başvurusu için öğrenciler, öncelikle staj yapmak istedikleri
   kurum veya kuruluşa irtibata geçerek yer ve projeyi belirler. Yurt
   içi/yurt dışı staj başvurusunda bulunacak öğrenciler, **Haziran ayı
   ortasına kadar** (mezuniyet hakkı olanlar için **Mayıs ayı başında**) staj
   talebini sunar. SSBF programına kayıtlı öğrenciler, **güz dönemi sonunda
   en az 3.00 GNO** sağlamalı ve **Mayıs'ın ilk haftasına kadar** bir öğretim
   üyesiyle iletişime geçerek araştırma projesini oluşturmalı, bu projeyi
   SSBF Dekanına başvurmalıdır.   Staj projesi onaylandıktan sonra
   öğrenciler, **OMS sistemine girerek Yeni Staj Projesi Formu** üzerinden
   projenin tanımını ve detayları doldurur. Staj danışmanı olarak atanan
   öğretim üyesi onayı gereklidir. Uluslararası staj başvuruları için, ikili
   anlaşmalar kapsamında **minimum 4 hafta** süresi zorunludur. Başvuru için
   **şubat ayından itibaren** özgeçmiş, transkript ve ek belgeler
   gönderilmelidir. Başvurular, öğretim üyesi mülak
```

**RAGAS Scores:**
- Faithfulness: 1.0
- Answer Correctness: 0.173
- Answer Relevancy: 0.308
- Composite Score: 0.648

================================================================================
## TEST Q42: IAESTE ile uluslararası staj nasıl yapılır?
================================================================================

**Expected Answer:**
> IAESTE ile uluslararası staj başvurusu, Kariyer Geliştirme ve Staj Ofisi
   tarafından duyurulan tarihlerde yapılır. Öğrenciler başvuru belgelerini
   tamamlayarak IAESTE programı kapsamında yurtdışında staj yapar. Kabul
   edilen öğrenciler Zorunlu Staj Belgesi alır ve stajı tamamladıktan sonra
   değerlendirme formlarını sunar.

**Qwen 32B Answer:**
```
IAESTE ile uluslararası staj yapmak için öncelikle başvuru kriterlerine
   uygun olunmalıdır: Lisans öğrencileri en az 2. sınıfta, Lisansüstü
   öğrencileri ise Yüksek Lisans düzeyinde olmak koşuluyla Genel Not
   Ortalaması (GNO) 2,00 ve üzeri, Diller Okulu tarafından yapılan yabancı
   dil sınavından 100 üzerinden en az 70 puan alınmalıdır. Başvurular her yıl
   Staj Ofisi tarafından ilan edilen son tarihe kadar yapılmalıdır, bu tarih
   genellikle Kasım ayının sonu olarak belirtilmiştir. Ekim ortasında Diller
   Okulu ile birlikte yabancı dil sınavı (İngilizce, Almanca, Fransızca)
   tarihleri belirlenir ve öğrenciler özgeçmişleriyle birlikte transkript
   göndererek başvururlar. Kasım sonuna kadar, ev sahibi kuruluşların
   doldurduğu İş Teklifi Formu ve üst yazı ile birlikte öğrenci listesi
   IAESTE Merkez’e gönderilir. Şubat sonuna kadar kontenjanlar istenilen
   özelliklere göre en yüksek puanlı öğrencilere eşleştirilir ve onay alınır.
   Onaylanmayan öğrenciler staja gidemez. Staj süresi en az 4 hafta olmak
   üzere, ev sahibi kuruluşun belirlediği maksimum süreye göre değişir. Staja
   giden öğrencilere ev sahibi kuruluşun öngördüğü meblağ kadar aylık ödeme
   yapılır.
```

**RAGAS Scores:**
- Faithfulness: 0.941
- Answer Correctness: 0.203
- Answer Relevancy: 0.647
- Composite Score: 0.607

================================================================================
## TEST Q44: What is the minimum GPA requirement for Erasmus?
================================================================================

**Expected Answer:**
> Minimum GPA is 2.20 for undergrad and 2.5 for graduate students.

**Qwen 32B Answer:**
```
The minimum GPA requirement for Erasmus+ Internship Mobility is **2.20 for
   undergraduate students** and **2.5 for graduate students**. For exchange
   programs, the requirement is the same: **2.20 for undergraduates** and
   **2.5 for graduates**. These thresholds apply to eligibility for
   participation and are part of the Erasmus+ Score calculation, which
   combines 50% GPA and 50% language proficiency score.
```

**RAGAS Scores:**
- Faithfulness: 0.375
- Answer Correctness: 0.65
- Answer Relevancy: 0.86
- Composite Score: 0.687

================================================================================
## TEST Q45: How many books can I borrow from the library?
================================================================================

**Expected Answer:**
> Undergraduate, graduate, and exchange students can borrow 60 books for 60
   days, 5 multimedia items for 7 days, and 2 bound periodicals for 7 days.

**Qwen 32B Answer:**
```
Undergraduate, graduate, and exchange students at Sabancı University can
   borrow up to 60 books for a period of 60 days. Book loans can be renewed
   unlimitedly for the same duration, provided no other user has requested
   the item. This limit applies specifically to the "General Book Collection"
   and "Large-Size Book Collection," as students cannot access the "Faculty
   Reserve Collection" or "Student Reserve Collection." Non-student groups,
   such as Package 2 or Package 3 users, have lower borrowing limits (e.g.,
   10 books for 30 days), but these do not pertain to regular student
   accounts.
```

**RAGAS Scores:**
- Faithfulness: 0.778
- Answer Correctness: 0.362
- Answer Relevancy: 0.382
- Composite Score: 0.649

================================================================================
## TEST Q47: Lisans mezuniyeti için kaç kredi gerekiyor?
================================================================================

**Expected Answer:**
> Lisans mezuniyeti için belirli bir sabit kredi sayısı yoktur. Kayıtlı
   olunan diploma programının gerektirdiği tüm ders ve kredi
   yükümlülüklerinin tamamlanması ve SÜ kredilerine göre hesaplanan genel not
   ortalamasının en az 2.00 olması gerekir.

**Qwen 32B Answer:**
```
Bu konuda bilgim yok. Verilen belgelerde lisans mezuniyeti için gerekli
   toplam kredi miktarı doğrudan belirtilmemiştir. Ancak, ders muafiyet
   prosedürlerinde "özel öğrenci olarak alınan dersler ve öğrenci değişim
   programları kapsamında alınan derslerin kredi toplamının mezuniyet
   kredisinin %33'ünü geçmemesi" gerektiği ifade edilmiştir. Lisans
   mezuniyeti kredisinin tam değerini öğrenmek için Sabancı Üniversitesi
   Lisans Eğitim ve Öğretim Yönetmeliği'ne başvurulması gerekir.
```

**RAGAS Scores:**
- Faithfulness: 0.5
- Answer Correctness: 0.176
- Answer Relevancy: 0.0
- Composite Score: 0.453

================================================================================
## TEST Q49: Çift anadal programı için minimum kaç ek kredi gerekir?
================================================================================

**Expected Answer:**
> Çift anadal diploma programlarında anadal diploma programı yükümlülükleri
   dışındaki fark derslerle en az 30 SÜ ve 60 AKTS kredisi tamamlanması
   gerekir.

**Qwen 32B Answer:**
```
Çift anadal programına kabul edilen öğrencilerin, anadal programı
   yükümlülükleri dışında en az **30 SÜ (başarılı birim)** ve **60 AKTS
   kredisi** tamamlaması gerekir. Bu kredi ve birim gereklilikleri, çift
   anadal programının mezuniyet koşullarına ek olarak, anadal programının
   zorunluluklarını da yerine getirmek şartıyla uygulanır.
```

**RAGAS Scores:**
- Faithfulness: 0.5
- Answer Correctness: 0.717
- Answer Relevancy: 0.408
- Composite Score: 0.676

================================================================================
## TEST Q50: Rektörlük Araştırma Fonu bütçe limiti ne kadardır?
================================================================================

**Expected Answer:**
> Rektörlük Araştırma Fonu bütçesi deneysel olmayan projeler için 300.000
   TL, deneysel projeler için 600.000 TL'dir. Proje süresi en fazla 36 aydır.
   Laboratuvar ekipmanı, sarf malzemesi ve deneysel saha çalışmalarının
   maliyetleri toplam proje bütçesinin %30'undan fazlasını oluşturması
   durumunda proje deneysel olarak değerlendirilir.

**Qwen 32B Answer:**
```
Rektörlük Araştırma Fonu bütçe limiti, araştırma türüne göre değişir.
   Teorik araştırma alanları için bütçe en fazla 300.000 TL, deneysel
   araştırma alanları için ise en fazla 600.000 TL olur. Ayrıca, laboratuvar
   ekipmanı, sarf malzemesi ve deneysel saha maliyetlerinin toplam proje
   bütçesinin %30'unu aşması durumunda proje "deneysel" olarak
   değerlendirilir. Bu durum, bütçe sınırlarının uygulanmasında
   belirleyicidir.
```

**RAGAS Scores:**
- Faithfulness: 0.667
- Answer Correctness: 0.362
- Answer Relevancy: 0.624
- Composite Score: 0.652

================================================================================
## TEST Q52: Yurtlardan ayrılma prosedürü nasıl işler?
================================================================================

**Expected Answer:**
> Üniversiteden 1 ayı aşan süre için ayrılan öğrencilerin yurt kaydı Öğrenci
   Kaynakları Birimi tarafından bildirilir ve yurt yönetimi tarafından
   silinir. Öğrenciye bildirim sonrası 2 gün içinde yurtla ilişkilerini
   kesmesi gerekir. Öğrenci Yurt Odası Teslim Formu doldurulur. Depozito
   iadesi hasar yoksa kesintisiz yapılır.

**Qwen 32B Answer:**
```
Yurtlardan ayrılma prosedürü, öğrencinin Üniversite’den 1 ayı aşan süre
   için ayrılması durumunda uygulanır. Öğrenci Kaynakları Birimi’nden gelen
   bilgi doğrultusunda yurt kaydı silinir ve öğrenciye bildirilir. Yurt kaydı
   silinen öğrenciler, bildirimden sonra 2 gün içinde yurtla tüm ilişkilerini
   kesmek zorundadır. İlişkinin kesilmesi kapsamında öğrenci, yurt odasını
   teslim etmek ve gerekli formları doldurmakla yükümlüdür. Yurtta kalan
   eşyalar öğrenci tarafından alınmazsa, bu eşyalar atık olarak
   değerlendirilir. Lisansüstü öğrenciler, ayrılmaları sırasında yatırdıkları
   depozito ücretini, yurtta herhangi bir zarar bırakmadıkları takdirde tam
   olarak iade alırlar.
```

**RAGAS Scores:**
- Faithfulness: 0.778
- Answer Correctness: 0.778
- Answer Relevancy: 0.448
- Composite Score: 0.659

================================================================================
## TEST Q56: Cinsel taciz şikayeti nasıl yapılır?
================================================================================

**Expected Answer:**
> Cinsel taciz veya ayrımcılık şikayetleri, herhangi bir zamanda Komiteye
   veya Üniversite Yönetimine sunulabilir. Komite, şikayetçinin
   akademik/çalışma hayatının olumsuz etkilenmemesini güvence altına alır.
   Şikayetler; çalışanlar, öğrenciler veya üniversite topluluğunun herhangi
   bir üyesi tarafından yapılabilir. Özel bir komite, iddiaları belirlenen
   süreler içinde inceler.

**Qwen 32B Answer:**
```
Cinsel taciz şikayeti için başvuran, olayı yaşadığını düşündüğü anda,
   olayın niteliğinden emin olmasa dahi, Cinsel Tacize Karşı Önlem ve Destek
   Komitesi veya güvenilen bir kişiyle iletişime geçebilir. Başvuran,
   yaşadığı durumu anlatır ve kendisine destek mekanizmaları, şikâyet süreci
   ve ilgili disiplin süreçleri hakkında bilgi verilir. Şikâyetçi tarafın
   onayı olmadan hiçbir işlem başlatılamaz. Başvuru süresi için bir zaman
   sınırı yoktur; ancak resmi şikayet süreci başladıktan sonra Ceza Kanunu’na
   göre en geç 6 ay içinde şikayetçi olmazsa soruşturma yapılamaz. Başvuran,
   güvenilen bir kişi ya da birime (akademik danışman, psikolog, dekanlık
   vb.) danışarak süreci başlatabilir. Ayrıca, cinsel tacize uğradığını
   düşünen kişi, delil olarak e-posta, mesaj gibi belgeleri saklamalı ve bu
   süreçte yalnız olmadığını bilmelidir.
```

**RAGAS Scores:**
- Faithfulness: 0.889
- Answer Correctness: 0.178
- Answer Relevancy: 0.0
- Composite Score: 0.383

================================================================================
## TEST Q57: Sabancı Üniversitesi nerede?
================================================================================

**Expected Answer:**
> Sabancı Üniversitesi İstanbul Tuzla'da bulunmaktadır.

**Qwen 32B Answer:**
```
Sabancı Üniversitesi Tuzla-İstanbul'da, Orhanlı bölgesinde yer alır.
   Üniversitenin adresi: **Orta Mahalle, Üniversite Caddesi No: 27, 34956
   Tuzla, İstanbul** olarak belirtilmiştir. Bu bilgi, belgede yer alan tüm
   ilgili bölümlerde (kalite, çevre, iş sağlığı ve güvenliği el kitapları)
   aynı şekilde tekrarlanmaktadır.
```

**RAGAS Scores:**
- Faithfulness: 0.5
- Answer Correctness: 0.578
- Answer Relevancy: 0.77
- Composite Score: 0.689

================================================================================
## TEST Q58: Kampüs içi trafik cezası itirazı nasıl yapılır?
================================================================================

**Expected Answer:**
> Trafik kurallarının ihlali halinde Güvenlik Müdürlüğünce e-mail yoluyla
   ilgili kişi uyarılır ve ceza puanı durumu bildirilir. Sürücünün itiraz
   hakkı vardır.

**Qwen 32B Answer:**
```
Kampüs içi trafik cezasına itiraz edilmesi durumunda, sürücü Güvenlik
   Müdürlüğüne başvurmalıdır. Güvenlik Müdürlüğü, itirazı inceleyerek karar
   verir. Ayrıca, verilen cezaya karşı itiraz hakkı kullanılmadan idari
   yargıya da başvurulabilir. Üniversite Yönetim Kurulu, itirazları onbeş gün
   içinde karara bağlamakla yükümlüdür. Trafik kurallarına uymayan
   sürücülerin araçları, yasak yerlere bırakılması durumunda kampüsten
   çekilebilir. Uygun olmayan davranışlar tekrar ederse ceza puanı iki katına
   çıkarılır. Puan kayıtları ve taşıt pullarının iptali Güvenlik Yöneticiliği
   tarafından yapılır.
```

**RAGAS Scores:**
- Faithfulness: 0.857
- Answer Correctness: 0.434
- Answer Relevancy: 0.502
- Composite Score: 0.588

================================================================================
## TEST Q59: How does the lost and found procedure work on campus?
================================================================================

**Expected Answer:**
> Lost property found on campus is turned over to the Security Directorate.
   The Security Directorate records the items and attempts to return them to
   their owners. Items that remain unclaimed within 30 days are processed in
   accordance with the relevant procedure.

**Qwen 32B Answer:**
```
The lost and found procedure on campus involves reporting found items by
   calling extension 3565 or delivering them to the Security Office in
   Building D2. Security officers prepare a two-copy Found Item Return Form,
   signed by both the finder and the officer, with the original retained by
   the Security Unit and the second copy given to the finder. Items are
   numbered (e.g., "01-24/01/01" format) and stored, with valuables separated
   into labeled bags. Unclaimed items are kept for one year. In cases of loss
   or theft on campus, individuals must notify the Security Unit via phone
   (5555) and the IT Unit through their ticketing system.
```

**RAGAS Scores:**
- Faithfulness: 1.0
- Answer Correctness: 0.188
- Answer Relevancy: 0.921
- Composite Score: 0.661

================================================================================
## TEST Q61: Üniversiteden uzaklaştırma cezası alan öğrenci yurtta kalabilir mi?
================================================================================

**Expected Answer:**
> Üniversiteden uzaklaştırma cezası alan öğrenciler, ceza süresi boyunca
   yurtlarda konaklayamaz. Yurtlar Yönergesi'ne göre disiplin suçu nedeniyle
   uzaklaştırma durumunda yurt kaydı iptal edilir ve öğrenci bildirildikten
   sonra 2 gün içinde yurtla ilişkisini kesmek zorundadır.

**Qwen 32B Answer:**
```
Üniversite'den uzaklaştırma cezası alan öğrenciler, ceza süresi boyunca
   yurtlarda konaklayamaz. Yurtlar Yönergesi'ne göre, uzaklaştırma cezası
   süresi içinde yurtta kalma talepleri kabul edilmez. Ayrıca, disiplin
   cezası nedeniyle uzaklaştırılan öğrencilerin yurtta kalma durumu,
   "Yurtlardan Ayrılma Prosedürü" kapsamında değerlendirilir. Bu sürede
   üniversite kampüsüne girmelerine izin verilmediğinden, yurtta kalma imkânı
   doğrudan ortadan kalkar. Uzaklaştırma cezası alındığında, öğrenci
   yurtlardan ayrılır ve ceza süresi boyunca tekrar kabul edilmez.
```

**RAGAS Scores:**
- Faithfulness: 0.833
- Answer Correctness: 0.392
- Answer Relevancy: 0.546
- Composite Score: 0.575

================================================================================
## TEST Q65: Uzaklaştırma cezası alan öğrenci ücret iadesi alabilir mi?
================================================================================

**Expected Answer:**
> Uzaklaştırma cezası alan öğrencilere herhangi bir ücret iadesi yapılmaz.
   Disiplin suçu nedeniyle uzaklaştırılan öğrenciler, ceza dönemindeki
   öğrenim ücretini ödemekle yükümlüdür. Burs alan öğrencilerin bursu
   uzaklaştırma süresince dondurulur.

**Qwen 32B Answer:**
```
Uzaklaştırma cezası alan öğrencilere herhangi bir ücret iadesi yapılmaz.
   Disiplin suçu nedeniyle Üniversiteden uzaklaştırılan öğrenciler, ödemekle
   yükümlü oldukları dönem öğrenim ücretini tamamıyla ödemek zorundadır. Bu
   durum, Sabancı Üniversitesi Temel Geliştirme Yılı Eğitim ve Öğretim
   Yönetmeliği'nde açıkça belirtilmiştir. Ayrıca, burs ve mali destek
   yönergesine göre, bir dönem uzaklaştırma cezası alan öğrencilerin bursu
   kesilir ve bu dönem için öğrenim ücretinin %25’i ödenir; ancak bu ödeme,
   ücret iadesi değil, bir yükümlülük olarak kalmaktadır.
```

**RAGAS Scores:**
- Faithfulness: 0.833
- Answer Correctness: 0.585
- Answer Relevancy: 0.523
- Composite Score: 0.671

================================================================================
## TEST Q67: Sabancı Üniversitesi'nde kaç fakülte vardır?
================================================================================

**Expected Answer:**
> Sabancı Üniversitesi'nde üç fakülte vardır: Mühendislik ve Doğa Bilimleri
   Fakültesi (FENS), Sanat ve Sosyal Bilimler Fakültesi (FASS) ve Yönetim
   Bilimleri Fakültesi (SOM).

**Qwen 32B Answer:**
```
Sabancı Üniversitesi'nde üç fakülte vardır. Bu fakülteler; Mühendislik ve
   Doğa Bilimleri Fakültesi, Sanat ve Sosyal Bilimler Fakültesi ile Yönetim
   Bilimleri Fakültesi'dir. Tüm belgelerde bu üç fakülte birlikte
   listelenmiştir. Ek bir fakülteye dair bilgi bulunmamaktadır.
```

**RAGAS Scores:**
- Faithfulness: 0.5
- Answer Correctness: 0.556
- Answer Relevancy: 0.711
- Composite Score: 0.673

================================================================================
## TEST Q78: Yurt başvuru ve kabul koşulları nelerdir?
================================================================================

**Expected Answer:**
> Yurt başvurusu https://dormapp.sabanciuniv.edu sistemi üzerinden yapılır.
   Kabul öncelik sırasına göre değerlendirme yapılır. Burs tipi odalarda burs
   alan öğrenciler konaklar. Yaz dönemi için ayrıca başvuru alınır. Ödeme
   koşulları Student Housing Instruction Letter ile belirlenir.

**Qwen 32B Answer:**
```
Yurt başvurusu ve kabul koşulları şunlardır:  1. **Başvuru Süreci**: Yurt
   başvuruları online sistem üzerinden (https://dormapp.sabanciuniv.edu)
   yapılır. Türk öğrenciler evrak yükleme zorunluluğu taşımaz, ancak yurt
   başvurusu sırasında Yurt Giriş Formu doldurulur.  2. **Yaz Dönemi
   Başvuru**: Yaz döneminde yurtta kalacak öğrenciler, bahar dönemi sona
   ermeden 3 hafta önce online başvuruda bulunmalıdır.  3. **Kabul
   Kriterleri**: Üniversite Yönetimi tarafından belirlenen öncelik sıralaması
   (burs sahipleri öncelikli), yurt binası kapasitesi ve bakım-onarım
   durumları dikkate alınır.  4. **Hukuki Şartlar**: 6 aydan fazla hapis
   cezası almış (taksirli suçlar hariç) ve kesinleşmiş mahkumiyeti bulunan
   öğrenciler kabul edilmez.  5. **Kabul Süresi**: Yurt kabulü en fazla bir
   akademik yıl için geçerlidir. Yeni dönem için yeniden başvuruda
   bulunulması gerekir.  6. **Devam Koşulu**: Güz dönemi yurta yerleşen
   öğrenciler,
```

**RAGAS Scores:**
- Faithfulness: 0.929
- Answer Correctness: 0.479
- Answer Relevancy: 0.493
- Composite Score: 0.696

================================================================================
## TEST Q82: Mezuniyet veya kayıt dondurma durumunda yurt kaydı ne olur?
================================================================================

**Expected Answer:**
> Kayıt dondurma, mezuniyet, izin ya da disiplin cezası gibi sebeplerle 1
   ayı aşan süre için ayrılan öğrencilerin yurt kaydı Öğrenci Kaynakları
   Birimi bilgisi doğrultusunda silinir. Yurt kaydı silinen öğrenci
   bildirildikten sonra 2 gün içinde yurtla ilişkisini keser. Depozito iadesi
   hasar yoksa yapılır.

**Qwen 32B Answer:**
```
Mezuniyet durumunda öğrenci yurt kaydı, mezuniyet tarihinden itibaren yurt
   yönetimi tarafından silinir ve öğrenci 2 gün içinde yurtla tüm
   ilişiklerini kesmek zorundadır. Kayıt dondurma durumunda ise öğrenci,
   kayıt dondurma tarihinden itibaren yurt kaydı silinir ve aynı 2 günlük
   sürede yurtla ilişkisini sona erdirmek gerekir. Ayrıca, mezuniyet veya
   kayıt dondurma nedeniyle üniversiteden ayrılan öğrencilerin ikamet
   izinleri de ilgili tarihten itibaren iptal edilir. Yurt ücreti iadesi için
   barınma süresi ve içinde bulunduğu ayın ücreti tamamen alınır, kalan
   ayların %50'si iade edilir. Yurt yönetimi ve mali işler, ücret iadesini
   hesap dökümüne göre yapar.
```

**RAGAS Scores:**
- Faithfulness: 1.0
- Answer Correctness: 0.432
- Answer Relevancy: 0.432
- Composite Score: 0.661

================================================================================
## TEST Q88: How does the electric vehicle charging process work?
================================================================================

**Expected Answer:**
> Electric vehicles must be used in ECO mode. When the charge indicator is
   at 45% or below, the vehicle must be plugged in for charging before
   returning the key. If the vehicle is charged off-campus at a non-
   contracted station, the expense must be declared with an invoice issued in
   the name of SU.

**Qwen 32B Answer:**
```
The electric vehicle charging process at Sabancı University involves the
   use of Electric Vehicle Charging Stations located within the SU Tuzla
   campus. These stations are part of the electrical systems managed by the
   Electrical Works Department, which ensures their maintenance and
   operational functionality. Vehicles must be registered with the Security
   Directorate for entry, and registered vehicles can freely access the
   campus, including charging stations, through Gate A. However, if there is
   suspicion about the driver’s identity (e.g., student or parent) or if
   visitors are present in the vehicle, the driver must stop for identity
   verification. Vehicles are required to be used solely for the purpose
   stated in their request, and the process of returning a vehicle includes
   ensuring it is charged. Specific technical or procedural details about the
   charging operation itself (e.g., payment methods, charging duration) are
   not outlined in the provided context.
```

**RAGAS Scores:**
- Faithfulness: 0.75
- Answer Correctness: 0.161
- Answer Relevancy: 0.596
- Composite Score: 0.363

================================================================================
## TEST Q89: Kampüsteki sağlık merkezinin acil hattı numarası nedir?
================================================================================

**Expected Answer:**
> Kampüsteki Sağlık Merkezi Acil Hattı 6666'dır. Ambulans gereken durumlarda
   en yakın telefondan bu hat aranmalıdır.

**Qwen 32B Answer:**
```
Kampüs içinde acil durumlar için Sağlık Merkezi Acil Hattı **6666**
   numarası aranmalıdır. Kampüs dışından arama yapılıyorsa, numara **0216 568
   7666** olarak kullanılır. Bu bilgi, tüm belgelerde (ambulans prosedürleri,
   sağlık hizmetleri ve acil durum planları) aynı şekilde teyit edilmiştir.
```

**RAGAS Scores:**
- Faithfulness: 0.667
- Answer Correctness: 0.505
- Answer Relevancy: 0.555
- Composite Score: 0.685

================================================================================
## TEST Q93: Entegrasyon Projelerine nasıl başvurulur?
================================================================================

**Expected Answer:**
> EPD başvurusu tam zamanlı öğretim üyelerinin işe başlama tarihinden
   itibaren 1 yıl içinde yapılmalıdır. PY, EPD Teklif Formunu doldurur ve
   proje partnerine yönlendirir. Proje partneri teknik uygunluğu
   değerlendirir. Bütçe 50.000 TL'ye kadardır. Başvuru formu proje özeti,
   yönetim planı ve harcama kalemlerinin gerekçesini içerir.

**Qwen 32B Answer:**
```
Entegrasyon Projeleri Desteği (EPD) başvurusu, Sabancı Üniversitesi'ne tam
   zamanlı öğretim üyesi olarak atananlar tarafından, işe başlama tarihinden
   itibaren **1 yıl içinde** yapılır. Başvuru süreci **başvuru, değerlendirme
   ve kabul** aşamalarını kapsar. Proje bütçesi **50.000 TL** olup,
   makine/teçhizat alımı, seyahat giderleri (yurtdışı için günlük 300 USD),
   öğrenci çalışma giderleri (30 TL/saat) ve temsil/tanıtma masrafları gibi
   kalemlerle detaylandırılır. Başvuruda proje özeti, iş-zaman çizelgesi ve
   bütçe gerekçesi zorunludur. Desteği alan projelerin sonunda **en az iki
   dış kaynaklı hibe başvurusu** veya bir garantili dış finansman sağlanması
   beklenir. Proje yürütücülerinin bu koşulları karşılayamaması, akademik
   performans değerlendirmesine yansır.
```

**RAGAS Scores:**
- Faithfulness: 1.0
- Answer Correctness: 0.492
- Answer Relevancy: 0.561
- Composite Score: 0.682

