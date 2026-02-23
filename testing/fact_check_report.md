# RAG Test Suite — Fact-Checking Report
## Sabancı Üniversitesi Source Document Verification

---

## Q1: Kütüphaneden (Bilgi Merkezi) öğrenciler kaç kitap, kaç gün ödünç alabilir?

**Source File:** `iic-c840-02.json` — Ödünç Alma Yönergesi (IIC-C840-02)

**Exact Text (Section 1.3):**
> "Sabancı Üniversitesi Lisans, Lisansüstü ve Değişim Öğrencileri, **60 gün süre ile 60 adet kitap**, 7 gün süre ile 5 adet multimedya, 7 gün süre ile 2 adet ciltli süreli yayın, 3 gün süre ile 5 popüler dergi ödünç alabilir."

**Note on "Paket 2, 30 gün, 10 adet":** This refers to **Sanayi ve Ticaret Kuruluşları (Dış Kullanıcılar)** — external industrial/commercial members, NOT students.
> "Paket 2 kullanıcıları, yayınlardan yerinde yararlanacağı gibi, 30 gün süre ile 10 adet kitap ödünç alabilir."

**Recommended Expected Answer:**
Sabancı Üniversitesi Lisans, Lisansüstü ve Değişim Öğrencileri, 60 gün süre ile 60 adet kitap ödünç alabilir. Ayrıca 7 gün süre ile 5 adet multimedya, 7 gün süre ile 2 adet ciltli süreli yayın, 3 gün süre ile 5 popüler dergi ödünç alabilir. (Kaynak: IIC-C840-02, Bölüm 1.3)

**Verdict:** RAG answer (60 kitap, 60 gün) is **CORRECT**. The old expected answer ("Paket 2, 30 gün, 10 adet") was **WRONG** — it applies to external corporate users, not students.

---

## Q2: Kütüphanelerarası ödünç (ILL) hizmeti nasıl işler?

**Source File:** `iic-c820-04.json` — Kütüphanelerarası Ödünç Yönergesi (IIC-C820-04)

**Key Facts (from eval report cross-reference and iic-c840-03/iic-c840-04):**
- Sabancı Üniversitesi mensupları ILL hizmetinden yararlanabilir
- İstekler web form üzerinden yapılır
- Bir seferde en fazla 2 adet kaynak istenebilir
- Gelen kaynaklar 1 hafta içinde teslim alınmalıdır
- Teslim alınmayan kaynaklar için 6 ay süre ile hizmetten yararlanma engellenir

**Recommended Expected Answer:**
Kütüphanelerarası ödünç (ILL) hizmeti, Bilgi Merkezi bünyesinde bulunmayan kitapların diğer kurumlardan ödünç getirtilmesi hizmetidir. İstekler web form üzerinden yapılır. Bir seferde en fazla 2 adet kaynak istenebilir. Gelen kaynaklar 1 hafta içinde teslim alınmalıdır. Teslim alınmayan kaynaklar için kullanıcı 6 ay süre ile hizmetten yararlanamaz.

---

## Q3: Sabancı Üniversitesi'ndeki disiplin cezaları nelerdir?

**Source File:** `isr-c210-01.json` — Öğrenci Disiplin Yönergesi (ISR-C210-01)

**Exact Text (Section 3 — DİSİPLİN CEZALARI):**
1. **Uyarma:** Öğrencinin, öğrencilikle ilgili davranışlarında daha dikkatli olması gerektiği hususunda yazılı olarak ikaz edilmesi
2. **Kınama:** Öğrenciye öğrencilik ile ilgili kusurlu davranışlarından dolayı kınandığının yazı ile bildirilmesi
3. **Uzaklaştırma (1 haftadan 1 aya kadar):** Yükseköğretim kurumundan bir haftadan bir aya kadar uzaklaştırılma
4. **Uzaklaştırma (Bir yarıyıl için):** Yükseköğretim kurumundan bir yarıyıl uzaklaştırılma
5. **Uzaklaştırma (İki yarıyıl için):** Yükseköğretim kurumundan iki yarıyıl uzaklaştırılma
6. **Yükseköğretim Kurumundan Çıkarma:** Bir daha çıkarıldığı yükseköğretim kurumuna alınmamak üzere çıkarılma

**Recommended Expected Answer:**
Disiplin cezaları: 1) Uyarma, 2) Kınama, 3) Yükseköğretim kurumundan bir haftadan bir aya kadar uzaklaştırma, 4) Bir yarıyıl için uzaklaştırma, 5) İki yarıyıl için uzaklaştırma, 6) Yükseköğretim kurumundan çıkarma. Uyarma, kınama ve 1 haftadan 1 aya kadar uzaklaştırma cezaları Dekan/Enstitü Müdürü tarafından; 1-2 yarıyıl uzaklaştırma ve çıkarma cezaları Disiplin Kurulu tarafından verilir. (Kaynak: ISR-C210-01, Madde 1.4 ve Bölüm 3)

**Verdict:** There are effectively **6 distinct penalty categories** (not 5). The "1-2 yarıyıl uzaklaştırma" is actually two separate categories.

---

## Q4: Disiplin soruşturma sistemi nasıl işler?

**Source File:** `isr-c210-01.json` — Öğrenci Disiplin Yönergesi (ISR-C210-01)

**Key Facts (Section 2):**
- **Disiplin Amiri:** Dekanlar, Enstitü Müdürleri (müşterek alanlarda Rektör)
- **Soruşturmacı/Soruşturma Kurulu:** Üniversite Disiplin Kurulu Havuzu'ndan atanır (2 öğretim elemanı)
- **Üniversite Disiplin Kurulu Havuzu:** 25 öğretim elemanı + 5 fakülte idari çalışanı
- Disiplin Amiri soruşturma başlatılıp başlatılmayacağına karar verir
- Soruşturmacı raporu hazırlayıp Disiplin Amirine gönderir
- Disiplin Amiri nihai kararı verir
- Ceza, ÖK tarafından öğrenciye tebliğ edilir

**Recommended Expected Answer:**
Disiplin Amiri (Dekan/Enstitü Müdürü) soruşturma başlatılıp başlatılmayacağına karar verir. Soruşturmacı veya Soruşturma Kurulu (Üniversite Disiplin Kurulu Havuzu'ndan seçilen 2 öğretim elemanı) atanır. Soruşturmacı delilleri toplar, Soruşturma Raporu hazırlar ve Disiplin Amirine gönderir. Disiplin Amiri nihai kararı verir. Ceza ÖK tarafından öğrenciye tebliğ edilir. Havuz, 25 öğretim elemanı ve 5 idari çalışandan oluşur. (Kaynak: ISR-C210-01, Bölüm 2)

---

## Q5: Sınavda kopya çekmek hangi cezayı gerektirir?

**Source File:** `isr-c210-01.json` — Öğrenci Disiplin Yönergesi (ISR-C210-01)

**Exact Text (Section 3):**
- **Kınama** cezasını gerektiren fiiller: "5. Sınavlarda kopyaya **teşebbüs** etmek."
- **Uzaklaştırma (Bir yarıyıl için)** cezasını gerektiren fiiller: "6. Sınavlarda kopya **çekmek veya çektirmek**"  
- **Uzaklaştırma (İki yarıyıl için):** "Sınavlarda tehditle kopya çekmek, kopya çeken öğrencilerin sınav salonundan çıkarılmasına engel olmak, kendi yerine başkasını sınava sokmak veya başkasının yerine sınava girmek"

**Recommended Expected Answer:**
Sınavda kopya çekmek veya çektirmek, **bir yarıyıl için uzaklaştırma** cezasını gerektirir. Kopyaya teşebbüs etmek ise kınama cezasını gerektirir. Tehditle kopya çekmek veya başkasının yerine sınava girmek ise iki yarıyıl uzaklaştırma cezasını gerektirir. (Kaynak: ISR-C210-01, Bölüm 3)

**Verdict:** Important distinction — "kopya çekmek" = **bir yarıyıl uzaklaştırma** (not "1 haftadan 1 aya kadar uzaklaştırma" as some RAG answers suggest). "Kopyaya teşebbüs" = kınama.

---

## Q6: Burs devam koşulları nelerdir? (Üstün Akademik Başarı bursu için GNO)

**Source File:** `isr-c160-01.json` — Burs ve Mali Destek Yönergesi (ISR-C160-01)

**Exact Text (Section 1.2.i.d — Üstün Akademik Başarı Bursu):**
> "Burs değerlendirmesine alınabilmesi için öğrencilerin ... aşağıdaki koşulları sağlaması gerekir:
> - Lisans öğreniminde normal öğrenim süresi içerisinde olunması,
> - İzinli geçirilen süreler dışında lisans seviyesinde en az iki dönemin bitirilmiş olması,
> - Lisans programının ilk yılında olan öğrenciler için **34 SÜ kredinin**; ara sınıflarda olan öğrenciler için ise son iki dönemde en az **30 SÜ kredinin** kazanılmış olması,
> - Ağırlıklı genel not ortalamasının (GNO) **en az 3.00** olması,
> - Son iki döneminden birinde dönem not ortalamasının (DNO) **en az 3.00** olması"

**"Akademik Başarı ve İhtiyaç" Bursu (Section 1.2.ii.e):**
> "Ağırlıklı genel not ortalamasının (GNO) **en az 2.50** olması"

**ÖSYM Giriş Bursları (Section 1.4.i):**
> "Üniversite'ye ilk girişte sağlanan burslar, akademik başarı durumuna **bakılmaksızın**, SÜ'ye kayıt tarihi itibarıyla normal öğrenim süresince kesintisiz devam eder."

**Recommended Expected Answer:**
Üstün Akademik Başarı bursu için GNO en az 3.00 ve son iki dönemden birinde DNO en az 3.00 olmalıdır. Akademik Başarı ve İhtiyaç bursu için GNO en az 2.50 gerekir. ÖSYM giriş bursları ise akademik başarıya bakılmaksızın normal öğrenim süresince devam eder. Disiplin cezası burs durumunu etkiler: kınama/1 haftadan 1 aya kadar uzaklaştırma → burs devam eder; 1-2 dönem uzaklaştırma → burs kesilir, %25 ödenir; çıkarma → burs tamamen kesilir. (Kaynak: ISR-C160-01, Bölüm 1.2 ve 1.5)

---

## Q7: Lisansüstü tez savunma jürisinin yapısı nasıldır?

**Source Files Searched:** `isr-c520-01.json`, `isr-c520-02.json`, `isr-c510-01.json`, `isr-c510-02.json`

**Finding:** The ISR-C520 files cover **Ders Açma/Kapatma** (curriculum), and ISR-C510 files cover **Program Açma/Kapatma**. These do NOT contain thesis jury composition rules.

Thesis jury information is governed by the **Lisansüstü Eğitim ve Öğretim Yönetmeliği** (Graduate Education Regulation), which appears to be a YÖK-level regulation, not a Sabancı-specific yönerge in these files. The `ihr-s420-01.json` contains **faculty appointment** jüri details, not student thesis jüri.

**Status:** NOT FOUND in the preprocessed yönerge documents. The Lisansüstü Eğitim ve Öğretim Yönetmeliği may be available as a PDF in the `_converted_doc` directory or not in the current corpus.

**Note:** The test expected answer likely comes from the YÖK Lisansüstü Eğitim ve Öğretim Yönetmeliği, which specifies jüri composition (typically 3 or 5 members, at least one external).

---

## Q8: Çift anadal programına başvuru için minimum GNO ne olmalı?

**Source File:** `cift-anadal-yonergesi-isr-c290-02.json` — Çift Anadal Yönergesi (ISR-C290-02)

**Exact Text (Section 2.c):**
> "Öğrencinin çift anadal diploma programına başvurusunun geçerli olabilmesi için aşağıda belirtilen koşulları sağlaması gerekir:
> i. Programa kayıt olunacak dönemin başına kadar anadal diploma programında alınan tüm derslerden başarılı olunması
> ii. Başvuru yapılan dönemin not girişleri sonrasında oluşan GNO'nun **en az 3,20** olması
> iii. Anadal diploma programının ilgili sınıfında **başarı sıralamasında ilk %20**'de yer alınması"

**Başvuru Zamanı (Section 2.b):**
> "Öğrenciler, lisans öğrenimlerinin en erken üçüncü, en geç beşinci dönemlerinin başında programa başlayacakları şekilde; en erken ikinci en geç dördüncü dönemlerinde çift anadal başvurusu yapabilirler."

**Recommended Expected Answer:**
Çift anadal diploma programına başvuru için GNO en az 3.20 olmalı ve öğrenci anadal diploma programının ilgili sınıfında başarı sıralamasında ilk %20'de yer almalıdır. Başvuru en erken 2. dönemde, en geç 4. dönemde yapılabilir. Tüm derslerden başarılı olunması gerekir. (Kaynak: ISR-C290-02, Bölüm 2)

---

## Q9: Ders ekleme-bırakma (add/drop) süresi ne kadardır?

**Source File:** `isr-c270-01.json` — Akademik Takvim Yönergesi (ISR-C270-01)

**Exact Text:**
> "Ders Ekleme-Bırakma: ... Lisans ve Lisansüstü için, 'Derslerin Başlaması' tarihinden itibaren **2. hafta içinde**. BT, ETM, VA, CYSEC ve MBA için, 'Derslerin Başlaması' tarihinden itibaren **2. ya da 3. hafta içinde**."

**Recommended Expected Answer:**
Lisans ve lisansüstü öğrenciler için ders ekleme-bırakma (add/drop) süresi, derslerin başlamasından itibaren 2. hafta içindedir. BT, ETM, VA, CYSEC ve MBA programları için ise 2. veya 3. hafta içindedir. Kesin tarihler her dönem Akademik Takvim'de belirlenir. (Kaynak: ISR-C270-01)

---

## Q10: Kayıt dondurma (dönem izni) koşulları nelerdir?

**Source Files Searched:** `isr-c210-05.json`, `isr-c130-01.json`, `isr-c160-01.json`

**Partial Findings:**
- `isr-c210-05.json` (Yabancı Uyruklu İkamet) mentions: "kayıt donduran öğrencinin ikamet izni …kayıt dondurma tarihinden itibaren ikamet kartları iptal olur"
- `isr-c160-01.json` (Burs) mentions the effect on scholarships: "burslar, öğrencinin dönem izni alması durumunda dondurulur"
- Specific conditions for requesting kayıt dondurma/dönem izni are likely in **Lisans Eğitim ve Öğretim Yönetmeliği** or a specific prosedür, not found as a standalone yönerge JSON.

**Status:** Kayıt dondurma/dönem izni procedure details NOT found as a separate yönerge. Effects on burs and ikamet are documented. The detailed conditions are likely in the Yönetmelik (regulation) documents.

---

## Q11: Mezuniyet için gereken minimum kredi sayısı nedir?

**Source Files Searched:** No specific `isr-c240*` graduation directive was read in detail.

**Status:** Graduation credit requirements are in the **Lisans Eğitim ve Öğretim Yönetmeliği** and program-specific requirements. The `isr-c240-01.json` (Mezuniyet Denetimi ve Diploma Düzenleme Yönergesi) and `isr-c240-03.json` (Yan Dal) exist but weren't searched for credit totals.

---

## Q12: Bir dersi en fazla kaç kez tekrar alabilir?

**Status:** NOT searched in detail. This is governed by the **Lisans Eğitim ve Öğretim Yönetmeliği**.

---

## Q13: Staj başvurusu nasıl yapılır?

**Source File (Reference):** `IIPAR-C710-02` — Uluslararası Staj Yönergesi

Q1 of the test output shows the Erasmus staj minimum GNO: Lisans 2.20, Lisansüstü 2.5

**Status:** Staj yönerge files exist but were not read in detail for the domestic staj process.

---

## Q14: Transkript nasıl alınır?

**Status:** NOT searched. Likely in a PSR (prosedür) document.

---

## Q15: Öğrenci belgesi nasıl alınır?

**Status:** NOT searched. Likely in a PSR document.

---

## Q16: Cinsel taciz şikâyeti nasıl yapılır?

**Source File (Reference):** `cinsel-taciz-karsisinda-uygulanacak-yontem-ve-alinacak-onlemler-ipo-a510-01.json` — exists in yonerge directory.

**From isr-c210-01.json (Discipline Directive):**
- "Yükseköğretim kurumlarında cinsel tacizde bulunmak" → **İki yarıyıl için uzaklaştırma**
- "Kişilerin vücudu üzerinde cinsel davranışlarda bulunmak suretiyle cinsel dokunulmazlıklarını ihlal etmek" → **Yükseköğretim kurumundan çıkarma**

**Status:** The specific complaint procedure file exists but was not read for detailed content.

---

## Q17: Disiplin cezasına itiraz süresi ne kadardır?

**Source File:** `isr-c210-01.json` — Öğrenci Disiplin Yönergesi (ISR-C210-01)

**Exact Text (Section 1.8):**
> "Disiplin amirleri ve kurullarınca verilen disiplin cezalarına karşı **onbeş gün** içinde Üniversite Yönetim Kuruluna itiraz edilebilir. İtiraz halinde, itiraz mercii olan Üniversite Yönetim Kurulu, itirazı **onbeş gün** içinde kesin olarak karara bağlar."
> "İtiraz halinde, itiraz mercii olan Üniversite Yönetim Kurulu kararı inceleyerek verilen cezayı aynen kabul veya reddeder."
> "Verilen disiplin cezalarına karşı, itiraz hakkı kullanılmadan da idari yargı yoluna başvurulabilir."

**Recommended Expected Answer:**
Disiplin cezalarına karşı 15 gün içinde Üniversite Yönetim Kuruluna itiraz edilebilir. Üniversite Yönetim Kurulu itirazı 15 gün içinde kesin karara bağlar. Cezayı aynen kabul veya reddeder. Red halinde, disiplin kurulu/yetkili disiplin amiri red gerekçesini göz önünde bulundurarak itirazı karara bağlar. İtiraz hakkı kullanılmadan da idari yargı yoluna başvurulabilir. (Kaynak: ISR-C210-01, Madde 1.8)

---

## Q18: Erasmus hibe/destek miktarı nedir?

**Status:** The exchange programs yönerge (`degisim-programlari-kapsaminda-gelen-ogrenciler-yonergesi-iiro-c420-02.json`) was found but it covers INCOMING students, not outgoing. Outgoing student grant details are likely in `degisim-programlari-kapsaminda-giden-ogrenciler-yonergesi` or Erasmus prosedür files.

---

## Q19: Yatay geçiş için GNO şartı nedir?

**Source Files Searched:** Yatay geçiş references found in `cift-anadal-yonergesi-isr-c290-02.json` and `iciad-a320-05.json` but these are secondary mentions.

**From isr-c160-01.json (Burs Yönergesi):**
> "SÜ Lisans programlarına yatay geçişle kabul edilen öğrenciler burssuz öğrenci statüsü ile kayıt yaptırabilirler."

**Status:** The specific yatay geçiş GNO requirement is likely in the **Lisans Eğitim ve Öğretim Yönetmeliği** or a specific yatay geçiş yönergesi/prosedürü, not found as a standalone yönerge JSON.

---

## Summary Table

| Q# | Topic | Source File | Facts Found | Status |
|----|-------|------------|-------------|--------|
| Q1 | Kütüphane ödünç | iic-c840-02.json | 60 kitap / 60 gün (öğrenci) | ✅ VERIFIED — RAG correct, old expected WRONG |
| Q2 | ILL hizmeti | iic-c820-04.json | Web form, 2 kaynak limiti | ✅ VERIFIED (from eval report) |
| Q3 | Disiplin cezaları | isr-c210-01.json | 6 ceza türü (uyarma → çıkarma) | ✅ VERIFIED |
| Q4 | Soruşturma sistemi | isr-c210-01.json | Disiplin Amiri → Soruşturmacı → Rapor | ✅ VERIFIED |
| Q5 | Kopya çekmek cezası | isr-c210-01.json | Kopya çekmek = 1 yarıyıl uzaklaştırma | ✅ VERIFIED |
| Q6 | Burs devam koşulları | isr-c160-01.json | Üstün Akademik: GNO ≥ 3.00 | ✅ VERIFIED |
| Q7 | Tez jürisi yapısı | — | Not found in yönerge corpus | ❌ NOT FOUND |
| Q8 | Çift anadal GNO | isr-c290-02.json | GNO ≥ 3.20 + ilk %20 | ✅ VERIFIED |
| Q9 | Ders ekleme-bırakma | isr-c270-01.json | 2. hafta (Lisans/LÜ) | ✅ VERIFIED |
| Q10 | Kayıt dondurma | — | Partial (effects found, conditions not) | ⚠️ PARTIAL |
| Q11 | Mezuniyet kredisi | — | Not searched in detail | ❌ NOT SEARCHED |
| Q12 | Ders tekrar limiti | — | Not searched | ❌ NOT SEARCHED |
| Q13 | Staj başvurusu | IIPAR-C710-02 | Erasmus staj GNO found (Q1 test) | ⚠️ PARTIAL |
| Q14 | Transkript | — | Not searched | ❌ NOT SEARCHED |
| Q15 | Öğrenci belgesi | — | Not searched | ❌ NOT SEARCHED |
| Q16 | Cinsel taciz şikayeti | ipo-a510-01.json (exists) | Ceza: 2 yarıyıl/çıkarma (isr-c210-01) | ⚠️ PARTIAL |
| Q17 | İtiraz süresi | isr-c210-01.json | 15 gün | ✅ VERIFIED |
| Q18 | Erasmus hibe | — | Not found for outgoing | ❌ NOT FOUND |
| Q19 | Yatay geçiş GNO | — | Not found as standalone | ❌ NOT FOUND |

---

## Critical Corrections for Test Suite

### 1. Q1 — Library Borrowing (HIGHEST PRIORITY FIX)
**Current expected:** "Paket 2 kullanıcıları 30 gün süre ile 10 adet kitap ödünç alabilir."
**Correct expected:** "Sabancı Üniversitesi Lisans, Lisansüstü ve Değişim Öğrencileri, 60 gün süre ile 60 adet kitap ödünç alabilir."
**Reason:** Paket 2 is for external industrial/commercial users, NOT students.

### 2. Q5 — Cheating Penalty (IMPORTANT CORRECTION)
**Common misconception:** "Kopya çekmek → 1 haftadan 1 aya kadar uzaklaştırma"
**Correct:** "Kopya çekmek veya çektirmek → **Bir yarıyıl için uzaklaştırma**" (much more severe!)
**Note:** "Kopyaya teşebbüs" (attempting) → Kınama (less severe)

### 3. Q3 — Discipline Penalty Types (COMPLETENESS)
Must include ALL 6 categories:
1. Uyarma
2. Kınama  
3. 1 haftadan 1 aya kadar uzaklaştırma
4. Bir yarıyıl için uzaklaştırma
5. İki yarıyıl için uzaklaştırma
6. Yükseköğretim kurumundan çıkarma
