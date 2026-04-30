"""
Generation agent for context building and answer generation.

This agent handles:
- Context building from retrieved chunks
- Language-appropriate system prompts
- Answer generation with LLM
- Hallucination detection (number verification)
"""

from typing import Dict, List, Optional, Set
import re
import time
import textwrap

from services.core.llm_service import LLMService
from services.core.data_loader import DataLoaderService
from services.config.settings import RAGConfig


class GenerationAgent:
    """Handles context building and answer generation."""

    def __init__(
        self,
        llm_service: LLMService,
        data_loader: DataLoaderService,
        config: RAGConfig
    ):
        """
        Initialize generation agent.

        Args:
            llm_service: Service for LLM operations.
            data_loader: Service for document summaries.
            config: RAG configuration.
        """
        self.llm = llm_service
        self.data_loader = data_loader
        self.config = config

    def generate(
        self,
        query: str,
        chunks: List[Dict],
        language: Optional[str] = None,
        kg_facts: str = ""
    ) -> str:
        """
        Generate answer from context.

        Steps:
        1. Build context from chunks
        2. Construct prompt with optional KG facts
        3. Generate answer
        4. Verify numbers (hallucination check)

        Args:
            query: User question.
            chunks: Retrieved and ranked chunks.
            language: Detected language.
            kg_facts: Optional knowledge graph facts.

        Returns:
            Generated answer string.
        """
        # Build context
        context = self.build_context(chunks)

        # Get system prompt
        system_prompt = self._get_system_prompt(language)

        # Build full prompt
        full_prompt = self._build_prompt(query, context, kg_facts)

        # Generate answer with retry logic for empty responses
        answer = None
        max_retries = 3
        for attempt in range(max_retries):
            raw = self.llm.chat(full_prompt, system_prompt, temperature=0.0, max_tokens=900)
            if raw and raw.strip() and not raw.strip().startswith("Empty response"):
                answer = raw
                break
            wait = 2 ** attempt
            print(f"[Generation] Empty response on attempt {attempt + 1}/{max_retries}, "
                  f"retrying in {wait}s...")
            time.sleep(wait)

        if not answer or not answer.strip() or answer.strip().startswith("Empty response"):
            print(f"[Generation] All {max_retries} attempts returned empty. Using fallback.")
            if language == "tr":
                answer = ("Üzgünüm, bu soruya şu an yanıt üretilemedi. "
                          "Lütfen sorunuzu yeniden deneyin.")
            else:
                answer = ("Sorry, I was unable to generate an answer to this question "
                          "at this time. Please try again.")

        # Post-processing pipeline
        answer = self._fix_repetition(answer, full_prompt, system_prompt, language)
        answer = self._strip_doc_codes(answer)
        #answer = self.verify_hallucinations(answer, context, language)

        return answer

    def build_context(
        self,
        chunks: List[Dict],
        include_summaries: bool = True,
        max_summary_chars: int = 300
    ) -> str:
        """
        Build structured context from chunks with document grouping and chunk cap.

        Groups chunks by source document using XML-like tags so the LLM can
        clearly see document boundaries and section context.

        Args:
            chunks: List of chunk dicts.
            include_summaries: Whether to include document summaries.
            max_summary_chars: Max characters for summary truncation.

        Returns:
            Structured context string with document/chunk markers.
        """
        # Enforce chunk cap from config
        max_chunks = self.config.retrieval.max_docs_context
        chunks = chunks[:max_chunks]

        # Group chunks by source document (preserving order)
        from collections import OrderedDict
        doc_groups: OrderedDict = OrderedDict()
        for i, ch in enumerate(chunks):
            meta = ch.get("meta", {}) or {}
            path = meta.get("source_path") or meta.get("doc_path", "unknown")
            if path not in doc_groups:
                doc_groups[path] = {
                    "title": meta.get("title", ""),
                    "chunks": [],
                }
            doc_groups[path]["chunks"].append((i, ch))

        parts = []
        doc_summaries_added: Set[str] = set()
        chunk_num = 0

        for path, group in doc_groups.items():
            title = group["title"]
            doc_lines = []

            # Document header
            doc_lines.append(f'<document title="{title}">')

            # Add document summary once per document
            if (
                include_summaries
                and self.config.features.use_doc_summaries
                and path
                and path not in doc_summaries_added
            ):
                summary = self.data_loader.get_doc_summary(path)
                if summary:
                    truncated = summary[:max_summary_chars]
                    if len(summary) > max_summary_chars:
                        truncated += "..."
                    doc_lines.append(f"  <summary>{truncated}</summary>")
                    doc_summaries_added.add(path)

            # Add chunks with section context
            for _, ch in group["chunks"]:
                chunk_num += 1
                meta = ch.get("meta", {}) or {}
                section = meta.get("section_header", "")
                text = ch.get("text", "")

                section_attr = f' section="{section}"' if section else ""
                doc_lines.append(f'  <chunk id="{chunk_num}"{section_attr}>')
                doc_lines.append(f"    {text}")
                doc_lines.append("  </chunk>")

            doc_lines.append("</document>")
            parts.append("\n".join(doc_lines))

        return "\n\n".join(parts)

    def _get_system_prompt(self, language: Optional[str]) -> str:
        """
        Get language-appropriate system prompt.

        Args:
            language: Detected language.

        Returns:
            System prompt string.
        """
        if language == "tr":
            return textwrap.dedent("""
                Sen Sabancı Üniversitesi'nin bilgi sistemi asistanısın. KESİNLİKLE TÜRKÇE yanıtla.

                TEMEL KURALLAR:
                1) Tüm <chunk> öğelerini baştan sona oku. Cevap genellikle context'te mevcuttur.
                2) Context'teki sayıları, tarihleri, süreleri, koşulları BİREBİR kullan. Uydurma YASAK.
                3) Context'te AÇIKÇA yazmayan bilgi EKLEME. Bilmiyorsan "bu konuda bilgim yok" de.
                4) Sorulan soruyu DOĞRUDAN yanıtla. Soruyu tekrarlama, giriş cümlesi kurma.

                BAĞLAM OKUMA:
                5) VARSAYILAN KİŞİ: Soruyu ÖĞRENCİ soruyor kabul et.
                   - KYK, devlet kurumu vb. dış kurum bilgilerini DEĞİL, Sabancı Üniversitesi'nin
                     kendi iç prosedürünü yanıtla.
                6) Bilgi birden fazla chunk'a yayılmışsa, hepsini birleştirerek tam cevap ver.
                7) Birden fazla chunk aynı konuyu işliyorsa en SPESİFİK olanı tercih et.
                   - ÖNEMLİ: Eğer kural, sayı veya GNO Lisans (Undergraduate) ve Lisansüstü (Graduate) için FARKLIYSA, İKİSİNİ BİRDEN KESİNLİKLE YAZ. Sadece birini yazıp bırakma.

                SORU TİPİNE GÖRE YANIT:
                8) SÜRE/SAYI SORUSU: "ne kadar sürede", "kaç gün", "ne zaman" gibi sorularda
                   context'ten İLGİLİ SAYILARI (gün, ay, yıl, dönem) MUTLAKA çıkar ve yaz.
                9) CEZA SORUSU: Context'te bir fiil için ÖZEL ceza belirtilmişse (örn. kopya çekmek
                   için bir yarıyıl uzaklaştırma), genel ceza tanımlarıyla KARIŞTIRMA, eğer özel bir ceza yoksa o fiil için özel bir ceza bulamadığını söyle ve genel ceza tanımlarını belirt.
                10) HİBE/HESAPLAMA SORUSU: Seçim kriterleri (puan hesaplama) ile hibe ödeme
                    sürecini (taksit, sözleşme, gün bazında hesap) AYIR.
                11) ŞART/KOŞUL SORUSU: "şartları nelerdir", "koşulları nelerdir", "başvuru için ne gerekir"
                    gibi sorularda ÖNCE sayısal kriterleri yaz (GNO eşiği, kredi sayısı, dönem sayısı,
                    puan sınırı), SONRA prosedürel adımları özetle. Prosedürel detaylara takılıp
                    sayısal eşikleri atlamayı YASAK.

                BAĞLAM ÖNCELİĞİ:
                12) Context'teki chunk'lar ilgililik sırasına göre sıralanmıştır. İLK chunk'lar en
                    ilgili olanlardır — öncelikli olarak onlara odaklan.

                CEVAP BİÇİMİ:
                13) KAPSAMLI OL: Cevap duruma göre değişiyorsa TÜM varyasyonları listele.
                14) KISA VE ÖZ OL: Prosedür soruları için 2-5 cümlede özetle.
                15) Aynı cümleyi veya paragrafı KESİNLİKLE TEKRARLAMA. Her cümle yeni bilgi içermeli.
                16) Belge kodu (PSR-C210-0101 gibi), chunk id, madde numarası YAZMA.
            """).strip()
        else:
            return textwrap.dedent("""
                You are Sabancı University's knowledge assistant. ALWAYS answer in ENGLISH.

                CORE RULES:
                1) Read ALL <chunk> elements. The answer is usually IN the context.
                2) Use EXACT numbers, dates, durations from the context. Do NOT invent any.
                3) Do NOT add information not EXPLICITLY in the context. If unsure, say so.
                4) Start your answer DIRECTLY. No preamble, no repeating the question.

                CONTEXT READING:
                5) DEFAULT PERSONA: Assume the question is from a STUDENT.
                   - If a table/list has multiple user categories (Package 1/2/3,
                     student/staff/alumni), use ONLY the STUDENT row.
                   - Answer about Sabancı University's OWN procedures, not external institutions.
                6) If info is spread across chunks, synthesize them into one complete answer.
                7) Prefer the most SPECIFIC chunk when multiple discuss the same topic.
                   - CRITICAL: If a rule, number, or GPA differs for Undergraduate (Lisans) vs. Graduate (Lisansüstü) students, YOU MUST STATE BOTH. Do not just state one.

                QUESTION-TYPE RULES:
                8) DURATION/NUMBER QUESTIONS: When asked "how long", "how many days", always
                   EXTRACT the relevant numbers (days, months, semesters) from context.
                9) PENALTY QUESTIONS: If context maps a SPECIFIC act to a SPECIFIC penalty
                   (e.g. cheating = one semester suspension), report that specific mapping.
                   Do NOT substitute the generic penalty definition.
                10) GRANT/CALCULATION QUESTIONS: Distinguish between selection criteria (scoring)
                    and grant payment process (installments, contract, day-based calculation).
                11) REQUIREMENT/CONDITION QUESTIONS: When asked "what are the requirements",
                    "conditions", "eligibility criteria" — FIRST state all quantitative thresholds
                    (GPA minimums, credit counts, semester counts, score cutoffs), THEN
                    summarize procedural steps. Do NOT skip numeric criteria in favor of procedures.

                CONTEXT PRIORITY:
                12) Chunks are sorted by relevance — the FIRST chunks are the most relevant.
                    Prioritize information from the first chunks.

                ANSWER FORMAT:
                13) BE COMPREHENSIVE: If the answer varies by condition, state ALL variations.
                14) BE CONCISE: Summarize in 2-5 sentences. Do NOT copy every procedural step.
                15) NEVER repeat the same sentence. Each sentence must add new information.
                16) Do NOT include document codes (like PSR-C210-0101), chunk IDs, or article
                    numbers. State information naturally.
            """).strip()

    def _build_prompt(
        self,
        query: str,
        context: str,
        kg_facts: str = ""
    ) -> str:
        """
        Build full prompt with structured context and optional KG facts.

        Args:
            query: User question.
            context: Structured context string with document/chunk markers.
            kg_facts: Optional knowledge graph facts.

        Returns:
            Full prompt string.
        """
        instructions = """INSTRUCTIONS:
1. Scan ALL <document> blocks and every <chunk> element before answering.
2. EXTRACT specific numbers, GPAs, durations, credit counts, deadlines, conditions.
3. If different conditions apply to different groups (e.g. Undergraduate vs. Graduate), YOU MUST state ALL of them clearly. Do not stop at the first group!
4. If the question asks for a list, enumerate ALL items found.
5. Use ONLY information from the context. Do NOT add from your own knowledge.
6. For student questions, use the STUDENT/ÖĞRENCİ section (not staff/alumni/Paket 2).
7. NEVER repeat the same sentence. Each sentence must provide NEW information.
8. Keep your answer between 2-8 sentences unless a detailed list is specifically needed.
9. Do NOT include document codes like (PSR-XXX-XXXX) or (ISR-XXX-XX) in your answer.
10. If context has NO relevant information for the question, say you don't have that information."""

        if kg_facts:
            return f"""Context:
{context}

{kg_facts}

NOTE: KG facts above are supplementary hints. CROSS-CHECK them against the Context.
If a fact conflicts with the Context, IGNORE the fact and use the Context.

Question: {query}

{instructions}

Answer:"""
        else:
            return f"""Context:
{context}

Question: {query}

{instructions}

Answer:"""

    def _fix_repetition(
        self,
        answer: str,
        full_prompt: str,
        system_prompt: str,
        language: Optional[str]
    ) -> str:
        """
        Detect and fix repetition loops in generated answers.

        If more than 50% of sentences are duplicates, regenerate with
        an explicit anti-repetition instruction.

        Args:
            answer: Generated answer to check.
            full_prompt: Original prompt for regeneration.
            system_prompt: System prompt for regeneration.
            language: Detected language.

        Returns:
            Original answer if clean, or regenerated answer.
        """
        if not answer or len(answer) < 100:
            return answer

        # Split into sentences
        sentences = [s.strip() for s in re.split(r'[.!?。]\s+', answer) if s.strip()]

        if len(sentences) < 3:
            return answer

        # Count unique sentences
        unique = set(sentences)
        repetition_ratio = 1.0 - (len(unique) / len(sentences))

        if repetition_ratio < 0.5:
            return answer

        print(f"[Generation] WARNING: Repetition detected ({repetition_ratio:.0%}). "
              f"Regenerating...")

        # Regenerate with explicit anti-repetition guard
        if language == "tr":
            guard = ("\n\nÖNEMLİ: Önceki yanıt tekrar döngüsüne girdi. "
                     "Her cümle FARKLI bilgi içermeli. Cevap bulamıyorsan "
                     "\"Bu konuda yeterli bilgi bulunamadı\" de.")
        else:
            guard = ("\n\nIMPORTANT: Your previous answer was a repetition loop. "
                     "Every sentence must contain DIFFERENT information. "
                     "If you cannot find relevant information, say so.")

        try:
            new_answer = self.llm.chat(
                full_prompt + guard,
                system_prompt,
                temperature=0.1,
                max_tokens=900
            )

            if new_answer and not new_answer.startswith(("Ollama Error", "LLM error")):
                # Verify the new answer is not also repetitive
                new_sentences = [s.strip() for s in re.split(r'[.!?。]\s+', new_answer) if s.strip()]
                if len(new_sentences) >= 2:
                    new_unique = set(new_sentences)
                    new_ratio = 1.0 - (len(new_unique) / len(new_sentences))
                    if new_ratio < 0.5:
                        print("[Generation] Regeneration successful.")
                        return new_answer

            print("[Generation] Regeneration still repetitive, returning fallback.")
        except Exception as e:
            print(f"[Generation] Regeneration failed: {e}")

        # Last resort: return just the unique sentences
        seen = set()
        deduped = []
        for s in sentences:
            if s not in seen:
                seen.add(s)
                deduped.append(s)
        return ". ".join(deduped) + "."

    def _strip_doc_codes(self, answer: str) -> str:
        """
        Remove document/procedure codes from the answer.

        Strips patterns like (PSR-C210-0101), (ISR-C220-01), (PIPAR-C710-0201)
        that leak into answers despite prompt instructions.

        Args:
            answer: Answer text to clean.

        Returns:
            Cleaned answer without document codes.
        """
        if not answer:
            return answer

        # Remove parenthesized codes: (PSR-C210-0101), (ISR-C220-01), etc.
        cleaned = re.sub(
            r'\s*\([A-Z]{2,6}-[A-Z0-9]+-[0-9]+(?:-[0-9]+)?\)',
            '',
            answer
        )

        # Remove inline codes without parens when followed by comma or period
        # e.g., "PSR-C210-0101 göre," → "göre,"
        cleaned = re.sub(
            r'\b[A-Z]{2,6}-[A-Z][0-9]+-[0-9]+(?:-[0-9]+)?\b',
            '',
            cleaned
        )

        # Clean up double spaces
        cleaned = re.sub(r'  +', ' ', cleaned)
        # Clean up orphaned punctuation
        cleaned = re.sub(r' ,', ',', cleaned)
        cleaned = re.sub(r' \.', '.', cleaned)

        if cleaned != answer:
            print("[Generation] Stripped document codes from answer.")

        return cleaned.strip()

    def verify_hallucinations(
        self,
        answer: str,
        context: str,
        language: Optional[str]
    ) -> str:
        """
        Verify that the generated answer does not contain hallucinations using an LLM.
        This provides much smarter context-aware verification than regex number checking.

        Args:
            answer: Generated answer.
            context: Source context.
            language: Detected language.

        Returns:
            Original answer if completely factual, or a corrected version.
        """
        if not answer or len(answer.strip()) < 5:
            return answer

        print("[Generation] Running LLM hallucination check...")

        if language == "tr":
            system_prompt = (
                "Sen katı bir doğruluk kontrolörüsün. Sağlanan 'Bağlam' (Context) metni ile üretilen 'Yanıt' (Answer) metnini karşılaştır.\n"
                "Görevlerin:\n"
                "1. Yanıttaki sayıların, katsayıların, tarihlerin, GNO'ların veya kuralların Bağlam'da geçip geçmediğini dikkatlice incele.\n"
                "2. Yanıtta listeleri numaralandırmak için kullanılan (1., 2. gibi) sayılar halisünasyon değildir, bunları yok say.\n"
                "3. Eğer Yanıtta Bağlam'da BULUNMAYAN kritik uydurma bir kural veya matematiksel sayı varsa, bunu düzelt veya ilgili kısmı çıkar.\n"
                "4. YALNIZCA düzeltilmiş yanıtı döndür. Eğer orijinal yanıt doğrulandıysa veya sadece liste numaraları varsa orijinali AYNEN geri döndür. Yorum YOK."
            )
            prompt = f"Bağlam:\n{context}\n\nOrijinal Yanıt:\n{answer}\n\nDoğrulanmış ve Düzeltilmiş Yanıt:"
        else:
            system_prompt = (
                "You are a strict fact-checker. Compare the 'Context' with the generated 'Answer'.\n"
                "Your tasks:\n"
                "1. Scrutinize if ANY specific numbers, GPAs, dates, or rules in the Answer are missing from the Context.\n"
                "2. Generic list numbering (like 1., 2.) are NOT hallucinations, ignore them.\n"
                "3. If the Answer contains hallucinations (invented facts or unauthorized mathematical logic), correct or remove them.\n"
                "4. RETURN ONLY the corrected answer text. If the original answer is factual, return it exactly as is. NO additional comments."
            )
            prompt = f"Context:\n{context}\n\nOriginal Answer:\n{answer}\n\nVerified and Corrected Answer:"

        try:
            corrected = self.llm.chat(prompt, system_prompt, temperature=0.0, max_tokens=900)
            
            if not corrected or corrected.startswith(("Ollama Error", "LLM error", "Connection", "Empty response")):
                print("[Generation] Hallucination check failed (timeout/error), keeping original.")
                return answer
            
            if corrected.strip() != answer.strip():
                print("[Generation] LLM applied corrections for hallucinations.")
                return corrected.strip()
            else:
                print("[Generation] LLM confirmed answer is factual.")
                return answer

        except Exception as e:
            print(f"[Generation] LLM hallucination check call failed: {e}")
            return answer

    def generate_with_citations(
        self,
        query: str,
        chunks: List[Dict],
        language: Optional[str] = None
    ) -> Dict:
        """
        Generate answer with source citations.

        Args:
            query: User question.
            chunks: Retrieved chunks.
            language: Detected language.

        Returns:
            Dict with 'answer' and 'citations' list.
        """
        answer = self.generate(query, chunks, language)

        # Build citations from chunks
        citations = []
        for i, ch in enumerate(chunks):
            meta = ch.get("meta", {}) or {}
            citations.append({
                "index": i + 1,
                "title": meta.get("title", ""),
                "section": meta.get("section_header", ""),
                "excerpt": ch.get("text", "")[:200] + "...",
                "source_path": meta.get("source_path", ""),
            })

        return {
            "answer": answer,
            "citations": citations,
        }
