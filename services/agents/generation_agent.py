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

        # Generate answer
        answer = self.llm.chat(full_prompt, system_prompt, temperature=0.0)

        # Verify numbers (hallucination check)
        answer = self.verify_answer_numbers(answer, context, language)

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
                Sen Sabancı Üniversitesi'nin kurum içi bilgi sistemine bağlı Türkçe konuşan asistansın.

                KRİTİK KURALLAR:
                1) HER <chunk> öğesini BAŞTAN SONA oku — cevap genellikle Context'te VARDIR.
                2) Context'te geçen sayıları, tarihleri, süreleri, koşulları AYNEN ve EKSİKSİZ aktar.
                   Örnek: Context'te "60 gün süre ile 60 adet kitap" yazıyorsa, yanıtta da aynı sayılar olmalı.
                3) Context'te OLMAYAN hiçbir sayı, tarih veya değer YAZMA. Uydurma kesinlikle yasaktır.
                4) Context'te olmayan bilgileri KESİNLİKLE UYDURMA.
                5) "Bu bilgi bağlamda yok" SADECE hiçbir chunk'ta ilgili bilgi gerçekten YOKSA söylenebilir.
                   Eğer herhangi bir chunk'ta ilgili sayı veya bilgi varsa, onu KULLAN.

                BAĞLAM OKUMA KURALLARI:
                6) Soru bir öğrenci sorusuysa (ör: "kaç kitap ödünç alabilirim", "GNO şartı nedir"),
                   Context'te ÖĞRENCİYE AİT (lisans/lisansüstü/değişim) bölümü bul ve ORADAN yanıtla.
                   Paket 2 (personel), Paket 3 (misafir) gibi farklı kullanıcı gruplarını KARMA.
                7) Birden fazla chunk'ta aynı konuda bilgi varsa, EN SPESİFİK olanı tercih et.

                YANIT BİÇİMİ KURALLARI:
                8) Soru birden fazla madde/öğe soruyorsa (ör: "nelerdir", "hangileri", "kaç tür",
                   "sıralayınız", "listele"), TÜM maddeleri numaralı liste halinde ver.
                9) Context'teki TÜM ilgili bilgileri dahil et - yalnızca bir kısmını verme.
                10) Farklı gruplar için farklı değerler varsa (lisans/lisansüstü vb.), HEPSİNİ belirt.
                11) En az 2 cümle ile yanıt ver (basit evet/hayır soruları hariç).
            """).strip()
        else:
            return textwrap.dedent("""
                You are an assistant for Sabancı University's internal knowledge system.

                CRITICAL RULES:
                1) READ EVERY <chunk> element from start to end — the answer is usually IN the context.
                2) Use EXACT numbers, dates, durations, conditions from the context.
                   Example: if the context says "60 books for 60 days", your answer must include those numbers.
                3) Do NOT invent or estimate ANY numbers — use ONLY what's written in the context.
                4) Do NOT invent information not in the context.
                5) ONLY say "not found in context" if you truly cannot find it in ANY chunk.
                   If ANY chunk contains relevant numbers or facts, you MUST use them.

                CONTEXT READING RULES:
                6) If the question is from a student's perspective (e.g. "how many books can I borrow",
                   "what is the GPA requirement"), find the STUDENT-specific section in the context
                   (undergrad/graduate/exchange) and answer from THAT section.
                   Do NOT confuse with staff, alumni, or visitor rules.
                7) When multiple chunks discuss the same topic, prefer the most SPECIFIC one.

                ANSWER FORMAT RULES:
                8) If the question asks for multiple items (e.g. "what are", "which ones",
                   "how many types", "list"), provide ALL items as a numbered list.
                9) Include ALL relevant information from the context — do not give partial answers.
                10) If different values apply to different groups, mention ALL of them.
                11) Provide at least 2 sentences (except for simple yes/no questions).
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
1. Scan ALL <document> blocks and every <chunk> element — do not stop after the first match.
2. EXTRACT every specific number, GPA, duration, credit count, deadline, and condition from the context.
3. If different conditions apply to different groups (e.g. lisans/lisansüstü, undergrad/graduate), state ALL of them.
4. If the question asks for a list of items, enumerate ALL items found in the context.
5. Use ONLY information from the context. Do NOT add information from your own knowledge.
6. If the question is about students, find the STUDENT-applicable section (not staff/alumni).
7. NEVER say the information is not available if ANY chunk contains relevant numbers or facts."""

        if kg_facts:
            return f"""{kg_facts}

Context:
{context}

Question: {query}

IMPORTANT: The facts above MAY be helpful hints. CROSS-CHECK them against the Context.
If a fact seems inconsistent with the Context, IGNORE the fact and use the Context instead.

{instructions}

Answer:"""
        else:
            return f"""Context:
{context}

Question: {query}

{instructions}

Answer:"""

    def verify_answer_numbers(
        self,
        answer: str,
        context: str,
        language: Optional[str]
    ) -> str:
        """
        Verify that numbers in answer exist in context.
        If hallucinated numbers are detected, re-generate with an explicit
        number guard to correct them.

        Args:
            answer: Generated answer.
            context: Source context.
            language: Detected language.

        Returns:
            Original answer if clean, or corrected answer if hallucination detected.
        """
        # Extract numbers from answer and context
        answer_numbers = set(re.findall(r'\d+[.,]?\d*', answer))
        context_numbers = set(re.findall(r'\d+[.,]?\d*', context))

        def normalize_num(n: str) -> str:
            return n.replace(',', '.')

        context_numbers_normalized = {normalize_num(n) for n in context_numbers}

        # Filter significant numbers (ignore very small integers like 1)
        # Note: numbers like 2, 3 are important in this domain (e.g. 2 ay, 3 yarıyıl)
        significant = {
            n for n in answer_numbers
            if float(normalize_num(n)) >= 4 or '.' in n or ',' in n
        }

        if not significant:
            return answer

        # Check for hallucinated numbers
        hallucinated = {
            n for n in significant
            if normalize_num(n) not in context_numbers_normalized
        }

        if not hallucinated:
            return answer

        print(f"[Generation] WARNING: Possible hallucinated numbers: {hallucinated}")
        print(f"[Generation] Numbers in context: "
              f"{sorted(list(context_numbers_normalized)[:20])}")

        # Active correction: re-prompt the LLM to fix hallucinated numbers
        print("[Generation] Re-generating with number guard...")

        context_nums_display = ", ".join(sorted(context_numbers)[:40])

        if language == "tr":
            guard_system = (
                "Verilen yanıttaki yanlış sayıları düzelt. "
                "Sadece kaynak bağlamda geçen sayıları kullan."
            )
            guard_prompt = (
                f"Aşağıdaki yanıtta kaynak bağlamda BULUNMAYAN sayılar olabilir.\n\n"
                f"Yanıt:\n{answer}\n\n"
                f"Kaynak bağlamda geçen sayılar: {context_nums_display}\n\n"
                f"Yanıtı aynı yapıda tut, ancak kaynak bağlamda olmayan sayıları "
                f"kaynak bağlamdan doğru sayı ile değiştir veya çıkar.\n\n"
                f"Düzeltilmiş yanıt:"
            )
        else:
            guard_system = (
                "Correct wrong numbers in the given answer. "
                "Use only numbers that appear in the source context."
            )
            guard_prompt = (
                f"The following answer may contain numbers NOT found in the source context.\n\n"
                f"Answer:\n{answer}\n\n"
                f"Numbers found in source context: {context_nums_display}\n\n"
                f"Keep the same structure but replace numbers not in the source context "
                f"with the correct number from the context, or remove them.\n\n"
                f"Corrected answer:"
            )

        try:
            corrected = self.llm.chat(guard_prompt, guard_system, temperature=0.0)
        except Exception as e:
            print(f"[Generation] Number guard LLM call failed: {e}")
            return answer

        if not corrected or corrected.startswith(("Ollama Error", "LLM error", "Connection")):
            print("[Generation] Number guard failed, keeping original.")
            return answer

        # Verify the correction actually improved things
        corrected_nums = set(re.findall(r'\d+[.,]?\d*', corrected))
        corrected_significant = {
            n for n in corrected_nums
            if float(normalize_num(n)) >= 5 or '.' in n or ',' in n
        }
        corrected_hallucinated = {
            n for n in corrected_significant
            if normalize_num(n) not in context_numbers_normalized
        }

        if len(corrected_hallucinated) < len(hallucinated):
            print(f"[Generation] Number guard applied. "
                  f"Hallucinated: {len(hallucinated)} -> {len(corrected_hallucinated)}")
            return corrected
        else:
            print("[Generation] Number guard did not improve. Keeping original.")
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
