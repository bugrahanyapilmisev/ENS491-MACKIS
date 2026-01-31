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
        Build context string from chunks with formatting.

        Args:
            chunks: List of chunk dicts.
            include_summaries: Whether to include document summaries.
            max_summary_chars: Max characters for summary truncation.

        Returns:
            Formatted context string.
        """
        parts = []
        doc_summaries_added: Set[str] = set()

        for i, ch in enumerate(chunks):
            meta = ch.get("meta", {}) or {}
            path = meta.get("source_path") or meta.get("doc_path", "")
            title = meta.get("title", "")
            section = meta.get("section_header", "")
            lang = meta.get("doc_lang", "")

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
                    parts.append(f"[Document Overview: {title}]\n{truncated}")
                    doc_summaries_added.add(path)

            # Build chunk header
            header_parts = [f"[{i + 1}]"]
            if title:
                header_parts.append(f"Title: {title}")
            if section:
                header_parts.append(f"Section: {section}")
            if lang:
                header_parts.append(f"Lang: {lang}")

            header = " | ".join(header_parts)
            parts.append(header + "\n" + ch.get("text", ""))

        return "\n\n-----\n\n".join(parts)

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
                1) Context'i DİKKATLİ OKU - cevap genellikle Context'te VARDIR.
                2) Context'te geçen sayıları, tarihleri, süreleri, koşulları AYNEN kullan.
                3) SAYI veya DEĞERLERİ KENDİN UYDURMA - Context'te yazanı yaz.
                4) Context'te olmayan bilgileri KESİNLİKLE UYDURMA.
                5) SADECE hiçbir yerde bulamadığında "bu bilgi bağlamda yok" de.
                6) Cevapların kısa ve net olsun.

                DİKKAT: "GNO", "not ortalaması", "minimum" gibi kelimeler Context'te farklı şekillerde geçebilir.
                Lisans=undergrad, Lisansüstü=graduate için farklı değerler olabilir, İKİSİNİ de belirt.

                ÖRNEK: Context'te "Lisans için 2.20, Lisansüstü için 2.5" varsa, tam olarak bunu yaz.
            """).strip()
        else:
            return textwrap.dedent("""
                You are an assistant for Sabancı University's internal knowledge system.

                CRITICAL RULES:
                1) READ the context CAREFULLY - the answer is usually IN the context.
                2) Use EXACT numbers, dates, durations, conditions from the context.
                3) Do NOT invent numbers - use what's written in the context.
                4) Do NOT invent information not in the context.
                5) ONLY say "not in context" if you truly cannot find it anywhere.
                6) Be brief and direct.

                NOTE: Terms like "GPA", "GNO", "minimum" may appear in different forms.
                Undergrad vs Graduate may have different values - mention BOTH if present.

                EXAMPLE: If context has "2.20 for undergrad, 2.5 for graduate", write exactly that.
            """).strip()

    def _build_prompt(
        self,
        query: str,
        context: str,
        kg_facts: str = ""
    ) -> str:
        """
        Build full prompt with context and optional KG facts.

        Args:
            query: User question.
            context: Formatted context string.
            kg_facts: Optional knowledge graph facts.

        Returns:
            Full prompt string.
        """
        if kg_facts:
            return f"""{kg_facts}

Context:
{context}

Question: {query}

INSTRUCTIONS:
1. The facts above MAY be helpful hints. CROSS-CHECK them against the Context below.
2. If a fact seems inconsistent with the Context (e.g., GPA > 4.0), IGNORE IT and use the Context instead.
3. SEARCH the context for specific numbers, values, requirements, conditions, durations, or limits.
4. If the question asks about GNO/GPA, look for phrases like "en az", "minimum", "2.20", "2.5", etc.
5. If different conditions apply to different groups (lisans/lisansüstü), mention ALL of them.
6. EXTRACT and state the relevant information directly from the Context.

Answer:"""
        else:
            return f"""Context:
{context}

Question: {query}

INSTRUCTIONS:
1. SEARCH the entire context above for information related to the question.
2. Look for specific numbers, values, requirements, conditions, durations, or limits.
3. If the question asks about GNO/GPA, look for phrases like "en az", "minimum", "2.20", "2.5", etc.
4. If different conditions apply to different groups (lisans/lisansüstü), mention ALL of them.
5. EXTRACT and state the relevant information directly.

Answer:"""

    def verify_answer_numbers(
        self,
        answer: str,
        context: str,
        language: Optional[str]
    ) -> str:
        """
        Verify that numbers in answer exist in context.

        Logs warnings for potential hallucinated numbers but doesn't
        modify the answer (to avoid removing valid content).

        Args:
            answer: Generated answer.
            context: Source context.
            language: Detected language.

        Returns:
            Original answer (unchanged, but warnings logged).
        """
        # Extract numbers from answer and context
        answer_numbers = set(re.findall(r'\d+[.,]?\d*', answer))
        context_numbers = set(re.findall(r'\d+[.,]?\d*', context))

        def normalize_num(n: str) -> str:
            return n.replace(',', '.')

        context_numbers_normalized = {normalize_num(n) for n in context_numbers}

        # Filter significant numbers (ignore small integers like 1, 2, 3)
        significant = {
            n for n in answer_numbers
            if float(normalize_num(n)) >= 5 or '.' in n or ',' in n
        }

        if not significant:
            return answer

        # Check for hallucinated numbers
        hallucinated = {
            n for n in significant
            if normalize_num(n) not in context_numbers_normalized
        }

        if hallucinated:
            print(f"[Generation] WARNING: Possible hallucinated numbers: {hallucinated}")
            print(f"[Generation] Numbers in context: "
                  f"{sorted(list(context_numbers_normalized)[:20])}")

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
