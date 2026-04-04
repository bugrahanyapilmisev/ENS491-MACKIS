"""
Query analysis agent for intent detection and query understanding.

This agent handles:
- Language detection (Turkish/English)
- Intent classification (list, count, describe, other)
- Followup detection using semantic similarity
- Semantic tag inference using LLM
- Negation term extraction using LLM
"""

from typing import Dict, List, Optional, Tuple
import re
import textwrap

from services.core.embedding_service import EmbeddingService
from services.core.llm_service import LLMService
from services.config.settings import RAGConfig

# Language detection (optional dependency)
try:
    from langdetect import detect as ld_detect
except ImportError:
    ld_detect = None

# Turkish diacritics for fallback detection
TR_DIACRITICS = "çğıöşüÇĞİÖŞÜ"


class QueryAnalysisAgent:
    """Analyzes queries for intent, language, followup status, and tags."""

    def __init__(
        self,
        embedding_service: EmbeddingService,
        llm_service: LLMService,
        config: RAGConfig
    ):
        """
        Initialize query analysis agent.

        Args:
            embedding_service: Service for text embeddings.
            llm_service: Service for LLM chat operations.
            config: RAG configuration.
        """
        self.embedding = embedding_service
        self.llm = llm_service
        self.config = config

    # Negation keywords — if none appear, skip the LLM negation call entirely
    _NEGATION_KEYWORDS_TR = {"değil", "hariç", "dışında", "haricinde", "olmadan", "olmayan", "disinda", "haric"}
    _NEGATION_KEYWORDS_EN = {"not", "except", "excluding", "other than", "aside from", "without"}
    _NEGATION_KEYWORDS = _NEGATION_KEYWORDS_TR | _NEGATION_KEYWORDS_EN

    def _query_has_negation_cue(self, query: str) -> bool:
        """
        Fast check whether the query contains any negation keyword.
        Used to short-circuit the expensive LLM negation call.
        """
        q_lower = query.lower()
        return any(kw in q_lower for kw in self._NEGATION_KEYWORDS)

    def analyze(
        self,
        query: str,
        history: Optional[List[Dict]] = None
    ) -> Dict:
        """
        Perform complete query analysis.

        Optimizations vs. original:
        - Tags + query expansion are inferred in a SINGLE LLM call
        - Negation extraction is short-circuited: the LLM is only called
          when the query contains an explicit negation keyword.

        Args:
            query: User query to analyze.
            history: Optional conversation history.

        Returns:
            Analysis dict with:
            - language: "tr" | "en" | None
            - intent: "list_names" | "count_items" | "describe" | "other"
            - is_followup: bool
            - anchor_query: Optional previous query if followup
            - tags: List of semantic tags
            - negated_terms: List of excluded terms
            - expanded_queries: List of expanded query strings (includes original)
        """
        history = history or []

        language = self.detect_language(query)
        intent = self.detect_intent(query)
        is_followup, anchor = self.detect_followup(query, history)

        # Single LLM call for tags + expansion.
        # When follow-up, pass anchor so the LLM can resolve pronouns like "bu", "onların".
        tags, expanded_queries = self.infer_tags_and_expand(
            query, language,
            anchor_query=anchor if is_followup else None
        )

        # Short-circuit: only call LLM for negation if query has negation cue
        if self._query_has_negation_cue(query):
            negated = self.extract_negated_terms(query, language)
        else:
            negated = []
            print("[QueryAnalysis] No negation cue found — skipping LLM negation call")

        return {
            "language": language,
            "intent": intent,
            "is_followup": is_followup,
            "anchor_query": anchor,
            "tags": tags,
            "negated_terms": negated,
            "expanded_queries": expanded_queries,
        }

    def detect_language(self, text: str) -> Optional[str]:
        """
        Detect query language (Turkish or English).

        Args:
            text: Text to analyze.

        Returns:
            "tr", "en", or None if undetermined.
        """
        s = (text or "").strip()
        if not s:
            return None

        sample = s[:4000]

        # Try langdetect library if available
        if ld_detect is not None:
            try:
                code = ld_detect(sample).lower()
                if code.startswith("tr"):
                    return "tr"
                if code.startswith("en"):
                    return "en"
            except Exception:
                pass

        # Fallback: check for Turkish diacritics
        if re.search(f"[{TR_DIACRITICS}]", sample):
            return "tr"

        return None

    def detect_intent(self, query: str) -> str:
        """
        Detect query intent type.

        Args:
            query: User query.

        Returns:
            One of: "list_names", "count_items", "describe", "other"
        """
        if not query:
            return "other"

        q = query.strip().lower()

        # Count intent triggers (Turkish and English)
        count_triggers = [
            "kaç tane", "kaç adet", "sayısı kaç", "toplam kaç",
            "kac tane", "kac adet", "sayisi kac", "toplam kac",
            "how many", "number of", "count of",
        ]
        if any(t in q for t in count_triggers):
            return "count_items"

        # List intent triggers
        list_triggers = [
            "isimlerini say", "isimlerini listele", "adlarını say",
            "isimlerini listele", "adlarini say",
            "list the names", "list all", "enumerate",
        ]
        if any(t in q for t in list_triggers):
            return "list_names"

        # Describe intent triggers
        describe_triggers = [
            "nedir", "ne demek", "açıkla", "acikla",
            "explain", "describe", "what is", "what are",
        ]
        if any(t in q for t in describe_triggers):
            return "describe"

        return "other"

    def detect_followup(
        self,
        query: str,
        history: List[Dict],
        threshold: float = 0.60
    ) -> Tuple[bool, Optional[str]]:
        """
        Check if query is a followup using semantic similarity.

        Args:
            query: Current user query.
            history: Conversation history with role and content.
            threshold: Similarity threshold for followup detection.

        Returns:
            Tuple of (is_followup, anchor_query).
        """
        if not history:
            return False, None

        # Find last user query in history
        last_q = None
        for msg in reversed(history):
            if msg.get("role") == "user":
                last_q = (msg.get("content") or "").strip()
                if last_q:
                    break

        if not last_q:
            return False, None

        # Minimum length guard: very short messages (greetings, one-word replies)
        # can't be meaningful anchors, and their embeddings are too generic.
        if len(last_q.split()) < 3:
            return False, None

        # Check semantic similarity
        try:
            q_vec = self.embedding.embed(query)
            last_vec = self.embedding.embed(last_q)
            sim = self.embedding.cosine_similarity(q_vec, last_vec)

            print(f"[QueryAnalysis] Follow-up similarity={sim:.3f} "
                  f"(threshold={threshold}) | anchor='{last_q[:60]}'")

            if sim >= threshold:
                return True, last_q

        except Exception as e:
            print(f"[QueryAnalysis] Followup detection error: {e}")

        return False, None

    def infer_tags(
        self,
        query: str,
        lang: Optional[str] = None
    ) -> List[str]:
        """
        Infer semantic tags for query using LLM.

        Args:
            query: User query.
            lang: Detected language.

        Returns:
            List of semantic tags (lowercase snake_case).
        """
        sys_prompt = textwrap.dedent("""
            You are a classifier for search queries in a university system.
            Return a JSON object with semantic tags for the query.

            JSON schema: {"tags": ["tag1", "tag2", ...]}

            Rules:
            - 2-6 tags, lowercase snake_case
            - Use English tags even for Turkish queries
            - Tags should be specific topics, not generic words like "information" or "question"
            - Focus on the domain: scholarships, exchange_programs, library, discipline, etc.

            Output STRICT JSON ONLY, no explanation.
        """).strip()

        result = self.llm.chat_json(
            f"Query: {query}",
            sys_prompt,
            temperature=0.0
        )

        if not result:
            return []

        tags = result.get("tags", [])
        if not isinstance(tags, list):
            return []

        return [
            t.strip().lower()
            for t in tags
            if isinstance(t, str) and t.strip()
        ]

    def infer_tags_and_expand(
        self,
        query: str,
        lang: Optional[str] = None,
        anchor_query: Optional[str] = None
    ) -> tuple:
        """
        Infer semantic tags AND generate expanded queries in a single LLM call.

        When anchor_query is provided (follow-up case), the LLM receives the full
        conversation context so tags and expansions are anchored to the original topic
        rather than the ambiguous pronoun-heavy follow-up.

        Args:
            query: User query (may contain pronouns like "bu", "onların" in follow-ups).
            lang: Detected language.
            anchor_query: The previous user query if this is a follow-up. When provided,
                          both tags and expanded queries will reflect the combined topic.

        Returns:
            Tuple of (tags: List[str], expanded_queries: List[str]).
            expanded_queries always starts with the original query.
        """
        if anchor_query:
            # Follow-up mode: give the LLM full context so it understands what
            # pronouns like "bu", "onların", "bu şartlar" refer to.
            sys_prompt = textwrap.dedent("""
                You are a search assistant for a university information system.
                The user is asking a FOLLOW-UP question that contains pronouns or references
                to a previous question. You must resolve those references using the context.

                Given:
                - PREVIOUS question (what "bu", "onların", "bu şartlar" etc. refer to)
                - CURRENT follow-up question

                Produce TWO things for the COMBINED topic:

                1. **tags**: 2-6 lowercase_snake_case semantic tags for the COMBINED topic.
                   IMPORTANT: Tags must reflect the original topic (from the previous question),
                   not just the surface words of the follow-up.
                   Use English tags: exchange_programs, erasmus, scholarships, gpa, discipline, etc.

                2. **queries**: 2-4 search queries that capture what the user is REALLY asking,
                   with pronouns fully resolved using the previous question's context.
                   Each query must be self-contained (no unresolved pronouns).
                   Keep the same language as the current query.
                   Include relevant synonyms:
                   - GNO ↔ GPA ↔ genel not ortalaması
                   - ÇAP ↔ çift anadal ↔ double major
                   - ECTS ↔ kredi
                   - Erasmus ↔ değişim programı ↔ exchange program
                   Do NOT include the original current query (with pronouns) in "queries".

                JSON schema: {"tags": ["tag1", ...], "queries": ["resolved query 1", ...]}
                Output STRICT JSON ONLY, no explanation.
            """).strip()

            user_input = (
                f"PREVIOUS question: {anchor_query}\n"
                f"CURRENT follow-up: {query}"
            )
        else:
            sys_prompt = textwrap.dedent("""
                You are a search assistant for a university information system.
                Given a user query, produce TWO things:

                1. **tags**: 2-6 lowercase_snake_case semantic tags classifying the query topic.
                   Use English tags even for Turkish queries.
                   Focus on specific domains: scholarships, exchange_programs, library, discipline, gpa, etc.

                2. **queries**: 2-4 alternative phrasings / synonyms of the query for search expansion.
                   Keep the same language as the original query.
                   Include relevant Turkish ↔ English synonyms when applicable, e.g.:
                   - GNO ↔ GPA ↔ genel not ortalaması
                   - ÇAP ↔ çift anadal ↔ double major
                   - yandal ↔ minor
                   - ECTS ↔ kredi ↔ credit
                   - burs ↔ scholarship
                   Do NOT include the original query in "queries".

                JSON schema: {"tags": ["tag1", ...], "queries": ["alt1", "alt2", ...]}
                Output STRICT JSON ONLY, no explanation.
            """).strip()

            user_input = f"Query: {query}"

        result = self.llm.chat_json(user_input, sys_prompt, temperature=0.0)

        tags = []
        expanded = [query]  # always include original query first

        if result:
            raw_tags = result.get("tags", [])
            if isinstance(raw_tags, list):
                tags = [
                    t.strip().lower()
                    for t in raw_tags
                    if isinstance(t, str) and t.strip()
                ]

            raw_queries = result.get("queries", [])
            if isinstance(raw_queries, list):
                for q in raw_queries:
                    if isinstance(q, str) and q.strip() and q.strip() != query:
                        expanded.append(q.strip())

        ctx = f" [anchor='{anchor_query[:40]}...']" if anchor_query else ""
        print(f"[QueryAnalysis] Combined tags={tags}, expanded={len(expanded)} queries{ctx}")
        return tags, expanded

    def extract_negated_terms(
        self,
        query: str,
        lang: Optional[str] = None
    ) -> List[str]:
        """
        Extract negated/excluded terms from query using LLM.

        Args:
            query: User query.
            lang: Detected language.

        Returns:
            List of negated terms (lowercase).
        """
        if not query:
            return []

        sys_prompt = textwrap.dedent("""
            You are a negation extractor for search queries.
            Find terms that are explicitly NEGATED or EXCLUDED from what the user wants.

            JSON schema: {"negated_terms": ["term1", "term2", ...]}

            Rules:
            - ONLY include terms that come AFTER negation words:
              Turkish: "değil", "hariç", "dışında", "haricinde"
              English: "not", "except", "excluding", "other than", "aside from"
            - Questions asking ABOUT something are NOT negation:
              "Disiplin cezaları nelerdir?" -> [] (asking about discipline, not excluding it)
              "What are the penalties?" -> [] (asking about penalties)
            - Questions EXCLUDING something are negation:
              "Erasmus dışında hangi programlar var?" -> ["erasmus"]
              "Burs değil kredi istiyorum" -> ["burs"]
            - Return lowercase terms
            - When in doubt, return EMPTY list []

            Output STRICT JSON ONLY.
        """).strip()

        result = self.llm.chat_json(
            f"Query: {query}",
            sys_prompt,
            temperature=0.0
        )

        if not result:
            return []

        terms = result.get("negated_terms", [])
        if not isinstance(terms, list):
            return []

        return [
            t.strip().lower()
            for t in terms
            if isinstance(t, str) and t.strip()
        ]

    def build_retrieval_query(
        self,
        query: str,
        analysis: Dict
    ) -> str:
        """
        Build the retrieval query, optionally combining with anchor.

        Args:
            query: Original user query.
            analysis: Analysis result from analyze().

        Returns:
            Query string for retrieval (may include anchor for followups).
        """
        if analysis.get("is_followup") and analysis.get("anchor_query"):
            return f"{analysis['anchor_query']}\n\n{query}"
        return query
