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

    def analyze(
        self,
        query: str,
        history: Optional[List[Dict]] = None
    ) -> Dict:
        """
        Perform complete query analysis.

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
        """
        history = history or []

        language = self.detect_language(query)
        intent = self.detect_intent(query)
        is_followup, anchor = self.detect_followup(query, history)
        tags = self.infer_tags(query, language)
        negated = self.extract_negated_terms(query, language)

        return {
            "language": language,
            "intent": intent,
            "is_followup": is_followup,
            "anchor_query": anchor,
            "tags": tags,
            "negated_terms": negated,
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

        # Check semantic similarity
        try:
            q_vec = self.embedding.embed(query)
            last_vec = self.embedding.embed(last_q)
            sim = self.embedding.cosine_similarity(q_vec, last_vec)

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
