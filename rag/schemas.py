from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from langchain_core.documents import Document


def _deduplicate(values: list[str]) -> list[str]:
    """Remove empty and duplicate values while preserving order."""

    seen: set[str] = set()
    unique_values: list[str] = []

    for value in values:
        cleaned_value = value.strip()

        if cleaned_value and cleaned_value not in seen:
            seen.add(cleaned_value)
            unique_values.append(cleaned_value)

    return unique_values


@dataclass
class LegalReference:
    """
    Canonical reference to a legal provision or one of its child parts.
    """

    provision_type: str
    base_number: str
    subsection_path: list[str] = field(default_factory=list)
    component_type: str | None = None
    original_citation: str = ""


@dataclass
class QueryPlan:
    """
    Describes how a user question should be retrieved.

    Existing fields are preserved so current router and retriever
    code continues to work.
    """

    original_question: str
    question_type: str
    concepts: list[str]
    section_number: str | None
    retrieval_queries: list[str]

    section_hints: list[str] = field(default_factory=list)
    answer_style: str = "general"

    # Multi-document support
    document_ids: list[str] = field(default_factory=list)
    document_hints: list[str] = field(default_factory=list)

    # Constitution uses Articles instead of Sections
    article_number: str | None = None

    # General field for both Sections and Articles
    provision_numbers: list[str] = field(default_factory=list)
    provision_type: str | None = None

    # F-03: structured subsection/clause references
    legal_references: list[LegalReference] = field(
        default_factory=list
    )

    def __post_init__(self) -> None:
        self.concepts = _deduplicate(self.concepts)
        self.retrieval_queries = _deduplicate(
            self.retrieval_queries
        )
        self.section_hints = _deduplicate(
            self.section_hints
        )
        self.document_ids = _deduplicate(
            self.document_ids
        )
        self.document_hints = _deduplicate(
            self.document_hints
        )
        self.provision_numbers = _deduplicate(
            self.provision_numbers
        )

        if (
            self.section_number
            and self.section_number
            not in self.provision_numbers
        ):
            self.provision_numbers.append(
                self.section_number
            )

        if (
            self.article_number
            and self.article_number
            not in self.provision_numbers
        ):
            self.provision_numbers.append(
                self.article_number
            )


@dataclass
class CandidateDocument:
    """
    A document returned by one retrieval query before fusion
    and final ranking.
    """

    document: Document
    relevance_score: float | None
    query_index: int
    query_text: str

    retrieval_method: str = "vector"
    retrieval_rank: int | None = None

    @property
    def metadata(self) -> dict[str, Any]:
        return self.document.metadata

    @property
    def document_id(self) -> str | None:
        return self.document.metadata.get(
            "document_id"
        )

    @property
    def provision_number(self) -> str | None:
        metadata = self.document.metadata

        return (
            metadata.get("provision_number")
            or metadata.get("section_number")
            or metadata.get("article_number")
        )


@dataclass
class RankedDocument:
    """
    Candidate document after reciprocal-rank fusion,
    metadata scoring and legal relevance ranking.
    """

    document: Document
    fusion_score: float

    relevance_score: float | None = None
    matched_queries: int = 1
    retrieval_methods: list[str] = field(default_factory=list)
    matched_query_indices: list[int] = field(default_factory=list)
    retrieval_routes: list[str] = field(default_factory=list)
    keyword_overlap: float = 0.0
    concept_overlap: float = 0.0

    # Existing field retained for compatibility
    section_boost: float = 0.0

    # Multi-document scoring
    document_boost: float = 0.0
    provision_boost: float = 0.0

    special_penalty: float = 0.0
    duplicate_penalty: float = 0.0
    final_score: float = 0.0

    @property
    def metadata(self) -> dict[str, Any]:
        return self.document.metadata

    @property
    def document_id(self) -> str | None:
        return self.document.metadata.get(
            "document_id"
        )

    @property
    def provision_number(self) -> str | None:
        metadata = self.document.metadata

        return (
            metadata.get("provision_number")
            or metadata.get("section_number")
            or metadata.get("article_number")
        )


@dataclass
class RetrievalConfidence:
    """Confidence information calculated after retrieval."""

    label: str
    score: float
    top_similarity: float
    average_similarity: float
    section_count: int
    concept_coverage: float

    # Multi-document confidence information
    document_count: int = 0
    document_coverage: float = 0.0
    exact_document_match: bool = False
    exact_provision_match: bool = False


@dataclass
class AnswerResult:
    """Final answer returned by the complete RAG pipeline."""

    answer: str
    sources: list[dict[str, Any]]
    question_type: str
    detected_concepts: list[str]
    retrieved_document_count: int
    confidence: RetrievalConfidence

    # LLM and fallback tracking
    used_llm: bool = False
    llm_provider: str | None = None
    fallback_used: bool = False
