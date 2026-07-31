from dataclasses import dataclass, field
from typing import Any

from langchain_core.documents import Document


@dataclass
class QueryPlan:
    original_question: str
    question_type: str
    concepts: list[str]
    section_number: str | None
    retrieval_queries: list[str]
    section_hints: list[str] = field(default_factory=list)
    answer_style: str = "general"


@dataclass
class CandidateDocument:
    document: Document
    relevance_score: float
    query_index: int
    query_text: str


@dataclass
class RankedDocument:
    document: Document
    fusion_score: float
    relevance_score: float | None = None
    matched_queries: int = 1
    keyword_overlap: float = 0.0
    concept_overlap: float = 0.0
    section_boost: float = 0.0
    special_penalty: float = 0.0
    final_score: float = 0.0


@dataclass
class RetrievalConfidence:
    label: str
    score: float
    top_similarity: float
    average_similarity: float
    section_count: int
    concept_coverage: float


@dataclass
class AnswerResult:
    answer: str
    sources: list[dict[str, Any]]
    question_type: str
    detected_concepts: list[str]
    retrieved_document_count: int
    confidence: RetrievalConfidence
