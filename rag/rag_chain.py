import math
import os
import re
import time
from dataclasses import dataclass
from typing import Any

from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_google_genai import ChatGoogleGenerativeAI
from qdrant_client.models import (
    FieldCondition,
    Filter,
    MatchAny,
    MatchValue,
)

from rag.vector_store import create_vector_store


load_dotenv()


# -------------------------------------------------------------------
# Environment settings
# -------------------------------------------------------------------

CHAT_MODEL = os.getenv(
    "CHAT_MODEL",
    "gemini-3.6-flash",
)

TOP_K = int(
    os.getenv(
        "TOP_K",
        "6",
    )
)

MAX_CONTEXT_DOCUMENTS = int(
    os.getenv(
        "MAX_CONTEXT_DOCUMENTS",
        "10",
    )
)

MAX_CONTEXT_SECTIONS = int(
    os.getenv(
        "MAX_CONTEXT_SECTIONS",
        "4",
    )
)

MIN_RELEVANCE_SCORE = float(
    os.getenv(
        "MIN_RELEVANCE_SCORE",
        "0.0",
    )
)

ENABLE_NEIGHBOR_RETRIEVAL = (
    os.getenv(
        "ENABLE_NEIGHBOR_RETRIEVAL",
        "True",
    ).strip().lower()
    in {"1", "true", "yes", "on"}
)

NEIGHBOR_SECTION_RADIUS = int(
    os.getenv(
        "NEIGHBOR_SECTION_RADIUS",
        "2",
    )
)

SEMANTIC_DEDUP_THRESHOLD = float(
    os.getenv(
        "SEMANTIC_DEDUP_THRESHOLD",
        "0.94",
    )
)

ADAPTIVE_SECTION_SCORE_GAP = float(
    os.getenv(
        "ADAPTIVE_SECTION_SCORE_GAP",
        "0.18",
    )
)

INTENT_SCORE_WEIGHT = float(
    os.getenv(
        "INTENT_SCORE_WEIGHT",
        "0.20",
    )
)

MMR_LAMBDA = float(
    os.getenv(
        "MMR_LAMBDA",
        "0.75",
    )
)

MAX_ALTERNATIVE_SECTIONS = int(
    os.getenv(
        "MAX_ALTERNATIVE_SECTIONS",
        "1",
    )
)

DEBUG_RETRIEVAL = (
    os.getenv(
        "DEBUG_RETRIEVAL",
        "False",
    ).strip().lower()
    in {"1", "true", "yes", "on"}
)

DEBUG_PREVIEW_CHARS = int(
    os.getenv(
        "DEBUG_PREVIEW_CHARS",
        "800",
    )
)

GOOGLE_API_MAX_RETRIES = int(
    os.getenv(
        "GOOGLE_API_MAX_RETRIES",
        "4",
    )
)

GOOGLE_API_RETRY_DELAY = float(
    os.getenv(
        "GOOGLE_API_RETRY_DELAY",
        "5",
    )
)

QUESTION_MIN_LENGTH = 3
QUESTION_MAX_LENGTH = 4000


def validate_rag_settings() -> None:
    """Validate retrieval and ranking settings."""

    if TOP_K <= 0:
        raise ValueError("TOP_K must be greater than zero.")

    if MAX_CONTEXT_DOCUMENTS <= 0:
        raise ValueError(
            "MAX_CONTEXT_DOCUMENTS must be greater than zero."
        )

    if MAX_CONTEXT_SECTIONS <= 0:
        raise ValueError(
            "MAX_CONTEXT_SECTIONS must be greater than zero."
        )

    if not 0.0 <= MIN_RELEVANCE_SCORE <= 1.0:
        raise ValueError(
            "MIN_RELEVANCE_SCORE must be between 0 and 1."
        )

    if not 0.0 <= SEMANTIC_DEDUP_THRESHOLD <= 1.0:
        raise ValueError(
            "SEMANTIC_DEDUP_THRESHOLD must be between 0 and 1."
        )

    if not 0.0 <= ADAPTIVE_SECTION_SCORE_GAP <= 1.0:
        raise ValueError(
            "ADAPTIVE_SECTION_SCORE_GAP must be between 0 and 1."
        )

    if not 0.0 <= INTENT_SCORE_WEIGHT <= 1.0:
        raise ValueError(
            "INTENT_SCORE_WEIGHT must be between 0 and 1."
        )

    if not 0.0 <= MMR_LAMBDA <= 1.0:
        raise ValueError(
            "MMR_LAMBDA must be between 0 and 1."
        )

    if MAX_ALTERNATIVE_SECTIONS < 0:
        raise ValueError(
            "MAX_ALTERNATIVE_SECTIONS cannot be negative."
        )


# -------------------------------------------------------------------
# Multi-document registry used for document detection and filtering
# -------------------------------------------------------------------

DOCUMENT_REGISTRY: dict[str, dict[str, Any]] = {
    "substantive_criminal_law": {
        "display_name": "Pakistan Penal Code",
        "aliases": (
            "pakistan penal code",
            "penal code",
            "ppc",
        ),
    },
    "criminal_procedure": {
        "display_name": "Code of Criminal Procedure",
        "aliases": (
            "code of criminal procedure",
            "criminal procedure code",
            "criminal procedure",
            "crpc",
        ),
    },
    "constitutional_law": {
        "display_name": "Constitution of Pakistan",
        "aliases": (
            "constitution of pakistan",
            "pakistan constitution",
            "constitution",
        ),
    },
    "law_of_evidence": {
        "display_name": "Qanun-e-Shahadat Order",
        "aliases": (
            "qanun-e-shahadat order",
            "qanun e shahadat order",
            "qanun-e-shahadat",
            "qanun e shahadat",
            "law of evidence",
            "evidence law",
            "evidence act",
        ),
    },
}


@dataclass(frozen=True)
class DocumentSelection:
    """Documents explicitly requested or inferred from the user query."""

    document_types: tuple[str, ...]
    display_names: tuple[str, ...]
    aliases_found: tuple[str, ...]

    @property
    def is_document_specific(self) -> bool:
        return bool(self.document_types)

    @property
    def is_cross_document(self) -> bool:
        return len(self.document_types) > 1


# -------------------------------------------------------------------
# Legal knowledge maps used only for retrieval and ranking
# -------------------------------------------------------------------

SCENARIO_MARKERS: dict[str, int] = {
    "entrusted": 3,
    "embezzled": 3,
    "misappropriated": 3,
    "accountant": 2,
    "employee": 2,
    "clerk": 2,
    "servant": 2,
    "agent": 2,
    "murder": 3,
    "killed": 3,
    "caused death": 3,
    "riot": 2,
    "mob": 2,
    "stampede": 2,
    "death": 2,
    "injury": 2,
    "fraud": 2,
    "forged": 2,
    "kidnapped": 3,
    "used for personal": 3,
    "personal expenses": 3,
    "company funds": 3,
    "which provision applies": 2,
    "which provisions apply": 2,
    "which sections apply": 2,
    "under the law": 1,
    "charged": 2,
    "accused": 2,
    "defendant": 2,
    "found": 2,
    "lost wallet": 3,
    "found on the road": 3,
    "later decided to keep": 3,
    "kept it for personal use": 3,
}

LEGAL_CONCEPTS: dict[str, dict[str, Any]] = {
    "murder": {
        "markers": (
            "murder",
            "qatl-e-amd",
            "intentional killing",
            "killed a person",
            "commit a murder",
            "committed murder",
        ),
        "queries": (
            "Section 300 definition of qatl-e-amd",
            "Section 302 punishment of qatl-e-amd",
        ),
        "preferred_sections": {"300", "302"},
        "keywords": {
            "murder",
            "qatl",
            "death",
            "intention",
            "knowledge",
            "punishment",
        },
    },
    "theft": {
        "markers": (
            "theft",
            "stole",
            "stolen",
            "stealing",
        ),
        "queries": (
            "Section 378 definition of theft",
            "Section 379 punishment for theft",
        ),
        "preferred_sections": {"378", "379"},
        "keywords": {
            "theft",
            "dishonestly",
            "movable",
            "property",
            "punishment",
        },
    },
    "breach_of_trust": {
        "markers": (
            "embezzle",
            "embezzled",
            "entrusted",
            "company funds",
            "accountant",
            "employee funds",
            "personal expenses",
            "misappropriated funds",
            "breach of trust",
        ),
        "queries": (
            "Section 405 criminal breach of trust",
            "Section 408 criminal breach of trust by clerk or servant",
            "Section 409 criminal breach of trust by public servant banker merchant or agent",
        ),
        "preferred_sections": {"405", "408", "409"},
        "keywords": {
            "entrusted",
            "dominion",
            "dishonestly",
            "misappropriates",
            "converts",
            "clerk",
            "servant",
            "agent",
        },
    },
    "falsification": {
        "markers": (
            "false entry",
            "falsified",
            "altered records",
            "altered accounts",
            "destroyed accounts",
            "omitted entries",
            "books of account",
        ),
        "queries": (
            "Section 477-A falsification of accounts",
        ),
        "preferred_sections": {"477-A"},
        "keywords": {
            "falsification",
            "accounts",
            "false",
            "entry",
            "defraud",
        },
    },
    "unlawful_assembly": {
        "markers": (
            "mob",
            "riot",
            "rioting",
            "unlawful assembly",
            "common object",
            "stampede",
        ),
        "queries": (
            "Unlawful assembly and common object liability",
            "Rioting and use of force by unlawful assembly",
        ),
        "preferred_sections": set(),
        "keywords": {
            "assembly",
            "riot",
            "rioting",
            "common",
            "object",
            "force",
            "violence",
        },
    },
    "abetment": {
        "markers": (
            "incite",
            "inciting",
            "instigate",
            "instigating",
            "encourage",
            "abet",
            "abetment",
        ),
        "queries": (
            "Abetment by instigation or intentional aid",
            "Abetment of an offence by the public or by more than ten persons",
        ),
        "preferred_sections": {"107"},
        "keywords": {
            "abetment",
            "instigates",
            "conspiracy",
            "intentionally",
            "aids",
        },
    },
    "found_property": {
        "markers": (
            "found property",
            "found wallet",
            "lost wallet",
            "found on the road",
            "lost property",
            "later decided to keep",
            "kept it for personal use",
            "finder",
        ),
        "queries": (
            "Section 403 dishonest misappropriation of found property",
            "Section 378 theft property not in another person's possession",
        ),
        "preferred_sections": {"403", "378"},
        "keywords": {
            "found",
            "lost",
            "owner",
            "misappropriation",
            "possession",
            "dishonestly",
        },
    },
}

SPECIAL_SECTION_REQUIREMENTS: dict[str, set[str]] = {
    "317": {
        "heir",
        "inheritance",
        "succession",
        "beneficiary",
        "will",
        "estate",
    },
    "396": {
        "dacoity",
        "robbery",
        "five or more",
        "five persons",
        "gang",
    },
    "404": {
        "deceased",
        "death of owner",
        "estate",
        "servant of deceased",
    },
    "477-A": {
        "false entry",
        "falsified",
        "altered",
        "omitted",
        "accounts",
        "accounting records",
        "books",
    },
    "390": {
        "force",
        "fear",
        "threat",
        "hurt",
        "restraint",
        "weapon",
        "robbery",
    },
    "405": {
        "entrusted",
        "entrustment",
        "dominion",
        "agent",
        "employee",
        "servant",
        "clerk",
    },
    "410": {
        "stolen property",
        "receiving stolen property",
        "possession of stolen property",
        "transfer of stolen property",
        "classification of property",
        "property obtained by theft",
        "property obtained by robbery",
        "property obtained by extortion",
    },
}

STOP_WORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "has",
    "he",
    "her",
    "his",
    "in",
    "is",
    "it",
    "of",
    "on",
    "or",
    "person",
    "that",
    "the",
    "their",
    "this",
    "to",
    "under",
    "was",
    "were",
    "which",
    "who",
    "with",
}


# -------------------------------------------------------------------
# Retrieval result models
# -------------------------------------------------------------------

@dataclass
class RankedDocument:
    document: Document
    fusion_score: float
    relevance_score: float | None = None
    matched_queries: int = 1
    keyword_overlap: float = 0.0
    concept_overlap: float = 0.0
    section_boost: float = 0.0
    intent_score: float = 0.0
    special_penalty: float = 0.0
    final_score: float = 0.0
    context_role: str = "supporting"


@dataclass
class RetrievalConfidence:
    label: str
    score: float
    top_similarity: float
    average_similarity: float
    section_count: int
    concept_coverage: float


# -------------------------------------------------------------------
# Gemini model
# -------------------------------------------------------------------

def get_chat_model() -> ChatGoogleGenerativeAI:
    """Create the Gemini model used for grounded answer generation."""

    api_key = os.getenv("GOOGLE_API_KEY")

    if not api_key:
        raise ValueError(
            "GOOGLE_API_KEY is missing from the .env file."
        )

    return ChatGoogleGenerativeAI(
        model=CHAT_MODEL,
        google_api_key=api_key,
    )


# -------------------------------------------------------------------
# Adaptive retriever
# -------------------------------------------------------------------

class AdaptiveRetriever:
    """
    Backward-compatible retrieval wrapper with filtered/scored search.
    """

    def __init__(self, vector_store) -> None:
        self.vector_store = vector_store

    def invoke(
        self,
        query: str,
        k: int | None = None,
        metadata_filter: Filter | None = None,
    ) -> list[Document]:
        return self.vector_store.similarity_search(
            query=query,
            k=k or TOP_K,
            filter=metadata_filter,
        )

    def search_with_scores(
        self,
        query: str,
        k: int,
        metadata_filter: Filter | None = None,
    ) -> list[tuple[Document, float]]:
        try:
            results = self.vector_store.similarity_search_with_score(
                query=query,
                k=k,
                filter=metadata_filter,
            )

            return [
                (document, float(score))
                for document, score in results
            ]

        except (AttributeError, TypeError):
            documents = self.invoke(
                query=query,
                k=k,
                metadata_filter=metadata_filter,
            )

            return [
                (document, 0.0)
                for document in documents
            ]


def create_rag_components():
    """
    Return an adaptive retriever, Gemini model, and Qdrant client.
    """

    validate_rag_settings()

    vector_store, client = create_vector_store(
        reset=False
    )

    return (
        AdaptiveRetriever(vector_store),
        get_chat_model(),
        client,
    )


# -------------------------------------------------------------------
# Question analysis
# -------------------------------------------------------------------

def normalize_text(text: str) -> str:
    """Normalize text for deterministic comparison."""

    return re.sub(
        r"\s+",
        " ",
        text.lower(),
    ).strip()


def tokenize(text: str) -> set[str]:
    """Tokenize text for lightweight lexical scoring."""

    tokens = re.findall(
        r"[a-z0-9]+(?:-[a-z0-9]+)?",
        text.lower(),
    )

    return {
        token
        for token in tokens
        if len(token) > 1
        and token not in STOP_WORDS
    }


def extract_section_number(
    question: str,
) -> str | None:
    """
    Extract an explicitly requested section or constitutional article.

    Both values are stored in the common `section_number` metadata field.
    """

    match = re.search(
        r"\b(?:section|article|art\.)\s+"
        r"(\d+(?:-[A-Za-z]+)?[A-Za-z]?)\b",
        question,
        re.IGNORECASE,
    )

    return (
        match.group(1).upper()
        if match
        else None
    )


def detect_requested_documents(
    question: str,
) -> DocumentSelection:
    """
    Detect one or more explicitly named indexed legal documents.

    Longer aliases are matched before shorter aliases so that
    `Constitution of Pakistan` is preferred over `Constitution`.
    """

    normalized_question = normalize_text(
        question
    )

    matched: list[
        tuple[int, int, str, str, str]
    ] = []

    for document_type, config in (
        DOCUMENT_REGISTRY.items()
    ):
        display_name = str(
            config["display_name"]
        )

        aliases = sorted(
            (
                str(alias)
                for alias in config["aliases"]
            ),
            key=len,
            reverse=True,
        )

        for alias in aliases:
            normalized_alias = normalize_text(
                alias
            )

            match = re.search(
                rf"(?<![a-z0-9])"
                rf"{re.escape(normalized_alias)}"
                rf"(?![a-z0-9])",
                normalized_question,
            )

            if match:
                matched.append(
                    (
                        match.start(),
                        -len(normalized_alias),
                        document_type,
                        display_name,
                        alias,
                    )
                )
                break

    matched.sort()

    document_types: list[str] = []
    display_names: list[str] = []
    aliases_found: list[str] = []

    for (
        _position,
        _negative_length,
        document_type,
        display_name,
        alias,
    ) in matched:
        if document_type in document_types:
            continue

        document_types.append(
            document_type
        )
        display_names.append(
            display_name
        )
        aliases_found.append(
            alias
        )

    return DocumentSelection(
        document_types=tuple(
            document_types
        ),
        display_names=tuple(
            display_names
        ),
        aliases_found=tuple(
            aliases_found
        ),
    )


def format_document_selection(
    selection: DocumentSelection,
) -> str:
    """Return a readable summary of selected documents."""

    if not selection.display_names:
        return "All indexed legal documents"

    return ", ".join(
        selection.display_names
    )


def detect_legal_concepts(
    question: str,
) -> list[str]:
    """Detect relevant legal concept families from the question."""

    normalized = normalize_text(question)
    detected: list[str] = []

    for concept_name, concept_data in LEGAL_CONCEPTS.items():
        if any(
            marker in normalized
            for marker in concept_data["markers"]
        ):
            detected.append(concept_name)

    return detected


def calculate_scenario_score(
    question: str,
) -> int:
    """Compute a weighted factual-scenario score."""

    normalized = normalize_text(question)

    return sum(
        weight
        for marker, weight in SCENARIO_MARKERS.items()
        if marker in normalized
    )


def classify_question(question: str) -> str:
    """Classify the user's legal query."""

    normalized = normalize_text(question)

    if any(
        phrase in normalized
        for phrase in (
            "difference between",
            "compare",
            "distinguish between",
        )
    ):
        return "comparison"

    if extract_section_number(question):
        return "section_lookup"

    scenario_score = calculate_scenario_score(
        question
    )

    if (
        scenario_score >= 3
        or (
            len(normalized.split()) >= 18
            and detect_legal_concepts(question)
        )
    ):
        return "fact_scenario"

    if any(
        term in normalized
        for term in (
            "punishment",
            "penalty",
            "sentence",
            "imprisonment",
            "fine",
        )
    ):
        return "punishment"

    if any(
        phrase in normalized
        for phrase in (
            "definition",
            "define",
            "meaning of",
            "legal meaning",
            "what is",
        )
    ):
        return "definition"

    return "general"


def get_retrieval_k(
    question_type: str,
) -> int:
    """Choose retrieval depth based on question complexity."""

    return {
        "section_lookup": max(5, TOP_K),
        "definition": max(7, TOP_K),
        "punishment": max(8, TOP_K),
        "comparison": max(14, TOP_K),
        "fact_scenario": max(16, TOP_K),
        "general": max(10, TOP_K),
    }.get(
        question_type,
        TOP_K,
    )


def build_retrieval_queries(
    question: str,
    question_type: str,
    detected_concepts: list[str] | None = None,
    document_selection: DocumentSelection | None = None,
) -> list[str]:
    """Build original, task-specific, and legal-concept queries."""

    original = question.strip()
    queries: list[str] = [original]
    section_number = extract_section_number(original)
    concepts = detected_concepts or detect_legal_concepts(
        original
    )
    selection = (
        document_selection
        or DocumentSelection(
            document_types=(),
            display_names=(),
            aliases_found=(),
        )
    )
    selected_document_label = (
        format_document_selection(
            selection
        )
    )

    if question_type == "section_lookup" and section_number:
        queries.append(
            f"{selected_document_label} Section {section_number} "
            "exact statutory text, heading, explanation, "
            "illustrations and punishment"
        )

    elif question_type == "punishment":
        queries.append(
            f"{original} exact punishment provision, imprisonment "
            "term, fine, conditions and legal capacity"
        )

    elif question_type == "definition":
        queries.append(
            f"{original} exact statutory definition, legal elements, "
            "explanations and illustrations"
        )

    elif question_type == "comparison":
        queries.append(
            f"{original} compare statutory definitions, elements, "
            "conditions and punishments"
        )

    elif question_type == "fact_scenario":
        queries.append(
            f"{selected_document_label} provisions directly connected "
            f"to this factual scenario: {original}"
        )

    for concept_name in concepts:
        queries.extend(
            LEGAL_CONCEPTS[concept_name]["queries"]
        )

    return list(
        dict.fromkeys(queries)
    )


# -------------------------------------------------------------------
# Metadata filters
# -------------------------------------------------------------------

def build_document_conditions(
    document_types: tuple[str, ...] | list[str] | None,
) -> list[FieldCondition]:
    """Build optional Qdrant conditions for selected document types."""

    normalized_types = [
        str(document_type)
        for document_type in (
            document_types or []
        )
        if str(document_type).strip()
    ]

    if not normalized_types:
        return []

    if len(normalized_types) == 1:
        match = MatchValue(
            value=normalized_types[0],
        )
    else:
        match = MatchAny(
            any=normalized_types,
        )

    return [
        FieldCondition(
            key="metadata.document_type",
            match=match,
        )
    ]


def build_retrieval_filter(
    document_types: tuple[str, ...] | list[str] | None = None,
    section_numbers: list[str] | None = None,
) -> Filter:
    """
    Build a reusable document-aware Qdrant filter.

    With no document type, retrieval searches the whole collection.
    """

    must_conditions: list[Any] = []

    must_conditions.extend(
        build_document_conditions(
            document_types
        )
    )

    if section_numbers:
        normalized_sections = [
            str(section).upper()
            for section in section_numbers
        ]

        section_match = (
            MatchValue(
                value=normalized_sections[0],
            )
            if len(normalized_sections) == 1
            else MatchAny(
                any=normalized_sections,
            )
        )

        must_conditions.append(
            FieldCondition(
                key="metadata.section_number",
                match=section_match,
            )
        )

    must_conditions.extend(
        [
            FieldCondition(
                key="metadata.heading_only_chunk",
                match=MatchValue(
                    value=False,
                ),
            ),
            FieldCondition(
                key="metadata.section_body_present",
                match=MatchValue(
                    value=True,
                ),
            ),
        ]
    )

    return Filter(
        must=must_conditions
    )


def build_section_filter(
    section_number: str,
    document_types: tuple[str, ...] | list[str] | None = None,
) -> Filter:
    """Build a document-aware Qdrant filter for one exact section."""

    return build_retrieval_filter(
        document_types=document_types,
        section_numbers=[
            section_number,
        ],
    )


def build_sections_filter(
    section_numbers: list[str],
    document_types: tuple[str, ...] | list[str] | None = None,
) -> Filter:
    """Build a document-aware Qdrant filter for several sections."""

    return build_retrieval_filter(
        document_types=document_types,
        section_numbers=section_numbers,
    )


def build_document_filter(
    document_types: tuple[str, ...] | list[str],
) -> Filter:
    """Build a Qdrant filter that searches selected documents only."""

    return build_retrieval_filter(
        document_types=document_types,
    )


def is_usable_document(
    document: Document,
) -> bool:
    """Reject empty, heading-only, and bodyless candidates."""

    metadata = document.metadata

    return bool(
        document.page_content.strip()
        and not metadata.get(
            "heading_only_chunk",
            False,
        )
        and metadata.get(
            "section_body_present",
            True,
        )
    )


# -------------------------------------------------------------------
# Ranking helpers
# -------------------------------------------------------------------

def _document_key(
    document: Document,
) -> tuple[Any, ...]:
    metadata = document.metadata

    return (
        metadata.get("document_id")
        or metadata.get("document_name"),
        metadata.get("section_number"),
        metadata.get("section_part_number", 1),
        metadata.get("page_start"),
        metadata.get("page_end"),
        metadata.get("chunk_number"),
    )


def _section_key(
    document: Document,
) -> tuple[Any, ...]:
    metadata = document.metadata
    section_number = metadata.get(
        "section_number"
    )

    if section_number:
        return (
            metadata.get("document_id")
            or metadata.get("document_name"),
            str(section_number),
        )

    return (
        metadata.get("document_id")
        or metadata.get("document_name"),
        "unsectioned",
        metadata.get("chunk_number"),
    )


def normalize_similarity_score(
    score: float,
) -> float:
    """
    Normalize common Qdrant cosine-score values into 0..1.

    Qdrant cosine scores are usually already higher-is-better. Values
    outside 0..1 are bounded conservatively.
    """

    if math.isnan(score) or math.isinf(score):
        return 0.0

    return max(
        0.0,
        min(1.0, score),
    )


def keyword_overlap_score(
    question: str,
    document: Document,
) -> float:
    """Calculate Jaccard-like keyword overlap."""

    question_tokens = tokenize(question)
    document_tokens = tokenize(
        (
            str(
                document.metadata.get(
                    "section_title",
                    "",
                )
            )
            + " "
            + document.page_content[:1200]
        )
    )

    if not question_tokens:
        return 0.0

    overlap = question_tokens & document_tokens

    return min(
        1.0,
        len(overlap)
        / max(1, len(question_tokens)),
    )


def concept_overlap_score(
    document: Document,
    detected_concepts: list[str],
) -> float:
    """Measure overlap with detected concept keywords/sections."""

    if not detected_concepts:
        return 0.0

    metadata = document.metadata
    section_number = str(
        metadata.get(
            "section_number",
            "",
        )
    ).upper()

    document_text = normalize_text(
        (
            str(
                metadata.get(
                    "section_title",
                    "",
                )
            )
            + " "
            + document.page_content[:1500]
        )
    )

    scores: list[float] = []

    for concept_name in detected_concepts:
        concept = LEGAL_CONCEPTS[concept_name]
        keyword_hits = sum(
            keyword in document_text
            for keyword in concept["keywords"]
        )

        keyword_score = keyword_hits / max(
            1,
            len(concept["keywords"]),
        )

        section_score = (
            1.0
            if section_number
            in concept["preferred_sections"]
            else 0.0
        )

        scores.append(
            min(
                1.0,
                0.65 * keyword_score
                + 0.35 * section_score,
            )
        )

    return max(scores, default=0.0)


def section_match_boost(
    document: Document,
    question: str,
    detected_concepts: list[str],
) -> float:
    """Boost exact section, offence-title, and preferred-section matches."""

    metadata = document.metadata
    section_number = str(
        metadata.get(
            "section_number",
            "",
        )
    ).upper()

    explicit_section = extract_section_number(
        question
    )

    if (
        explicit_section
        and section_number == explicit_section
    ):
        return 1.0

    title = normalize_text(
        str(
            metadata.get(
                "section_title",
                "",
            )
        )
    )

    question_normalized = normalize_text(
        question
    )

    title_tokens = tokenize(title)

    if (
        title
        and title in question_normalized
    ):
        return 0.9

    if (
        title_tokens
        and len(
            title_tokens
            & tokenize(question)
        )
        >= min(2, len(title_tokens))
    ):
        return 0.6

    for concept_name in detected_concepts:
        if section_number in LEGAL_CONCEPTS[
            concept_name
        ]["preferred_sections"]:
            return 0.7

    return 0.0


def special_section_penalty(
    document: Document,
    question: str,
) -> float:
    """Penalize special provisions whose required facts are absent."""

    section_number = str(
        document.metadata.get(
            "section_number",
            "",
        )
    ).upper()

    requirements = SPECIAL_SECTION_REQUIREMENTS.get(
        section_number
    )

    if not requirements:
        return 0.0

    normalized_question = normalize_text(
        question
    )

    if any(
        requirement in normalized_question
        for requirement in requirements
    ):
        return 0.0

    return 0.40


def intent_alignment_score(
    document: Document,
    question_type: str,
    question: str,
    detected_concepts: list[str],
) -> float:
    """
    Score how well a section's legal function matches the user's intent.

    This distinguishes offence definitions, punishment provisions,
    supporting definitions, and alternatives before final selection.
    """

    metadata = document.metadata
    title = normalize_text(
        str(
            metadata.get(
                "section_title",
                "",
            )
        )
    )
    section_number = str(
        metadata.get(
            "section_number",
            "",
        )
    ).upper()
    normalized_question = normalize_text(
        question
    )

    score = 0.0

    if question_type == "section_lookup":
        explicit = extract_section_number(question)

        if explicit and section_number == explicit:
            return 1.0

    if question_type == "punishment":
        punishment_markers = (
            "punishment",
            "imprisonment",
            "fine",
            "shall be punished",
        )

        if any(
            marker in title
            or marker in normalize_text(
                document.page_content[:900]
            )
            for marker in punishment_markers
        ):
            score += 0.85

    elif question_type == "definition":
        definition_markers = (
            "definition",
            "is said to",
            "means",
            "whoever",
        )

        if any(
            marker in title
            or marker in normalize_text(
                document.page_content[:900]
            )
            for marker in definition_markers
        ):
            score += 0.75

    elif question_type == "comparison":
        if any(
            section_number
            in LEGAL_CONCEPTS[concept]["preferred_sections"]
            for concept in detected_concepts
        ):
            score += 0.75

    elif question_type == "fact_scenario":
        if any(
            section_number
            in LEGAL_CONCEPTS[concept]["preferred_sections"]
            for concept in detected_concepts
        ):
            score += 0.90

        if "punishment" in title:
            score -= 0.10

    if title and title in normalized_question:
        score += 0.20

    return max(
        0.0,
        min(1.0, score),
    )


def classify_context_role(
    item: RankedDocument,
    question_type: str,
    question: str,
    detected_concepts: list[str],
    top_score: float,
) -> str:
    """
    Assign a functional context role to a retrieved legal section.
    """

    metadata = item.document.metadata
    section_number = str(
        metadata.get(
            "section_number",
            "",
        )
    ).upper()
    title = normalize_text(
        str(
            metadata.get(
                "section_title",
                "",
            )
        )
    )

    explicit = extract_section_number(
        question
    )

    if explicit and section_number == explicit:
        return "primary"

    if item.final_score >= top_score - 0.03:
        return "primary"

    punishment_markers = (
        "punishment",
        "imprisonment",
        "fine",
        "shall be punished",
    )

    if (
        question_type == "punishment"
        and any(
            marker in title
            or marker in normalize_text(
                item.document.page_content[:900]
            )
            for marker in punishment_markers
        )
    ):
        return "punishment"

    if any(
        section_number
        in LEGAL_CONCEPTS[concept]["preferred_sections"]
        for concept in detected_concepts
    ):
        if item.final_score >= top_score - 0.10:
            return "supporting"

        return "alternative"

    if question_type == "definition":
        return "definition"

    if item.final_score >= top_score - 0.12:
        return "supporting"

    return "alternative"


def mmr_select_groups(
    ranked_groups: list[list[RankedDocument]],
    limit: int,
) -> list[list[RankedDocument]]:
    """
    Select diverse section groups using a lightweight MMR strategy.

    Relevance comes from final_score. Redundancy is measured using token
    overlap between the leading chunks of already selected groups.
    """

    if not ranked_groups or limit <= 0:
        return []

    remaining = list(ranked_groups)
    selected: list[list[RankedDocument]] = []

    while remaining and len(selected) < limit:
        best_group = None
        best_score = float("-inf")

        for group in remaining:
            relevance = max(
                item.final_score
                for item in group
            )

            if not selected:
                mmr_score = relevance
            else:
                candidate_text = " ".join(
                    item.document.page_content[:1000]
                    for item in group
                )

                redundancy = max(
                    text_similarity(
                        candidate_text,
                        " ".join(
                            item.document.page_content[:1000]
                            for item in selected_group
                        ),
                    )
                    for selected_group in selected
                )

                mmr_score = (
                    MMR_LAMBDA * relevance
                    - (1.0 - MMR_LAMBDA)
                    * redundancy
                )

            if mmr_score > best_score:
                best_score = mmr_score
                best_group = group

        if best_group is None:
            break

        selected.append(best_group)
        remaining.remove(best_group)

    return selected


def role_based_group_selection(
    ranked_groups: list[list[RankedDocument]],
    question_type: str,
    limit: int,
) -> list[list[RankedDocument]]:
    """
    Keep only roles needed for the detected question type.
    """

    if not ranked_groups:
        return []

    selected: list[list[RankedDocument]] = []
    alternative_count = 0

    role_priorities = {
        "section_lookup": {
            "primary",
            "definition",
            "supporting",
        },
        "definition": {
            "primary",
            "definition",
            "supporting",
        },
        "punishment": {
            "primary",
            "punishment",
            "definition",
            "supporting",
        },
        "comparison": {
            "primary",
            "supporting",
            "alternative",
            "definition",
            "punishment",
        },
        "fact_scenario": {
            "primary",
            "supporting",
            "alternative",
            "punishment",
        },
        "general": {
            "primary",
            "supporting",
            "definition",
        },
    }

    allowed_roles = role_priorities.get(
        question_type,
        role_priorities["general"],
    )

    for group in ranked_groups:
        group_role = group[0].context_role

        if group_role not in allowed_roles:
            continue

        if group_role == "alternative":
            if alternative_count >= MAX_ALTERNATIVE_SECTIONS:
                continue

            alternative_count += 1

        selected.append(group)

        if len(selected) >= limit:
            break

    return selected


def calculate_final_score(
    item: RankedDocument,
) -> float:
    """Combine semantic, section, keyword, concept, and quality signals."""

    semantic = normalize_similarity_score(
        item.relevance_score or 0.0
    )

    base_weight = max(
        0.0,
        1.0 - INTENT_SCORE_WEIGHT,
    )

    base_score = (
        0.55 * semantic
        + 0.20 * item.section_boost
        + 0.15 * item.keyword_overlap
        + 0.10 * item.concept_overlap
    )

    score = (
        base_weight * base_score
        + INTENT_SCORE_WEIGHT * item.intent_score
    )

    # RRF remains a small stability signal across multiple queries.
    score += min(
        0.08,
        item.fusion_score * 2.0,
    )

    score -= item.special_penalty

    if item.document.metadata.get(
        "page_quality_suspicious",
        False,
    ):
        score -= 0.05

    return max(
        0.0,
        min(1.0, score),
    )


def text_similarity(
    first: str,
    second: str,
) -> float:
    """Compute lightweight token similarity for semantic deduplication."""

    first_tokens = tokenize(first)
    second_tokens = tokenize(second)

    if not first_tokens or not second_tokens:
        return 0.0

    intersection = len(
        first_tokens & second_tokens
    )
    union = len(
        first_tokens | second_tokens
    )

    return intersection / max(1, union)


def deduplicate_ranked_documents(
    items: list[RankedDocument],
) -> list[RankedDocument]:
    """Remove near-duplicate chunks while retaining split section parts."""

    selected: list[RankedDocument] = []

    for item in items:
        metadata = item.document.metadata
        section_number = metadata.get(
            "section_number"
        )
        part_number = metadata.get(
            "section_part_number",
            1,
        )

        duplicate = False

        for existing in selected:
            existing_metadata = (
                existing.document.metadata
            )

            # Different parts of the same section within the same
            # document must be preserved. Identical section numbers in
            # different laws are separate legal provisions.
            same_section_identity = (
                metadata.get(
                    "section_identity"
                )
                == existing_metadata.get(
                    "section_identity"
                )
            )

            if (
                same_section_identity
                and part_number
                != existing_metadata.get(
                    "section_part_number",
                    1,
                )
            ):
                continue

            if (
                text_similarity(
                    item.document.page_content,
                    existing.document.page_content,
                )
                >= SEMANTIC_DEDUP_THRESHOLD
            ):
                duplicate = True
                break

        if not duplicate:
            selected.append(item)

    return selected


# -------------------------------------------------------------------
# Neighbor retrieval and section merging
# -------------------------------------------------------------------

def parse_numeric_section(
    section_number: str,
) -> int | None:
    """Return the numeric prefix for simple neighboring-section lookup."""

    match = re.fullmatch(
        r"(\d+)",
        section_number.strip(),
    )

    return (
        int(match.group(1))
        if match
        else None
    )


def build_neighbor_section_numbers(
    section_number: str,
) -> list[str]:
    """Build a bounded list of neighboring simple numeric sections."""

    numeric = parse_numeric_section(
        section_number
    )

    if numeric is None:
        return [section_number]

    start = max(
        1,
        numeric - NEIGHBOR_SECTION_RADIUS,
    )
    end = numeric + NEIGHBOR_SECTION_RADIUS

    return [
        str(value)
        for value in range(start, end + 1)
    ]


def retrieve_neighbor_documents(
    retriever: AdaptiveRetriever,
    section_number: str,
    question: str,
    document_types: tuple[str, ...] | list[str] | None = None,
) -> list[tuple[Document, float]]:
    """Retrieve neighboring sections when explicitly enabled."""

    if not ENABLE_NEIGHBOR_RETRIEVAL:
        return []

    neighboring_sections = (
        build_neighbor_section_numbers(
            section_number
        )
    )

    if len(neighboring_sections) <= 1:
        return []

    try:
        return retriever.search_with_scores(
            query=question,
            k=max(
                len(neighboring_sections) * 2,
                TOP_K,
            ),
            metadata_filter=build_sections_filter(
                neighboring_sections,
                document_types=document_types,
            ),
        )

    except Exception:
        # Neighbor retrieval is an enhancement, not a fatal dependency.
        return []


def merge_section_parts(
    documents: list[Document],
) -> list[Document]:
    """
    Merge all selected parts of a section into one context document.

    Metadata records the contributing chunks and page range.
    """

    grouped: dict[
        tuple[Any, ...],
        list[Document],
    ] = {}

    for document in documents:
        grouped.setdefault(
            _section_key(document),
            [],
        ).append(document)

    merged_documents: list[Document] = []

    for group_documents in grouped.values():
        ordered = sorted(
            group_documents,
            key=lambda document: (
                int(
                    document.metadata.get(
                        "section_part_number",
                        1,
                    )
                    or 1
                ),
                int(
                    document.metadata.get(
                        "chunk_number",
                        0,
                    )
                    or 0
                ),
            ),
        )

        if len(ordered) == 1:
            merged_documents.append(
                ordered[0]
            )
            continue

        base_metadata = dict(
            ordered[0].metadata
        )

        combined_parts: list[str] = []
        contributing_chunks: list[Any] = []
        source_pages: set[int] = set()

        for document in ordered:
            metadata = document.metadata
            combined_parts.append(
                document.page_content.strip()
            )
            contributing_chunks.append(
                metadata.get(
                    "chunk_number"
                )
            )

            for page in metadata.get(
                "source_pages",
                [],
            ):
                if isinstance(page, int):
                    source_pages.add(page)

        base_metadata.update(
            {
                "section_part_number": 1,
                "section_part_count": 1,
                "section_was_merged": True,
                "merged_chunk_numbers": (
                    contributing_chunks
                ),
                "source_pages": sorted(
                    source_pages
                ),
                "page_start": (
                    min(source_pages)
                    if source_pages
                    else base_metadata.get(
                        "page_start"
                    )
                ),
                "page_end": (
                    max(source_pages)
                    if source_pages
                    else base_metadata.get(
                        "page_end"
                    )
                ),
            }
        )

        merged_documents.append(
            Document(
                page_content=(
                    "\n\n".join(
                        combined_parts
                    )
                ),
                metadata=base_metadata,
            )
        )

    return merged_documents


# -------------------------------------------------------------------
# Section collision detection
# -------------------------------------------------------------------

def discover_section_documents(
    retriever: AdaptiveRetriever,
    section_number: str,
) -> list[dict[str, str]]:
    """
    Find indexed documents that contain an exact section/article number.

    A generous candidate count is used because one section may be split
    into several chunks in each document.
    """

    results = retriever.search_with_scores(
        query=(
            f"Section {section_number} exact legal text"
        ),
        k=max(
            50,
            TOP_K * 8,
        ),
        metadata_filter=build_section_filter(
            section_number
        ),
    )

    discovered: dict[
        str,
        dict[str, str],
    ] = {}

    for document, _score in results:
        if not is_usable_document(document):
            continue

        metadata = document.metadata
        document_id = str(
            metadata.get(
                "document_id",
                "",
            )
        )

        if not document_id:
            continue

        discovered.setdefault(
            document_id,
            {
                "document_id": document_id,
                "document_name": str(
                    metadata.get(
                        "document_name",
                        "Unknown document",
                    )
                ),
                "document_title": str(
                    metadata.get(
                        "document_title",
                        metadata.get(
                            "document_name",
                            "Unknown document",
                        ),
                    )
                ),
                "document_type": str(
                    metadata.get(
                        "document_type",
                        "legal_document",
                    )
                ),
            },
        )

    return sorted(
        discovered.values(),
        key=lambda item: item[
            "document_title"
        ].lower(),
    )


def build_section_clarification(
    section_number: str,
    matching_documents: list[dict[str, str]],
) -> str:
    """Create a safe clarification message for a section collision."""

    choices = "\n".join(
        f"- {document['document_title']}"
        for document in matching_documents
    )

    return (
        f"Section {section_number} exists in multiple indexed laws. "
        "Please specify the document you mean:\n"
        f"{choices}"
    )


# -------------------------------------------------------------------
# Retrieval pipeline
# -------------------------------------------------------------------

def retrieve_documents(
    retriever: AdaptiveRetriever,
    queries: list[str],
    question_type: str,
    original_question: str,
    detected_concepts: list[str],
    section_number: str | None = None,
    requested_document_types: tuple[str, ...] = (),
    maximum_documents: int = MAX_CONTEXT_DOCUMENTS,
) -> tuple[list[Document], list[RankedDocument]]:
    """
    Retrieve, rerank, deduplicate, group, and merge section context.
    """

    dynamic_k = get_retrieval_k(
        question_type
    )

    # Exact section lookup remains authoritative.
    if section_number:
        exact_results = retriever.search_with_scores(
            query=queries[0],
            k=dynamic_k,
            metadata_filter=build_section_filter(
                section_number,
                document_types=requested_document_types,
            ),
        )

        exact_documents = [
            document
            for document, score in exact_results
            if is_usable_document(document)
            and str(
                document.metadata.get(
                    "section_number",
                    "",
                )
            ).upper()
            == section_number.upper()
            and (
                MIN_RELEVANCE_SCORE <= 0
                or score >= MIN_RELEVANCE_SCORE
            )
        ]

        if exact_documents:
            merged = merge_section_parts(
                exact_documents
            )

            return (
                merged[:maximum_documents],
                [
                    RankedDocument(
                        document=document,
                        fusion_score=1.0,
                        relevance_score=1.0,
                        matched_queries=1,
                        final_score=1.0,
                    )
                    for document in merged[
                        :maximum_documents
                    ]
                ],
            )

    fused_results: dict[
        tuple[Any, ...],
        RankedDocument,
    ] = {}

    for query_index, query in enumerate(
        queries
    ):
        metadata_filter = (
            build_document_filter(
                requested_document_types
            )
            if requested_document_types
            else None
        )

        scored_documents = (
            retriever.search_with_scores(
                query=query,
                k=dynamic_k,
                metadata_filter=metadata_filter,
            )
        )

        query_weight = (
            1.35
            if query_index == 0
            else 1.0
        )

        for rank, (
            document,
            relevance_score,
        ) in enumerate(
            scored_documents,
            start=1,
        ):
            if not is_usable_document(document):
                continue

            normalized_relevance = (
                normalize_similarity_score(
                    relevance_score
                )
            )

            if (
                MIN_RELEVANCE_SCORE > 0
                and normalized_relevance
                < MIN_RELEVANCE_SCORE
            ):
                continue

            key = _document_key(
                document
            )

            if key not in fused_results:
                fused_results[key] = (
                    RankedDocument(
                        document=document,
                        fusion_score=0.0,
                        relevance_score=(
                            normalized_relevance
                        ),
                        matched_queries=0,
                    )
                )

            item = fused_results[key]
            item.fusion_score += (
                query_weight / (60 + rank)
            )
            item.matched_queries += 1
            item.relevance_score = max(
                item.relevance_score or 0.0,
                normalized_relevance,
            )

    # Retrieve useful neighboring provisions around the strongest
    # preferred sections discovered from legal concepts.
    preferred_sections: set[str] = set()

    for concept_name in detected_concepts:
        preferred_sections.update(
            LEGAL_CONCEPTS[
                concept_name
            ]["preferred_sections"]
        )

    for preferred_section in sorted(
        preferred_sections
    ):
        for document, relevance_score in (
            retrieve_neighbor_documents(
                retriever=retriever,
                section_number=preferred_section,
                question=original_question,
                document_types=requested_document_types,
            )
        ):
            if not is_usable_document(document):
                continue

            key = _document_key(document)

            if key not in fused_results:
                fused_results[key] = (
                    RankedDocument(
                        document=document,
                        fusion_score=0.005,
                        relevance_score=(
                            normalize_similarity_score(
                                relevance_score
                            )
                        ),
                        matched_queries=1,
                    )
                )

    ranked_items = list(
        fused_results.values()
    )

    for item in ranked_items:
        item.keyword_overlap = (
            keyword_overlap_score(
                original_question,
                item.document,
            )
        )

        item.concept_overlap = (
            concept_overlap_score(
                item.document,
                detected_concepts,
            )
        )

        item.section_boost = (
            section_match_boost(
                item.document,
                original_question,
                detected_concepts,
            )
        )

        item.intent_score = (
            intent_alignment_score(
                document=item.document,
                question_type=question_type,
                question=original_question,
                detected_concepts=detected_concepts,
            )
        )

        item.special_penalty = (
            special_section_penalty(
                item.document,
                original_question,
            )
        )

        item.final_score = (
            calculate_final_score(
                item
            )
        )

    ranked_items.sort(
        key=lambda item: (
            item.final_score,
            item.matched_queries,
            item.relevance_score or 0.0,
        ),
        reverse=True,
    )

    ranked_items = (
        deduplicate_ranked_documents(
            ranked_items
        )
    )

    top_ranked_score = (
        ranked_items[0].final_score
        if ranked_items
        else 0.0
    )

    for item in ranked_items:
        item.context_role = classify_context_role(
            item=item,
            question_type=question_type,
            question=original_question,
            detected_concepts=detected_concepts,
            top_score=top_ranked_score,
        )

    grouped: dict[
        tuple[Any, ...],
        list[RankedDocument],
    ] = {}

    for item in ranked_items:
        grouped.setdefault(
            _section_key(
                item.document
            ),
            [],
        ).append(item)

    ranked_groups = sorted(
        grouped.values(),
        key=lambda group: max(
            item.final_score
            for item in group
        ),
        reverse=True,
    )

    # Remove weak sections when their score falls substantially below
    # the strongest section. This prevents low-value context such as a
    # general definition or downstream consequence from being included
    # merely because space remains in the context window.
    if ranked_groups:
        top_group_score = max(
            item.final_score
            for item in ranked_groups[0]
        )

        minimum_group_score = max(
            0.0,
            top_group_score - ADAPTIVE_SECTION_SCORE_GAP,
        )

        filtered_ranked_groups = [
            group
            for group in ranked_groups
            if max(
                item.final_score
                for item in group
            )
            >= minimum_group_score
        ]

        # Always keep at least the strongest group.
        ranked_groups = (
            filtered_ranked_groups
            or ranked_groups[:1]
        )

    selected_items: list[
        RankedDocument
    ] = []

    section_limit = (
        3
        if question_type == "fact_scenario"
        else MAX_CONTEXT_SECTIONS
    )

    ranked_groups = mmr_select_groups(
        ranked_groups=ranked_groups,
        limit=section_limit,
    )

    ranked_groups = role_based_group_selection(
        ranked_groups=ranked_groups,
        question_type=question_type,
        limit=section_limit,
    )

    for group in ranked_groups:
        selected_items.extend(
            sorted(
                group,
                key=lambda item: int(
                    item.document.metadata.get(
                        "section_part_number",
                        1,
                    )
                    or 1
                ),
            )
        )

        if len(selected_items) >= (
            maximum_documents
        ):
            break

    selected_items = selected_items[
        :maximum_documents
    ]

    selected_documents: list[Document] = []

    for item in selected_items:
        item.document.metadata[
            "context_role"
        ] = item.context_role
        selected_documents.append(
            item.document
        )

    merged_documents = merge_section_parts(
        selected_documents
    )

    return (
        merged_documents[
            :section_limit
        ],
        selected_items,
    )


# -------------------------------------------------------------------
# Confidence scoring
# -------------------------------------------------------------------

def calculate_retrieval_confidence(
    ranked_items: list[RankedDocument],
    selected_documents: list[Document],
    detected_concepts: list[str],
) -> RetrievalConfidence:
    """Calculate a transparent retrieval-confidence estimate."""

    if not ranked_items:
        return RetrievalConfidence(
            label="Low",
            score=0.0,
            top_similarity=0.0,
            average_similarity=0.0,
            section_count=0,
            concept_coverage=0.0,
        )

    similarities = [
        item.relevance_score or 0.0
        for item in ranked_items[:10]
    ]

    top_similarity = max(
        similarities,
        default=0.0,
    )

    average_similarity = (
        sum(similarities)
        / max(1, len(similarities))
    )

    selected_sections = {
        str(
            document.metadata.get(
                "section_number",
                "",
            )
        )
        for document in selected_documents
        if document.metadata.get(
            "section_number"
        )
    }

    covered_concepts = 0

    for concept_name in detected_concepts:
        preferred = LEGAL_CONCEPTS[
            concept_name
        ]["preferred_sections"]

        if (
            not preferred
            or selected_sections & preferred
        ):
            covered_concepts += 1

    concept_coverage = (
        covered_concepts
        / max(1, len(detected_concepts))
        if detected_concepts
        else 1.0
    )

    score = (
        0.45 * top_similarity
        + 0.30 * average_similarity
        + 0.15 * min(
            1.0,
            len(selected_sections) / 3,
        )
        + 0.10 * concept_coverage
    )

    if score >= 0.72:
        label = "High"
    elif score >= 0.48:
        label = "Medium"
    else:
        label = "Low"

    return RetrievalConfidence(
        label=label,
        score=round(score, 3),
        top_similarity=round(
            top_similarity,
            3,
        ),
        average_similarity=round(
            average_similarity,
            3,
        ),
        section_count=len(
            selected_sections
        ),
        concept_coverage=round(
            concept_coverage,
            3,
        ),
    )


# -------------------------------------------------------------------
# Context formatting and debugging
# -------------------------------------------------------------------

def format_page_range(
    metadata: dict[str, Any],
) -> str:
    """Format a document's page range."""

    page_start = metadata.get(
        "page_start"
    )
    page_end = metadata.get(
        "page_end"
    )

    if page_start is None:
        return str(
            metadata.get(
                "page_number",
                "Unknown",
            )
        )

    if page_end in {
        None,
        page_start,
    }:
        return str(page_start)

    return (
        f"{page_start}-{page_end}"
    )


def format_context(
    documents: list[Document],
) -> str:
    """Format merged legal sections as labeled prompt context."""

    context_parts: list[str] = []

    for index, document in enumerate(
        documents,
        start=1,
    ):
        metadata = document.metadata

        context_parts.append(
            f"""
[Source {index}]
Document: {metadata.get("document_title", metadata.get("document_name", "Unknown document"))}
Document type: {metadata.get("document_type", "legal_document")}
Document ID: {metadata.get("document_id", "unknown")}
Section/Article: {metadata.get("section_number", "Unsectioned")} — {metadata.get("section_title", "")}
Context role: {metadata.get("context_role", "supporting")}
Pages: {format_page_range(metadata)}
Merged section parts: {metadata.get("section_was_merged", False)}
Source chunks: {metadata.get("merged_chunk_numbers", [metadata.get("chunk_number")])}
Quality: {metadata.get("page_quality_status", "unknown")}

{document.page_content}
""".strip()
        )

    return "\n\n---\n\n".join(
        context_parts
    )


def display_retrieved_documents(
    documents: list[Document],
    ranked_items: list[RankedDocument],
) -> None:
    """Display final context and ranking diagnostics."""

    score_by_section: dict[
        tuple[str, str],
        float,
    ] = {}

    for item in ranked_items:
        metadata = item.document.metadata
        document_id = str(
            metadata.get(
                "document_id",
                "unknown",
            )
        )
        section_number = str(
            metadata.get(
                "section_number",
                "Unsectioned",
            )
        )
        score_key = (
            document_id,
            section_number,
        )

        score_by_section[score_key] = max(
            score_by_section.get(
                score_key,
                0.0,
            ),
            item.final_score,
        )

    print("\n" + "=" * 70)
    print("SELECTED RETRIEVAL CONTEXT")
    print("=" * 70)

    for index, document in enumerate(
        documents,
        start=1,
    ):
        metadata = document.metadata
        section_number = str(
            metadata.get(
                "section_number",
                "Unsectioned",
            )
        )

        document_id = str(
            metadata.get(
                "document_id",
                "unknown",
            )
        )
        document_title = str(
            metadata.get(
                "document_title",
                metadata.get(
                    "document_name",
                    "Unknown document",
                ),
            )
        )
        score_key = (
            document_id,
            section_number,
        )

        content = document.page_content.strip()
        preview = content[
            :DEBUG_PREVIEW_CHARS
        ]

        if len(content) > DEBUG_PREVIEW_CHARS:
            preview += (
                "\n...[preview shortened]"
            )

        matching_items = [
            item
            for item in ranked_items
            if (
                str(
                    item.document.metadata.get(
                        "document_id",
                        "unknown",
                    )
                )
                == document_id
                and str(
                    item.document.metadata.get(
                        "section_number",
                        "Unsectioned",
                    )
                )
                == section_number
            )
        ]

        best_item = (
            max(
                matching_items,
                key=lambda item: item.final_score,
            )
            if matching_items
            else None
        )

        print(
            f"\n{index}. Document: {document_title} | "
            f"Section/Article: {section_number} | "
            f"Pages: {format_page_range(metadata)} | "
            f"Role: "
            f"{best_item.context_role if best_item else 'unknown'} | "
            f"Intent score: "
            f"{best_item.intent_score if best_item else 0.0:.3f} | "
            f"Final score: "
            f"{score_by_section.get(score_key, 0.0):.3f}"
        )
        print("-" * 70)
        print(preview)


# -------------------------------------------------------------------
# Gemini response handling
# -------------------------------------------------------------------

def extract_response_text(
    response: Any,
) -> str:
    """Extract plain text from a Gemini response."""

    response_text = getattr(
        response,
        "text",
        None,
    )

    if (
        isinstance(response_text, str)
        and response_text.strip()
    ):
        return response_text.strip()

    content = getattr(
        response,
        "content",
        "",
    )

    if isinstance(content, str):
        return content.strip()

    if isinstance(content, list):
        text_parts: list[str] = []

        for item in content:
            item_text = (
                item.get("text")
                if isinstance(item, dict)
                else getattr(
                    item,
                    "text",
                    None,
                )
            )

            if (
                isinstance(item_text, str)
                and item_text.strip()
            ):
                text_parts.append(
                    item_text.strip()
                )

        return "\n".join(
            text_parts
        ).strip()

    return str(content).strip()


def invoke_chat_model_with_retry(
    chat_model: ChatGoogleGenerativeAI,
    prompt: str,
):
    """Invoke Gemini with bounded exponential retries."""

    retry_markers = {
        "429",
        "resource_exhausted",
        "quota",
        "rate limit",
        "temporarily unavailable",
        "503",
        "502",
        "504",
        "timeout",
    }

    for attempt in range(
        1,
        GOOGLE_API_MAX_RETRIES + 1,
    ):
        try:
            return chat_model.invoke(
                prompt
            )

        except Exception as error:
            error_text = str(
                error
            ).lower()

            is_retryable = any(
                marker in error_text
                for marker in retry_markers
            )

            if (
                not is_retryable
                or attempt
                == GOOGLE_API_MAX_RETRIES
            ):
                raise

            delay = (
                GOOGLE_API_RETRY_DELAY
                * (2 ** (attempt - 1))
            )

            print(
                "Gemini request temporarily failed. "
                f"Retrying in {delay:.0f} seconds "
                f"({attempt}/{GOOGLE_API_MAX_RETRIES})..."
            )

            time.sleep(delay)

    raise RuntimeError(
        "Gemini request failed after all retries."
    )


# -------------------------------------------------------------------
# Sources
# -------------------------------------------------------------------

def create_sources(
    documents: list[Document],
) -> list[dict[str, Any]]:
    """Create source records from final merged context sections."""

    sources: list[
        dict[str, Any]
    ] = []

    for index, document in enumerate(
        documents,
        start=1,
    ):
        metadata = document.metadata

        sources.append(
            {
                "label": f"Source {index}",
                "document_name": metadata.get(
                    "document_name",
                    "Unknown document",
                ),
                "document_title": metadata.get(
                    "document_title",
                    metadata.get(
                        "document_name",
                        "Unknown document",
                    ),
                ),
                "document_id": metadata.get(
                    "document_id",
                    "unknown",
                ),
                "document_type": metadata.get(
                    "document_type",
                    "legal_document",
                ),
                "section_number": metadata.get(
                    "section_number",
                ),
                "section_title": metadata.get(
                    "section_title",
                ),
                "context_role": metadata.get(
                    "context_role",
                    "supporting",
                ),
                "page_start": metadata.get(
                    "page_start",
                ),
                "page_end": metadata.get(
                    "page_end",
                ),
                "page_number": metadata.get(
                    "page_number",
                    "Unknown",
                ),
                # Keep both keys for backward compatibility.
                # query_cli.py expects the singular `chunk_number`,
                # while merged sections may contain several chunks.
                "chunk_number": metadata.get(
                    "chunk_number",
                    (
                        metadata.get(
                            "merged_chunk_numbers",
                            ["Unknown"],
                        )[0]
                        if metadata.get(
                            "merged_chunk_numbers"
                        )
                        else "Unknown"
                    ),
                ),
                "chunk_numbers": metadata.get(
                    "merged_chunk_numbers",
                    [
                        metadata.get(
                            "chunk_number",
                            "Unknown",
                        )
                    ],
                ),
            }
        )

    return sources


def filter_sources_used_in_answer(
    answer: str,
    sources: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Return only source labels explicitly cited in the answer."""

    used_labels = {
        int(number)
        for number in re.findall(
            r"\[Source\s+(\d+)\]",
            answer,
            re.IGNORECASE,
        )
    }

    if not used_labels:
        return sources

    return [
        source
        for index, source in enumerate(
            sources,
            start=1,
        )
        if index in used_labels
    ]


# -------------------------------------------------------------------
# Prompt construction
# -------------------------------------------------------------------

def build_grounded_prompt(
    question: str,
    question_type: str,
    context: str,
    confidence: RetrievalConfidence,
    document_selection: DocumentSelection,
) -> str:
    """Build a structured, confidence-aware legal prompt."""

    task_instructions = ""

    if question_type == "fact_scenario":
        task_instructions = """
Give a concise factual-scenario answer using only the sections needed.

Use this structure when supported:
1. Primary provision — state why it fits.
2. Close alternative — include only when needed to distinguish the
   primary provision or when one material fact is missing.
3. Missing facts — list only facts that could change the classification.

Do not force all headings when the answer needs only one provision and a
short explanation. Normally discuss no more than two provisions.

When the facts expressly mention entrustment, prioritize criminal breach
of trust over ordinary dishonest misappropriation. Mention ordinary
misappropriation only if entrustment is uncertain.

Do not include falsification-of-accounts provisions unless the question
states or strongly indicates false entries, altered records, omissions,
destruction of records, or another accounting concealment act.

Do not discuss a retrieved section merely to reject it unless it is a
close legal alternative required to explain the distinction.

For found-property scenarios, prioritize Section 403. Use Section 378
only to explain why property found outside another person's possession
may not amount to theft. Do not include Section 410 unless the question
asks about stolen-property classification, possession, transfer, or
receiving stolen property.
""".strip()

    elif question_type == "punishment":
        task_instructions = """
Separate:
- Offence or triggering conduct
- Punishment provision
- Maximum imprisonment
- Fine
- Required legal status, capacity or condition
Do not merge a definition section and punishment section without clearly
distinguishing their separate functions.
""".strip()

    elif question_type == "section_lookup":
        task_instructions = """
State the requested section heading and summarize only its operative
text. Include explanations, illustrations or punishment only when they
appear in that section.
""".strip()

    else:
        task_instructions = """
Identify the primary offence or legal issue, explain why the strongest
retrieved section applies, identify weaker alternatives, and state the
facts that remain necessary.
""".strip()

    selected_documents_text = (
        format_document_selection(
            document_selection
        )
    )

    document_scope_instruction = (
        "The user explicitly selected these documents: "
        f"{selected_documents_text}. Do not use excerpts from other "
        "documents."
        if document_selection.is_document_specific
        else (
            "The user did not name a document. Use the strongest "
            "relevant excerpts across the indexed legal documents and "
            "identify the source law clearly."
        )
    )

    return f"""
You are a retrieval-grounded assistant for a multi-document Pakistan
legal knowledge base.

Use only the retrieved excerpts.

Document scope: {selected_documents_text}
{document_scope_instruction}

Retrieval confidence: {confidence.label}
Confidence score: {confidence.score}
Top similarity: {confidence.top_similarity}
Average similarity: {confidence.average_similarity}
Selected section count: {confidence.section_count}
Concept coverage: {confidence.concept_coverage}

Rules:
1. Do not use outside legal knowledge.
2. Do not invent sections, elements, exceptions, punishments or facts.
3. Cite every legal claim using [Source N].
4. Mention a section only when it appears in the excerpts.
5. Rank the strongest directly supported provision first.
6. Do not list every retrieved provision.
7. A special provision applies only when the question states its special
   circumstance. For example, do not include dacoity-with-murder without
   dacoity facts, or succession consequences without heir facts.
8. Treat factual application as conditional and do not determine guilt.
9. Identify the exact missing facts needed for alternative provisions.
10. When retrieval confidence is Low, explicitly state that the retrieved
    material may be insufficient for a reliable classification.
11. Say "The answer was not found in the indexed document" only when none
    of the excerpts is relevant.
12. Do not present the response as personalized legal advice.
13. Prefer concise answers. Do not repeat the same legal requirement in
    multiple sections of the response.
14. Use context roles:
    - primary: directly answers the question;
    - supporting: explains a required element;
    - punishment: provides the penalty;
    - definition: defines a legal term;
    - alternative: include only if materially necessary.
15. Do not mention an alternative section merely because it was retrieved.
    Mention it only when it changes or clarifies the legal classification.
16. Always identify the source law when two or more documents are used.
17. Never treat the same section number in different documents as the
    same legal provision.
18. For cross-document comparisons, discuss each document separately
    before stating the comparison.

Question type: {question_type}

{task_instructions}

Retrieved excerpts:

{context}

User question:

{question}

Grounded answer:
""".strip()


# -------------------------------------------------------------------
# Main RAG function
# -------------------------------------------------------------------

def answer_question(
    question: str,
    retriever: AdaptiveRetriever,
    chat_model: ChatGoogleGenerativeAI,
) -> dict[str, Any]:
    """Retrieve, rerank, merge, and answer a legal question."""

    if (
        not question
        or len(question.strip())
        < QUESTION_MIN_LENGTH
    ):
        raise ValueError(
            "The question must contain at least "
            f"{QUESTION_MIN_LENGTH} characters."
        )

    original_question = question.strip()

    if (
        len(original_question)
        > QUESTION_MAX_LENGTH
    ):
        raise ValueError(
            "The question must not exceed "
            f"{QUESTION_MAX_LENGTH} characters."
        )

    question_type = classify_question(
        original_question
    )

    document_selection = (
        detect_requested_documents(
            original_question
        )
    )

    detected_concepts = (
        detect_legal_concepts(
            original_question
        )
    )

    section_number = extract_section_number(
        original_question
    )

    # An unqualified exact section lookup is ambiguous when the same
    # number exists in several indexed laws. Cross-document comparison
    # questions are intentionally exempt from clarification.
    if (
        section_number
        and not document_selection.is_document_specific
        and question_type != "comparison"
    ):
        matching_documents = (
            discover_section_documents(
                retriever=retriever,
                section_number=section_number,
            )
        )

        if len(matching_documents) > 1:
            clarification = (
                build_section_clarification(
                    section_number=section_number,
                    matching_documents=matching_documents,
                )
            )

            return {
                "answer": clarification,
                "sources": [],
                "question_type": question_type,
                "detected_concepts": detected_concepts,
                "retrieved_document_count": 0,
                "requires_clarification": True,
                "matching_documents": (
                    matching_documents
                ),
                "confidence": {
                    "label": "Ambiguous",
                    "score": 0.0,
                },
            }

        if len(matching_documents) == 1:
            only_document_type = str(
                matching_documents[0][
                    "document_type"
                ]
            )
            only_document_title = str(
                matching_documents[0][
                    "document_title"
                ]
            )

            document_selection = (
                DocumentSelection(
                    document_types=(
                        only_document_type,
                    ),
                    display_names=(
                        only_document_title,
                    ),
                    aliases_found=(),
                )
            )

    retrieval_queries = build_retrieval_queries(
        question=original_question,
        question_type=question_type,
        detected_concepts=detected_concepts,
        document_selection=document_selection,
    )

    (
        retrieved_documents,
        ranked_items,
    ) = retrieve_documents(
        retriever=retriever,
        queries=retrieval_queries,
        question_type=question_type,
        original_question=original_question,
        detected_concepts=detected_concepts,
        section_number=section_number,
        requested_document_types=(
            document_selection.document_types
        ),
        maximum_documents=MAX_CONTEXT_DOCUMENTS,
    )

    if not retrieved_documents:
        return {
            "answer": (
                "The answer was not found in the selected "
                "indexed legal document(s)."
            ),
            "sources": [],
            "question_type": question_type,
            "confidence": {
                "label": "Low",
                "score": 0.0,
            },
        }

    confidence = calculate_retrieval_confidence(
        ranked_items=ranked_items,
        selected_documents=(
            retrieved_documents
        ),
        detected_concepts=detected_concepts,
    )

    if DEBUG_RETRIEVAL:
        print("\nOriginal question:")
        print(original_question)
        print(
            f"\nQuestion type: {question_type}"
        )
        print(
            "Scenario score: "
            f"{calculate_scenario_score(original_question)}"
        )
        print(
            "Detected legal concepts: "
            f"{detected_concepts or ['None']}"
        )
        print(
            "Explicit section/article: "
            f"{section_number or 'None'}"
        )
        print(
            "Requested documents: "
            f"{format_document_selection(document_selection)}"
        )
        print(
            "Requested document types: "
            f"{list(document_selection.document_types) or ['All']}"
        )
        print(
            "Dynamic retrieval k: "
            f"{get_retrieval_k(question_type)}"
        )
        print(
            "Adaptive section score gap: "
            f"{ADAPTIVE_SECTION_SCORE_GAP}"
        )
        print(
            "Intent score weight: "
            f"{INTENT_SCORE_WEIGHT}"
        )
        print(
            "MMR lambda: "
            f"{MMR_LAMBDA}"
        )

        print("\nRetrieval queries:")

        for index, retrieval_query in enumerate(
            retrieval_queries,
            start=1,
        ):
            print(
                f"{index}. {retrieval_query}"
            )

        display_retrieved_documents(
            retrieved_documents,
            ranked_items,
        )

        print("\nRetrieval confidence:")
        print(confidence)

    context = format_context(
        retrieved_documents
    )

    prompt = build_grounded_prompt(
        question=original_question,
        question_type=question_type,
        context=context,
        confidence=confidence,
        document_selection=document_selection,
    )

    response = invoke_chat_model_with_retry(
        chat_model=chat_model,
        prompt=prompt,
    )

    answer = extract_response_text(
        response
    )

    all_sources = create_sources(
        retrieved_documents
    )

    used_sources = filter_sources_used_in_answer(
        answer,
        all_sources,
    )

    return {
        "answer": answer,
        "sources": used_sources,
        "question_type": question_type,
        "detected_concepts": (
            detected_concepts
        ),
        "requested_documents": list(
            document_selection.display_names
        ),
        "requested_document_types": list(
            document_selection.document_types
        ),
        "requires_clarification": False,
        "retrieved_document_count": len(
            retrieved_documents
        ),
        "confidence": {
            "label": confidence.label,
            "score": confidence.score,
            "top_similarity": (
                confidence.top_similarity
            ),
            "average_similarity": (
                confidence.average_similarity
            ),
            "section_count": (
                confidence.section_count
            ),
            "concept_coverage": (
                confidence.concept_coverage
            ),
        },
    }