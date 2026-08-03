from __future__ import annotations

import re
from dataclasses import dataclass

from rag.concept_registry import (
    LEGAL_CONCEPTS,
    SCENARIO_MARKERS,
)
from rag.schemas import QueryPlan


# -------------------------------------------------------------------
# Document routing configuration
# -------------------------------------------------------------------

@dataclass(frozen=True)
class DocumentRoute:
    document_id: str
    short_name: str
    full_name: str
    provision_type: str
    aliases: tuple[str, ...]


DOCUMENT_ROUTES: tuple[DocumentRoute, ...] = (
    DocumentRoute(
        document_id="ppc_1860",
        short_name="PPC",
        full_name="Pakistan Penal Code, 1860",
        provision_type="section",
        aliases=(
            "ppc",
            "pakistan penal code",
            "penal code",
        ),
    ),
    DocumentRoute(
        document_id="constitution_1973",
        short_name="Constitution",
        full_name=(
            "Constitution of the Islamic Republic "
            "of Pakistan, 1973"
        ),
        provision_type="article",
        aliases=(
            "constitution",
            "constitution of pakistan",
            "pakistan constitution",
            "constitutional",
        ),
    ),
    DocumentRoute(
        document_id="ata_1997",
        short_name="ATA",
        full_name="Anti-Terrorism Act, 1997",
        provision_type="section",
        aliases=(
            "ata",
            "anti terrorism act",
            "anti-terrorism act",
            "terrorism act",
        ),
    ),
    DocumentRoute(
        document_id="amla_2010",
        short_name="AMLA",
        full_name="Anti-Money Laundering Act, 2010",
        provision_type="section",
        aliases=(
            "amla",
            "anti money laundering act",
            "anti-money laundering act",
            "money laundering act",
        ),
    ),
)


DOCUMENT_ROUTE_BY_ID = {
    route.document_id: route
    for route in DOCUMENT_ROUTES
}


# -------------------------------------------------------------------
# Text normalization
# -------------------------------------------------------------------

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


def normalize_text(text: str) -> str:
    """Normalize text for rule-based intent detection."""

    normalized = text.lower()
    normalized = normalized.replace("–", "-")
    normalized = normalized.replace("—", "-")
    normalized = re.sub(
        r"[^a-z0-9\-\s]",
        " ",
        normalized,
    )
    normalized = re.sub(
        r"\s+",
        " ",
        normalized,
    )

    return normalized.strip()


def tokenize(text: str) -> set[str]:
    """Return meaningful lowercase tokens from a question."""

    tokens = re.findall(
        r"[a-z0-9]+(?:-[a-z0-9]+)?",
        normalize_text(text),
    )

    return {
        token
        for token in tokens
        if (
            len(token) > 1
            and token not in STOP_WORDS
        )
    }


def deduplicate_strings(
    values: list[str],
) -> list[str]:
    """Remove empty and duplicate strings while preserving order."""

    seen: set[str] = set()
    unique_values: list[str] = []

    for value in values:
        cleaned_value = str(value).strip()

        if (
            cleaned_value
            and cleaned_value not in seen
        ):
            seen.add(cleaned_value)
            unique_values.append(
                cleaned_value
            )

    return unique_values


# -------------------------------------------------------------------
# Document and provision extraction
# -------------------------------------------------------------------

def detect_document_ids(
    question: str,
) -> list[str]:
    """Detect explicitly named legal documents."""

    normalized = normalize_text(
        question
    )
    detected_ids: list[str] = []

    for route in DOCUMENT_ROUTES:
        for alias in route.aliases:
            normalized_alias = normalize_text(
                alias
            )

            if re.search(
                rf"(?<![a-z0-9])"
                rf"{re.escape(normalized_alias)}"
                rf"(?![a-z0-9])",
                normalized,
            ):
                detected_ids.append(
                    route.document_id
                )
                break

    return deduplicate_strings(
        detected_ids
    )


def extract_section_numbers(
    question: str,
) -> list[str]:
    """Extract all explicit Section references."""

    matches = re.findall(
        r"\b(?:section|sec\.?|s\.)\s*"
        r"(\d+(?:-[A-Za-z]+)?[A-Za-z]?)\b",
        question,
        flags=re.IGNORECASE,
    )

    return deduplicate_strings(
        [
            match.upper()
            for match in matches
        ]
    )


def extract_article_numbers(
    question: str,
) -> list[str]:
    """Extract all explicit Article references."""

    matches = re.findall(
        r"\b(?:article|art\.?)\s*"
        r"(\d+(?:-[A-Za-z]+)?[A-Za-z]?)\b",
        question,
        flags=re.IGNORECASE,
    )

    return deduplicate_strings(
        [
            match.upper()
            for match in matches
        ]
    )


def extract_section_number(
    question: str,
) -> str | None:
    """
    Backward-compatible helper returning the first Section number.
    """

    section_numbers = extract_section_numbers(
        question
    )

    return (
        section_numbers[0]
        if section_numbers
        else None
    )


def extract_article_number(
    question: str,
) -> str | None:
    """Return the first explicit Article number."""

    article_numbers = extract_article_numbers(
        question
    )

    return (
        article_numbers[0]
        if article_numbers
        else None
    )


def infer_provision_type(
    document_ids: list[str],
    section_numbers: list[str],
    article_numbers: list[str],
) -> str | None:
    """Infer whether the query targets Sections or Articles."""

    if article_numbers and not section_numbers:
        return "article"

    if section_numbers and not article_numbers:
        return "section"

    if len(document_ids) == 1:
        route = DOCUMENT_ROUTE_BY_ID.get(
            document_ids[0]
        )

        if route is not None:
            return route.provision_type

    return None


# -------------------------------------------------------------------
# Legal concept detection
# -------------------------------------------------------------------

def marker_matches(
    normalized_question: str,
    marker: str,
) -> bool:
    """Match one concept marker without unsafe partial-word matches."""

    normalized_marker = normalize_text(
        marker
    )

    if not normalized_marker:
        return False

    return bool(
        re.search(
            rf"(?<![a-z0-9])"
            rf"{re.escape(normalized_marker)}"
            rf"(?![a-z0-9])",
            normalized_question,
        )
    )


def detect_legal_concepts(
    question: str,
) -> list[str]:
    """Detect legal concepts defined in concept_registry.py."""

    normalized = normalize_text(
        question
    )
    detected: list[str] = []

    for concept_name, concept_data in (
        LEGAL_CONCEPTS.items()
    ):
        markers = concept_data.get(
            "markers",
            [],
        )

        if any(
            marker_matches(
                normalized,
                str(marker),
            )
            for marker in markers
        ):
            detected.append(
                concept_name
            )

    return deduplicate_strings(
        detected
    )


def get_concept_document_hints(
    concepts: list[str],
) -> list[str]:
    """
    Read optional preferred document IDs from the concept registry.

    Existing registries without preferred_documents remain compatible.
    """

    document_ids: list[str] = []

    for concept_name in concepts:
        concept_data = LEGAL_CONCEPTS.get(
            concept_name,
            {},
        )

        preferred_documents = (
            concept_data.get(
                "preferred_documents",
                [],
            )
            or concept_data.get(
                "document_ids",
                [],
            )
        )

        document_ids.extend(
            str(document_id)
            for document_id
            in preferred_documents
            if str(document_id).strip()
        )

    return deduplicate_strings(
        document_ids
    )


def get_section_hints(
    concepts: list[str],
) -> list[str]:
    """Collect backward-compatible preferred Section hints."""

    section_hints: list[str] = []

    for concept_name in concepts:
        concept_data = LEGAL_CONCEPTS.get(
            concept_name,
            {},
        )

        preferred_sections = concept_data.get(
            "preferred_sections",
            [],
        )

        section_hints.extend(
            str(section).upper()
            for section in preferred_sections
            if str(section).strip()
        )

    return deduplicate_strings(
        section_hints
    )


# -------------------------------------------------------------------
# Question classification
# -------------------------------------------------------------------

def calculate_scenario_score(
    question: str,
) -> int:
    """Calculate the weighted fact-scenario score."""

    normalized = normalize_text(
        question
    )

    return sum(
        int(weight)
        for marker, weight
        in SCENARIO_MARKERS.items()
        if marker_matches(
            normalized,
            str(marker),
        )
    )


def classify_question(
    question: str,
) -> str:
    """Classify the user's legal question."""

    normalized = normalize_text(
        question
    )

    # Comparison remains highest priority.
    if (
        any(
            phrase in normalized
            for phrase in (
                "difference between",
                "compare",
                "comparison",
                "distinguish between",
            )
        )
        or re.search(
            r"\b(?:vs|versus)\b",
            normalized,
        )
    ):
        return "comparison"

    section_numbers = extract_section_numbers(
        question
    )
    article_numbers = extract_article_numbers(
        question
    )

    # A detailed factual narrative should be treated as a scenario even
    # when it also mentions a Section or Article.
    scenario_score = calculate_scenario_score(
        question
    )
    detected_concepts = detect_legal_concepts(
        question
    )

    is_scenario = (
        scenario_score >= 3
        or (
            len(normalized.split()) >= 18
            and bool(detected_concepts)
        )
    )

    if is_scenario:
        return "fact_scenario"

    if article_numbers:
        return "article_lookup"

    if section_numbers:
        return "section_lookup"

    if any(
        term in normalized
        for term in (
            "punishment",
            "penalty",
            "sentence",
            "imprisonment",
            "fine",
            "punishable",
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
            "what constitutes",
            "elements of",
        )
    ):
        return "definition"

    if any(
        phrase in normalized
        for phrase in (
            "right to",
            "fundamental right",
            "constitutional right",
        )
    ):
        return "constitutional_right"

    return "general"


def _is_theft_fact_scenario(
    question_type: str,
    concepts: list[str],
) -> bool:
    """Check whether a fact scenario should target PPC theft provisions."""

    if question_type != "fact_scenario":
        return False

    return "theft" in concepts


def get_retrieval_k(
    question_type: str,
    top_k: int,
) -> int:
    """Return adaptive retrieval depth by question type."""

    return {
        "section_lookup": max(
            6,
            top_k,
        ),
        "article_lookup": max(
            6,
            top_k,
        ),
        "definition": max(
            8,
            top_k,
        ),
        "punishment": max(
            10,
            top_k,
        ),
        "constitutional_right": max(
            10,
            top_k,
        ),
        "comparison": max(
            16,
            top_k,
        ),
        "fact_scenario": max(
            25,
            top_k,
        ),
        "general": max(
            12,
            top_k,
        ),
    }.get(
        question_type,
        top_k,
    )


# -------------------------------------------------------------------
# Retrieval-query construction
# -------------------------------------------------------------------

def get_document_label(
    document_id: str,
) -> str:
    """Return a readable law name for a known document ID."""

    route = DOCUMENT_ROUTE_BY_ID.get(
        document_id
    )

    if route is None:
        return document_id

    return route.full_name


def build_retrieval_queries(
    question: str,
    question_type: str,
    detected_concepts: list[str] | None = None,
    document_ids: list[str] | None = None,
    section_numbers: list[str] | None = None,
    article_numbers: list[str] | None = None,
) -> list[str]:
    """Create document-aware semantic retrieval queries."""

    original = question.strip()

    if not original:
        raise ValueError(
            "Question cannot be empty."
        )

    concepts = (
        detected_concepts
        if detected_concepts is not None
        else detect_legal_concepts(
            original
        )
    )

    documents = (
        document_ids
        if document_ids is not None
        else detect_document_ids(
            original
        )
    )

    sections = (
        section_numbers
        if section_numbers is not None
        else extract_section_numbers(
            original
        )
    )

    articles = (
        article_numbers
        if article_numbers is not None
        else extract_article_numbers(
            original
        )
    )

    queries: list[str] = [
        original
    ]

    document_context = " ".join(
        get_document_label(document_id)
        for document_id in documents
    ).strip()

    if (
        question_type == "section_lookup"
        and sections
    ):
        for section_number in sections:
            queries.append(
                (
                    f"{document_context} Section "
                    f"{section_number} exact statutory text "
                    "heading subsections explanation illustration "
                    "conditions exceptions and punishment"
                ).strip()
            )

    elif (
        question_type == "article_lookup"
        and articles
    ):
        for article_number in articles:
            queries.append(
                (
                    f"{document_context or 'Constitution of Pakistan'} "
                    f"Article {article_number} exact constitutional "
                    "text right scope limitation exception and remedy"
                ).strip()
            )

    elif question_type == "punishment":
        queries.append(
            (
                f"{document_context} {original} exact punishment "
                "imprisonment term fine conditions offence elements "
                "and applicable statutory provision"
            ).strip()
        )

    elif question_type == "definition":
        queries.append(
            (
                f"{document_context} {original} exact statutory "
                "definition legal elements conditions exceptions "
                "explanations and illustrations"
            ).strip()
        )

    elif question_type == "constitutional_right":
        queries.append(
            (
                f"{document_context or 'Constitution of Pakistan'} "
                f"{original} relevant constitutional article right "
                "scope safeguard limitation and enforcement"
            ).strip()
        )

    elif question_type == "comparison":
        queries.append(
            (
                f"{document_context} {original} compare statutory "
                "definitions legal elements scope conditions "
                "exceptions and punishments"
            ).strip()
        )

    elif question_type == "fact_scenario":
        queries.append(
            (
                f"{document_context or 'Pakistan criminal laws'} "
                f"legal provisions directly applicable to this "
                f"factual scenario: {original}"
            ).strip()
        )

    else:
        queries.append(
            (
                f"{document_context or 'Pakistan laws'} "
                f"relevant legal provisions for: {original}"
            ).strip()
        )

    for concept_name in concepts:
        concept_data = LEGAL_CONCEPTS.get(
            concept_name,
            {},
        )

        concept_queries = concept_data.get(
            "queries",
            [],
        )

        queries.extend(
            str(query)
            for query in concept_queries
            if str(query).strip()
        )

    return deduplicate_strings(
        queries
    )


# -------------------------------------------------------------------
# Public router
# -------------------------------------------------------------------

def route_question(
    question: str,
) -> QueryPlan:
    """Build a complete multi-document retrieval plan."""

    original_question = question.strip()

    if not original_question:
        raise ValueError(
            "Question cannot be empty."
        )

    question_type = classify_question(
        original_question
    )

    concepts = detect_legal_concepts(
        original_question
    )

    explicit_document_ids = detect_document_ids(
        original_question
    )

    concept_document_ids = (
        get_concept_document_hints(
            concepts
        )
    )

    document_ids = deduplicate_strings(
        explicit_document_ids
        + concept_document_ids
    )

    section_numbers = extract_section_numbers(
        original_question
    )

    article_numbers = extract_article_numbers(
        original_question
    )

    provision_numbers = deduplicate_strings(
        section_numbers
        + article_numbers
    )

    if _is_theft_fact_scenario(
        question_type=question_type,
        concepts=concepts,
    ):
        provision_numbers = deduplicate_strings(
            provision_numbers
            + sorted(
                get_section_hints(
                    concepts
                )
            )
        )

    provision_type = infer_provision_type(
        document_ids=document_ids,
        section_numbers=section_numbers,
        article_numbers=article_numbers,
    )

    retrieval_queries = build_retrieval_queries(
        question=original_question,
        question_type=question_type,
        detected_concepts=concepts,
        document_ids=document_ids,
        section_numbers=section_numbers,
        article_numbers=article_numbers,
    )

    document_hints = [
        get_document_label(
            document_id
        )
        for document_id in document_ids
    ]

    return QueryPlan(
        original_question=original_question,
        question_type=question_type,
        concepts=concepts,
        section_number=(
            section_numbers[0]
            if section_numbers
            else None
        ),
        retrieval_queries=retrieval_queries,
        section_hints=get_section_hints(
            concepts
        ),
        answer_style=question_type,
        document_ids=document_ids,
        document_hints=document_hints,
        article_number=(
            article_numbers[0]
            if article_numbers
            else None
        ),
        provision_numbers=provision_numbers,
        provision_type=provision_type,
    )


# Backward-compatible alias.
build_query_plan = route_question


# -------------------------------------------------------------------
# Diagnostic test
# -------------------------------------------------------------------

def display_query_plan(
    question: str,
) -> None:
    """Print the detected routing plan for one question."""

    plan = route_question(
        question
    )

    print("\n" + "=" * 70)
    print("QUESTION")
    print("=" * 70)
    print(question)

    print("\nROUTING PLAN")
    print(
        f"Question type: {plan.question_type}"
    )
    print(
        f"Concepts: {plan.concepts}"
    )
    print(
        f"Document IDs: {plan.document_ids}"
    )
    print(
        f"Provision type: {plan.provision_type}"
    )
    print(
        f"Section number: {plan.section_number}"
    )
    print(
        f"Article number: {plan.article_number}"
    )
    print(
        f"Provision numbers: "
        f"{plan.provision_numbers}"
    )
    print(
        f"Section hints: {plan.section_hints}"
    )

    print("\nRetrieval queries:")

    for index, query in enumerate(
        plan.retrieval_queries,
        start=1,
    ):
        print(
            f"{index}. {query}"
        )


def main() -> None:
    """Run representative multi-document routing tests."""

    questions = [
        "What is Section 379 of the PPC?",
        "What does Article 10A of the Constitution provide?",
        "What is the punishment for money laundering under AMLA?",
        "Explain Section 7 of the Anti-Terrorism Act.",
        (
            "A person collected illegal funds and transferred them "
            "through several accounts to conceal their source. "
            "Which law and provisions may apply?"
        ),
        (
            "Compare theft under the PPC with money laundering "
            "under AMLA."
        ),
    ]

    for question in questions:
        display_query_plan(
            question
        )


if __name__ == "__main__":
    main()
