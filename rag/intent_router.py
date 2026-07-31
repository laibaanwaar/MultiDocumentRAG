import re

from rag.concept_registry import (
    LEGAL_CONCEPTS,
    SCENARIO_MARKERS,
)
from rag.schemas import QueryPlan


def normalize_text(text: str) -> str:
    return re.sub(
        r"\s+",
        " ",
        text.lower(),
    ).strip()


def tokenize(text: str) -> set[str]:
    tokens = re.findall(
        r"[a-z0-9]+(?:-[a-z0-9]+)?",
        text.lower(),
    )

    return {
        token
        for token in tokens
        if len(token) > 1
        and token not in {
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
    }


def extract_section_number(question: str) -> str | None:
    match = re.search(
        r"\bsection\s+(\d+(?:-[A-Za-z]+)?[A-Za-z]?)\b",
        question,
        re.IGNORECASE,
    )

    return match.group(1).upper() if match else None


def detect_legal_concepts(question: str) -> list[str]:
    normalized = normalize_text(question)
    detected: list[str] = []

    for concept_name, concept_data in LEGAL_CONCEPTS.items():
        if any(
            marker in normalized
            for marker in concept_data["markers"]
        ):
            detected.append(concept_name)

    return detected


def calculate_scenario_score(question: str) -> int:
    normalized = normalize_text(question)

    return sum(
        weight
        for marker, weight in SCENARIO_MARKERS.items()
        if marker in normalized
    )


def classify_question(question: str) -> str:
    normalized = normalize_text(question)

    # 1. Comparison should stay highest priority
    if any(
        phrase in normalized
        for phrase in (
            "difference between",
            "compare",
            "distinguish between",
        )
    ) or re.search(r"\bvs\b|\bversus\b", normalized):
        return "comparison"

    # 2. Detect scenario BEFORE section/punishment/definition
    scenario_score = calculate_scenario_score(question)
    detected_concepts = detect_legal_concepts(question)

    is_scenario = (
        scenario_score >= 3
        or (
            len(normalized.split()) >= 18
            and detected_concepts
        )
    )

    if is_scenario:
        return "fact_scenario"

    # 3. Explicit section lookup
    if extract_section_number(question):
        return "section_lookup"

    # 4. Punishment question
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

    # 5. Definition question
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


def get_retrieval_k(question_type: str, top_k: int) -> int:
    return {
        "section_lookup": max(5, top_k),
        "definition": max(7, top_k),
        "punishment": max(8, top_k),
        "comparison": max(14, top_k),
        "fact_scenario": max(25, top_k),
        "general": max(10, top_k),
    }.get(question_type, top_k)


def build_retrieval_queries(
    question: str,
    question_type: str,
    detected_concepts: list[str] | None = None,
) -> list[str]:
    original = question.strip()
    queries: list[str] = [original]
    section_number = extract_section_number(original)
    concepts = detected_concepts or detect_legal_concepts(original)

    if question_type == "section_lookup" and section_number:
        queries.append(
            f"Section {section_number} exact statutory text, heading, explanation, illustrations and punishment"
        )
    elif question_type == "punishment":
        queries.append(
            f"{original} exact punishment provision, imprisonment term, fine, conditions and legal capacity"
        )
    elif question_type == "definition":
        queries.append(
            f"{original} exact statutory definition, legal elements, explanations and illustrations"
        )
    elif question_type == "comparison":
        queries.append(
            f"{original} compare statutory definitions, elements, conditions and punishments"
        )
    elif question_type == "fact_scenario":
        queries.append(
            f"Pakistan Penal Code provisions directly connected to this factual scenario: {original}"
        )

    for concept_name in concepts:
        queries.extend(LEGAL_CONCEPTS[concept_name]["queries"])

    return list(dict.fromkeys(queries))


def route_question(question: str) -> QueryPlan:
    original_question = question.strip()
    question_type = classify_question(original_question)
    concepts = detect_legal_concepts(original_question)
    section_number = extract_section_number(original_question)
    retrieval_queries = build_retrieval_queries(
        question=original_question,
        question_type=question_type,
        detected_concepts=concepts,
    )

    section_hints: list[str] = []
    for concept_name in concepts:
        section_hints.extend(
            sorted(LEGAL_CONCEPTS[concept_name]["preferred_sections"])
        )

    return QueryPlan(
        original_question=original_question,
        question_type=question_type,
        concepts=concepts,
        section_number=section_number,
        retrieval_queries=retrieval_queries,
        section_hints=list(dict.fromkeys(section_hints)),
        answer_style=question_type,
    )
