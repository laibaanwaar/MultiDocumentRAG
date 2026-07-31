import math
from collections import Counter
from typing import Any

from langchain_core.documents import Document

from rag.concept_registry import LEGAL_CONCEPTS, SPECIAL_SECTION_REQUIREMENTS
from rag.intent_router import extract_section_number, normalize_text, tokenize
from rag.schemas import (
    CandidateDocument,
    QueryPlan,
    RankedDocument,
    RetrievalConfidence,
)


def _document_key(document: Document) -> tuple[Any, ...]:
    metadata = document.metadata
    return (
        metadata.get("document_id") or metadata.get("document_name"),
        metadata.get("section_number"),
        metadata.get("section_part_number", 1),
        metadata.get("page_start"),
        metadata.get("page_end"),
        metadata.get("chunk_number"),
    )


def _section_key(document: Document) -> tuple[Any, ...]:
    metadata = document.metadata
    section_number = metadata.get("section_number")

    if section_number:
        return (
            metadata.get("document_id") or metadata.get("document_name"),
            str(section_number),
        )

    return (
        metadata.get("document_id") or metadata.get("document_name"),
        "unsectioned",
        metadata.get("chunk_number"),
    )


def normalize_similarity_score(score: float) -> float:
    if math.isnan(score) or math.isinf(score):
        return 0.0

    return max(0.0, min(1.0, score))


def keyword_overlap_score(question: str, document: Document) -> float:
    question_tokens = tokenize(question)
    document_tokens = tokenize(
        str(document.metadata.get("section_title", "")) + " " + document.page_content[:1200]
    )

    if not question_tokens:
        return 0.0

    overlap = question_tokens & document_tokens
    return min(1.0, len(overlap) / max(1, len(question_tokens)))


def concept_overlap_score(
    document: Document,
    detected_concepts: list[str],
) -> float:
    if not detected_concepts:
        return 0.0

    metadata = document.metadata
    section_number = str(metadata.get("section_number", "")).upper()
    document_text = normalize_text(
        str(metadata.get("section_title", "")) + " " + document.page_content[:1500]
    )

    scores: list[float] = []
    for concept_name in detected_concepts:
        concept = LEGAL_CONCEPTS[concept_name]
        keyword_hits = sum(keyword in document_text for keyword in concept["keywords"])
        keyword_score = keyword_hits / max(1, len(concept["keywords"]))
        section_score = 1.0 if section_number in concept["preferred_sections"] else 0.0
        scores.append(min(1.0, 0.65 * keyword_score + 0.35 * section_score))

    return max(scores, default=0.0)


def section_match_boost(
    document: Document,
    question: str,
    detected_concepts: list[str],
) -> float:
    metadata = document.metadata
    section_number = str(metadata.get("section_number", "")).upper()
    explicit_section = extract_section_number(question)

    if explicit_section and section_number == explicit_section:
        return 1.0

    title = normalize_text(str(metadata.get("section_title", "")))
    question_normalized = normalize_text(question)
    title_tokens = tokenize(title)

    if title and title in question_normalized:
        return 0.9

    if title_tokens and len(title_tokens & tokenize(question)) >= min(2, len(title_tokens)):
        return 0.6

    for concept_name in detected_concepts:
        if section_number in LEGAL_CONCEPTS[concept_name]["preferred_sections"]:
            return 0.7

    return 0.0


def special_section_penalty(document: Document, question: str) -> float:
    section_number = str(document.metadata.get("section_number", "")).upper()
    requirements = SPECIAL_SECTION_REQUIREMENTS.get(section_number)

    if not requirements:
        return 0.0

    normalized_question = normalize_text(question)
    if any(requirement in normalized_question for requirement in requirements):
        return 0.0

    return 0.40


def calculate_final_score(item: RankedDocument) -> float:
    semantic = normalize_similarity_score(item.relevance_score or 0.0)
    score = (
        0.55 * semantic
        + 0.20 * item.section_boost
        + 0.15 * item.keyword_overlap
        + 0.10 * item.concept_overlap
    )
    score += min(0.08, item.fusion_score * 2.0)
    score -= item.special_penalty

    if item.document.metadata.get("page_quality_suspicious", False):
        score -= 0.05

    return max(0.0, min(1.0, score))


def text_similarity(first: str, second: str) -> float:
    first_tokens = tokenize(first)
    second_tokens = tokenize(second)

    if not first_tokens or not second_tokens:
        return 0.0

    intersection = len(first_tokens & second_tokens)
    union = len(first_tokens | second_tokens)
    return intersection / max(1, union)


def deduplicate_ranked_documents(
    items: list[RankedDocument],
    semantic_threshold: float,
) -> list[RankedDocument]:
    selected: list[RankedDocument] = []

    for item in items:
        metadata = item.document.metadata
        section_number = metadata.get("section_number")
        part_number = metadata.get("section_part_number", 1)
        duplicate = False

        for existing in selected:
            existing_metadata = existing.document.metadata

            if (
                section_number == existing_metadata.get("section_number")
                and part_number != existing_metadata.get("section_part_number", 1)
            ):
                continue

            if (
                text_similarity(
                    item.document.page_content,
                    existing.document.page_content,
                )
                >= semantic_threshold
            ):
                duplicate = True
                break

        if not duplicate:
            selected.append(item)

    return selected


def rank_candidates(
    question: str,
    detected_concepts: list[str],
    candidates: list[CandidateDocument],
    semantic_threshold: float,
) -> list[RankedDocument]:
    fused_results: dict[tuple[Any, ...], RankedDocument] = {}

    for candidate in candidates:
        key = _document_key(candidate.document)

        if key not in fused_results:
            fused_results[key] = RankedDocument(
                document=candidate.document,
                fusion_score=0.0,
                relevance_score=normalize_similarity_score(candidate.relevance_score),
                matched_queries=0,
            )

        item = fused_results[key]
        item.fusion_score += 1.0 / (60 + candidate.query_index + 1)
        item.matched_queries += 1
        item.relevance_score = max(
            item.relevance_score or 0.0,
            normalize_similarity_score(candidate.relevance_score),
        )

    ranked_items = list(fused_results.values())

    for item in ranked_items:
        item.keyword_overlap = keyword_overlap_score(question, item.document)
        item.concept_overlap = concept_overlap_score(item.document, detected_concepts)
        item.section_boost = section_match_boost(item.document, question, detected_concepts)
        item.special_penalty = special_section_penalty(item.document, question)
        item.final_score = calculate_final_score(item)

    ranked_items.sort(
        key=lambda item: (
            item.final_score,
            item.matched_queries,
            item.relevance_score or 0.0,
        ),
        reverse=True,
    )

    return deduplicate_ranked_documents(
        ranked_items,
        semantic_threshold=semantic_threshold,
    )


def merge_section_parts(documents: list[Document]) -> list[Document]:
    grouped: dict[tuple[Any, ...], list[Document]] = {}

    for document in documents:
        grouped.setdefault(_section_key(document), []).append(document)

    merged_documents: list[Document] = []

    for group_documents in grouped.values():
        ordered = sorted(
            group_documents,
            key=lambda document: (
                int(document.metadata.get("section_part_number", 1) or 1),
                int(document.metadata.get("chunk_number", 0) or 0),
            ),
        )

        if len(ordered) == 1:
            merged_documents.append(ordered[0])
            continue

        base_metadata = dict(ordered[0].metadata)
        combined_parts: list[str] = []
        contributing_chunks: list[Any] = []
        source_pages: set[int] = set()

        for document in ordered:
            metadata = document.metadata
            combined_parts.append(document.page_content.strip())
            contributing_chunks.append(metadata.get("chunk_number"))

            for page in metadata.get("source_pages", []):
                if isinstance(page, int):
                    source_pages.add(page)

        base_metadata.update(
            {
                "section_part_number": 1,
                "section_part_count": 1,
                "section_was_merged": True,
                "merged_chunk_numbers": contributing_chunks,
                "source_pages": sorted(source_pages),
                "page_start": min(source_pages) if source_pages else base_metadata.get("page_start"),
                "page_end": max(source_pages) if source_pages else base_metadata.get("page_end"),
            }
        )

        merged_documents.append(
            Document(
                page_content="\n\n".join(combined_parts),
                metadata=base_metadata,
            )
        )

    return merged_documents


def select_context_documents(
    ranked_items: list[RankedDocument],
    question_type: str,
    maximum_documents: int,
    max_context_sections: int,
    final_k: int | None = None,
) -> list[Document]:
    grouped: dict[tuple[Any, ...], list[RankedDocument]] = {}

    for item in ranked_items:
        grouped.setdefault(_section_key(item.document), []).append(item)

    ranked_groups = sorted(
        grouped.values(),
        key=lambda group: max(item.final_score for item in group),
        reverse=True,
    )

    selected_items: list[RankedDocument] = []
    section_limit = 3 if question_type == "fact_scenario" else max_context_sections

    for group in ranked_groups[:section_limit]:
        selected_items.extend(
            sorted(
                group,
                key=lambda item: int(item.document.metadata.get("section_part_number", 1) or 1),
            )
        )
        if len(selected_items) >= maximum_documents:
            break

    selected_documents = [item.document for item in selected_items[:maximum_documents]]
    merged_documents = merge_section_parts(selected_documents)[:section_limit]
    final_limit = final_k if final_k is not None else (8 if question_type in {"fact_scenario", "comparison"} else 5)

    return merged_documents[:final_limit]


def calculate_retrieval_confidence(
    ranked_items: list[RankedDocument],
    selected_documents: list[Document],
    detected_concepts: list[str],
) -> RetrievalConfidence:
    if not ranked_items:
        return RetrievalConfidence(
            label="Low",
            score=0.0,
            top_similarity=0.0,
            average_similarity=0.0,
            section_count=0,
            concept_coverage=0.0,
        )

    similarities = [item.relevance_score or 0.0 for item in ranked_items[:10]]
    top_similarity = max(similarities, default=0.0)
    average_similarity = sum(similarities) / max(1, len(similarities))

    selected_sections = {
        str(document.metadata.get("section_number", ""))
        for document in selected_documents
        if document.metadata.get("section_number")
    }

    covered_concepts = 0
    for concept_name in detected_concepts:
        preferred = LEGAL_CONCEPTS[concept_name]["preferred_sections"]
        if not preferred or selected_sections & preferred:
            covered_concepts += 1

    concept_coverage = (
        covered_concepts / max(1, len(detected_concepts))
        if detected_concepts
        else 1.0
    )

    score = (
        0.45 * top_similarity
        + 0.30 * average_similarity
        + 0.15 * min(1.0, len(selected_sections) / 3)
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
        top_similarity=round(top_similarity, 3),
        average_similarity=round(average_similarity, 3),
        section_count=len(selected_sections),
        concept_coverage=round(concept_coverage, 3),
    )
