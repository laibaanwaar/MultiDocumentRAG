from __future__ import annotations

import math
from collections import Counter
from typing import Any, Iterable

from langchain_core.documents import Document

from rag.concept_registry import (
    LEGAL_CONCEPTS,
    SPECIAL_SECTION_REQUIREMENTS,
)
from rag.intent_router import (
    extract_article_number,
    extract_section_number,
    normalize_text,
    tokenize,
)
from rag.schemas import (
    CandidateDocument,
    RankedDocument,
    RetrievalConfidence,
)

def _normalize_value(
    value: Any,
) -> str:
    """Return a normalized uppercase metadata value."""

    if value is None:
        return ""

    return str(value).strip().upper()


def _normalize_values(
    values: Iterable[Any] | None,
) -> set[str]:
    """Normalize a collection of metadata values."""

    if values is None:
        return set()

    return {
        normalized
        for value in values
        if (
            normalized := _normalize_value(
                value
            )
        )
    }


def get_document_id(
    document: Document,
) -> str:
    """Return the stable legal-document identity."""

    metadata = document.metadata

    return str(
        metadata.get(
            "document_id",
            metadata.get(
                "document_name",
                "",
            ),
        )
    ).strip()


def get_provision_type(
    document: Document,
) -> str | None:
    """Return `section` or `article` from chunk metadata."""

    metadata = document.metadata

    explicit_type = str(
        metadata.get(
            "provision_type",
            "",
        )
    ).strip().lower()

    if explicit_type in {
        "section",
        "article",
    }:
        return explicit_type

    if metadata.get(
        "article_number"
    ):
        return "article"

    if metadata.get(
        "section_number"
    ):
        return "section"

    if (
        metadata.get(
            "document_type"
        )
        == "constitutional_law"
    ):
        return "article"

    return None


def get_provision_number(
    document: Document,
) -> str:
    """Return a Section or Article number using compatible keys."""

    metadata = document.metadata

    return _normalize_value(
        metadata.get(
            "provision_number"
        )
        or metadata.get(
            "section_number"
        )
        or metadata.get(
            "article_number"
        )
    )


def get_provision_title(
    document: Document,
) -> str:
    """Return the best available Section or Article title."""

    metadata = document.metadata

    return str(
        metadata.get(
            "provision_title"
        )
        or metadata.get(
            "section_title"
        )
        or metadata.get(
            "article_title"
        )
        or ""
    ).strip()


def get_part_number(
    document: Document,
) -> int:
    """Return a safe provision-part number."""

    metadata = document.metadata

    raw_value = metadata.get(
        "provision_part_number",
        metadata.get(
            "section_part_number",
            1,
        ),
    )

    try:
        return max(
            1,
            int(raw_value or 1),
        )
    except (
        TypeError,
        ValueError,
    ):
        return 1


def get_part_count(
    document: Document,
) -> int:
    """Return a safe provision-part count."""

    metadata = document.metadata

    raw_value = metadata.get(
        "provision_part_count",
        metadata.get(
            "section_part_count",
            1,
        ),
    )

    try:
        return max(
            1,
            int(raw_value or 1),
        )
    except (
        TypeError,
        ValueError,
    ):
        return 1


def _document_key(
    document: Document,
) -> tuple[Any, ...]:
    """Return a stable chunk-level fusion key."""

    metadata = document.metadata

    chunk_id = metadata.get(
        "chunk_id"
    )

    if chunk_id:
        return (
            get_document_id(document),
            str(chunk_id),
        )

    return (
        get_document_id(document),
        get_provision_type(
            document
        ),
        get_provision_number(
            document
        ),
        get_part_number(
            document
        ),
        metadata.get(
            "page_start"
        ),
        metadata.get(
            "page_end"
        ),
        metadata.get(
            "document_chunk_number",
            metadata.get(
                "chunk_number"
            ),
        ),
    )


def _provision_key(
    document: Document,
) -> tuple[Any, ...]:
    """Return a document-aware Section or Article grouping key."""

    metadata = document.metadata
    provision_number = get_provision_number(
        document
    )

    if provision_number:
        return (
            get_document_id(
                document
            ),
            get_provision_type(
                document
            )
            or "provision",
            provision_number,
        )

    return (
        get_document_id(
            document
        ),
        "unsectioned",
        metadata.get(
            "document_chunk_number",
            metadata.get(
                "chunk_number"
            ),
        ),
    )


# Backward-compatible name used by older code.
_section_key = _provision_key


def normalize_similarity_score(
    score: float,
) -> float:
    """Clamp a similarity score to the expected 0–1 range."""

    if math.isnan(
        score
    ) or math.isinf(
        score
    ):
        return 0.0

    return max(
        0.0,
        min(
            1.0,
            float(score),
        ),
    )


def keyword_overlap_score(
    question: str,
    document: Document,
) -> float:
    """Measure lexical overlap with title, law name, and chunk text."""

    question_tokens = tokenize(
        question
    )

    if not question_tokens:
        return 0.0

    metadata = document.metadata

    searchable_text = " ".join(
        (
            str(
                metadata.get(
                    "document_title",
                    "",
                )
            ),
            str(
                metadata.get(
                    "document_short_name",
                    "",
                )
            ),
            get_provision_title(
                document
            ),
            document.page_content[:1600],
        )
    )

    document_tokens = tokenize(
        searchable_text
    )

    overlap = (
        question_tokens
        & document_tokens
    )

    return min(
        1.0,
        len(overlap)
        / max(
            1,
            len(question_tokens),
        ),
    )


def _concept_document_ids(
    concept_data: dict[str, Any],
) -> set[str]:
    """Read optional preferred-document hints."""

    return {
        str(document_id).strip()
        for document_id in (
            concept_data.get(
                "preferred_documents",
                [],
            )
            or concept_data.get(
                "document_ids",
                [],
            )
        )
        if str(document_id).strip()
    }


def _concept_provisions(
    concept_data: dict[str, Any],
) -> set[str]:
    """Read Section and Article hints from the concept registry."""

    values: list[Any] = []

    values.extend(
        concept_data.get(
            "preferred_sections",
            [],
        )
    )

    values.extend(
        concept_data.get(
            "preferred_articles",
            [],
        )
    )

    values.extend(
        concept_data.get(
            "preferred_provisions",
            [],
        )
    )

    return _normalize_values(
        values
    )


def concept_overlap_score(
    document: Document,
    detected_concepts: list[str],
) -> float:
    """
    Score concept keywords, preferred laws, and preferred provisions.
    """

    if not detected_concepts:
        return 0.0

    provision_number = (
        get_provision_number(
            document
        )
    )
    document_id = get_document_id(
        document
    )

    document_text = normalize_text(
        " ".join(
            (
                get_provision_title(
                    document
                ),
                document.page_content[:1800],
            )
        )
    )

    scores: list[float] = []

    for concept_name in detected_concepts:
        concept = LEGAL_CONCEPTS.get(
            concept_name,
            {},
        )

        keywords = [
            normalize_text(
                str(keyword)
            )
            for keyword in concept.get(
                "keywords",
                [],
            )
            if str(keyword).strip()
        ]

        keyword_hits = sum(
            keyword in document_text
            for keyword in keywords
        )

        keyword_score = (
            keyword_hits
            / max(
                1,
                len(keywords),
            )
            if keywords
            else 0.0
        )

        preferred_provisions = (
            _concept_provisions(
                concept
            )
        )

        provision_score = (
            1.0
            if (
                provision_number
                and provision_number
                in preferred_provisions
            )
            else 0.0
        )

        preferred_documents = (
            _concept_document_ids(
                concept
            )
        )

        document_score = (
            1.0
            if (
                document_id
                and document_id
                in preferred_documents
            )
            else 0.0
        )

        if preferred_documents:
            combined_score = (
                0.55
                * keyword_score
                + 0.25
                * provision_score
                + 0.20
                * document_score
            )
        else:
            combined_score = (
                0.65
                * keyword_score
                + 0.35
                * provision_score
            )

        scores.append(
            min(
                1.0,
                combined_score,
            )
        )

    return max(
        scores,
        default=0.0,
    )


def provision_match_boost(
    document: Document,
    question: str,
    detected_concepts: list[str],
) -> float:
    """
    Boost exact Section/Article matches, titles, and concept hints.
    """

    provision_number = (
        get_provision_number(
            document
        )
    )
    provision_type = get_provision_type(
        document
    )

    explicit_section = (
        extract_section_number(
            question
        )
    )
    explicit_article = (
        extract_article_number(
            question
        )
    )

    if (
        provision_type == "section"
        and explicit_section
        and provision_number
        == _normalize_value(
            explicit_section
        )
    ):
        return 1.0

    if (
        provision_type == "article"
        and explicit_article
        and provision_number
        == _normalize_value(
            explicit_article
        )
    ):
        return 1.0

    title = normalize_text(
        get_provision_title(
            document
        )
    )
    question_normalized = (
        normalize_text(
            question
        )
    )
    title_tokens = tokenize(
        title
    )

    if (
        title
        and title
        in question_normalized
    ):
        return 0.90

    if (
        title_tokens
        and len(
            title_tokens
            & tokenize(
                question
            )
        )
        >= min(
            2,
            len(
                title_tokens
            ),
        )
    ):
        return 0.65

    document_id = get_document_id(
        document
    )

    for concept_name in detected_concepts:
        concept = LEGAL_CONCEPTS.get(
            concept_name,
            {},
        )

        preferred_provisions = (
            _concept_provisions(
                concept
            )
        )

        preferred_documents = (
            _concept_document_ids(
                concept
            )
        )

        provision_matches = (
            provision_number
            and provision_number
            in preferred_provisions
        )

        document_matches = (
            not preferred_documents
            or document_id
            in preferred_documents
        )

        if (
            provision_matches
            and document_matches
        ):
            return 0.75

    return 0.0


# Backward-compatible function name.
section_match_boost = (
    provision_match_boost
)


def special_section_penalty(
    document: Document,
    question: str,
) -> float:
    """
    Penalize special PPC Sections unless their required facts appear.

    Plain numeric keys in SPECIAL_SECTION_REQUIREMENTS are treated as
    PPC-specific to avoid penalizing the same Section number in ATA or
    AMLA. A registry may also use `document_id:provision` keys.
    """

    provision_number = (
        get_provision_number(
            document
        )
    )

    if not provision_number:
        return 0.0

    document_id = get_document_id(
        document
    )

    scoped_key = (
        f"{document_id}:"
        f"{provision_number}"
    )

    requirements = (
        SPECIAL_SECTION_REQUIREMENTS.get(
            scoped_key
        )
    )

    if (
        requirements is None
        and document_id == "ppc_1860"
    ):
        requirements = (
            SPECIAL_SECTION_REQUIREMENTS.get(
                provision_number
            )
        )

    if not requirements:
        return 0.0

    normalized_question = normalize_text(
        question
    )

    if any(
        normalize_text(
            str(requirement)
        )
        in normalized_question
        for requirement in requirements
    ):
        return 0.0

    return 0.40


def calculate_final_score(
    item: RankedDocument,
) -> float:
    """Combine semantic, lexical, routing, and fusion signals."""

    semantic = normalize_similarity_score(
        item.relevance_score
        or 0.0
    )

    score = (
        0.55
        * semantic
        + 0.20
        * item.section_boost
        + 0.15
        * item.keyword_overlap
        + 0.10
        * item.concept_overlap
    )

    score += min(
        0.08,
        item.fusion_score
        * 2.0,
    )

    score -= item.special_penalty

    if item.document.metadata.get(
        "page_quality_suspicious",
        False,
    ):
        score -= 0.05

    if item.document.metadata.get(
        "heading_only_chunk",
        False,
    ):
        score -= 0.25

    return max(
        0.0,
        min(
            1.0,
            score,
        ),
    )


# -------------------------------------------------------------------
# Deduplication and reciprocal-rank fusion
# -------------------------------------------------------------------

def text_similarity(
    first: str,
    second: str,
) -> float:
    """Calculate Jaccard token similarity between two chunks."""

    first_tokens = tokenize(
        first
    )
    second_tokens = tokenize(
        second
    )

    if (
        not first_tokens
        or not second_tokens
    ):
        return 0.0

    intersection = len(
        first_tokens
        & second_tokens
    )
    union = len(
        first_tokens
        | second_tokens
    )

    return intersection / max(
        1,
        union,
    )


def deduplicate_ranked_documents(
    items: list[RankedDocument],
    semantic_threshold: float,
) -> list[RankedDocument]:
    """
    Remove near-duplicate chunks without dropping separate provision parts.
    """

    selected: list[
        RankedDocument
    ] = []

    for item in items:
        item_key = _provision_key(
            item.document
        )
        item_part = get_part_number(
            item.document
        )
        duplicate = False

        for existing in selected:
            existing_key = (
                _provision_key(
                    existing.document
                )
            )
            existing_part = (
                get_part_number(
                    existing.document
                )
            )

            # Keep different parts of the same long provision.
            if (
                item_key == existing_key
                and item_part
                != existing_part
            ):
                continue

            # Never deduplicate across different laws or provisions.
            if item_key != existing_key:
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
            selected.append(
                item
            )

    return selected


def rank_candidates(
    question: str,
    detected_concepts: list[str],
    candidates: list[CandidateDocument],
    semantic_threshold: float,
) -> list[RankedDocument]:
    """
    Fuse multi-query candidates and rank legal chunks.
    """

    if not 0.0 <= semantic_threshold <= 1.0:
        raise ValueError(
            "semantic_threshold must be between 0 and 1."
        )

    fused_results: dict[
        tuple[Any, ...],
        RankedDocument,
    ] = {}

    query_ranks: Counter[int] = (
        Counter()
    )

    for candidate in candidates:
        key = _document_key(
            candidate.document
        )

        query_ranks[
            candidate.query_index
        ] += 1

        candidate_rank = query_ranks[
            candidate.query_index
        ]

        if key not in fused_results:
            fused_results[key] = (
                RankedDocument(
                    document=(
                        candidate.document
                    ),
                    fusion_score=0.0,
                    relevance_score=(
                        normalize_similarity_score(
                            candidate.relevance_score
                        )
                    ),
                    matched_queries=0,
                )
            )

        item = fused_results[
            key
        ]

        # Reciprocal-rank fusion uses the rank within each query.
        item.fusion_score += (
            1.0
            / (
                60
                + candidate_rank
            )
        )
        item.matched_queries += 1
        item.relevance_score = max(
            item.relevance_score
            or 0.0,
            normalize_similarity_score(
                candidate.relevance_score
            ),
        )

    ranked_items = list(
        fused_results.values()
    )

    for item in ranked_items:
        item.keyword_overlap = (
            keyword_overlap_score(
                question,
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
            provision_match_boost(
                item.document,
                question,
                detected_concepts,
            )
        )
        item.special_penalty = (
            special_section_penalty(
                item.document,
                question,
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
            item.relevance_score
            or 0.0,
            -get_part_number(
                item.document
            ),
        ),
        reverse=True,
    )

    return deduplicate_ranked_documents(
        ranked_items,
        semantic_threshold=(
            semantic_threshold
        ),
    )

def merge_provision_parts(
    documents: list[Document],
) -> list[Document]:
    """Merge split parts of the same Section or Article."""

    grouped: dict[
        tuple[Any, ...],
        list[Document],
    ] = {}

    for document in documents:
        grouped.setdefault(
            _provision_key(
                document
            ),
            [],
        ).append(
            document
        )

    merged_documents: list[
        Document
    ] = []

    for group_documents in (
        grouped.values()
    ):
        ordered = sorted(
            group_documents,
            key=lambda document: (
                get_part_number(
                    document
                ),
                int(
                    document.metadata.get(
                        "document_chunk_number",
                        document.metadata.get(
                            "chunk_number",
                            0,
                        ),
                    )
                    or 0
                ),
            ),
        )

        if len(
            ordered
        ) == 1:
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

            content = (
                document.page_content.strip()
            )

            if (
                content
                and content
                not in combined_parts
            ):
                combined_parts.append(
                    content
                )

            contributing_chunks.append(
                metadata.get(
                    "document_chunk_number",
                    metadata.get(
                        "chunk_number"
                    ),
                )
            )

            raw_source_pages = (
                metadata.get(
                    "source_pages",
                    [],
                )
            )

            if isinstance(
                raw_source_pages,
                (
                    list,
                    tuple,
                    set,
                ),
            ):
                for page in raw_source_pages:
                    if isinstance(
                        page,
                        int,
                    ):
                        source_pages.add(
                            page
                        )

        base_metadata.update(
            {
                "provision_part_number": 1,
                "provision_part_count": 1,
                "section_part_number": 1,
                "section_part_count": 1,
                "provision_was_merged": True,
                "section_was_merged": True,
                "merged_chunk_numbers": (
                    contributing_chunks
                ),
                "source_pages": sorted(
                    source_pages
                ),
                "page_start": (
                    min(
                        source_pages
                    )
                    if source_pages
                    else base_metadata.get(
                        "page_start"
                    )
                ),
                "page_end": (
                    max(
                        source_pages
                    )
                    if source_pages
                    else base_metadata.get(
                        "page_end"
                    )
                ),
            }
        )

        merged_documents.append(
            Document(
                page_content="\n\n".join(
                    combined_parts
                ),
                metadata=base_metadata,
            )
        )

    return merged_documents


# Backward-compatible function name.
merge_section_parts = (
    merge_provision_parts
)


def select_context_documents(
    ranked_items: list[RankedDocument],
    question_type: str,
    maximum_documents: int,
    max_context_sections: int,
    final_k: int | None = None,
) -> list[Document]:
    """
    Select top Sections/Articles while preserving all selected parts.
    """

    if maximum_documents <= 0:
        return []

    if max_context_sections <= 0:
        return []

    grouped: dict[
        tuple[Any, ...],
        list[RankedDocument],
    ] = {}

    for item in ranked_items:
        grouped.setdefault(
            _provision_key(
                item.document
            ),
            [],
        ).append(
            item
        )

    ranked_groups = sorted(
        grouped.values(),
        key=lambda group: (
            max(
                item.final_score
                for item in group
            ),
            max(
                item.relevance_score
                or 0.0
                for item in group
            ),
        ),
        reverse=True,
    )

    if question_type == "fact_scenario":
        provision_limit = max(
            3,
            max_context_sections,
        )
    elif question_type == "comparison":
        provision_limit = max(
            4,
            max_context_sections,
        )
    else:
        provision_limit = (
            max_context_sections
        )

    selected_items: list[
        RankedDocument
    ] = []

    for group in ranked_groups[
        :provision_limit
    ]:
        ordered_group = sorted(
            group,
            key=lambda item: (
                get_part_number(
                    item.document
                )
            ),
        )

        selected_items.extend(
            ordered_group
        )

        if len(
            selected_items
        ) >= maximum_documents:
            break

    selected_documents = [
        item.document
        for item in selected_items[
            :maximum_documents
        ]
    ]

    merged_documents = (
        merge_provision_parts(
            selected_documents
        )
    )

    if final_k is not None:
        final_limit = max(
            0,
            final_k,
        )
    elif question_type in {
        "fact_scenario",
        "comparison",
    }:
        final_limit = 8
    else:
        final_limit = 5

    return merged_documents[
        :min(
            provision_limit,
            final_limit,
        )
    ]



def _selected_provision_keys(
    selected_documents: list[Document],
) -> set[
    tuple[Any, ...]
]:
    """Return selected document-aware provision identities."""

    return {
        _provision_key(
            document
        )
        for document in selected_documents
    }


def _concept_is_covered(
    concept_name: str,
    selected_documents: list[Document],
) -> bool:
    """Check whether selected context covers one detected concept."""

    concept = LEGAL_CONCEPTS.get(
        concept_name,
        {},
    )

    preferred_provisions = (
        _concept_provisions(
            concept
        )
    )
    preferred_documents = (
        _concept_document_ids(
            concept
        )
    )

    if (
        not preferred_provisions
        and not preferred_documents
    ):
        return True

    for document in selected_documents:
        document_matches = (
            not preferred_documents
            or get_document_id(
                document
            )
            in preferred_documents
        )

        provision_matches = (
            not preferred_provisions
            or get_provision_number(
                document
            )
            in preferred_provisions
        )

        if (
            document_matches
            and provision_matches
        ):
            return True

    return False


def calculate_retrieval_confidence(
    ranked_items: list[RankedDocument],
    selected_documents: list[Document],
    detected_concepts: list[str],
) -> RetrievalConfidence:
    """
    Estimate confidence from similarity, diversity, and concept coverage.

    `section_count` is retained for schema compatibility and now counts
    unique Sections or Articles.
    """

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
        normalize_similarity_score(
            item.relevance_score
            or 0.0
        )
        for item in ranked_items[
            :10
        ]
    ]

    top_similarity = max(
        similarities,
        default=0.0,
    )

    average_similarity = (
        sum(
            similarities
        )
        / max(
            1,
            len(
                similarities
            ),
        )
    )

    selected_provisions = (
        _selected_provision_keys(
            selected_documents
        )
    )

    if detected_concepts:
        covered_concepts = sum(
            _concept_is_covered(
                concept_name,
                selected_documents,
            )
            for concept_name
            in detected_concepts
        )

        concept_coverage = (
            covered_concepts
            / len(
                detected_concepts
            )
        )
    else:
        concept_coverage = 1.0

    provision_diversity = min(
        1.0,
        len(
            selected_provisions
        )
        / 3,
    )

    score = (
        0.45
        * top_similarity
        + 0.30
        * average_similarity
        + 0.15
        * provision_diversity
        + 0.10
        * concept_coverage
    )

    if score >= 0.72:
        label = "High"
    elif score >= 0.48:
        label = "Medium"
    else:
        label = "Low"

    return RetrievalConfidence(
        label=label,
        score=round(
            score,
            3,
        ),
        top_similarity=round(
            top_similarity,
            3,
        ),
        average_similarity=round(
            average_similarity,
            3,
        ),
        section_count=len(
            selected_provisions
        ),
        concept_coverage=round(
            concept_coverage,
            3,
        ),
    )