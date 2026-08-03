from __future__ import annotations

from typing import Any, Iterable

from langchain_core.documents import Document
from qdrant_client.models import (
    FieldCondition,
    Filter,
    MatchAny,
    MatchValue,
)

from rag.concept_registry import LEGAL_CONCEPTS
from rag.schemas import CandidateDocument


METADATA_PREFIX = "metadata"


# -------------------------------------------------------------------
# Metadata normalization
# -------------------------------------------------------------------

def normalize_provision_number(
    value: str | int | None,
) -> str | None:
    """Normalize a Section or Article number for metadata matching."""

    if value is None:
        return None

    normalized = str(value).strip().upper()

    return normalized or None


def normalize_string_list(
    values: Iterable[str] | None,
) -> list[str]:
    """Remove empty and duplicate strings while preserving order."""

    if values is None:
        return []

    seen: set[str] = set()
    normalized_values: list[str] = []

    for value in values:
        normalized = str(value).strip()

        if (
            normalized
            and normalized not in seen
        ):
            seen.add(normalized)
            normalized_values.append(
                normalized
            )

    return normalized_values


def get_document_id(
    document: Document,
) -> str:
    """Return a stored document ID."""

    return str(
        document.metadata.get(
            "document_id",
            "",
        )
    ).strip()


def get_document_provision_type(
    document: Document,
) -> str | None:
    """Return `section` or `article` from stored chunk metadata."""

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

    if metadata.get("article_number"):
        return "article"

    if metadata.get("section_number"):
        return "section"

    if (
        metadata.get("document_type")
        == "constitutional_law"
    ):
        return "article"

    return None


def get_document_provision_number(
    document: Document,
) -> str | None:
    """Return a provision number from new or legacy metadata keys."""

    metadata = document.metadata

    return normalize_provision_number(
        metadata.get("provision_number")
        or metadata.get("section_number")
        or metadata.get("article_number")
    )


def get_document_identity(
    document: Document,
) -> str:
    """Build a stable identity used to deduplicate exact hits."""

    metadata = document.metadata

    chunk_id = str(
        metadata.get(
            "chunk_id",
            "",
        )
    ).strip()

    if chunk_id:
        return chunk_id

    return "|".join(
        (
            get_document_id(document),
            get_document_provision_type(
                document
            )
            or "",
            get_document_provision_number(
                document
            )
            or "",
            str(
                metadata.get(
                    "page_start",
                    "",
                )
            ),
            str(
                metadata.get(
                    "page_end",
                    "",
                )
            ),
            str(
                metadata.get(
                    "document_chunk_number",
                    "",
                )
            ),
            document.page_content.strip(),
        )
    )


# -------------------------------------------------------------------
# Qdrant filter builders
# -------------------------------------------------------------------

def build_document_filter(
    document_ids: list[str],
) -> Filter:
    """Build a filter restricted to one or more legal documents."""

    normalized_ids = normalize_string_list(
        document_ids
    )

    if not normalized_ids:
        raise ValueError(
            "document_ids cannot be empty."
        )

    return Filter(
        must=[
            FieldCondition(
                key=f"{METADATA_PREFIX}.document_id",
                match=MatchAny(
                    any=normalized_ids
                ),
            ),
            FieldCondition(
                key=(
                    f"{METADATA_PREFIX}."
                    "heading_only_chunk"
                ),
                match=MatchValue(
                    value=False
                ),
            ),
        ]
    )


def build_provision_filter(
    provision_numbers: list[str],
    provision_type: str | None = None,
    document_ids: list[str] | None = None,
) -> Filter:
    """
    Build a document-aware Section or Article filter.

    The new chunking pipeline stores a generic `provision_number`.
    `provision_type` distinguishes Sections from Constitution Articles.
    """

    normalized_numbers = normalize_string_list(
        [
            normalize_provision_number(number)
            or ""
            for number in provision_numbers
        ]
    )

    if not normalized_numbers:
        raise ValueError(
            "provision_numbers cannot be empty."
        )

    normalized_type = (
        str(provision_type).strip().lower()
        if provision_type
        else None
    )

    if normalized_type not in {
        None,
        "section",
        "article",
    }:
        raise ValueError(
            "provision_type must be 'section', 'article', or None."
        )

    conditions: list[FieldCondition] = [
        FieldCondition(
            key=f"{METADATA_PREFIX}.provision_number",
            match=MatchAny(
                any=normalized_numbers
            ),
        ),
        FieldCondition(
            key=(
                f"{METADATA_PREFIX}."
                "heading_only_chunk"
            ),
            match=MatchValue(
                value=False
            ),
        ),
    ]

    if normalized_type:
        conditions.append(
            FieldCondition(
                key=f"{METADATA_PREFIX}.provision_type",
                match=MatchValue(
                    value=normalized_type
                ),
            )
        )

    normalized_document_ids = (
        normalize_string_list(
            document_ids
        )
    )

    if normalized_document_ids:
        conditions.append(
            FieldCondition(
                key=f"{METADATA_PREFIX}.document_id",
                match=MatchAny(
                    any=normalized_document_ids
                ),
            )
        )

    return Filter(
        must=conditions
    )


def build_section_filter(
    section_number: str,
    document_ids: list[str] | None = None,
) -> Filter:
    """Build a backward-compatible exact Section filter."""

    normalized_number = normalize_provision_number(
        section_number
    )

    if not normalized_number:
        raise ValueError(
            "section_number cannot be empty."
        )

    conditions: list[FieldCondition] = [
        FieldCondition(
            key=f"{METADATA_PREFIX}.section_number",
            match=MatchValue(
                value=normalized_number
            ),
        ),
        FieldCondition(
            key=(
                f"{METADATA_PREFIX}."
                "heading_only_chunk"
            ),
            match=MatchValue(
                value=False
            ),
        ),
    ]

    normalized_document_ids = normalize_string_list(
        document_ids
    )

    if normalized_document_ids:
        conditions.append(
            FieldCondition(
                key=f"{METADATA_PREFIX}.document_id",
                match=MatchAny(
                    any=normalized_document_ids
                ),
            )
        )

    return Filter(
        must=conditions
    )


def build_sections_filter(
    section_numbers: list[str],
    document_ids: list[str] | None = None,
) -> Filter:
    """Build a backward-compatible multi-Section filter."""

    normalized_numbers = normalize_string_list(
        [
            normalize_provision_number(number)
            or ""
            for number in section_numbers
        ]
    )

    if not normalized_numbers:
        raise ValueError(
            "section_numbers cannot be empty."
        )

    conditions: list[FieldCondition] = [
        FieldCondition(
            key=f"{METADATA_PREFIX}.section_number",
            match=MatchAny(
                any=normalized_numbers
            ),
        ),
        FieldCondition(
            key=(
                f"{METADATA_PREFIX}."
                "heading_only_chunk"
            ),
            match=MatchValue(
                value=False
            ),
        ),
    ]

    normalized_document_ids = normalize_string_list(
        document_ids
    )

    if normalized_document_ids:
        conditions.append(
            FieldCondition(
                key=f"{METADATA_PREFIX}.document_id",
                match=MatchAny(
                    any=normalized_document_ids
                ),
            )
        )

    return Filter(
        must=conditions
    )


def build_article_filter(
    article_number: str,
    document_ids: list[str] | None = None,
) -> Filter:
    """Build an exact Constitution Article filter."""

    normalized_number = normalize_provision_number(
        article_number
    )

    if not normalized_number:
        raise ValueError(
            "article_number cannot be empty."
        )

    effective_document_ids = (
        normalize_string_list(
            document_ids
        )
        or ["constitution_1973"]
    )

    return Filter(
        must=[
            FieldCondition(
                key=f"{METADATA_PREFIX}.article_number",
                match=MatchValue(
                    value=normalized_number
                ),
            ),
            FieldCondition(
                key=f"{METADATA_PREFIX}.document_id",
                match=MatchAny(
                    any=effective_document_ids
                ),
            ),
            FieldCondition(
                key=(
                    f"{METADATA_PREFIX}."
                    "heading_only_chunk"
                ),
                match=MatchValue(
                    value=False
                ),
            ),
        ]
    )


def build_articles_filter(
    article_numbers: list[str],
    document_ids: list[str] | None = None,
) -> Filter:
    """Build a multi-Article Constitution filter."""

    normalized_numbers = normalize_string_list(
        [
            normalize_provision_number(number)
            or ""
            for number in article_numbers
        ]
    )

    if not normalized_numbers:
        raise ValueError(
            "article_numbers cannot be empty."
        )

    effective_document_ids = (
        normalize_string_list(
            document_ids
        )
        or ["constitution_1973"]
    )

    return Filter(
        must=[
            FieldCondition(
                key=f"{METADATA_PREFIX}.article_number",
                match=MatchAny(
                    any=normalized_numbers
                ),
            ),
            FieldCondition(
                key=f"{METADATA_PREFIX}.document_id",
                match=MatchAny(
                    any=effective_document_ids
                ),
            ),
            FieldCondition(
                key=(
                    f"{METADATA_PREFIX}."
                    "heading_only_chunk"
                ),
                match=MatchValue(
                    value=False
                ),
            ),
        ]
    )


# -------------------------------------------------------------------
# Document validation
# -------------------------------------------------------------------

def is_usable_document(
    document: Document,
) -> bool:
    """Reject empty, heading-only, or bodyless chunks."""

    metadata = document.metadata

    body_present = metadata.get(
        "provision_body_present",
        metadata.get(
            "section_body_present",
            True,
        ),
    )

    return bool(
        document.page_content.strip()
        and not metadata.get(
            "heading_only_chunk",
            False,
        )
        and body_present is not False
    )


def document_matches_route(
    document: Document,
    document_ids: list[str] | None = None,
    provision_type: str | None = None,
    provision_numbers: list[str] | None = None,
) -> bool:
    """Confirm that a returned document matches the requested route."""

    normalized_document_ids = normalize_string_list(
        document_ids
    )

    if (
        normalized_document_ids
        and get_document_id(document)
        not in normalized_document_ids
    ):
        return False

    normalized_type = (
        str(provision_type).strip().lower()
        if provision_type
        else None
    )

    if (
        normalized_type
        and get_document_provision_type(
            document
        )
        != normalized_type
    ):
        return False

    normalized_numbers = {
        number
        for number in (
            normalize_provision_number(value)
            for value in (
                provision_numbers
                or []
            )
        )
        if number
    }

    if (
        normalized_numbers
        and get_document_provision_number(
            document
        )
        not in normalized_numbers
    ):
        return False

    return True


# -------------------------------------------------------------------
# Neighbor provision helpers
# -------------------------------------------------------------------

def parse_numeric_provision(
    provision_number: str,
) -> int | None:
    """Return an integer only for purely numeric provisions."""

    normalized = normalize_provision_number(
        provision_number
    )

    if (
        normalized is None
        or not normalized.isdigit()
    ):
        return None

    return int(normalized)


def parse_numeric_section(
    section_number: str,
) -> int | None:
    """Backward-compatible alias."""

    return parse_numeric_provision(
        section_number
    )


def build_neighbor_provision_numbers(
    provision_number: str,
    radius: int,
) -> list[str]:
    """Return nearby numeric Section or Article numbers."""

    normalized = normalize_provision_number(
        provision_number
    )

    if not normalized:
        return []

    numeric = parse_numeric_provision(
        normalized
    )

    if numeric is None or radius <= 0:
        return [
            normalized
        ]

    start = max(
        1,
        numeric - radius,
    )
    end = numeric + radius

    return [
        str(value)
        for value in range(
            start,
            end + 1,
        )
    ]


def build_neighbor_section_numbers(
    section_number: str,
    radius: int,
) -> list[str]:
    """Backward-compatible alias."""

    return build_neighbor_provision_numbers(
        provision_number=section_number,
        radius=radius,
    )


# -------------------------------------------------------------------
# Vector-store wrapper
# -------------------------------------------------------------------

class AdaptiveRetriever:
    """Thin retrieval wrapper around QdrantVectorStore."""

    def __init__(
        self,
        vector_store: Any,
    ) -> None:
        self.vector_store = vector_store

    def invoke(
        self,
        query: str,
        k: int,
        metadata_filter: Filter | None = None,
    ) -> list[Document]:
        """Run dense semantic retrieval."""

        if not query.strip():
            return []

        return self.vector_store.similarity_search(
            query=query,
            k=k,
            filter=metadata_filter,
        )

    def search_with_scores(
        self,
        query: str,
        k: int,
        metadata_filter: Filter | None = None,
    ) -> list[tuple[Document, float]]:
        """Run retrieval and preserve Qdrant cosine similarity scores."""

        if not query.strip():
            return []

        try:
            results = (
                self.vector_store
                .similarity_search_with_score(
                    query=query,
                    k=k,
                    filter=metadata_filter,
                )
            )

            return [
                (
                    document,
                    float(score),
                )
                for document, score
                in results
            ]

        except (
            AttributeError,
            TypeError,
        ):
            documents = self.invoke(
                query=query,
                k=k,
                metadata_filter=metadata_filter,
            )

            return [
                (
                    document,
                    0.0,
                )
                for document in documents
            ]


# -------------------------------------------------------------------
# Exact and neighbor retrieval
# -------------------------------------------------------------------

def deduplicate_scored_documents(
    items: list[tuple[Document, float]],
) -> list[tuple[Document, float]]:
    """Keep the highest-scoring copy of each returned chunk."""

    best_items: dict[
        str,
        tuple[Document, float],
    ] = {}

    for document, score in items:
        identity = get_document_identity(
            document
        )

        current = best_items.get(
            identity
        )

        if (
            current is None
            or score > current[1]
        ):
            best_items[identity] = (
                document,
                score,
            )

    return sorted(
        best_items.values(),
        key=lambda item: item[1],
        reverse=True,
    )


def retrieve_exact_provision_documents(
    retriever: AdaptiveRetriever,
    question: str,
    provision_numbers: list[str],
    provision_type: str | None,
    document_ids: list[str] | None,
    top_k: int,
) -> list[tuple[Document, float]]:
    """
    Fetch exact provision chunks using generic and legacy metadata keys.
    """

    normalized_numbers = normalize_string_list(
        [
            normalize_provision_number(number)
            or ""
            for number in provision_numbers
        ]
    )

    if not normalized_numbers:
        return []

    filters: list[Filter] = [
        build_provision_filter(
            provision_numbers=normalized_numbers,
            provision_type=provision_type,
            document_ids=document_ids,
        )
    ]

    if provision_type == "article":
        filters.append(
            build_articles_filter(
                article_numbers=normalized_numbers,
                document_ids=document_ids,
            )
        )

    elif provision_type == "section":
        filters.append(
            build_sections_filter(
                section_numbers=normalized_numbers,
                document_ids=document_ids,
            )
        )

    else:
        filters.append(
            build_sections_filter(
                section_numbers=normalized_numbers,
                document_ids=document_ids,
            )
        )
        filters.append(
            build_articles_filter(
                article_numbers=normalized_numbers,
                document_ids=document_ids,
            )
        )

    scored_results: list[
        tuple[Document, float]
    ] = []

    for metadata_filter in filters:
        try:
            results = retriever.search_with_scores(
                query=question,
                k=max(
                    top_k,
                    len(normalized_numbers) * 4,
                ),
                metadata_filter=metadata_filter,
            )
        except Exception:
            continue

        for document, score in results:
            if (
                is_usable_document(document)
                and document_matches_route(
                    document=document,
                    document_ids=document_ids,
                    provision_type=provision_type,
                    provision_numbers=normalized_numbers,
                )
            ):
                scored_results.append(
                    (
                        document,
                        score,
                    )
                )

    return deduplicate_scored_documents(
        scored_results
    )


def retrieve_neighbor_documents(
    retriever: AdaptiveRetriever,
    section_number: str,
    question: str,
    radius: int,
    top_k: int,
    enabled: bool,
    document_ids: list[str] | None = None,
    provision_type: str = "section",
) -> list[tuple[Document, float]]:
    """
    Retrieve nearby Sections or Articles from the same routed document.
    """

    if not enabled:
        return []

    neighboring_numbers = (
        build_neighbor_provision_numbers(
            provision_number=section_number,
            radius=radius,
        )
    )

    if len(neighboring_numbers) <= 1:
        return []

    try:
        return retriever.search_with_scores(
            query=question,
            k=max(
                len(neighboring_numbers) * 2,
                top_k,
            ),
            metadata_filter=build_provision_filter(
                provision_numbers=neighboring_numbers,
                provision_type=provision_type,
                document_ids=document_ids,
            ),
        )

    except Exception:
        return []


# -------------------------------------------------------------------
# Candidate retrieval
# -------------------------------------------------------------------

def get_concept_preferred_sections(
    detected_concepts: list[str],
) -> set[str]:
    """Collect Section hints from the existing concept registry."""

    preferred_sections: set[str] = set()

    for concept_name in detected_concepts:
        concept_data = LEGAL_CONCEPTS.get(
            concept_name,
            {},
        )

        preferred_sections.update(
            normalize_string_list(
                concept_data.get(
                    "preferred_sections",
                    [],
                )
            )
        )

    return preferred_sections


def get_concept_document_ids(
    concept_name: str,
) -> list[str]:
    """Read optional multi-document hints from concept_registry.py."""

    concept_data = LEGAL_CONCEPTS.get(
        concept_name,
        {},
    )

    return normalize_string_list(
        concept_data.get(
            "preferred_documents",
            [],
        )
        or concept_data.get(
            "document_ids",
            [],
        )
    )


def fetch_candidates(
    retriever: AdaptiveRetriever,
    question: str,
    queries: list[str],
    question_type: str,
    section_number: str | None,
    detected_concepts: list[str],
    top_k: int,
    neighbor_radius: int,
    enable_neighbor_retrieval: bool,
    min_relevance_score: float,
    document_ids: list[str] | None = None,
    provision_type: str | None = None,
    provision_numbers: list[str] | None = None,
    article_number: str | None = None,
) -> tuple[
    list[CandidateDocument],
    list[Document],
]:
    """
    Fetch semantic candidates and exact Section/Article hits.

    The original positional arguments are preserved. New routing fields
    are optional so existing callers remain compatible.
    """

    if top_k <= 0:
        raise ValueError(
            "top_k must be greater than zero."
        )

    normalized_document_ids = (
        normalize_string_list(
            document_ids
        )
    )

    explicit_provision_numbers = (
        normalize_string_list(
            provision_numbers
        )
    )

    if section_number:
        normalized_section = (
            normalize_provision_number(
                section_number
            )
        )

        if normalized_section:
            explicit_provision_numbers.append(
                normalized_section
            )

    if article_number:
        normalized_article = (
            normalize_provision_number(
                article_number
            )
        )

        if normalized_article:
            explicit_provision_numbers.append(
                normalized_article
            )

    explicit_provision_numbers = (
        normalize_string_list(
            explicit_provision_numbers
        )
    )

    effective_provision_type = (
        str(provision_type).strip().lower()
        if provision_type
        else None
    )

    if (
        article_number
        and effective_provision_type is None
    ):
        effective_provision_type = "article"

    elif (
        section_number
        and effective_provision_type is None
    ):
        effective_provision_type = "section"

    if (
        effective_provision_type == "article"
        and not normalized_document_ids
    ):
        normalized_document_ids = [
            "constitution_1973"
        ]

    candidate_items: list[
        CandidateDocument
    ] = []
    exact_documents: list[Document] = []

    if explicit_provision_numbers:
        exact_results = (
            retrieve_exact_provision_documents(
                retriever=retriever,
                question=question,
                provision_numbers=(
                    explicit_provision_numbers
                ),
                provision_type=(
                    effective_provision_type
                ),
                document_ids=(
                    normalized_document_ids
                ),
                top_k=top_k,
            )
        )

        exact_documents = [
            document
            for document, _score
            in exact_results
        ]

        if (
            exact_documents
            and question_type
            in {
                "section_lookup",
                "article_lookup",
                "fact_scenario",
            }
        ):
            return (
                [],
                exact_documents,
            )

    semantic_filter = (
        build_document_filter(
            normalized_document_ids
        )
        if normalized_document_ids
        else None
    )

    for query_index, query in enumerate(
        queries
    ):
        scored_documents = (
            retriever.search_with_scores(
                query=query,
                k=top_k,
                metadata_filter=semantic_filter,
            )
        )

        for (
            document,
            relevance_score,
        ) in scored_documents:
            if not is_usable_document(
                document
            ):
                continue

            if (
                min_relevance_score > 0
                and relevance_score
                < min_relevance_score
            ):
                continue

            candidate_items.append(
                CandidateDocument(
                    document=document,
                    relevance_score=float(
                        relevance_score
                    ),
                    query_index=query_index,
                    query_text=query,
                )
            )

    preferred_sections = (
        get_concept_preferred_sections(
            detected_concepts
        )
    )

    for concept_name in detected_concepts:
        concept_document_ids = (
            get_concept_document_ids(
                concept_name
            )
        )

        if normalized_document_ids:
            neighbor_document_ids = (
                normalized_document_ids
            )

        else:
            neighbor_document_ids = (
                concept_document_ids
            )

        for preferred_section in sorted(
            preferred_sections
        ):
            neighbor_results = (
                retrieve_neighbor_documents(
                    retriever=retriever,
                    section_number=(
                        preferred_section
                    ),
                    question=question,
                    radius=neighbor_radius,
                    top_k=top_k,
                    enabled=(
                        enable_neighbor_retrieval
                    ),
                    document_ids=(
                        neighbor_document_ids
                    ),
                    provision_type="section",
                )
            )

            for (
                document,
                relevance_score,
            ) in neighbor_results:
                if not is_usable_document(
                    document
                ):
                    continue

                if (
                    min_relevance_score > 0
                    and relevance_score
                    < min_relevance_score
                ):
                    continue

                candidate_items.append(
                    CandidateDocument(
                        document=document,
                        relevance_score=float(
                            relevance_score
                        ),
                        query_index=len(
                            queries
                        ),
                        query_text=(
                            "neighbor:"
                            f"{preferred_section}"
                        ),
                    )
                )

    return (
        candidate_items,
        exact_documents,
    )
