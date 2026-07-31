from typing import Any

from langchain_core.documents import Document
from qdrant_client.models import (
    FieldCondition,
    Filter,
    MatchAny,
    MatchValue,
)

from rag.concept_registry import LEGAL_CONCEPTS
from rag.schemas import CandidateDocument


def build_section_filter(section_number: str) -> Filter:
    return Filter(
        must=[
            FieldCondition(
                key="metadata.section_number",
                match=MatchValue(value=section_number),
            ),
            FieldCondition(
                key="metadata.heading_only_chunk",
                match=MatchValue(value=False),
            ),
            FieldCondition(
                key="metadata.section_body_present",
                match=MatchValue(value=True),
            ),
        ]
    )


def build_sections_filter(section_numbers: list[str]) -> Filter:
    return Filter(
        must=[
            FieldCondition(
                key="metadata.section_number",
                match=MatchAny(any=section_numbers),
            ),
            FieldCondition(
                key="metadata.heading_only_chunk",
                match=MatchValue(value=False),
            ),
            FieldCondition(
                key="metadata.section_body_present",
                match=MatchValue(value=True),
            ),
        ]
    )


def is_usable_document(document: Document) -> bool:
    metadata = document.metadata

    return bool(
        document.page_content.strip()
        and not metadata.get("heading_only_chunk", False)
        and metadata.get("section_body_present", True)
    )


def parse_numeric_section(section_number: str) -> int | None:
    if not section_number:
        return None

    stripped = section_number.strip()

    if stripped.isdigit():
        return int(stripped)

    return None


def build_neighbor_section_numbers(section_number: str, radius: int) -> list[str]:
    numeric = parse_numeric_section(section_number)

    if numeric is None:
        return [section_number]

    start = max(1, numeric - radius)
    end = numeric + radius

    return [str(value) for value in range(start, end + 1)]


class AdaptiveRetriever:
    """
    Thin retrieval wrapper around the vector store.
    """

    def __init__(self, vector_store) -> None:
        self.vector_store = vector_store

    def invoke(
        self,
        query: str,
        k: int,
        metadata_filter: Filter | None = None,
    ) -> list[Document]:
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
        try:
            results = self.vector_store.similarity_search_with_score(
                query=query,
                k=k,
                filter=metadata_filter,
            )
            return [(document, float(score)) for document, score in results]
        except (AttributeError, TypeError):
            return [(document, 0.0) for document in self.invoke(query, k, metadata_filter)]


def retrieve_neighbor_documents(
    retriever: AdaptiveRetriever,
    section_number: str,
    question: str,
    radius: int,
    top_k: int,
    enabled: bool,
) -> list[tuple[Document, float]]:
    if not enabled:
        return []

    neighboring_sections = build_neighbor_section_numbers(
        section_number=section_number,
        radius=radius,
    )

    if len(neighboring_sections) <= 1:
        return []

    try:
        return retriever.search_with_scores(
            query=question,
            k=max(len(neighboring_sections) * 2, top_k),
            metadata_filter=build_sections_filter(neighboring_sections),
        )
    except Exception:
        return []


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
) -> tuple[list[CandidateDocument], list[Document]]:
    """
    Fetch raw candidate chunks and any exact section hits.
    """

    candidate_items: list[CandidateDocument] = []
    exact_documents: list[Document] = []

    if section_number:
        exact_results = retriever.search_with_scores(
            query=queries[0],
            k=top_k,
            metadata_filter=build_section_filter(section_number),
        )

        for document, score in exact_results:
            if (
                is_usable_document(document)
                and str(document.metadata.get("section_number", "")).upper()
                == section_number.upper()
                and (min_relevance_score <= 0 or score >= min_relevance_score)
            ):
                exact_documents.append(document)

        if exact_documents:
            return [], exact_documents

    for query_index, query in enumerate(queries):
        scored_documents = retriever.search_with_scores(
            query=query,
            k=top_k,
        )

        for document, relevance_score in scored_documents:
            if not is_usable_document(document):
                continue

            if min_relevance_score > 0 and relevance_score < min_relevance_score:
                continue

            candidate_items.append(
                CandidateDocument(
                    document=document,
                    relevance_score=float(relevance_score),
                    query_index=query_index,
                    query_text=query,
                )
            )

    preferred_sections: set[str] = set()
    for concept_name in detected_concepts:
        preferred_sections.update(
            LEGAL_CONCEPTS[concept_name]["preferred_sections"]
        )

    for preferred_section in sorted(preferred_sections):
        for document, relevance_score in retrieve_neighbor_documents(
            retriever=retriever,
            section_number=preferred_section,
            question=question,
            radius=neighbor_radius,
            top_k=top_k,
            enabled=enable_neighbor_retrieval,
        ):
            if not is_usable_document(document):
                continue

            candidate_items.append(
                CandidateDocument(
                    document=document,
                    relevance_score=float(relevance_score),
                    query_index=len(queries),
                    query_text=f"neighbor:{preferred_section}",
                )
            )

    return candidate_items, []
