from __future__ import annotations

import logging
from time import perf_counter
from typing import Any, Iterable

from langchain_core.documents import Document
from qdrant_client.models import (
    FieldCondition,
    Filter,
    MatchAny,
    MatchValue,
)

from rag.concept_registry import LEGAL_CONCEPTS
from rag.legal_reference_normalizer import (
    canonicalize_provision_number,
)
from rag.lexical_retriever import (
    LexicalIndex,
)
from rag.retrieval_errors import (
    MandatoryRetrievalError,
    OptionalRetrievalError,
    RetrievalError,
)
from rag.schemas import CandidateDocument, LegalReference


METADATA_PREFIX = "metadata"
EXACT_PROVISION_SCROLL_LIMIT = 128
logger = logging.getLogger(__name__)


def _get_collection_name(retriever: Any) -> str | None:
    """Return the backing collection name if the retriever exposes one."""

    vector_store = getattr(
        retriever,
        "vector_store",
        None,
    )

    if vector_store is None:
        return getattr(
            retriever,
            "collection_name",
            None,
        )

    return getattr(
        vector_store,
        "collection_name",
        getattr(
            retriever,
            "collection_name",
            None,
        ),
    )


def _raise_mandatory_retrieval_error(
    *,
    operation: str,
    route: str,
    collection: str | None,
    original_exception: Exception,
    elapsed_seconds: float | None = None,
) -> None:
    """Log and raise a mandatory retrieval failure."""

    logger.exception(
        (
            "%s failed for route=%s collection=%s "
            "after %.3fs"
        ),
        operation,
        route,
        collection or "unknown",
        elapsed_seconds or 0.0,
    )

    raise MandatoryRetrievalError(
        f"{operation} failed.",
        operation=operation,
        route=route,
        collection=collection,
        category=type(original_exception).__name__,
        original_exception=original_exception,
    ) from original_exception


def _raise_optional_retrieval_error(
    *,
    operation: str,
    route: str,
    collection: str | None,
    original_exception: Exception,
    elapsed_seconds: float | None = None,
) -> None:
    """Log and raise a supporting retrieval failure."""

    logger.warning(
        (
            "%s failed for route=%s collection=%s "
            "after %.3fs (%s)"
        ),
        operation,
        route,
        collection or "unknown",
        elapsed_seconds or 0.0,
        type(original_exception).__name__,
        exc_info=True,
    )

    raise OptionalRetrievalError(
        f"{operation} failed.",
        operation=operation,
        route=route,
        collection=collection,
        category=type(original_exception).__name__,
        original_exception=original_exception,
    ) from original_exception

def normalize_provision_number(
    value: str | int | None,
) -> str | None:
    """Normalize a Section or Article number for metadata matching."""

    return canonicalize_provision_number(
        value
    )


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


def normalize_legal_reference_path(
    subsection_path: Iterable[str] | None,
) -> list[str]:
    """Normalize a structured subsection or clause path."""

    if subsection_path is None:
        return []

    normalized_path: list[str] = []

    for component in subsection_path:
        normalized = str(component).strip().lower()

        if normalized:
            normalized_path.append(normalized)

    return normalized_path


def get_legal_reference_subsection_path_key(
    subsection_path: Iterable[str] | None,
) -> str:
    """Return the canonical dotted key for a legal reference path."""

    return ".".join(
        normalize_legal_reference_path(
            subsection_path
        )
    )


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


def get_document_base_provision_number(
    document: Document,
) -> str | None:
    """Return a base provision number from new or legacy metadata."""

    metadata = document.metadata

    return normalize_provision_number(
        metadata.get("base_provision_number")
        or metadata.get("provision_number")
        or metadata.get("section_number")
        or metadata.get("article_number")
    )


def get_document_subsection_path_key(
    document: Document,
) -> str:
    """Return the stored dotted subsection path key, if any."""

    metadata = document.metadata
    path_value = metadata.get(
        "subsection_path_key"
    )

    if isinstance(path_value, list):
        return get_legal_reference_subsection_path_key(
            path_value
        )

    return str(
        path_value or ""
    ).strip().lower()


def get_document_component_type(
    document: Document,
) -> str | None:
    """Return the stored child component type, if any."""

    component_type = str(
        document.metadata.get(
            "component_type",
            "",
        )
    ).strip().lower()

    return component_type or None


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


def _safe_int(
    value: Any,
    default: int = 0,
) -> int:
    """Return a safe integer for source-order sorting."""

    try:
        return int(value)
    except (
        TypeError,
        ValueError,
    ):
        return default


def get_document_order_key(
    document: Document,
) -> tuple[Any, ...]:
    """Return a stable source-order key for exact provision chunks."""

    metadata = document.metadata

    preferred_order = metadata.get(
        "document_chunk_number"
    )
    fallback_chunk_order = metadata.get(
        "chunk_number"
    )
    fallback_page = metadata.get(
        "page_start",
        metadata.get(
            "page_number",
            0,
        ),
    )

    return (
        _safe_int(
            preferred_order,
            default=_safe_int(
                fallback_chunk_order,
                default=_safe_int(
                    fallback_page,
                    default=0,
                ),
            ),
        ),
        _safe_int(
            metadata.get(
                "provision_part_number",
                metadata.get(
                    "section_part_number",
                    1,
                ),
            ),
            default=1,
        ),
        _safe_int(
            fallback_chunk_order,
            default=0,
        ),
        _safe_int(
            metadata.get(
                "page_start",
                metadata.get(
                    "page_number",
                    0,
                ),
            ),
            default=0,
        ),
        _safe_int(
            metadata.get(
                "page_end",
                metadata.get(
                    "page_number",
                    0,
                ),
            ),
            default=0,
        ),
        str(
            metadata.get(
                "chunk_id",
                "",
            )
        ),
    )

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


def get_document_provision_ordinal(
    document: Document,
) -> int | None:
    """Return the stored source-order provision ordinal, if present."""

    raw_value = document.metadata.get(
        "provision_ordinal"
    )

    try:
        ordinal = int(raw_value)
    except (
        TypeError,
        ValueError,
    ):
        return None

    return ordinal if ordinal > 0 else None


def build_provision_ordinal_filter(
    provision_ordinals: list[int],
    provision_type: str | None = None,
    document_ids: list[str] | None = None,
) -> Filter:
    """Build a source-order provision filter keyed by ordinal."""

    normalized_ordinals = [
        ordinal
        for ordinal in (
            int(value)
            for value in provision_ordinals
        )
        if ordinal > 0
    ]

    if not normalized_ordinals:
        raise ValueError(
            "provision_ordinals cannot be empty."
        )

    conditions: list[FieldCondition] = [
        FieldCondition(
            key=f"{METADATA_PREFIX}.provision_ordinal",
            match=MatchAny(
                any=normalized_ordinals
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

    normalized_type = (
        str(provision_type).strip().lower()
        if provision_type
        else None
    )

    if normalized_type:
        conditions.append(
            FieldCondition(
                key=f"{METADATA_PREFIX}.provision_type",
                match=MatchValue(
                    value=normalized_type
                ),
            )
        )

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


def build_child_provision_filter(
    reference: LegalReference,
    document_ids: list[str] | None = None,
) -> Filter:
    """Build an exact filter for one structured subsection reference."""

    provision_type = str(
        reference.provision_type
    ).strip().lower()

    if provision_type not in {
        "section",
        "article",
    }:
        raise ValueError(
            "reference.provision_type must be 'section' or 'article'."
        )

    base_provision_number = (
        normalize_provision_number(
            reference.base_number
        )
    )

    if not base_provision_number:
        raise ValueError(
            "reference.base_number cannot be empty."
        )

    subsection_path_key = (
        get_legal_reference_subsection_path_key(
            reference.subsection_path
        )
    )

    if not subsection_path_key:
        raise ValueError(
            "reference.subsection_path cannot be empty."
        )

    conditions: list[FieldCondition] = [
        FieldCondition(
            key=f"{METADATA_PREFIX}.provision_type",
            match=MatchValue(
                value=provision_type
            ),
        ),
        FieldCondition(
            key=(
                f"{METADATA_PREFIX}."
                "base_provision_number"
            ),
            match=MatchValue(
                value=base_provision_number
            ),
        ),
        FieldCondition(
            key=(
                f"{METADATA_PREFIX}."
                "subsection_path_key"
            ),
            match=MatchValue(
                value=subsection_path_key
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

    component_type = str(
        reference.component_type or ""
    ).strip().lower()

    if component_type:
        conditions.append(
            FieldCondition(
                key=(
                    f"{METADATA_PREFIX}."
                    "component_type"
                ),
                match=MatchValue(
                    value=component_type
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

class AdaptiveRetriever:
    """Thin retrieval wrapper around QdrantVectorStore."""

    def __init__(
        self,
        vector_store: Any,
    ) -> None:
        self.vector_store = vector_store
        self._lexical_index: LexicalIndex | None = None
        self.supports_similarity_search_with_score = callable(
            getattr(
                vector_store,
                "similarity_search_with_score",
                None,
            )
        )

        if not self.supports_similarity_search_with_score:
            logger.warning(
                "Vector store does not expose similarity_search_with_score; "
                "score-aware fallback will return None scores."
            )

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
    ) -> list[tuple[Document, float | None]]:
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
            logger.warning(
                "Vector score retrieval unavailable; falling back to "
                "rank-preserving unscored retrieval."
            )
            documents = self.invoke(
                query=query,
                k=k,
                metadata_filter=metadata_filter,
            )

            return [
                (
                    document,
                    None,
                )
                for document in documents
            ]

    def _load_lexical_documents(self) -> list[Document]:
        """Load all usable Qdrant chunks for the cached BM25 index."""

        client = getattr(
            self.vector_store,
            "client",
            None,
        )
        collection_name = getattr(
            self.vector_store,
            "collection_name",
            "",
        )

        if client is None or not collection_name:
            return []

        documents: list[Document] = []
        offset: Any = None

        while True:
            scroll_result = client.scroll(
                collection_name=collection_name,
                scroll_filter=None,
                limit=256,
                offset=offset,
                with_payload=True,
                with_vectors=False,
            )

            if isinstance(
                scroll_result,
                tuple,
            ):
                points, next_offset = scroll_result
            else:
                points = getattr(
                    scroll_result,
                    "points",
                    [],
                )
                next_offset = getattr(
                    scroll_result,
                    "next_page_offset",
                    None,
                )

            if not points:
                break

            for point in points:
                document = self._document_from_qdrant_point(
                    point
                )

                if is_usable_document(document):
                    documents.append(document)

            if next_offset is None:
                break

            offset = next_offset

        return documents

    def _get_lexical_index(self) -> LexicalIndex | None:
        """Build the BM25 corpus lazily and cache it for this retriever."""

        if self._lexical_index is None:
            self._lexical_index = LexicalIndex.from_documents(
                self._load_lexical_documents()
            )

        return self._lexical_index

    def search_lexical(
        self,
        query: str,
        k: int,
        document_ids: list[str] | None = None,
    ) -> list[tuple[Document, float]]:
        """Run deterministic BM25 retrieval over cached chunk payloads."""

        if not query.strip():
            return []

        lexical_index = self._get_lexical_index()

        if lexical_index is None:
            return []

        return lexical_index.search(
            query=query,
            k=k,
            document_ids=document_ids,
        )

    def _document_from_qdrant_point(
        self,
        point: Any,
    ) -> Document:
        """Convert one Qdrant point into the LangChain Document shape."""

        payload = getattr(
            point,
            "payload",
            {},
        ) or {}

        content_payload_key = getattr(
            self.vector_store,
            "content_payload_key",
            "page_content",
        )
        metadata_payload_key = getattr(
            self.vector_store,
            "metadata_payload_key",
            "metadata",
        )

        metadata = dict(
            payload.get(
                metadata_payload_key,
                {},
            )
            or {}
        )
        metadata["_id"] = getattr(
            point,
            "id",
            None,
        )
        metadata["_collection_name"] = getattr(
            self.vector_store,
            "collection_name",
            "",
        )

        return Document(
            page_content=payload.get(
                content_payload_key,
                "",
            ),
            metadata=metadata,
        )

    def scroll_documents(
        self,
        metadata_filter: Filter,
        page_size: int = EXACT_PROVISION_SCROLL_LIMIT,
    ) -> list[Document]:
        """Fetch all payload-matched documents from Qdrant with pagination."""

        client = getattr(
            self.vector_store,
            "client",
            None,
        )
        collection_name = getattr(
            self.vector_store,
            "collection_name",
            "",
        )

        if client is None or not collection_name:
            raise AttributeError(
                "The vector store does not expose a Qdrant client and collection name."
            )

        documents: list[Document] = []
        offset: Any = None

        while True:
            scroll_result = client.scroll(
                collection_name=collection_name,
                scroll_filter=metadata_filter,
                limit=page_size,
                offset=offset,
                with_payload=True,
                with_vectors=False,
            )

            if isinstance(
                scroll_result,
                tuple,
            ):
                points, next_offset = scroll_result
            else:
                points = getattr(
                    scroll_result,
                    "points",
                    [],
                )
                next_offset = getattr(
                    scroll_result,
                    "next_page_offset",
                    None,
                )

            if not points:
                break

            for point in points:
                documents.append(
                    self._document_from_qdrant_point(
                        point
                    )
                )

            if next_offset is None:
                break

            offset = next_offset

        return documents

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


def deduplicate_documents_in_order(
    documents: list[Document],
) -> list[Document]:
    """Keep the first copy of each chunk while preserving source order."""

    ordered_documents: list[Document] = []
    seen_identities: set[str] = set()

    for document in documents:
        identity = get_document_identity(
            document
        )

        if identity in seen_identities:
            continue

        seen_identities.add(identity)
        ordered_documents.append(
            document
        )

    return ordered_documents


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

    normalized_document_ids = normalize_string_list(
        document_ids
    )
    collection_name = _get_collection_name(
        retriever
    )
    route = (
        "provision:"
        f"type={provision_type or 'any'};"
        f"numbers={','.join(normalized_numbers)};"
        f"document_ids={','.join(normalized_document_ids) or 'any'}"
    )

    try:
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
    except Exception as exc:
        _raise_mandatory_retrieval_error(
            operation="build_exact_provision_filter",
            route=route,
            collection=collection_name,
            original_exception=exc,
        )

    exact_documents: list[Document] = []

    for metadata_filter in filters:
        started_at = perf_counter()
        try:
            results = (
                retriever.scroll_documents(
                    metadata_filter=metadata_filter,
                    page_size=(
                        EXACT_PROVISION_SCROLL_LIMIT
                    ),
                )
            )
        except AttributeError as exc:
            try:
                scored_results = (
                    retriever.search_with_scores(
                        query=question,
                        k=max(
                            top_k,
                            len(normalized_numbers)
                            * 4,
                        ),
                        metadata_filter=metadata_filter,
                    )
                )
                results = [
                    document
                    for document, _score
                    in scored_results
                ]
            except Exception as search_exc:
                _raise_mandatory_retrieval_error(
                    operation="search_exact_provision_fallback",
                    route=route,
                    collection=collection_name,
                    original_exception=search_exc,
                    elapsed_seconds=(
                        perf_counter() - started_at
                    ),
                )
        except Exception as exc:
            _raise_mandatory_retrieval_error(
                operation="scroll_exact_provision",
                route=route,
                collection=collection_name,
                original_exception=exc,
                elapsed_seconds=(
                    perf_counter() - started_at
                ),
            )

        matching_documents = [
            document
            for document in results
            if (
                is_usable_document(
                    document
                )
                and document_matches_route(
                    document=document,
                    document_ids=document_ids,
                    provision_type=provision_type,
                    provision_numbers=normalized_numbers,
                )
            )
        ]

        if matching_documents:
            exact_documents.extend(
                matching_documents
            )
            break

    if not exact_documents:
        return []

    ordered_documents = sorted(
        deduplicate_documents_in_order(
            exact_documents
        ),
        key=get_document_order_key,
    )

    return [
        (
            document,
            1.0,
        )
        for document in ordered_documents
    ]


def _document_matches_legal_reference(
    document: Document,
    reference: LegalReference,
    document_ids: list[str] | None = None,
) -> bool:
    """Validate that a returned document matches one structured citation."""

    if not is_usable_document(
        document
    ):
        return False

    normalized_document_ids = normalize_string_list(
        document_ids
    )

    if (
        normalized_document_ids
        and get_document_id(document)
        not in normalized_document_ids
    ):
        return False

    provision_type = str(
        reference.provision_type
    ).strip().lower()

    if (
        get_document_provision_type(
            document
        )
        != provision_type
    ):
        return False

    if (
        get_document_base_provision_number(
            document
        )
        != normalize_provision_number(
            reference.base_number
        )
    ):
        return False

    if (
        get_document_subsection_path_key(
            document
        )
        != get_legal_reference_subsection_path_key(
            reference.subsection_path
        )
    ):
        return False

    expected_component_type = str(
        reference.component_type or ""
    ).strip().lower()
    actual_component_type = (
        get_document_component_type(
            document
        )
    )

    if (
        expected_component_type
        and actual_component_type
        != expected_component_type
    ):
        return False

    return True


def _document_is_parent_provision_chunk(
    document: Document,
) -> bool:
    """Return True when a retrieved document is a parent provision chunk."""

    return not (
        get_document_subsection_path_key(
            document
        )
        or get_document_component_type(
            document
        )
    )


def retrieve_exact_legal_reference_documents(
    retriever: AdaptiveRetriever,
    question: str,
    legal_references: list[LegalReference],
    document_ids: list[str] | None,
    top_k: int,
) -> list[Document]:
    """Fetch exact child references with deterministic parent fallback."""

    exact_documents: list[Document] = []
    collection_name = _get_collection_name(
        retriever
    )

    for reference in legal_references:
        if normalize_legal_reference_path(
            reference.subsection_path
        ):
            route = (
                "child:"
                f"type={reference.provision_type};"
                f"base={normalize_provision_number(reference.base_number) or reference.base_number};"
                f"path={get_legal_reference_subsection_path_key(reference.subsection_path)};"
                f"document_ids={','.join(normalize_string_list(document_ids)) or 'any'}"
            )
            try:
                child_filter = build_child_provision_filter(
                    reference=reference,
                    document_ids=document_ids,
                )
                started_at = perf_counter()
                child_results = (
                    retriever.scroll_documents(
                        metadata_filter=child_filter,
                        page_size=(
                            EXACT_PROVISION_SCROLL_LIMIT
                        ),
                    )
                )
            except Exception as exc:
                _raise_mandatory_retrieval_error(
                    operation="scroll_exact_child",
                    route=route,
                    collection=collection_name,
                    original_exception=exc,
                    elapsed_seconds=(
                        perf_counter() - started_at
                        if "started_at" in locals()
                        else None
                    ),
                )

            matching_children = [
                document
                for document in child_results
                if _document_matches_legal_reference(
                    document=document,
                    reference=reference,
                    document_ids=document_ids,
                )
            ]

            if matching_children:
                exact_documents.extend(
                    deduplicate_documents_in_order(
                        matching_children
                    )
                )
                continue

            subsection_path_key = (
                get_legal_reference_subsection_path_key(
                    reference.subsection_path
                )
            )
            logger.info(
                (
                    "Exact child %s %s(%s) not found; "
                    "falling back to parent provision."
                ),
                reference.provision_type,
                normalize_provision_number(
                    reference.base_number
                ),
                subsection_path_key,
            )

            parent_results = (
                retrieve_exact_provision_documents(
                    retriever=retriever,
                    question=question,
                    provision_numbers=[
                        reference.base_number
                    ],
                    provision_type=reference.provision_type,
                    document_ids=document_ids,
                    top_k=top_k,
                )
            )

            parent_documents = [
                document
                for document, _score
                in parent_results
                if _document_is_parent_provision_chunk(
                    document
                )
            ]

            exact_documents.extend(
                parent_documents
            )
            continue

        parent_results = retrieve_exact_provision_documents(
            retriever=retriever,
            question=question,
            provision_numbers=[
                reference.base_number
            ],
            provision_type=reference.provision_type,
            document_ids=document_ids,
            top_k=top_k,
        )

        exact_documents.extend(
            [
                document
                for document, _score
                in parent_results
            ]
        )

    return deduplicate_documents_in_order(
        exact_documents
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
) -> list[tuple[Document, float | None]]:
    """
    Retrieve nearby Sections or Articles from the same routed document.
    """

    if not enabled:
        return []

    normalized_provision_number = (
        normalize_provision_number(
            section_number
        )
    )

    if not normalized_provision_number:
        return []

    normalized_document_ids = normalize_string_list(
        document_ids
    )
    collection_name = _get_collection_name(
        retriever
    )
    base_route = (
        "neighbor:"
        f"type={str(provision_type).strip().lower() or 'any'};"
        f"number={normalized_provision_number};"
        f"radius={radius};"
        f"document_ids={','.join(normalized_document_ids) or 'any'}"
    )

    try:
        started_at = perf_counter()
        probe_documents = retriever.scroll_documents(
            metadata_filter=build_provision_filter(
                provision_numbers=[
                    normalized_provision_number
                ],
                provision_type=provision_type,
                document_ids=document_ids,
            ),
            page_size=EXACT_PROVISION_SCROLL_LIMIT,
        )
    except Exception as exc:
        _raise_optional_retrieval_error(
            operation="probe_neighbor_provision",
            route=base_route,
            collection=collection_name,
            original_exception=exc,
            elapsed_seconds=(
                perf_counter() - started_at
                if "started_at" in locals()
                else None
            ),
        )

    if not probe_documents:
        numeric_target = parse_numeric_provision(
            normalized_provision_number
        )

        if (
            numeric_target is None
            or radius <= 0
            or not normalized_document_ids
        ):
            return []

        neighboring_numbers = (
            build_neighbor_provision_numbers(
                provision_number=normalized_provision_number,
                radius=radius,
            )
        )

        if len(neighboring_numbers) <= 1:
            return []

        try:
            started_at = perf_counter()
            return retriever.search_with_scores(
                query=question,
                k=max(
                    len(neighboring_numbers) * 2,
                    top_k,
                ),
                metadata_filter=build_provision_filter(
                    provision_numbers=neighboring_numbers,
                    provision_type=provision_type,
                    document_ids=normalized_document_ids,
                ),
            )
        except Exception as exc:
            _raise_optional_retrieval_error(
                operation="search_neighbor_documents",
                route=base_route,
                collection=collection_name,
                original_exception=exc,
                elapsed_seconds=(
                    perf_counter() - started_at
                    if "started_at" in locals()
                    else None
                ),
            )

    grouped_by_document_id: dict[str, list[Document]] = {}

    for document in probe_documents:
        if not is_usable_document(document):
            continue

        document_id = get_document_id(document)

        if (
            normalized_document_ids
            and document_id not in normalized_document_ids
        ):
            continue

        grouped_by_document_id.setdefault(
            document_id,
            [],
        ).append(document)

    if not grouped_by_document_id:
        return []

    ordered_results: list[tuple[Document, float | None]] = []

    for document_id, documents in grouped_by_document_id.items():
        ordinals = sorted(
            {
                ordinal
                for ordinal in (
                    get_document_provision_ordinal(
                        document
                    )
                    for document in documents
                )
                if ordinal is not None
            }
        )

        if ordinals:
            target_ordinal = ordinals[0]

            if len(ordinals) != 1:
                _raise_optional_retrieval_error(
                    operation="resolve_neighbor_ordinal",
                    route=(
                        f"{base_route};document_id={document_id}"
                    ),
                    collection=collection_name,
                    original_exception=ValueError(
                        "inconsistent provision ordinals"
                    ),
                )

            if radius <= 0:
                continue

            neighbor_ordinals = list(
                range(
                    max(
                        1,
                        target_ordinal - radius,
                    ),
                    target_ordinal + radius + 1,
                )
            )

            try:
                started_at = perf_counter()
                scored_neighbors = (
                    retriever.search_with_scores(
                        query=question,
                        k=max(
                            len(neighbor_ordinals) * 2,
                            top_k,
                        ),
                        metadata_filter=build_provision_ordinal_filter(
                            provision_ordinals=neighbor_ordinals,
                            provision_type=provision_type,
                            document_ids=[document_id],
                        ),
                    )
                )
            except Exception as exc:
                _raise_optional_retrieval_error(
                    operation="search_neighbor_documents",
                    route=(
                        f"{base_route};document_id={document_id};"
                        f"ordinal={target_ordinal}"
                    ),
                    collection=collection_name,
                    original_exception=exc,
                    elapsed_seconds=(
                        perf_counter() - started_at
                        if "started_at" in locals()
                        else None
                    ),
                )

            filtered_neighbors: list[tuple[Document, float | None]] = []
            seen_identities: set[str] = set()

            for document, relevance_score in scored_neighbors:
                if not is_usable_document(document):
                    continue

                if get_document_id(document) != document_id:
                    continue

                document_ordinal = (
                    get_document_provision_ordinal(
                        document
                    )
                )

                if document_ordinal not in neighbor_ordinals:
                    continue

                document_identity = get_document_identity(
                    document
                )

                if document_identity in seen_identities:
                    continue

                seen_identities.add(document_identity)
                filtered_neighbors.append(
                    (
                        document,
                        relevance_score,
                    )
                )

            filtered_neighbors.sort(
                key=lambda item: (
                    get_document_provision_ordinal(
                        item[0]
                    )
                    or 0,
                    get_document_order_key(
                        item[0]
                    ),
                )
            )
            ordered_results.extend(
                filtered_neighbors
            )
            continue

        numeric_target = parse_numeric_provision(
            normalized_provision_number
        )

        if numeric_target is None or radius <= 0:
            continue

        neighboring_numbers = (
            build_neighbor_provision_numbers(
                provision_number=normalized_provision_number,
                radius=radius,
            )
        )

        if len(neighboring_numbers) <= 1:
            continue

        try:
            started_at = perf_counter()
            scored_neighbors = retriever.search_with_scores(
                query=question,
                k=max(
                    len(neighboring_numbers) * 2,
                    top_k,
                ),
                metadata_filter=build_provision_filter(
                    provision_numbers=neighboring_numbers,
                    provision_type=provision_type,
                    document_ids=[document_id],
                ),
            )
        except Exception as exc:
            _raise_optional_retrieval_error(
                operation="search_neighbor_documents",
                route=(
                    f"{base_route};document_id={document_id}"
                ),
                collection=collection_name,
                original_exception=exc,
                elapsed_seconds=(
                    perf_counter() - started_at
                    if "started_at" in locals()
                    else None
                ),
            )

        filtered_neighbors = []
        seen_identities = set()

        for document, relevance_score in scored_neighbors:
            if not is_usable_document(document):
                continue

            if get_document_id(document) != document_id:
                continue

            document_number = get_document_provision_number(
                document
            )

            if document_number not in neighboring_numbers:
                continue

            document_identity = get_document_identity(
                document
            )

            if document_identity in seen_identities:
                continue

            seen_identities.add(document_identity)
            filtered_neighbors.append(
                (
                    document,
                    relevance_score,
                )
            )

        filtered_neighbors.sort(
            key=lambda item: get_document_order_key(
                item[0]
            )
        )
        ordered_results.extend(
            filtered_neighbors
        )

    return ordered_results

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


def _resolve_concept_neighbor_document_ids(
    *,
    concept_name: str,
    normalized_document_ids: list[str],
) -> list[str] | None:
    """Resolve the document scope for one concept-specific neighbor route."""

    concept_document_ids = (
        get_concept_document_ids(
            concept_name
        )
    )

    if normalized_document_ids:
        if concept_document_ids:
            compatible_document_ids = [
                document_id
                for document_id in normalized_document_ids
                if document_id in concept_document_ids
            ]

            return (
                compatible_document_ids
                or None
            )

        return normalized_document_ids

    if concept_document_ids:
        return concept_document_ids

    return None


def _neighbor_route_key(
    *,
    document_ids: list[str] | None,
    provision_type: str,
    preferred_provision_number: str,
) -> tuple[Any, ...]:
    """Build a stable identity for a concept neighbor route."""

    normalized_document_ids = (
        tuple(
            sorted(
                normalize_string_list(
                    document_ids
                )
            )
        )
        if document_ids
        else None
    )

    return (
        normalized_document_ids,
        provision_type.strip().lower(),
        normalize_provision_number(
            preferred_provision_number
        )
        or str(preferred_provision_number).strip(),
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
    legal_references: list[LegalReference] | None = None,
    retrieval_issues: list[RetrievalError] | None = None,
    retrieval_trace: dict[str, Any] | None = None,
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
    trace_events = (
        retrieval_trace.setdefault(
            "events",
            [],
        )
        if retrieval_trace is not None
        else None
    )

    def record_event(
        *,
        channel: str,
        route_id: str,
        started_at: float,
        status: str,
        accepted_count: int = 0,
        raw_count: int = 0,
        query_index: int | None = None,
        query_text: str | None = None,
        document_ids: list[str] | None = None,
        provision_type: str | None = None,
        preferred_provision_number: str | None = None,
        concept_name: str | None = None,
        error: Exception | None = None,
    ) -> None:
        if trace_events is None:
            return

        event: dict[str, Any] = {
            "channel": channel,
            "route_id": route_id,
            "status": status,
            "latency_ms": (perf_counter() - started_at) * 1000.0,
            "accepted_count": accepted_count,
            "raw_count": raw_count,
        }

        if query_index is not None:
            event["query_index"] = query_index

        if query_text is not None:
            event["query_text"] = query_text

        if document_ids is not None:
            event["document_ids"] = document_ids

        if provision_type is not None:
            event["provision_type"] = provision_type

        if preferred_provision_number is not None:
            event["preferred_provision_number"] = preferred_provision_number

        if concept_name is not None:
            event["concept_name"] = concept_name

        if error is not None:
            event["error_type"] = type(error).__name__
            event["error_message"] = str(error)

        trace_events.append(event)

    normalized_legal_references = list(
        legal_references or []
    )

    if normalized_legal_references:
        exact_started = perf_counter()
        try:
            exact_documents = (
                retrieve_exact_legal_reference_documents(
                    retriever=retriever,
                    question=question,
                    legal_references=(
                        normalized_legal_references
                    ),
                    document_ids=(
                        normalized_document_ids
                    ),
                    top_k=top_k,
                )
            )
        except MandatoryRetrievalError as exc:
            record_event(
                channel="exact",
                route_id="exact:legal_references",
                started_at=exact_started,
                status="error",
                accepted_count=0,
                raw_count=0,
                error=exc,
            )
            raise
        record_event(
            channel="exact",
            route_id="exact:legal_references",
            started_at=exact_started,
            status="success",
            accepted_count=len(exact_documents),
            raw_count=len(exact_documents),
        )

    elif explicit_provision_numbers:
        exact_started = perf_counter()
        try:
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
        except MandatoryRetrievalError as exc:
            record_event(
                channel="exact",
                route_id=(
                    "exact:"
                    f"{effective_provision_type or 'provision'}"
                ),
                started_at=exact_started,
                status="error",
                accepted_count=0,
                raw_count=0,
                document_ids=normalized_document_ids,
                provision_type=effective_provision_type,
                error=exc,
            )
            raise

        exact_documents = [
            document
            for document, _score
            in exact_results
        ]
        record_event(
            channel="exact",
            route_id=(
                "exact:"
                f"{effective_provision_type or 'provision'}"
            ),
            started_at=exact_started,
            status="success",
            accepted_count=len(exact_documents),
            raw_count=len(exact_results),
            document_ids=normalized_document_ids,
            provision_type=effective_provision_type,
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
        vector_started = perf_counter()
        vector_accepted = 0
        scored_documents = (
            retriever.search_with_scores(
                query=query,
                k=top_k,
                metadata_filter=semantic_filter,
            )
        )

        for retrieval_rank, (
            document,
            relevance_score,
        ) in enumerate(
            scored_documents,
            start=1,
        ):
            if not is_usable_document(
                document
            ):
                continue

            if (
                min_relevance_score > 0
                and relevance_score is not None
                and relevance_score < min_relevance_score
            ):
                continue

            candidate_items.append(
                CandidateDocument(
                    document=document,
                    relevance_score=relevance_score,
                    query_index=query_index,
                    query_text=query,
                    retrieval_method="vector",
                    retrieval_rank=retrieval_rank,
                )
            )
            vector_accepted += 1

        lexical_search = getattr(
            retriever,
            "search_lexical",
            None,
        )

        lexical_started = perf_counter()
        lexical_accepted = 0
        if callable(lexical_search):
            lexical_documents = lexical_search(
                query=query,
                k=top_k,
                document_ids=normalized_document_ids
                or None,
            )
        else:
            lexical_documents = []

        for document, lexical_score in lexical_documents:
            if not is_usable_document(
                document
            ):
                continue

            candidate_items.append(
                CandidateDocument(
                    document=document,
                    relevance_score=float(
                        lexical_score
                    ),
                    query_index=query_index,
                    query_text=query,
                    retrieval_method="lexical",
                )
            )
            lexical_accepted += 1

        record_event(
            channel="vector",
            route_id=f"vector:{query_index}:{query}",
            started_at=vector_started,
            status="success",
            accepted_count=vector_accepted,
            raw_count=len(scored_documents),
            query_index=query_index,
            query_text=query,
            document_ids=normalized_document_ids,
        )

        record_event(
            channel="lexical",
            route_id=f"lexical:{query_index}:{query}",
            started_at=lexical_started,
            status="success",
            accepted_count=lexical_accepted,
            raw_count=len(lexical_documents),
            query_index=query_index,
            query_text=query,
            document_ids=normalized_document_ids,
        )

    seen_neighbor_routes: set[tuple[Any, ...]] = set()

    for concept_name in detected_concepts:
        concept_sections = sorted(
            get_concept_preferred_sections(
                [concept_name]
            )
        )

        if not concept_sections:
            continue

        neighbor_document_ids = (
            _resolve_concept_neighbor_document_ids(
                concept_name=concept_name,
                normalized_document_ids=(
                    normalized_document_ids
                ),
            )
        )

        if (
            normalized_document_ids
            and neighbor_document_ids is None
        ):
            continue

        for preferred_section in concept_sections:
            route_key = _neighbor_route_key(
                document_ids=neighbor_document_ids,
                provision_type="section",
                preferred_provision_number=(
                    preferred_section
                ),
            )

            if route_key in seen_neighbor_routes:
                continue

            seen_neighbor_routes.add(
                route_key
            )

            neighbor_started = perf_counter()
            neighbor_accepted = 0
            try:
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
                        document_ids=neighbor_document_ids,
                        provision_type="section",
                    )
                )
            except OptionalRetrievalError as exc:
                if retrieval_issues is not None:
                    retrieval_issues.append(exc)
                record_event(
                    channel="neighbor",
                    route_id=(
                        "neighbor:"
                        f"{neighbor_document_ids or 'all'}:"
                        f"section:{preferred_section}"
                    ),
                    started_at=neighbor_started,
                    status="error",
                    accepted_count=0,
                    raw_count=0,
                    document_ids=neighbor_document_ids,
                    provision_type="section",
                    preferred_provision_number=preferred_section,
                    concept_name=concept_name,
                    error=exc,
                )
                continue

            for retrieval_rank, (
                document,
                relevance_score,
            ) in enumerate(
                neighbor_results,
                start=1,
            ):
                if not is_usable_document(
                    document
                ):
                    continue

                if (
                    min_relevance_score > 0
                    and relevance_score is not None
                    and relevance_score < min_relevance_score
                ):
                    continue

                candidate_items.append(
                    CandidateDocument(
                        document=document,
                        relevance_score=relevance_score,
                        query_index=len(
                            queries
                        ),
                        query_text=(
                            "neighbor:"
                            f"{preferred_section}"
                        ),
                        retrieval_method="neighbor",
                        retrieval_rank=retrieval_rank,
                    )
                )
                neighbor_accepted += 1

            record_event(
                channel="neighbor",
                route_id=(
                    "neighbor:"
                    f"{neighbor_document_ids or 'all'}:"
                    f"section:{preferred_section}"
                ),
                started_at=neighbor_started,
                status="success",
                accepted_count=neighbor_accepted,
                raw_count=len(neighbor_results),
                document_ids=neighbor_document_ids,
                provision_type="section",
                preferred_provision_number=preferred_section,
                concept_name=concept_name,
            )

    return (
        candidate_items,
        exact_documents,
    )
