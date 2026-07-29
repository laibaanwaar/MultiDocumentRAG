import hashlib
import os
import time
from collections import Counter
from uuid import NAMESPACE_URL, uuid5

from dotenv import load_dotenv
from langchain_core.documents import Document

from rag.document_loader import load_all_documents
from rag.text_splitter import create_chunks
from rag.vector_store import (
    COLLECTION_NAME,
    create_vector_store,
    get_collection_points_count,
)
from rag.embeddings import get_embedding_dimension


load_dotenv()


# -------------------------------------------------------------------
# Ingestion configuration
# -------------------------------------------------------------------

BATCH_SIZE = int(
    os.getenv(
        "INGEST_BATCH_SIZE",
        "20",
    )
)

DELAY_BETWEEN_BATCHES = int(
    os.getenv(
        "INGEST_BATCH_DELAY_SECONDS",
        "15",
    )
)

MAX_RETRIES = int(
    os.getenv(
        "INGEST_MAX_RETRIES",
        "8",
    )
)

MAX_RETRY_WAIT_SECONDS = int(
    os.getenv(
        "INGEST_MAX_RETRY_WAIT_SECONDS",
        "120",
    )
)

FAIL_ON_SUSPICIOUS_CHUNK = (
    os.getenv(
        "FAIL_ON_SUSPICIOUS_CHUNK",
        "False",
    ).lower()
    == "true"
)

ALLOW_UNSECTIONED_CHUNKS = (
    os.getenv(
        "ALLOW_UNSECTIONED_CHUNKS",
        "True",
    ).lower()
    == "true"
)

ALLOW_HEADING_ONLY_CHUNKS = (
    os.getenv(
        "ALLOW_HEADING_ONLY_CHUNKS",
        "False",
    ).lower()
    == "true"
)


RESET_COLLECTION = (
    os.getenv(
        "INGEST_RESET_COLLECTION",
        "False",
    ).strip().lower()
    in {"1", "true", "yes", "on"}
)

SCROLL_BATCH_SIZE = int(
    os.getenv(
        "INGEST_SCROLL_BATCH_SIZE",
        "256",
    )
)


# -------------------------------------------------------------------
# Validation helpers
# -------------------------------------------------------------------

def validate_ingestion_settings() -> None:
    """Validate ingestion environment settings."""

    if BATCH_SIZE <= 0:
        raise ValueError(
            "INGEST_BATCH_SIZE must be greater than zero."
        )

    if DELAY_BETWEEN_BATCHES < 0:
        raise ValueError(
            "INGEST_BATCH_DELAY_SECONDS cannot be negative."
        )

    if MAX_RETRIES <= 0:
        raise ValueError(
            "INGEST_MAX_RETRIES must be greater than zero."
        )

    if MAX_RETRY_WAIT_SECONDS <= 0:
        raise ValueError(
            "INGEST_MAX_RETRY_WAIT_SECONDS must be greater than zero."
        )


    if SCROLL_BATCH_SIZE <= 0:
        raise ValueError(
            "INGEST_SCROLL_BATCH_SIZE must be greater than zero."
        )


def is_section_chunk(chunk: Document) -> bool:
    """
    Return True when a chunk represents a legal section rather than
    introductory or structural text.
    """

    return not bool(
        chunk.metadata.get(
            "is_unsectioned_chunk",
            False,
        )
    )


def validate_chunk_metadata(
    chunk: Document,
    chunk_index: int,
) -> list[str]:
    """
    Validate metadata required for reliable retrieval and citations.

    Returns:
        A list of validation error messages.
    """

    errors: list[str] = []
    metadata = chunk.metadata

    if not chunk.page_content.strip():
        errors.append("empty page_content")

    required_document_fields = (
        "document_id",
        "document_name",
        "document_title",
        "document_type",
        "chunk_id",
        "document_chunk_number",
    )

    for field_name in required_document_fields:
        if not metadata.get(field_name):
            errors.append(
                f"missing {field_name}"
            )

    if is_section_chunk(chunk):
        section_number = metadata.get(
            "section_number"
        )

        if not section_number:
            errors.append(
                "missing section_number"
            )

        if not metadata.get(
            "section_identity"
        ):
            errors.append(
                "missing section_identity"
            )

        if metadata.get("page_start") is None:
            errors.append(
                "missing page_start"
            )

        if metadata.get("page_end") is None:
            errors.append(
                "missing page_end"
            )

        source_pages = metadata.get(
            "source_pages"
        )

        if not source_pages:
            errors.append(
                "missing source_pages"
            )

    elif not ALLOW_UNSECTIONED_CHUNKS:
        errors.append(
            "unsectioned chunks are disabled"
        )

    if (
        metadata.get(
            "heading_only_chunk",
            False,
        )
        and not ALLOW_HEADING_ONLY_CHUNKS
    ):
        errors.append(
            "heading-only chunk is not allowed"
        )

    if (
        metadata.get(
            "section_body_present"
        )
        is False
        and is_section_chunk(chunk)
    ):
        errors.append(
            "section body is missing"
        )

    if (
        FAIL_ON_SUSPICIOUS_CHUNK
        and metadata.get(
            "page_quality_suspicious",
            False,
        )
    ):
        errors.append(
            "chunk originates from a suspicious page"
        )

    if metadata.get("chunk_number") is None:
        errors.append(
            "missing chunk_number"
        )

    if metadata.get("chunk_length") is None:
        errors.append(
            "missing chunk_length"
        )

    if errors:
        return [
            f"Chunk {chunk_index}: {error}"
            for error in errors
        ]

    return []


def validate_chunks(
    chunks: list[Document],
) -> None:
    """
    Validate all chunks before resetting Qdrant.

    Qdrant is not modified unless every chunk passes validation.
    """

    if not chunks:
        raise ValueError(
            "No chunks were created. Ingestion cannot continue."
        )

    validation_errors: list[str] = []

    for index, chunk in enumerate(
        chunks,
        start=1,
    ):
        validation_errors.extend(
            validate_chunk_metadata(
                chunk=chunk,
                chunk_index=index,
            )
        )

    if validation_errors:
        preview = "\n".join(
            f"- {error}"
            for error in validation_errors[:25]
        )

        remaining = (
            len(validation_errors) - 25
        )

        if remaining > 0:
            preview += (
                f"\n- ...and {remaining} more validation errors"
            )

        raise RuntimeError(
            "Chunk validation failed before Qdrant reset:\n"
            f"{preview}"
        )


def validate_unique_chunk_ids(
    chunks: list[Document],
) -> list[str]:
    """
    Create deterministic IDs and ensure no two distinct chunks produce
    the same Qdrant point ID.
    """

    chunk_ids: list[str] = []
    seen_ids: dict[str, int] = {}

    for index, chunk in enumerate(
        chunks,
        start=1,
    ):
        chunk_id = create_deterministic_id(
            chunk
        )

        if chunk_id in seen_ids:
            first_index = seen_ids[
                chunk_id
            ]

            raise RuntimeError(
                "Duplicate deterministic Qdrant ID detected. "
                f"Chunks {first_index} and {index} generated "
                f"the same ID: {chunk_id}"
            )

        seen_ids[chunk_id] = index
        chunk_ids.append(chunk_id)

    return chunk_ids


# -------------------------------------------------------------------
# Stable Qdrant point IDs
# -------------------------------------------------------------------

def create_deterministic_id(
    chunk: Document,
) -> str:
    """
    Create a stable Qdrant ID from document, section, page, part,
    chunk number, and content.

    The same chunk always receives the same ID, so retries are
    idempotent and cannot create duplicate points.
    """

    metadata = chunk.metadata

    source_pages = metadata.get(
        "source_pages",
        [],
    )

    if isinstance(source_pages, list):
        source_pages_value = ",".join(
            str(page)
            for page in source_pages
        )
    else:
        source_pages_value = str(
            source_pages
        )

    identity_parts = [
        str(
            metadata.get(
                "document_id",
                "",
            )
        ),
        str(
            metadata.get(
                "document_name",
                "",
            )
        ),
        str(
            metadata.get(
                "document_type",
                "",
            )
        ),
        str(
            metadata.get(
                "section_identity",
                "",
            )
        ),
        str(
            metadata.get(
                "section_number",
                "",
            )
        ),
        str(
            metadata.get(
                "section_part_number",
                1,
            )
        ),
        str(
            metadata.get(
                "section_part_count",
                1,
            )
        ),
        str(
            metadata.get(
                "document_chunk_number",
                "",
            )
        ),
        str(
            metadata.get(
                "chunk_id",
                "",
            )
        ),
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
        source_pages_value,
        chunk.page_content.strip(),
    ]

    identity = "|".join(
        identity_parts
    )

    content_hash = hashlib.sha256(
        identity.encode("utf-8")
    ).hexdigest()

    return str(
        uuid5(
            NAMESPACE_URL,
            content_hash,
        )
    )


# -------------------------------------------------------------------
# Qdrant helpers
# -------------------------------------------------------------------

def get_points_count(client) -> int:
    """
    Return the exact number of points in the collection.

    Uses the vector_store helper when available.
    """

    return int(
        get_collection_points_count(
            client
        )
    )


def ensure_collection_is_empty(
    client,
) -> None:
    """Verify that reset=True created a truly empty collection."""

    initial_points = get_points_count(
        client
    )

    print(
        "Points present immediately after reset: "
        f"{initial_points}"
    )

    if initial_points != 0:
        raise RuntimeError(
            "The Qdrant collection was not cleared. "
            f"It still contains {initial_points} points."
        )



def get_existing_point_ids(
    client,
) -> set[str]:
    """
    Read every stored point ID from the current Qdrant collection.

    This enables safe resume ingestion without regenerating embeddings
    for chunks that are already stored.
    """

    existing_ids: set[str] = set()
    offset = None

    while True:
        points, next_offset = client.scroll(
            collection_name=COLLECTION_NAME,
            limit=SCROLL_BATCH_SIZE,
            offset=offset,
            with_payload=False,
            with_vectors=False,
        )

        for point in points:
            existing_ids.add(
                str(point.id)
            )

        if next_offset is None:
            break

        offset = next_offset

    return existing_ids


def validate_existing_collection(
    expected_ids: list[str],
    existing_ids: set[str],
) -> None:
    """
    Ensure the existing collection belongs to the current ingestion set.
    """

    expected_id_set = set(
        expected_ids
    )

    unexpected_ids = (
        existing_ids - expected_id_set
    )

    if unexpected_ids:
        preview = ", ".join(
            sorted(unexpected_ids)[:5]
        )

        raise RuntimeError(
            "The current Qdrant collection contains point IDs that do "
            "not belong to the current document/chunk set. "
            "Use INGEST_RESET_COLLECTION=True for a clean rebuild. "
            f"Unexpected ID examples: {preview}"
        )


def build_pending_ingestion_items(
    chunks: list[Document],
    chunk_ids: list[str],
    existing_ids: set[str],
) -> list[tuple[Document, str]]:
    """
    Return only chunks whose deterministic IDs are not yet in Qdrant.
    """

    return [
        (chunk, chunk_id)
        for chunk, chunk_id in zip(
            chunks,
            chunk_ids,
            strict=True,
        )
        if chunk_id not in existing_ids
    ]


# -------------------------------------------------------------------
# Retry handling
# -------------------------------------------------------------------

def is_rate_limit_error(
    error: Exception,
) -> bool:
    """Return True for Gemini quota or rate-limit failures."""

    message = str(error).lower()

    return any(
        indicator in message
        for indicator in (
            "429",
            "resource_exhausted",
            "rate limit",
            "quota",
            "too many requests",
        )
    )


def is_transient_error(
    error: Exception,
) -> bool:
    """Return True for temporary network/service failures."""

    message = str(error).lower()

    return any(
        indicator in message
        for indicator in (
            "503",
            "502",
            "504",
            "service unavailable",
            "temporarily unavailable",
            "timeout",
            "timed out",
            "connection reset",
            "connection aborted",
        )
    )


def calculate_retry_wait(
    attempt: int,
) -> int:
    """Calculate bounded exponential backoff."""

    return min(
        MAX_RETRY_WAIT_SECONDS,
        15 * (2 ** (attempt - 1)),
    )


def store_batch_with_retry(
    vector_store,
    batch: list[Document],
    batch_ids: list[str],
    batch_number: int,
    total_batches: int,
) -> None:
    """
    Store one batch in Qdrant with stable IDs and retry handling.
    """

    for attempt in range(
        1,
        MAX_RETRIES + 1,
    ):
        try:
            vector_store.add_documents(
                documents=batch,
                ids=batch_ids,
            )

            print(
                f"Stored batch {batch_number}/{total_batches} "
                f"({len(batch)} chunks)"
            )
            return

        except Exception as error:
            print(
                "\nEmbedding error: "
                f"{error}"
            )

            retryable = (
                is_rate_limit_error(error)
                or is_transient_error(error)
            )

            if not retryable:
                raise

            if attempt == MAX_RETRIES:
                raise RuntimeError(
                    "Embedding service remained unavailable after "
                    f"{MAX_RETRIES} attempts for batch "
                    f"{batch_number}/{total_batches}."
                ) from error

            wait_seconds = calculate_retry_wait(
                attempt
            )

            print(
                "\nTemporary embedding failure."
                f"\nWaiting {wait_seconds} seconds..."
                f"\nRetry attempt {attempt}/{MAX_RETRIES}"
                f"\nBatch {batch_number}/{total_batches}\n"
            )

            time.sleep(wait_seconds)


# -------------------------------------------------------------------
# Diagnostics
# -------------------------------------------------------------------

def display_ingestion_diagnostics(
    chunks: list[Document],
) -> None:
    """Display metadata and quality diagnostics before ingestion."""

    section_chunks = [
        chunk
        for chunk in chunks
        if is_section_chunk(chunk)
    ]

    unsectioned_chunks = [
        chunk
        for chunk in chunks
        if not is_section_chunk(chunk)
    ]

    heading_only_chunks = [
        chunk
        for chunk in chunks
        if chunk.metadata.get(
            "heading_only_chunk",
            False,
        )
    ]

    suspicious_chunks = [
        chunk
        for chunk in chunks
        if chunk.metadata.get(
            "page_quality_suspicious",
            False,
        )
    ]

    cross_page_chunks = [
        chunk
        for chunk in chunks
        if chunk.metadata.get(
            "spans_multiple_pages",
            False,
        )
    ]

    split_sections = {
        str(
            chunk.metadata.get(
                "section_identity"
            )
        )
        for chunk in chunks
        if chunk.metadata.get(
            "section_was_split",
            False,
        )
    }

    section_counts = Counter(
        str(
            chunk.metadata.get(
                "section_identity"
            )
        )
        for chunk in section_chunks
    )

    duplicate_section_parts = {
        section: count
        for section, count in section_counts.items()
        if count > 1
    }

    print("\n" + "=" * 70)
    print("INGESTION DIAGNOSTICS")
    print("=" * 70)
    print(
        f"Total chunks: {len(chunks)}"
    )
    print(
        f"Section chunks: {len(section_chunks)}"
    )
    print(
        f"Unsectioned chunks: {len(unsectioned_chunks)}"
    )
    print(
        f"Heading-only chunks: {len(heading_only_chunks)}"
    )
    print(
        f"Suspicious-source chunks: {len(suspicious_chunks)}"
    )
    print(
        f"Cross-page chunks: {len(cross_page_chunks)}"
    )
    print(
        f"Sections split into multiple parts: "
        f"{len(split_sections)}"
    )
    print(
        "Sections represented by multiple chunks: "
        f"{len(duplicate_section_parts)}"
    )

    document_counts = Counter(
        str(
            chunk.metadata.get(
                "document_id",
                "unknown",
            )
        )
        for chunk in chunks
    )

    print(
        f"Documents represented: "
        f"{len(document_counts)}"
    )

    print("\nPer-document chunk counts:")

    first_chunk_by_document: dict[
        str,
        Document,
    ] = {}

    for chunk in chunks:
        document_id = str(
            chunk.metadata.get(
                "document_id",
                "unknown",
            )
        )

        first_chunk_by_document.setdefault(
            document_id,
            chunk,
        )

    for document_id, count in sorted(
        document_counts.items(),
        key=lambda item: item[0],
    ):
        sample_chunk = first_chunk_by_document[
            document_id
        ]
        metadata = sample_chunk.metadata

        print(
            "- "
            f"{metadata.get('document_title', 'Unknown')} "
            f"[{metadata.get('document_type', 'legal_document')}]"
            f": {count} chunks"
        )


def build_document_chunk_summary(
    chunks: list[Document],
) -> list[dict[str, object]]:
    """
    Return chunk totals grouped by document identity.
    """

    grouped: dict[
        str,
        dict[str, object],
    ] = {}

    for chunk in chunks:
        metadata = chunk.metadata
        document_id = str(
            metadata.get(
                "document_id",
                "unknown",
            )
        )

        summary = grouped.setdefault(
            document_id,
            {
                "document_id": document_id,
                "document_name": metadata.get(
                    "document_name",
                    "Unknown",
                ),
                "document_title": metadata.get(
                    "document_title",
                    "Unknown",
                ),
                "document_type": metadata.get(
                    "document_type",
                    "legal_document",
                ),
                "chunks": 0,
                "section_chunks": 0,
                "unsectioned_chunks": 0,
                "cross_page_chunks": 0,
            },
        )

        summary["chunks"] = int(
            summary["chunks"]
        ) + 1

        if is_section_chunk(chunk):
            summary["section_chunks"] = int(
                summary["section_chunks"]
            ) + 1
        else:
            summary["unsectioned_chunks"] = int(
                summary["unsectioned_chunks"]
            ) + 1

        if metadata.get(
            "spans_multiple_pages",
            False,
        ):
            summary["cross_page_chunks"] = int(
                summary["cross_page_chunks"]
            ) + 1

    return sorted(
        grouped.values(),
        key=lambda item: str(
            item["document_title"]
        ).lower(),
    )


def display_document_chunk_summary(
    chunks: list[Document],
) -> None:
    """
    Print a clear per-document summary before and after ingestion.
    """

    summaries = build_document_chunk_summary(
        chunks
    )

    print("\n" + "=" * 70)
    print("MULTI-DOCUMENT INGESTION SUMMARY")
    print("=" * 70)

    for summary in summaries:
        print(
            f"\nDocument: "
            f"{summary['document_title']}"
        )
        print(
            f"Document ID: "
            f"{summary['document_id']}"
        )
        print(
            f"Document type: "
            f"{summary['document_type']}"
        )
        print(
            f"Chunks: "
            f"{summary['chunks']}"
        )
        print(
            f"Section chunks: "
            f"{summary['section_chunks']}"
        )
        print(
            f"Unsectioned chunks: "
            f"{summary['unsectioned_chunks']}"
        )
        print(
            f"Cross-page chunks: "
            f"{summary['cross_page_chunks']}"
        )


# -------------------------------------------------------------------
# Main ingestion workflow
# -------------------------------------------------------------------

def main() -> None:
    """Run the complete validated ingestion pipeline."""

    client = None

    try:
        validate_ingestion_settings()

        print("=" * 70)
        print("QDRANT DOCUMENT INGESTION")
        print("=" * 70)

        print(
            "\nStep 1: Loading and cleaning PDFs..."
        )

        documents = load_all_documents()

        print(
            "\nStep 2: Creating section-aware chunks..."
        )

        chunks = create_chunks(
            documents
        )

        display_ingestion_diagnostics(
            chunks
        )

        display_document_chunk_summary(
            chunks
        )

        print(
            "\nStep 3: Validating chunk metadata..."
        )

        # Critical: validate before deleting the existing collection.
        validate_chunks(
            chunks
        )

        chunk_ids = validate_unique_chunk_ids(
            chunks
        )

        print(
            "Chunk metadata validation passed."
        )
        print(
            "Deterministic ID validation passed."
        )

        collection_action = (
            "Recreating"
            if RESET_COLLECTION
            else "Opening"
        )

        print(
            f"\nStep 4: {collection_action} "
            "the Qdrant collection..."
        )

        vector_store, client = (
            create_vector_store(
                reset=RESET_COLLECTION
            )
        )

        if RESET_COLLECTION:
            ensure_collection_is_empty(
                client
            )

        existing_ids = get_existing_point_ids(
            client
        )

        validate_existing_collection(
            expected_ids=chunk_ids,
            existing_ids=existing_ids,
        )

        pending_items = build_pending_ingestion_items(
            chunks=chunks,
            chunk_ids=chunk_ids,
            existing_ids=existing_ids,
        )

        print(
            "\nResume status:"
        )
        print(
            f"Already stored points: {len(existing_ids)}"
        )
        print(
            f"Remaining chunks: {len(pending_items)}"
        )
        print(
            f"Expected final points: {len(chunks)}"
        )

        if not pending_items:
            print(
                "\nAll chunks are already present. "
                "No embeddings need to be generated."
            )
        else:
            print(
                "\nStep 5: Creating embeddings "
                "and storing remaining chunks..."
            )

            total_batches = (
                len(pending_items)
                + BATCH_SIZE
                - 1
            ) // BATCH_SIZE

            for batch_number, start_index in enumerate(
                range(
                    0,
                    len(pending_items),
                    BATCH_SIZE,
                ),
                start=1,
            ):
                batch_items = pending_items[
                    start_index:
                    start_index + BATCH_SIZE
                ]

                batch = [
                    chunk
                    for chunk, _chunk_id
                    in batch_items
                ]

                batch_ids = [
                    chunk_id
                    for _chunk, chunk_id
                    in batch_items
                ]

                store_batch_with_retry(
                    vector_store=vector_store,
                    batch=batch,
                    batch_ids=batch_ids,
                    batch_number=batch_number,
                    total_batches=total_batches,
                )

                current_points = get_points_count(
                    client
                )

                expected_points = (
                    len(existing_ids)
                    + min(
                        start_index + len(batch),
                        len(pending_items),
                    )
                )

                print(
                    f"Qdrant points: {current_points} "
                    f"(expected {expected_points})"
                )

                if current_points != expected_points:
                    raise RuntimeError(
                        "Qdrant point count became inconsistent during "
                        f"resume batch {batch_number}. Expected "
                        f"{expected_points}, found {current_points}."
                    )

                if (
                    batch_number < total_batches
                    and DELAY_BETWEEN_BATCHES > 0
                ):
                    print(
                        f"Pausing {DELAY_BETWEEN_BATCHES} seconds "
                        "before the next embedding batch..."
                    )

                    time.sleep(
                        DELAY_BETWEEN_BATCHES
                    )

        final_points = get_points_count(
            client
        )

        print("\n" + "=" * 70)
        print("INGESTION COMPLETED")
        print("=" * 70)
        print(
            f"PDF page documents: "
            f"{len(documents)}"
        )
        print(
            f"Chunks created: "
            f"{len(chunks)}"
        )
        print(
            "Points stored in Qdrant: "
            f"{final_points}"
        )
        print(
            "Vector dimensions: "
            f"{get_embedding_dimension()}"
        )
        print(
            f"Collection: {COLLECTION_NAME}"
        )
        print(
            "Ingestion mode: "
            f"{'reset' if RESET_COLLECTION else 'resume'}"
        )
        print(
            "Documents ingested: "
            f"{len(build_document_chunk_summary(chunks))}"
        )

        if final_points != len(chunks):
            raise RuntimeError(
                "Ingestion finished with an invalid point count. "
                f"Chunks: {len(chunks)}, "
                f"Qdrant points: {final_points}."
            )

        print(
            "\nValidation passed: every valid chunk has "
            "exactly one Qdrant point."
        )

    except KeyboardInterrupt:
        print(
            "\nIngestion stopped by the user. "
            "Stored points are preserved and the next run can resume."
        )

    except Exception as error:
        print(
            f"\nIngestion error: {error}"
        )

    finally:
        if client is not None:
            client.close()


if __name__ == "__main__":
    main()