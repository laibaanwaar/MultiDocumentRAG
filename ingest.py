from __future__ import annotations

import argparse
import hashlib
import os
import time
from collections import Counter
from dataclasses import dataclass
from uuid import NAMESPACE_URL, uuid5

from dotenv import load_dotenv
from langchain_core.documents import Document

from rag.document_loader import load_all_documents
from rag.embeddings import get_embedding_dimension
from rag.text_splitter import create_chunks
from rag.vector_store import (
    COLLECTION_NAME,
    create_vector_store,
    get_collection_points_count,
)


load_dotenv()


# -------------------------------------------------------------------
# Environment configuration
# -------------------------------------------------------------------

DEFAULT_BATCH_SIZE = int(
    os.getenv(
        "INGEST_BATCH_SIZE",
        "64",
    )
)

MAX_RETRIES = int(
    os.getenv(
        "INGEST_MAX_RETRIES",
        "3",
    )
)

MAX_RETRY_WAIT_SECONDS = int(
    os.getenv(
        "INGEST_MAX_RETRY_WAIT_SECONDS",
        "20",
    )
)

SCROLL_BATCH_SIZE = int(
    os.getenv(
        "INGEST_SCROLL_BATCH_SIZE",
        "256",
    )
)

ALLOW_UNSECTIONED_CHUNKS = (
    os.getenv(
        "ALLOW_UNSECTIONED_CHUNKS",
        "True",
    ).strip().lower()
    in {"1", "true", "yes", "on"}
)

ALLOW_HEADING_ONLY_CHUNKS = (
    os.getenv(
        "ALLOW_HEADING_ONLY_CHUNKS",
        "False",
    ).strip().lower()
    in {"1", "true", "yes", "on"}
)

FAIL_ON_SUSPICIOUS_CHUNK = (
    os.getenv(
        "FAIL_ON_SUSPICIOUS_CHUNK",
        "False",
    ).strip().lower()
    in {"1", "true", "yes", "on"}
)

RESET_COLLECTION_FROM_ENV = (
    os.getenv(
        "INGEST_RESET_COLLECTION",
        "False",
    ).strip().lower()
    in {"1", "true", "yes", "on"}
)


# -------------------------------------------------------------------
# Runtime options
# -------------------------------------------------------------------

@dataclass(frozen=True)
class IngestionOptions:
    reset: bool
    batch_size: int
    dry_run: bool


def parse_arguments() -> IngestionOptions:
    """Parse command-line options for ingestion."""

    parser = argparse.ArgumentParser(
        description=(
            "Load, chunk, embed, and store Pakistan legal PDFs "
            "inside a local Qdrant collection."
        )
    )

    parser.add_argument(
        "--reset",
        action="store_true",
        help=(
            "Delete and recreate the Qdrant collection before "
            "storing chunks."
        ),
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help=(
            "Number of chunks embedded and stored per batch. "
            f"Default: {DEFAULT_BATCH_SIZE}."
        ),
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help=(
            "Load, split, and validate documents without modifying "
            "Qdrant or generating embeddings."
        ),
    )

    arguments = parser.parse_args()

    return IngestionOptions(
        reset=bool(
            arguments.reset
            or RESET_COLLECTION_FROM_ENV
        ),
        batch_size=int(arguments.batch_size),
        dry_run=bool(arguments.dry_run),
    )


# -------------------------------------------------------------------
# General validation helpers
# -------------------------------------------------------------------

def validate_ingestion_settings(
    options: IngestionOptions,
) -> None:
    """Validate ingestion settings before processing documents."""

    if options.batch_size <= 0:
        raise ValueError(
            "--batch-size must be greater than zero."
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

    embedding_dimension = get_embedding_dimension()

    if embedding_dimension != 384:
        raise RuntimeError(
            "This project is configured for "
            "sentence-transformers/all-MiniLM-L6-v2, which must "
            "produce 384-dimensional vectors. "
            f"Configured dimension: {embedding_dimension}."
        )


def is_unsectioned_chunk(
    chunk: Document,
) -> bool:
    """Return True for introductory or structural text chunks."""

    return bool(
        chunk.metadata.get(
            "is_unsectioned_chunk",
            False,
        )
    )


def get_provision_number(
    chunk: Document,
) -> str | None:
    """
    Return a section or article number using backward-compatible keys.
    """

    metadata = chunk.metadata

    value = (
        metadata.get("provision_number")
        or metadata.get("section_number")
        or metadata.get("article_number")
    )

    if value is None:
        return None

    normalized = str(value).strip()

    return normalized or None


def get_provision_type(
    chunk: Document,
) -> str | None:
    """
    Return `section` or `article` using metadata and document type.
    """

    metadata = chunk.metadata

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


def get_provision_identity(
    chunk: Document,
) -> str | None:
    """Return a stable section/article identity from metadata."""

    metadata = chunk.metadata

    value = (
        metadata.get("provision_identity")
        or metadata.get("section_identity")
        or metadata.get("article_identity")
    )

    if value:
        return str(value).strip() or None

    document_id = str(
        metadata.get(
            "document_id",
            "",
        )
    ).strip()

    provision_number = get_provision_number(
        chunk
    )

    if document_id and provision_number:
        return (
            f"{document_id}::"
            f"{get_provision_type(chunk) or 'provision'}::"
            f"{provision_number}"
        )

    return None


def validate_chunk_metadata(
    chunk: Document,
    chunk_index: int,
) -> list[str]:
    """
    Validate metadata required for retrieval and source citations.

    Both Sections and Constitution Articles are supported.
    """

    errors: list[str] = []
    metadata = chunk.metadata

    if not chunk.page_content.strip():
        errors.append(
            "empty page_content"
        )

    required_document_fields = (
        "document_id",
        "document_name",
        "document_title",
        "document_type",
        "chunk_id",
        "document_chunk_number",
        "chunk_number",
        "chunk_length",
    )

    for field_name in required_document_fields:
        if metadata.get(field_name) in {
            None,
            "",
        }:
            errors.append(
                f"missing {field_name}"
            )

    if is_unsectioned_chunk(chunk):
        if not ALLOW_UNSECTIONED_CHUNKS:
            errors.append(
                "unsectioned chunks are disabled"
            )

    else:
        provision_number = get_provision_number(
            chunk
        )

        if not provision_number:
            errors.append(
                "missing provision_number/section_number/article_number"
            )

        if not get_provision_identity(chunk):
            errors.append(
                "missing provision identity"
            )

        if metadata.get("page_start") is None:
            errors.append(
                "missing page_start"
            )

        if metadata.get("page_end") is None:
            errors.append(
                "missing page_end"
            )

        if not metadata.get("source_pages"):
            errors.append(
                "missing source_pages"
            )

        body_present = metadata.get(
            "provision_body_present",
            metadata.get(
                "section_body_present",
                True,
            ),
        )

        if body_present is False:
            errors.append(
                "provision body is missing"
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
        FAIL_ON_SUSPICIOUS_CHUNK
        and metadata.get(
            "page_quality_suspicious",
            False,
        )
    ):
        errors.append(
            "chunk originates from a suspicious page"
        )

    page_start = metadata.get(
        "page_start"
    )
    page_end = metadata.get(
        "page_end"
    )

    if (
        isinstance(page_start, int)
        and isinstance(page_end, int)
        and page_start > page_end
    ):
        errors.append(
            "invalid page range"
        )

    return [
        f"Chunk {chunk_index}: {error}"
        for error in errors
    ]


def validate_chunks(
    chunks: list[Document],
) -> None:
    """
    Validate all chunks before opening or resetting Qdrant.

    This prevents a failed chunking run from deleting a valid
    collection.
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
            len(validation_errors)
            - 25
        )

        if remaining > 0:
            preview += (
                f"\n- ...and {remaining} more validation errors"
            )

        raise RuntimeError(
            "Chunk validation failed before Qdrant was modified:\n"
            f"{preview}"
        )


# -------------------------------------------------------------------
# Stable Qdrant point IDs
# -------------------------------------------------------------------

def create_deterministic_id(
    chunk: Document,
) -> str:
    """
    Create a stable UUID for a legal chunk.

    The same document, provision, pages, chunk part, and content always
    produce the same Qdrant point ID.
    """

    metadata = chunk.metadata

    source_pages = metadata.get(
        "source_pages",
        [],
    )

    if isinstance(
        source_pages,
        (list, tuple, set),
    ):
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
            get_provision_type(chunk)
            or ""
        ),
        str(
            get_provision_number(chunk)
            or ""
        ),
        str(
            get_provision_identity(chunk)
            or ""
        ),
        str(
            metadata.get(
                "section_part_number",
                metadata.get(
                    "provision_part_number",
                    1,
                ),
            )
        ),
        str(
            metadata.get(
                "section_part_count",
                metadata.get(
                    "provision_part_count",
                    1,
                ),
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


def validate_unique_chunk_ids(
    chunks: list[Document],
) -> list[str]:
    """
    Generate deterministic IDs and reject any duplicate point IDs.
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
# Qdrant helpers
# -------------------------------------------------------------------

def get_points_count(
    client,
) -> int:
    """Return the exact number of points in the collection."""

    return int(
        get_collection_points_count(
            client
        )
    )


def ensure_collection_is_empty(
    client,
) -> None:
    """Verify that reset=True created an empty collection."""

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
    Read all stored point IDs so interrupted ingestion can resume.
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


def validate_existing_point_set(
    expected_ids: list[str],
    existing_ids: set[str],
) -> None:
    """
    Ensure the existing collection belongs to the current chunk set.

    A collection containing points from old PDFs or old chunking rules
    must be rebuilt with --reset.
    """

    expected_id_set = set(
        expected_ids
    )

    unexpected_ids = (
        existing_ids
        - expected_id_set
    )

    if unexpected_ids:
        preview = ", ".join(
            sorted(unexpected_ids)[:5]
        )

        raise RuntimeError(
            "The Qdrant collection contains points that do not belong "
            "to the current documents or chunking configuration. "
            "Run `python ingest.py --reset` for a clean rebuild. "
            f"Unexpected ID examples: {preview}"
        )


def build_pending_ingestion_items(
    chunks: list[Document],
    chunk_ids: list[str],
    existing_ids: set[str],
) -> list[tuple[Document, str]]:
    """Return chunks not yet present in Qdrant."""

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
# Batch storage
# -------------------------------------------------------------------

def is_retryable_local_error(
    error: Exception,
) -> bool:
    """
    Identify temporary local storage or operating-system failures.

    No cloud quota or rate-limit handling is needed because embeddings
    are generated locally.
    """

    message = str(error).lower()

    return any(
        indicator in message
        for indicator in (
            "database is locked",
            "resource temporarily unavailable",
            "connection reset",
            "temporarily unavailable",
            "timed out",
            "timeout",
            "i/o error",
            "io error",
        )
    )


def calculate_retry_wait(
    attempt: int,
) -> int:
    """Return a bounded exponential retry delay."""

    return min(
        MAX_RETRY_WAIT_SECONDS,
        2 ** attempt,
    )


def store_batch_with_retry(
    vector_store,
    batch: list[Document],
    batch_ids: list[str],
    batch_number: int,
    total_batches: int,
) -> None:
    """
    Generate local MiniLM embeddings and store one Qdrant batch.
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
            if (
                not is_retryable_local_error(error)
                or attempt == MAX_RETRIES
            ):
                raise RuntimeError(
                    "Failed to embed or store "
                    f"batch {batch_number}/{total_batches}: {error}"
                ) from error

            wait_seconds = calculate_retry_wait(
                attempt
            )

            print(
                "Temporary local ingestion failure. "
                f"Retrying in {wait_seconds} seconds "
                f"({attempt}/{MAX_RETRIES})..."
            )

            time.sleep(
                wait_seconds
            )


# -------------------------------------------------------------------
# Diagnostics
# -------------------------------------------------------------------

def build_document_chunk_summary(
    chunks: list[Document],
) -> list[dict[str, object]]:
    """Build chunk totals grouped by legal document."""

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
                "article_chunks": 0,
                "unsectioned_chunks": 0,
                "cross_page_chunks": 0,
            },
        )

        summary["chunks"] = int(
            summary["chunks"]
        ) + 1

        if is_unsectioned_chunk(chunk):
            summary["unsectioned_chunks"] = int(
                summary["unsectioned_chunks"]
            ) + 1

        elif get_provision_type(chunk) == "article":
            summary["article_chunks"] = int(
                summary["article_chunks"]
            ) + 1

        else:
            summary["section_chunks"] = int(
                summary["section_chunks"]
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


def display_ingestion_diagnostics(
    chunks: list[Document],
) -> None:
    """Display quality and metadata diagnostics before ingestion."""

    unsectioned_chunks = [
        chunk
        for chunk in chunks
        if is_unsectioned_chunk(chunk)
    ]

    section_chunks = [
        chunk
        for chunk in chunks
        if (
            not is_unsectioned_chunk(chunk)
            and get_provision_type(chunk)
            != "article"
        )
    ]

    article_chunks = [
        chunk
        for chunk in chunks
        if (
            not is_unsectioned_chunk(chunk)
            and get_provision_type(chunk)
            == "article"
        )
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

    provision_counts = Counter(
        get_provision_identity(chunk)
        for chunk in chunks
        if (
            not is_unsectioned_chunk(chunk)
            and get_provision_identity(chunk)
        )
    )

    split_provisions = {
        identity
        for identity, count
        in provision_counts.items()
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
        f"Article chunks: {len(article_chunks)}"
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
        "Provisions represented by multiple chunks: "
        f"{len(split_provisions)}"
    )

    summaries = build_document_chunk_summary(
        chunks
    )

    print(
        f"Documents represented: {len(summaries)}"
    )

    print("\nPer-document chunk counts:")

    for summary in summaries:
        print(
            "- "
            f"{summary['document_title']} "
            f"[{summary['document_type']}]: "
            f"{summary['chunks']} chunks "
            f"({summary['section_chunks']} sections, "
            f"{summary['article_chunks']} articles, "
            f"{summary['unsectioned_chunks']} unsectioned)"
        )


def display_document_chunk_summary(
    chunks: list[Document],
) -> None:
    """Print a detailed per-document summary."""

    summaries = build_document_chunk_summary(
        chunks
    )

    print("\n" + "=" * 70)
    print("MULTI-DOCUMENT INGESTION SUMMARY")
    print("=" * 70)

    for summary in summaries:
        print(
            f"\nDocument: {summary['document_title']}"
        )
        print(
            f"Document ID: {summary['document_id']}"
        )
        print(
            f"Document type: {summary['document_type']}"
        )
        print(
            f"Chunks: {summary['chunks']}"
        )
        print(
            f"Section chunks: {summary['section_chunks']}"
        )
        print(
            f"Article chunks: {summary['article_chunks']}"
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
    """Run the complete local multi-document ingestion pipeline."""

    client = None

    try:
        options = parse_arguments()

        validate_ingestion_settings(
            options
        )

        print("=" * 70)
        print("PAKISTAN MULTI-DOCUMENT LEGAL RAG INGESTION")
        print("=" * 70)
        print(
            "Embedding model: "
            "sentence-transformers/all-MiniLM-L6-v2"
        )
        print(
            "Embedding dimension: "
            f"{get_embedding_dimension()}"
        )
        print(
            "Vector database: Qdrant local"
        )
        print(
            "Mode: "
            f"{'dry run' if options.dry_run else 'reset' if options.reset else 'resume'}"
        )

        print(
            "\nStep 1: Loading and cleaning PDFs..."
        )

        documents = load_all_documents()

        print(
            "\nStep 2: Creating provision-aware chunks..."
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

        # Validation happens before Qdrant is opened or reset.
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

        if options.dry_run:
            print("\n" + "=" * 70)
            print("DRY RUN COMPLETED")
            print("=" * 70)
            print(
                f"PDF page documents: {len(documents)}"
            )
            print(
                f"Validated chunks: {len(chunks)}"
            )
            print(
                "Qdrant was not modified and embeddings "
                "were not generated."
            )
            return

        collection_action = (
            "Recreating"
            if options.reset
            else "Opening"
        )

        print(
            f"\nStep 4: {collection_action} "
            "the Qdrant collection..."
        )

        vector_store, client = create_vector_store(
            reset=options.reset
        )

        if options.reset:
            ensure_collection_is_empty(
                client
            )

        existing_ids = get_existing_point_ids(
            client
        )

        validate_existing_point_set(
            expected_ids=chunk_ids,
            existing_ids=existing_ids,
        )

        pending_items = build_pending_ingestion_items(
            chunks=chunks,
            chunk_ids=chunk_ids,
            existing_ids=existing_ids,
        )

        print("\nResume status:")
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
                "\nAll chunks are already stored. "
                "No embeddings need to be generated."
            )

        else:
            print(
                "\nStep 5: Generating local MiniLM embeddings "
                "and storing chunks..."
            )

            total_batches = (
                len(pending_items)
                + options.batch_size
                - 1
            ) // options.batch_size

            for batch_number, start_index in enumerate(
                range(
                    0,
                    len(pending_items),
                    options.batch_size,
                ),
                start=1,
            ):
                batch_items = pending_items[
                    start_index:
                    start_index + options.batch_size
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
                        start_index
                        + len(batch),
                        len(pending_items),
                    )
                )

                print(
                    f"Qdrant points: {current_points} "
                    f"(expected {expected_points})"
                )

                if current_points != expected_points:
                    raise RuntimeError(
                        "Qdrant point count became inconsistent "
                        f"during batch {batch_number}. Expected "
                        f"{expected_points}, found {current_points}."
                    )

        final_points = get_points_count(
            client
        )

        print("\n" + "=" * 70)
        print("INGESTION COMPLETED")
        print("=" * 70)
        print(
            f"PDF page documents: {len(documents)}"
        )
        print(
            f"Chunks created: {len(chunks)}"
        )
        print(
            f"Points stored in Qdrant: {final_points}"
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
            f"{'reset' if options.reset else 'resume'}"
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
            "\nValidation passed: every chunk has exactly "
            "one Qdrant point."
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
        raise

    finally:
        if client is not None:
            client.close()


if __name__ == "__main__":
    main()