from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from langchain_qdrant import (
    QdrantVectorStore,
    RetrievalMode,
)
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    PayloadSchemaType,
    VectorParams,
)

from rag.embeddings import (
    get_embedding_dimension,
    get_embedding_model,
)
from rag.metadata_schema import (
    audit_collection_metadata,
    format_metadata_audit_result,
)


load_dotenv()

QDRANT_PATH = Path(
    os.getenv(
        "QDRANT_PATH",
        "./qdrant_storage",
    )
)

COLLECTION_NAME = os.getenv(
    "QDRANT_COLLECTION",
    "pakistan_legal_knowledge_base",
).strip()

DISTANCE_METRIC = Distance.COSINE

CONTENT_PAYLOAD_KEY = os.getenv(
    "QDRANT_CONTENT_PAYLOAD_KEY",
    "page_content",
).strip()

METADATA_PAYLOAD_KEY = os.getenv(
    "QDRANT_METADATA_PAYLOAD_KEY",
    "metadata",
).strip()

CREATE_PAYLOAD_INDEXES = (
    os.getenv(
        "QDRANT_CREATE_PAYLOAD_INDEXES",
        "True",
    ).lower()
    == "true"
)

STRICT_COLLECTION_VALIDATION = (
    os.getenv(
        "QDRANT_STRICT_COLLECTION_VALIDATION",
        "True",
    ).lower()
    == "true"
)

PAYLOAD_INDEXES: tuple[
    tuple[str, PayloadSchemaType],
    ...,
] = (
    (
        f"{METADATA_PAYLOAD_KEY}.document_id",
        PayloadSchemaType.KEYWORD,
    ),
    (
        f"{METADATA_PAYLOAD_KEY}.document_name",
        PayloadSchemaType.KEYWORD,
    ),
    (
        f"{METADATA_PAYLOAD_KEY}.document_title",
        PayloadSchemaType.KEYWORD,
    ),
    (
        f"{METADATA_PAYLOAD_KEY}.document_short_name",
        PayloadSchemaType.KEYWORD,
    ),
    (
        f"{METADATA_PAYLOAD_KEY}.document_type",
        PayloadSchemaType.KEYWORD,
    ),
    (
        f"{METADATA_PAYLOAD_KEY}.provision_type",
        PayloadSchemaType.KEYWORD,
    ),
    (
        f"{METADATA_PAYLOAD_KEY}.provision_number",
        PayloadSchemaType.KEYWORD,
    ),
    (
        f"{METADATA_PAYLOAD_KEY}.provision_ordinal",
        PayloadSchemaType.INTEGER,
    ),
    (
        f"{METADATA_PAYLOAD_KEY}.section_number",
        PayloadSchemaType.KEYWORD,
    ),
    (
        f"{METADATA_PAYLOAD_KEY}.article_number",
        PayloadSchemaType.KEYWORD,
    ),
    (
        f"{METADATA_PAYLOAD_KEY}.primary_section",
        PayloadSchemaType.KEYWORD,
    ),
    (
        f"{METADATA_PAYLOAD_KEY}.primary_article",
        PayloadSchemaType.KEYWORD,
    ),
    (
        f"{METADATA_PAYLOAD_KEY}.section_identity",
        PayloadSchemaType.KEYWORD,
    ),
    (
        f"{METADATA_PAYLOAD_KEY}.provision_identity",
        PayloadSchemaType.KEYWORD,
    ),
    (
        f"{METADATA_PAYLOAD_KEY}.base_provision_number",
        PayloadSchemaType.KEYWORD,
    ),
    (
        f"{METADATA_PAYLOAD_KEY}.previous_provision_number",
        PayloadSchemaType.KEYWORD,
    ),
    (
        f"{METADATA_PAYLOAD_KEY}.next_provision_number",
        PayloadSchemaType.KEYWORD,
    ),
    (
        f"{METADATA_PAYLOAD_KEY}.subsection_path_key",
        PayloadSchemaType.KEYWORD,
    ),
    (
        f"{METADATA_PAYLOAD_KEY}.component_type",
        PayloadSchemaType.KEYWORD,
    ),
    (
        f"{METADATA_PAYLOAD_KEY}.chapter_number",
        PayloadSchemaType.KEYWORD,
    ),
    (
        f"{METADATA_PAYLOAD_KEY}.part",
        PayloadSchemaType.KEYWORD,
    ),
    (
        f"{METADATA_PAYLOAD_KEY}.legal_topic",
        PayloadSchemaType.KEYWORD,
    ),
    (
        f"{METADATA_PAYLOAD_KEY}.content_type",
        PayloadSchemaType.KEYWORD,
    ),
    (
        f"{METADATA_PAYLOAD_KEY}.page_number",
        PayloadSchemaType.INTEGER,
    ),
    (
        f"{METADATA_PAYLOAD_KEY}.page_start",
        PayloadSchemaType.INTEGER,
    ),
    (
        f"{METADATA_PAYLOAD_KEY}.page_end",
        PayloadSchemaType.INTEGER,
    ),
    (
        f"{METADATA_PAYLOAD_KEY}.chunk_number",
        PayloadSchemaType.INTEGER,
    ),
    (
        f"{METADATA_PAYLOAD_KEY}.document_chunk_number",
        PayloadSchemaType.INTEGER,
    ),
    (
        f"{METADATA_PAYLOAD_KEY}.section_part_number",
        PayloadSchemaType.INTEGER,
    ),
    (
        f"{METADATA_PAYLOAD_KEY}.section_part_count",
        PayloadSchemaType.INTEGER,
    ),
    (
        f"{METADATA_PAYLOAD_KEY}.heading_only_chunk",
        PayloadSchemaType.BOOL,
    ),
    (
        f"{METADATA_PAYLOAD_KEY}.heading_only_page",
        PayloadSchemaType.BOOL,
    ),
    (
        f"{METADATA_PAYLOAD_KEY}.section_body_present",
        PayloadSchemaType.BOOL,
    ),
    (
        f"{METADATA_PAYLOAD_KEY}.page_quality_suspicious",
        PayloadSchemaType.BOOL,
    ),
    (
        f"{METADATA_PAYLOAD_KEY}.is_unsectioned_chunk",
        PayloadSchemaType.BOOL,
    ),
)

def validate_vector_store_settings() -> None:
    """Validate local Qdrant configuration before opening storage."""

    if not COLLECTION_NAME:
        raise ValueError(
            "QDRANT_COLLECTION cannot be empty."
        )

    if not CONTENT_PAYLOAD_KEY:
        raise ValueError(
            "QDRANT_CONTENT_PAYLOAD_KEY cannot be empty."
        )

    if not METADATA_PAYLOAD_KEY:
        raise ValueError(
            "QDRANT_METADATA_PAYLOAD_KEY cannot be empty."
        )

    embedding_dimension = get_embedding_dimension()

    if (
        not isinstance(embedding_dimension, int)
        or embedding_dimension <= 0
    ):
        raise ValueError(
            "The embedding dimension must be a positive integer. "
            f"Received: {embedding_dimension!r}"
        )


def prepare_storage_directory() -> Path:
    """
    Create and resolve the local Qdrant storage directory.

    Local mode persists all collection data inside this folder.
    """

    resolved_path = QDRANT_PATH.expanduser().resolve()

    if resolved_path.exists() and not resolved_path.is_dir():
        raise NotADirectoryError(
            "QDRANT_PATH points to a file instead of a directory:\n"
            f"{resolved_path}"
        )

    resolved_path.mkdir(
        parents=True,
        exist_ok=True,
    )

    return resolved_path


def _extract_vector_params(
    collection_info: Any,
) -> Any:
    """
    Extract vector configuration across qdrant-client response shapes.
    """

    try:
        return (
            collection_info
            .config
            .params
            .vectors
        )
    except AttributeError:
        return None


def _get_single_vector_params(
    vectors_config: Any,
) -> Any:
    """
    Return VectorParams for one unnamed dense vector.

    The current project does not use named or sparse vectors.
    """

    if isinstance(vectors_config, dict):
        if len(vectors_config) == 1:
            return next(
                iter(vectors_config.values())
            )

        return None

    return vectors_config


def _normalize_distance(
    distance: Any,
) -> str:
    """Normalize Qdrant distance values for comparison."""

    if isinstance(distance, Distance):
        return distance.value.lower()

    return str(distance).lower()


def validate_existing_collection(
    client: QdrantClient,
) -> None:
    """
    Validate an existing collection against MiniLM configuration.

    all-MiniLM-L6-v2 produces 384-dimensional vectors. A previous
    Gemini collection commonly uses 768 dimensions and must therefore
    be recreated before ingestion.
    """

    collection_info = client.get_collection(
        collection_name=COLLECTION_NAME
    )

    vectors_config = _extract_vector_params(
        collection_info
    )

    vector_params = _get_single_vector_params(
        vectors_config
    )

    if vector_params is None:
        message = (
            "The existing Qdrant collection does not use the expected "
            "single unnamed dense-vector configuration."
        )

        if STRICT_COLLECTION_VALIDATION:
            raise RuntimeError(
                f"{message}\n"
                "Run ingestion with --reset to recreate it."
            )

        print(f"Warning: {message}")
        return

    existing_size = getattr(
        vector_params,
        "size",
        None,
    )

    existing_distance = getattr(
        vector_params,
        "distance",
        None,
    )

    expected_size = get_embedding_dimension()

    dimension_matches = (
        existing_size == expected_size
    )

    distance_matches = (
        _normalize_distance(existing_distance)
        == _normalize_distance(
            DISTANCE_METRIC
        )
    )

    if dimension_matches and distance_matches:
        metadata_audit = audit_collection_metadata(
            client=client,
            collection_name=COLLECTION_NAME,
            metadata_payload_key=METADATA_PAYLOAD_KEY,
        )

        if metadata_audit.invalid_points == 0:
            return

        audit_message = format_metadata_audit_result(
            metadata_audit,
            collection_name=COLLECTION_NAME,
        )

        if STRICT_COLLECTION_VALIDATION:
            raise RuntimeError(
                f"{audit_message}\n"
                "The collection metadata is incompatible or corrupt. "
                "Run ingestion with --reset to recreate the collection."
            )

        print(f"Warning: {audit_message}")
        return

    mismatch_details: list[str] = []

    if not dimension_matches:
        mismatch_details.append(
            "vector dimension "
            f"{existing_size!r} does not match "
            f"{expected_size!r}"
        )

    if not distance_matches:
        mismatch_details.append(
            "distance metric "
            f"{existing_distance!r} does not match "
            f"{DISTANCE_METRIC.value!r}"
        )

    message = (
        f"Collection '{COLLECTION_NAME}' is incompatible: "
        + "; ".join(mismatch_details)
        + "."
    )

    if STRICT_COLLECTION_VALIDATION:
        raise RuntimeError(
            f"{message}\n"
            "Run ingestion with --reset to recreate the collection."
        )

    print(f"Warning: {message}")


def create_collection(
    client: QdrantClient,
) -> None:
    """Create a 384-dimensional dense-vector collection."""

    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(
            size=get_embedding_dimension(),
            distance=DISTANCE_METRIC,
        ),
    )

    print(
        "Created Qdrant collection: "
        f"{COLLECTION_NAME}"
    )


def create_payload_indexes(
    client: QdrantClient,
) -> None:
    """
    Create indexes for metadata fields used by legal retrieval filters.

    Local Qdrant may warn that payload indexes have no performance
    effect. Filtering still works in that mode.
    """

    if not CREATE_PAYLOAD_INDEXES:
        print(
            "Qdrant payload index creation is disabled."
        )
        return

    created_count = 0
    existing_count = 0
    unsupported_reported = False

    for field_name, field_schema in PAYLOAD_INDEXES:
        try:
            client.create_payload_index(
                collection_name=COLLECTION_NAME,
                field_name=field_name,
                field_schema=field_schema,
                wait=True,
            )

            created_count += 1

        except Exception as error:
            error_message = str(error).lower()

            already_exists = (
                "already exists" in error_message
                or "already indexed" in error_message
            )

            unsupported_in_local_mode = (
                "payload indexes have no effect" in error_message
                or "not supported" in error_message
                or "not implemented" in error_message
            )

            if already_exists:
                existing_count += 1
                continue

            if unsupported_in_local_mode:
                if not unsupported_reported:
                    print(
                        "Warning: this Qdrant local-mode version does "
                        "not use payload indexes for optimization. "
                        "Metadata filtering will still work."
                    )
                    unsupported_reported = True

                continue

            raise RuntimeError(
                "Failed to create Qdrant payload index "
                f"'{field_name}': {error}"
            ) from error

    print(
        "Qdrant payload indexes checked: "
        f"{created_count} created, "
        f"{existing_count} already present."
    )


# -------------------------------------------------------------------
# Collection count and reset verification
# -------------------------------------------------------------------

def get_collection_points_count(
    client: QdrantClient,
) -> int:
    """Return the exact number of stored vectors."""

    count_result = client.count(
        collection_name=COLLECTION_NAME,
        exact=True,
    )

    return int(
        count_result.count
    )


def verify_empty_collection(
    client: QdrantClient,
) -> None:
    """Ensure a newly created collection contains no vectors."""

    points_count = get_collection_points_count(
        client
    )

    if points_count != 0:
        raise RuntimeError(
            "The Qdrant collection was recreated but is not empty. "
            f"Points found: {points_count}."
        )


def reset_collection(
    client: QdrantClient,
) -> None:
    """Delete and recreate the configured collection."""

    if client.collection_exists(
        collection_name=COLLECTION_NAME
    ):
        print(
            "Deleting existing collection: "
            f"{COLLECTION_NAME}"
        )

        client.delete_collection(
            collection_name=COLLECTION_NAME
        )

    create_collection(client)
    verify_empty_collection(client)


def create_vector_store(
    reset: bool = False,
) -> tuple[
    QdrantVectorStore,
    QdrantClient,
]:
    """
    Create or open the local dense Qdrant vector store.

    Args:
        reset:
            Delete and recreate the collection. Use this after changing
            PDFs, cleaning, chunking, metadata, or embedding settings.

    Returns:
        A configured QdrantVectorStore and its QdrantClient.
    """

    validate_vector_store_settings()

    storage_path = prepare_storage_directory()

    client: QdrantClient | None = None

    try:
        client = QdrantClient(
            path=str(storage_path)
        )

        collection_exists = (
            client.collection_exists(
                collection_name=COLLECTION_NAME
            )
        )

        if reset:
            reset_collection(client)

        elif not collection_exists:
            create_collection(client)
            verify_empty_collection(client)

        else:
            validate_existing_collection(client)

        create_payload_indexes(client)

        vector_store = QdrantVectorStore(
            client=client,
            collection_name=COLLECTION_NAME,
            embedding=get_embedding_model(),
            retrieval_mode=RetrievalMode.DENSE,
            content_payload_key=(
                CONTENT_PAYLOAD_KEY
            ),
            metadata_payload_key=(
                METADATA_PAYLOAD_KEY
            ),
        )

        points_count = get_collection_points_count(
            client
        )

        print(
            "Opened Qdrant collection: "
            f"{COLLECTION_NAME}"
        )
        print(
            f"Qdrant storage: {storage_path}"
        )
        print(
            "Embedding dimension: "
            f"{get_embedding_dimension()}"
        )
        print(
            "Distance metric: "
            f"{DISTANCE_METRIC.value}"
        )
        print(
            "Points currently stored: "
            f"{points_count}"
        )

        return vector_store, client

    except Exception as error:
        if client is not None:
            client.close()

        error_message = str(error).lower()

        storage_locked = (
            "already accessed by another instance" in error_message
            or "resource temporarily unavailable" in error_message
            or "database is locked" in error_message
            or "lock" in error_message
        )

        if storage_locked:
            raise RuntimeError(
                "Qdrant local storage is already in use. Stop "
                "query_cli.py, ingestion, or another Python process "
                "using the same QDRANT_PATH, then try again."
            ) from error

        raise


# Backward-compatible alias for code that describes the operation as
# opening an existing vector store.
open_vector_store = create_vector_store

def display_vector_store_status() -> None:
    """Open Qdrant and print collection diagnostics."""

    client: QdrantClient | None = None

    try:
        _, client = create_vector_store(
            reset=False
        )

        collection_info = client.get_collection(
            collection_name=COLLECTION_NAME
        )

        print("\n" + "=" * 70)
        print("QDRANT COLLECTION STATUS")
        print("=" * 70)
        print(
            f"Collection: {COLLECTION_NAME}"
        )
        print(
            "Points count: "
            f"{get_collection_points_count(client)}"
        )
        print(
            "Embedding dimension: "
            f"{get_embedding_dimension()}"
        )
        print(
            "Distance metric: "
            f"{DISTANCE_METRIC.value}"
        )
        print(
            "Collection status: "
            f"{getattr(collection_info, 'status', 'Unknown')}"
        )

    finally:
        if client is not None:
            client.close()


if __name__ == "__main__":
    display_vector_store_status()
