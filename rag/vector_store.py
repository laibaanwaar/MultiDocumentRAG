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


load_dotenv()


# -------------------------------------------------------------------
# Environment configuration
# -------------------------------------------------------------------

QDRANT_PATH = Path(
    os.getenv(
        "QDRANT_PATH",
        "./qdrant_storage",
    )
)

COLLECTION_NAME = os.getenv(
    "QDRANT_COLLECTION",
    "pakistan_penal_code",
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


# LangChain stores document metadata under the `metadata` payload key.
# Qdrant supports nested payload paths using dot notation.
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
        f"{METADATA_PAYLOAD_KEY}.section_number",
        PayloadSchemaType.KEYWORD,
    ),
    (
        f"{METADATA_PAYLOAD_KEY}.primary_section",
        PayloadSchemaType.KEYWORD,
    ),
    (
        f"{METADATA_PAYLOAD_KEY}.chapter_number",
        PayloadSchemaType.KEYWORD,
    ),
    (
        f"{METADATA_PAYLOAD_KEY}.legal_topic",
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
)


# -------------------------------------------------------------------
# Configuration validation
# -------------------------------------------------------------------

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

    Local mode persists its database files inside this directory.
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


# -------------------------------------------------------------------
# Collection inspection
# -------------------------------------------------------------------

def _extract_vector_params(
    collection_info: Any,
) -> Any:
    """
    Extract vector configuration across supported qdrant-client
    response shapes.
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
    Return VectorParams for a collection with one unnamed vector.

    Named-vector collections return a dictionary and are not compatible
    with the current dense-only configuration.
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
    Validate that an existing collection matches the active embedding
    model and distance metric.

    A dimension mismatch must be fixed by recreating the collection;
    otherwise document insertion or retrieval can fail unpredictably.
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
            "single unnamed dense vector configuration."
        )

        if STRICT_COLLECTION_VALIDATION:
            raise RuntimeError(
                f"{message}\n"
                "Run ingestion with reset=True to recreate it."
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
            "Run ingestion with reset=True to recreate the collection."
        )

    print(f"Warning: {message}")


# -------------------------------------------------------------------
# Collection creation and indexes
# -------------------------------------------------------------------

def create_collection(
    client: QdrantClient,
) -> None:
    """Create a new dense-vector Qdrant collection."""

    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(
            size=get_embedding_dimension(),
            distance=DISTANCE_METRIC,
        ),
    )

    print(
        f"Created Qdrant collection: "
        f"{COLLECTION_NAME}"
    )


def create_payload_indexes(
    client: QdrantClient,
) -> None:
    """
    Create indexes for metadata fields used by legal retrieval filters.

    Payload indexes improve filtered retrieval. In some local Qdrant
    versions they may be accepted but have little performance impact;
    failures are reported without destroying the collection.
    """

    if not CREATE_PAYLOAD_INDEXES:
        return

    created_count = 0
    skipped_count = 0

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
                skipped_count += 1
                continue

            if unsupported_in_local_mode:
                print(
                    "Warning: payload indexes are not fully supported "
                    "by this Qdrant local-mode version. Retrieval will "
                    "still work, but metadata filtering may be slower."
                )
                return

            raise RuntimeError(
                "Failed to create Qdrant payload index "
                f"'{field_name}': {error}"
            ) from error

    if created_count or skipped_count:
        print(
            "Qdrant payload indexes ready: "
            f"{created_count} created, "
            f"{skipped_count} already present."
        )


# -------------------------------------------------------------------
# Collection count and reset verification
# -------------------------------------------------------------------

def get_collection_points_count(
    client: QdrantClient,
) -> int:
    """Return the exact number of points currently stored."""

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
    """Ensure a freshly reset collection contains no points."""

    points_count = get_collection_points_count(
        client
    )

    if points_count != 0:
        raise RuntimeError(
            "The Qdrant collection was recreated but is not empty. "
            f"Points found: {points_count}."
        )


# -------------------------------------------------------------------
# Public vector-store factory
# -------------------------------------------------------------------

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
            When True, delete the existing collection and create a
            completely fresh collection. Use this during ingestion after
            changing cleaning, chunking, metadata, or embedding settings.

    Returns:
        A tuple containing:
        - configured LangChain QdrantVectorStore;
        - underlying QdrantClient.

    Raises:
        RuntimeError:
            If the existing collection is incompatible, reset fails, or
            local storage is locked by another process.
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

        if reset and collection_exists:
            print(
                "Deleting existing collection: "
                f"{COLLECTION_NAME}"
            )

            client.delete_collection(
                collection_name=COLLECTION_NAME
            )

            collection_exists = False

        if not collection_exists:
            create_collection(client)
            verify_empty_collection(client)
        else:
            validate_existing_collection(
                client
            )

        create_payload_indexes(
            client
        )

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
            f"Points currently stored: "
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
                "query_cli.py, ingestion, or any other Python process "
                "using the same QDRANT_PATH, then try again."
            ) from error

        raise


# -------------------------------------------------------------------
# Diagnostic helper
# -------------------------------------------------------------------

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
            f"Collection status: "
            f"{getattr(collection_info, 'status', 'Unknown')}"
        )

    finally:
        if client is not None:
            client.close()


if __name__ == "__main__":
    display_vector_store_status()