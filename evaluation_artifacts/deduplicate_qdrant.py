import hashlib
import os

from dotenv import load_dotenv
from qdrant_client import QdrantClient, models


load_dotenv()

QDRANT_PATH = os.getenv(
    "QDRANT_PATH",
    "./qdrant_storage",
)

COLLECTION_NAME = os.getenv(
    "QDRANT_COLLECTION",
    "pakistan_penal_code",
)

# First run only checks duplicates.
# Change to False after confirming the count.
DRY_RUN = False


def create_chunk_key(payload: dict) -> tuple:
    """
    Create a unique identity from the chunk's source,
    page, position and text.
    """

    metadata = payload.get("metadata", {})

    source = (
        metadata.get("source")
        or metadata.get("source_path")
        or metadata.get("document_name")
        or "unknown"
    )

    page_number = metadata.get("page_number")
    start_index = metadata.get("start_index")

    text = (
        payload.get("page_content")
        or payload.get("text")
        or ""
    )

    text_hash = hashlib.sha256(
        text.encode("utf-8")
    ).hexdigest()

    return (
        source,
        page_number,
        start_index,
        text_hash,
    )


def find_duplicates(
    client: QdrantClient,
) -> tuple[int, list]:
    seen_keys: set[tuple] = set()
    duplicate_ids: list = []

    total_scanned = 0
    offset = None

    while True:
        records, next_offset = client.scroll(
            collection_name=COLLECTION_NAME,
            limit=100,
            offset=offset,
            with_payload=True,
            with_vectors=False,
        )

        for record in records:
            total_scanned += 1

            chunk_key = create_chunk_key(
                record.payload or {}
            )

            if chunk_key in seen_keys:
                duplicate_ids.append(record.id)
            else:
                seen_keys.add(chunk_key)

        if next_offset is None:
            break

        offset = next_offset

    return total_scanned, duplicate_ids


def delete_duplicates(
    client: QdrantClient,
    duplicate_ids: list,
) -> None:
    batch_size = 100

    for start in range(
        0,
        len(duplicate_ids),
        batch_size,
    ):
        ids_batch = duplicate_ids[
            start:start + batch_size
        ]

        client.delete(
            collection_name=COLLECTION_NAME,
            points_selector=models.PointIdsList(
                points=ids_batch
            ),
            wait=True,
        )

        print(
            f"Deleted {len(ids_batch)} duplicate points."
        )


def main() -> None:
    client = QdrantClient(path=QDRANT_PATH)

    try:
        total_scanned, duplicate_ids = find_duplicates(
            client
        )

        print("=" * 70)
        print("QDRANT DUPLICATE CHECK")
        print("=" * 70)
        print(f"Points scanned: {total_scanned}")
        print(f"Duplicates found: {len(duplicate_ids)}")
        print(f"Dry run: {DRY_RUN}")

        if not duplicate_ids:
            print("No duplicates were found.")
            return

        if DRY_RUN:
            print(
                "\nNo points were deleted. "
                "Check that 50 duplicates were found."
            )
            return

        delete_duplicates(
            client=client,
            duplicate_ids=duplicate_ids,
        )

        final_count = client.count(
            collection_name=COLLECTION_NAME,
            exact=True,
        ).count

        print("\nDuplicate cleanup completed.")
        print(f"Remaining exact points: {final_count}")

    finally:
        client.close()


if __name__ == "__main__":
    main()