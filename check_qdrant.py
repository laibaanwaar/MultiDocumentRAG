import os

from dotenv import load_dotenv
from qdrant_client import QdrantClient


load_dotenv()

QDRANT_PATH = os.getenv(
    "QDRANT_PATH",
    "./qdrant_storage",
)

COLLECTION_NAME = os.getenv(
    "QDRANT_COLLECTION",
    "pakistan_penal_code",
)


def main() -> None:
    client = QdrantClient(path=QDRANT_PATH)

    try:
        exact_result = client.count(
            collection_name=COLLECTION_NAME,
            exact=True,
        )

        collection_info = client.get_collection(
            collection_name=COLLECTION_NAME,
        )

        print("=" * 70)
        print("QDRANT COLLECTION CHECK")
        print("=" * 70)
        print(f"Collection: {COLLECTION_NAME}")
        print(f"Summary points_count: {collection_info.points_count}")
        print(f"Exact point count: {exact_result.count}")

    finally:
        client.close()


if __name__ == "__main__":
    main()