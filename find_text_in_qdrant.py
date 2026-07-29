from rag.vector_store import (
    COLLECTION_NAME,
    create_vector_store,
)


SEARCH_TERMS = [
    "punishment for theft",
    "379. punishment",
    "section 379",
]


def main() -> None:
    _, client = create_vector_store(reset=False)

    try:
        offset = None
        matches = []

        while True:
            records, next_offset = client.scroll(
                collection_name=COLLECTION_NAME,
                limit=100,
                offset=offset,
                with_payload=True,
                with_vectors=False,
            )

            for record in records:
                payload = record.payload or {}

                text = (
                    payload.get("page_content")
                    or payload.get("text")
                    or ""
                )

                normalized_text = text.lower()

                if any(
                    term in normalized_text
                    for term in SEARCH_TERMS
                ):
                    metadata = payload.get("metadata", {})

                    matches.append(
                        {
                            "page": metadata.get("page_number"),
                            "chunk": metadata.get("chunk_number"),
                            "text": text,
                        }
                    )

            if next_offset is None:
                break

            offset = next_offset

        print("=" * 70)
        print("QDRANT TEXT SEARCH")
        print("=" * 70)
        print(f"Matches found: {len(matches)}")

        for index, match in enumerate(matches, start=1):
            print("\n" + "-" * 70)
            print(f"MATCH {index}")
            print("-" * 70)
            print(f"Page: {match['page']}")
            print(f"Chunk: {match['chunk']}")
            print("\nText:")
            print(match["text"])

    finally:
        client.close()


if __name__ == "__main__":
    main()