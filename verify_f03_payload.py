from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue

from rag.vector_store import QDRANT_PATH, COLLECTION_NAME

client = QdrantClient(
    path=str(QDRANT_PATH.expanduser().resolve())
)

try:
    points, _ = client.scroll(
        collection_name=COLLECTION_NAME,
        scroll_filter=Filter(
            must=[
                FieldCondition(
                    key="metadata.document_id",
                    match=MatchValue(value="ata_1997"),
                ),
                FieldCondition(
                    key="metadata.base_provision_number",
                    match=MatchValue(value="11EE"),
                ),
                FieldCondition(
                    key="metadata.subsection_path_key",
                    match=MatchValue(value="2.b"),
                ),
            ]
        ),
        limit=20,
        with_payload=True,
        with_vectors=False,
    )

    print("MATCHES:", len(points))

    for point in points:
        metadata = point.payload.get("metadata", {})

        print("\n--- MATCH ---")
        print("document_id:", metadata.get("document_id"))
        print("provision_type:", metadata.get("provision_type"))
        print("provision_number:", metadata.get("provision_number"))
        print(
            "base_provision_number:",
            metadata.get("base_provision_number"),
        )
        print(
            "subsection_path:",
            metadata.get("subsection_path"),
        )
        print(
            "subsection_path_key:",
            metadata.get("subsection_path_key"),
        )
        print(
            "component_type:",
            metadata.get("component_type"),
        )
        print(
            "component_label:",
            metadata.get("component_label"),
        )
        print("page_start:", metadata.get("page_start"))
        print("page_end:", metadata.get("page_end"))
        print(
            "TEXT:",
            point.payload.get("page_content", "")[:800],
        )

finally:
    client.close()
