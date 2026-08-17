from __future__ import annotations

from copy import deepcopy
from types import SimpleNamespace

import pytest
from langchain_core.documents import Document

from rag import vector_store as vector_store_module
from rag.metadata_schema import (
    audit_collection_metadata,
    format_metadata_audit_result,
    validate_legal_chunk_metadata,
)
from rag.retrieval_errors import MandatoryRetrievalError
from rag.retriever import (
    AdaptiveRetriever,
    fetch_candidates,
    retrieve_exact_provision_documents,
    retrieve_neighbor_documents,
)
from rag.text_splitter import create_chunks


def make_page(
    *,
    document_id: str,
    document_title: str,
    provision_type: str,
    provision_number: str,
    title: str,
    body: str,
    page_number: int,
) -> Document:
    return Document(
        page_content=(
            f"{provision_number}. {title}\n"
            f"{body}"
        ),
        metadata={
            "document_id": document_id,
            "document_name": document_title,
            "document_title": document_title,
            "document_short_name": document_title,
            "document_type": "legal_document",
            "provision_type": provision_type,
            "page_number": page_number,
            "page": page_number - 1,
        },
    )


def make_long_body(label: str, repeats: int = 40) -> str:
    sentence = (
        f"{label} preserves the source order of legal provisions "
        "and keeps the context safe for retrieval."
    )
    return " ".join([sentence] * repeats)


def make_valid_metadata(
    *,
    document_id: str = "ppc_1860",
    document_title: str = "Pakistan Penal Code, 1860",
    document_type: str = "legal_document",
    provision_type: str = "section",
    provision_number: str | None = "379",
    provision_identity: str | None = None,
    chunk_number: int = 1,
    document_chunk_number: int = 1,
    provision_part_number: int = 1,
    provision_part_count: int = 1,
    provision_ordinal: int | None = 1,
    page_start: int = 1,
    page_end: int = 1,
    source_pages: list[int] | None = None,
    is_unsectioned_chunk: bool = False,
    subsection_path: list[str] | None = None,
    component_type: str | None = None,
    component_label: str | None = None,
    previous_provision_number: str | None = None,
    next_provision_number: str | None = None,
    previous_provision_identity: str | None = None,
    next_provision_identity: str | None = None,
) -> dict[str, object]:
    if source_pages is None:
        source_pages = [1]

    base_provision_number = provision_number
    resolved_provision_identity = provision_identity or (
        f"{document_id}::"
        f"{provision_type}::"
        f"{provision_number if provision_number else 'unsectioned'}"
    )
    chunk_id = (
        f"{document_id}::"
        f"{provision_type}::"
        f"{provision_number if provision_number else 'unsectioned'}::"
        f"part-{provision_part_number}::"
        f"chunk-{document_chunk_number}"
    )

    metadata: dict[str, object] = {
        "document_id": document_id,
        "document_name": document_title,
        "document_title": document_title,
        "document_short_name": document_title,
        "document_type": document_type,
        "provision_type": provision_type,
        "provision_number": provision_number,
        "base_provision_number": base_provision_number,
        "provision_identity": resolved_provision_identity,
        "chunk_id": chunk_id,
        "chunk_number": chunk_number,
        "document_chunk_number": document_chunk_number,
        "page_start": page_start,
        "page_end": page_end,
        "source_pages": list(source_pages),
        "provision_part_number": provision_part_number,
        "provision_part_count": provision_part_count,
        "heading_only_chunk": False,
        "provision_body_present": True,
        "section_body_present": True,
        "is_unsectioned_chunk": is_unsectioned_chunk,
        "previous_provision_number": previous_provision_number,
        "next_provision_number": next_provision_number,
        "previous_provision_identity": previous_provision_identity,
        "next_provision_identity": next_provision_identity,
    }

    if provision_ordinal is not None:
        metadata["provision_ordinal"] = provision_ordinal

    if provision_type == "section":
        metadata["section_number"] = provision_number
        metadata["section_identity"] = resolved_provision_identity
        metadata["article_number"] = None
        metadata["article_identity"] = None
    else:
        metadata["article_number"] = provision_number
        metadata["article_identity"] = resolved_provision_identity
        metadata["section_number"] = None
        metadata["section_identity"] = None

    if subsection_path is not None:
        metadata["subsection_path"] = list(subsection_path)
        metadata["subsection_path_key"] = ".".join(subsection_path)
        metadata["component_type"] = component_type
        metadata["component_label"] = component_label

    if is_unsectioned_chunk:
        metadata["provision_number"] = None
        metadata["base_provision_number"] = None
        metadata["provision_identity"] = (
            f"{document_id}::{provision_type}::unsectioned"
        )
        metadata["section_number"] = None
        metadata["article_number"] = None
        metadata["section_identity"] = None
        metadata["article_identity"] = None
        metadata.pop("provision_ordinal", None)

    return metadata


def make_point(point_id: str, metadata: dict[str, object]) -> SimpleNamespace:
    return SimpleNamespace(
        id=point_id,
        payload={
            "page_content": "hidden",
            "metadata": deepcopy(metadata),
        },
    )


class DummyCollectionInfo:
    def __init__(self) -> None:
        self.config = SimpleNamespace(
            params=SimpleNamespace(
                vectors=SimpleNamespace(
                    size=384,
                    distance="Cosine",
                )
            )
        )


class DummyQdrantClient:
    def __init__(self, points: list[SimpleNamespace]) -> None:
        self._points = points
        self._returned = False

    def get_collection(self, collection_name):
        return DummyCollectionInfo()

    def scroll(
        self,
        *,
        collection_name,
        scroll_filter,
        limit,
        offset=None,
        with_payload=True,
        with_vectors=False,
    ):
        if self._returned:
            return [], None

        self._returned = True
        return list(self._points), None


class FilterAwareRetriever:
    def __init__(self, documents: list[Document]) -> None:
        self.documents = documents

    def _matches(self, document: Document, metadata_filter) -> bool:
        if metadata_filter is None:
            return True

        for condition in getattr(metadata_filter, "must", []):
            key = str(getattr(condition, "key", ""))
            if key.startswith("metadata."):
                key = key.split(".", 1)[1]

            metadata_value = document.metadata.get(key)
            match = getattr(condition, "match", None)

            if hasattr(match, "value"):
                if metadata_value != match.value:
                    return False
                continue

            if hasattr(match, "any"):
                if metadata_value not in list(match.any):
                    return False
                continue

        return True

    def scroll_documents(self, metadata_filter, page_size=128):
        return [
            document
            for document in self.documents
            if self._matches(document, metadata_filter)
        ][:page_size]

    def search_with_scores(self, query: str, k: int, metadata_filter=None):
        matches = [
            document
            for document in self.documents
            if self._matches(document, metadata_filter)
        ][:k]

        return [
            (document, float(index + 1))
            for index, document in enumerate(matches)
        ]


def test_valid_section_metadata_passes() -> None:
    metadata = make_valid_metadata()
    result = validate_legal_chunk_metadata(metadata)

    assert result.valid
    assert result.issues == []


def test_valid_article_metadata_passes() -> None:
    metadata = make_valid_metadata(
        document_id="const_1973",
        document_title="Constitution of Pakistan, 1973",
        document_type="constitutional_law",
        provision_type="article",
        provision_number="10A",
    )
    result = validate_legal_chunk_metadata(metadata)

    assert result.valid
    assert result.issues == []


def test_missing_document_id_fails() -> None:
    metadata = make_valid_metadata()
    metadata.pop("document_id")

    result = validate_legal_chunk_metadata(metadata)

    assert not result.valid
    assert any(issue.field_name == "document_id" for issue in result.issues)


def test_missing_provision_number_fails_for_normal_chunk() -> None:
    metadata = make_valid_metadata()
    metadata["provision_number"] = None

    result = validate_legal_chunk_metadata(metadata)

    assert not result.valid
    assert any(issue.field_name == "provision_number" for issue in result.issues)


def test_invalid_provision_type_fails() -> None:
    metadata = make_valid_metadata()
    metadata["provision_type"] = "chapter"

    result = validate_legal_chunk_metadata(metadata)

    assert not result.valid
    assert any(issue.field_name == "provision_type" for issue in result.issues)


def test_noncanonical_provision_number_fails() -> None:
    metadata = make_valid_metadata(provision_number="153-A")

    result = validate_legal_chunk_metadata(metadata)

    assert not result.valid
    assert any(
        issue.field_name == "provision_number"
        for issue in result.issues
    )


def test_normal_section_missing_ordinal_fails() -> None:
    metadata = make_valid_metadata(provision_ordinal=None)

    result = validate_legal_chunk_metadata(metadata)

    assert not result.valid
    assert any(
        issue.field_name == "provision_ordinal"
        for issue in result.issues
    )


def test_normal_article_missing_ordinal_fails() -> None:
    metadata = make_valid_metadata(
        document_id="const_1973",
        document_title="Constitution of Pakistan, 1973",
        document_type="constitutional_law",
        provision_type="article",
        provision_number="10A",
        provision_ordinal=None,
    )

    result = validate_legal_chunk_metadata(metadata)

    assert not result.valid
    assert any(
        issue.field_name == "provision_ordinal"
        for issue in result.issues
    )


@pytest.mark.parametrize(
    ("field_name", "bad_value"),
    [
        ("provision_ordinal", "17"),
        ("source_pages", "29"),
        ("heading_only_chunk", "False"),
        ("subsection_path", "1"),
    ],
)
def test_wrong_metadata_types_fail(field_name, bad_value) -> None:
    metadata = make_valid_metadata(
        subsection_path=["2"],
        component_type="subsection",
        component_label="(2)",
    )
    metadata[field_name] = bad_value

    result = validate_legal_chunk_metadata(metadata)

    assert not result.valid
    assert any(issue.field_name == field_name for issue in result.issues)


def test_page_start_after_page_end_fails() -> None:
    metadata = make_valid_metadata(page_start=3, page_end=2)

    result = validate_legal_chunk_metadata(metadata)

    assert not result.valid
    assert any(issue.field_name == "page_start" for issue in result.issues)


def test_provision_part_number_exceeds_count_fails() -> None:
    metadata = make_valid_metadata(
        provision_part_number=3,
        provision_part_count=2,
    )

    result = validate_legal_chunk_metadata(metadata)

    assert not result.valid
    assert any(
        issue.field_name == "provision_part_number"
        for issue in result.issues
    )


def test_section_number_inconsistent_with_provision_number_fails() -> None:
    metadata = make_valid_metadata()
    metadata["section_number"] = "380"

    result = validate_legal_chunk_metadata(metadata)

    assert not result.valid
    assert any(issue.field_name == "section_number" for issue in result.issues)


def test_article_number_inconsistent_with_provision_number_fails() -> None:
    metadata = make_valid_metadata(
        document_id="const_1973",
        document_title="Constitution of Pakistan, 1973",
        document_type="constitutional_law",
        provision_type="article",
        provision_number="10A",
    )
    metadata["article_number"] = "11"

    result = validate_legal_chunk_metadata(metadata)

    assert not result.valid
    assert any(issue.field_name == "article_number" for issue in result.issues)


def test_duplicate_chunk_id_detected() -> None:
    metadata = make_valid_metadata()
    client = DummyQdrantClient(
        [
            make_point("point-1", metadata),
            make_point("point-2", metadata),
        ]
    )

    audit = audit_collection_metadata(
        client,
        "pakistan_legal_knowledge_base",
    )

    assert audit.duplicate_chunk_ids == [metadata["chunk_id"]]
    assert audit.invalid_points == 1
    assert audit.valid_points == 1


def test_all_chunks_of_one_provision_share_one_ordinal() -> None:
    metadata_a = make_valid_metadata(
        provision_ordinal=1,
        document_chunk_number=1,
    )
    metadata_b = make_valid_metadata(
        provision_ordinal=2,
        document_chunk_number=2,
    )
    client = DummyQdrantClient(
        [
            make_point("point-1", metadata_a),
            make_point("point-2", metadata_b),
        ]
    )

    audit = audit_collection_metadata(
        client,
        "pakistan_legal_knowledge_base",
    )

    assert audit.invalid_points == 2
    assert audit.consistency_errors["provision_ordinal"] >= 1


def test_distinct_provisions_cannot_share_one_ordinal() -> None:
    first = make_valid_metadata(
        provision_number="379",
        next_provision_number="380",
        next_provision_identity="ppc_1860::section::380",
    )
    second = make_valid_metadata(
        provision_number="380",
        chunk_number=2,
        document_chunk_number=2,
        provision_ordinal=1,
        previous_provision_number="379",
        previous_provision_identity="ppc_1860::section::379",
    )

    client = DummyQdrantClient(
        [
            make_point("point-1", first),
            make_point("point-2", second),
        ]
    )

    audit = audit_collection_metadata(
        client,
        "pakistan_legal_knowledge_base",
    )

    assert audit.invalid_points == 2
    assert audit.consistency_errors["provision_ordinal"] >= 1


def test_broken_previous_next_adjacency_is_detected() -> None:
    first = make_valid_metadata(
        provision_number="379",
        document_chunk_number=1,
        provision_ordinal=1,
        next_provision_number="380",
        next_provision_identity="ppc_1860::section::380",
    )
    second = make_valid_metadata(
        provision_number="380",
        document_chunk_number=2,
        provision_ordinal=2,
        previous_provision_number="379",
        previous_provision_identity="wrong",
    )

    client = DummyQdrantClient(
        [
            make_point("point-1", first),
            make_point("point-2", second),
        ]
    )

    audit = audit_collection_metadata(
        client,
        "pakistan_legal_knowledge_base",
    )

    assert audit.invalid_points == 1
    assert audit.consistency_errors["previous_provision_identity"] >= 1


def test_adjacency_never_crosses_document_boundaries() -> None:
    doc_one_a = make_valid_metadata(
        document_id="ppc_1860",
        document_title="Pakistan Penal Code, 1860",
        provision_number="379",
        document_chunk_number=1,
        provision_ordinal=1,
        next_provision_number="380",
        next_provision_identity="ppc_1860::section::380",
    )
    doc_one_b = make_valid_metadata(
        document_id="ppc_1860",
        document_title="Pakistan Penal Code, 1860",
        provision_number="380",
        document_chunk_number=2,
        provision_ordinal=2,
        previous_provision_number="379",
        previous_provision_identity="ppc_1860::section::379",
    )
    doc_two_a = make_valid_metadata(
        document_id="crpc_1898",
        document_title="Code of Criminal Procedure, 1898",
        provision_number="379",
        document_chunk_number=1,
        provision_ordinal=1,
        next_provision_number="380",
        next_provision_identity="crpc_1898::section::380",
    )
    doc_two_b = make_valid_metadata(
        document_id="crpc_1898",
        document_title="Code of Criminal Procedure, 1898",
        provision_number="380",
        document_chunk_number=2,
        provision_ordinal=2,
        previous_provision_number="379",
        previous_provision_identity="crpc_1898::section::379",
    )

    client = DummyQdrantClient(
        [
            make_point("p1", doc_one_a),
            make_point("p2", doc_one_b),
            make_point("p3", doc_two_a),
            make_point("p4", doc_two_b),
        ]
    )

    audit = audit_collection_metadata(
        client,
        "pakistan_legal_knowledge_base",
    )

    assert audit.invalid_points == 0


def test_child_chunk_inherits_correct_parent_identity() -> None:
    chunks = create_chunks(
        [
            make_page(
                document_id="ppc_1860",
                document_title="Pakistan Penal Code, 1860",
                provision_type="section",
                provision_number="11EE",
                title="Proscription of person",
                body=(
                    "(1) Parent subsection one text.\n"
                    "(2) Parent subsection two text.\n"
                    "(a) First condition.\n"
                    "(b) Second condition.\n"
                    "(3) Parent subsection three text."
                ),
                page_number=1,
            )
        ]
    )

    child_chunk = next(
        chunk
        for chunk in chunks
        if chunk.metadata.get("subsection_path")
    )

    result = validate_legal_chunk_metadata(child_chunk.metadata)

    assert result.valid
    assert child_chunk.metadata["base_provision_number"] == "11EE"
    assert child_chunk.metadata["provision_identity"].startswith(
        "ppc_1860::section::11EE"
    )


def test_legitimate_unsectioned_chunk_follows_allowed_schema() -> None:
    chunks = create_chunks(
        [
            Document(
                page_content=(
                    "Preamble text that does not contain a section heading."
                ),
                metadata={
                    "document_id": "ppc_1860",
                    "document_name": "Pakistan Penal Code, 1860",
                    "document_title": "Pakistan Penal Code, 1860",
                    "document_short_name": "Pakistan Penal Code, 1860",
                    "document_type": "legal_document",
                    "provision_type": "section",
                    "page_number": 1,
                    "page": 0,
                },
            )
        ]
    )

    unsectioned = chunks[0]

    assert unsectioned.metadata["is_unsectioned_chunk"] is True
    assert validate_legal_chunk_metadata(
        unsectioned.metadata
    ).valid


def test_unsectioned_chunk_with_none_ordinal_passes() -> None:
    metadata = make_valid_metadata(
        provision_number=None,
        provision_ordinal=None,
        is_unsectioned_chunk=True,
    )

    result = validate_legal_chunk_metadata(
        metadata,
        allow_unsectioned=True,
    )

    assert result.valid


def test_unsectioned_chunk_is_excluded_from_sequence_consistency_checks() -> None:
    first = make_valid_metadata(
        document_id="ata_1997",
        document_title="Anti-Terrorism Act, 1997",
        provision_number=None,
        provision_ordinal=None,
        is_unsectioned_chunk=True,
    )
    first.update(
        {
            "provision_identity": "ata_1997::section::unsectioned",
            "base_provision_number": None,
            "previous_provision_number": None,
            "next_provision_number": None,
            "previous_provision_identity": None,
            "next_provision_identity": None,
        }
    )

    second = make_valid_metadata(
        document_id="constitution_1973",
        document_title="Constitution of Pakistan, 1973",
        document_type="constitutional_law",
        provision_type="article",
        provision_number=None,
        provision_ordinal=None,
        is_unsectioned_chunk=True,
    )
    second.update(
        {
            "provision_identity": "constitution_1973::article::unsectioned",
            "base_provision_number": None,
            "previous_provision_number": None,
            "next_provision_number": None,
            "previous_provision_identity": None,
            "next_provision_identity": None,
        }
    )

    audit = audit_collection_metadata(
        DummyQdrantClient(
            [
                make_point("u1", first),
                make_point("u2", second),
            ]
        ),
        "pakistan_legal_knowledge_base",
    )

    assert audit.invalid_points == 0
    assert audit.consistency_errors == {}


def test_two_unsectioned_chunks_from_same_document_remain_valid() -> None:
    first = make_valid_metadata(
        provision_number=None,
        provision_ordinal=None,
        is_unsectioned_chunk=True,
    )
    first.update(
        {
            "provision_identity": "ppc_1860::section::unsectioned",
            "base_provision_number": None,
            "previous_provision_number": None,
            "next_provision_number": None,
            "previous_provision_identity": None,
            "next_provision_identity": None,
        }
    )

    second = make_valid_metadata(
        provision_number=None,
        provision_ordinal=None,
        is_unsectioned_chunk=True,
        document_chunk_number=2,
        chunk_number=2,
    )
    second.update(
        {
            "provision_identity": "ppc_1860::section::unsectioned",
            "base_provision_number": None,
            "previous_provision_number": None,
            "next_provision_number": None,
            "previous_provision_identity": None,
            "next_provision_identity": None,
        }
    )

    audit = audit_collection_metadata(
        DummyQdrantClient(
            [
                make_point("u1", first),
                make_point("u2", second),
            ]
        ),
        "pakistan_legal_knowledge_base",
    )

    assert audit.invalid_points == 0


def test_collection_audit_counts_valid_and_invalid_points_correctly() -> None:
    valid = make_valid_metadata()
    invalid = make_valid_metadata()
    invalid.pop("document_id")

    audit = audit_collection_metadata(
        DummyQdrantClient(
            [
                make_point("valid", valid),
                make_point("invalid", invalid),
            ]
        ),
        "pakistan_legal_knowledge_base",
    )

    assert audit.total_points == 2
    assert audit.valid_points == 1
    assert audit.invalid_points == 1
    assert audit.missing_required_fields["document_id"] == 1


def _make_vector_collection_client(points: list[SimpleNamespace]) -> DummyQdrantClient:
    return DummyQdrantClient(points)


def test_strict_collection_validation_rejects_bad_collection_metadata(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        vector_store_module,
        "STRICT_COLLECTION_VALIDATION",
        True,
    )

    client = _make_vector_collection_client(
        [
            make_point("valid", make_valid_metadata()),
            make_point("invalid", {**make_valid_metadata(), "document_id": ""}),
        ]
    )

    with pytest.raises(RuntimeError) as excinfo:
        vector_store_module.validate_existing_collection(client)

    assert "metadata is incompatible or corrupt" in str(excinfo.value)


def test_non_strict_collection_validation_warns_but_does_not_raise(
    monkeypatch,
    capsys,
) -> None:
    monkeypatch.setattr(
        vector_store_module,
        "STRICT_COLLECTION_VALIDATION",
        False,
    )

    client = _make_vector_collection_client(
        [
            make_point("valid", make_valid_metadata()),
            make_point("invalid", {**make_valid_metadata(), "document_id": ""}),
        ]
    )

    vector_store_module.validate_existing_collection(client)

    captured = capsys.readouterr()
    assert "Warning:" in captured.out


def test_f10_source_order_behavior_remains_unchanged() -> None:
    chunks = create_chunks(
        [
            make_page(
                document_id="ppc_1860",
                document_title="Pakistan Penal Code, 1860",
                provision_type="section",
                provision_number="11E",
                title="Provision 11E",
                body=make_long_body("Section 11E", repeats=4),
                page_number=1,
            ),
            make_page(
                document_id="ppc_1860",
                document_title="Pakistan Penal Code, 1860",
                provision_type="section",
                provision_number="11EE",
                title="Provision 11EE",
                body=make_long_body("Section 11EE", repeats=4),
                page_number=2,
            ),
            make_page(
                document_id="ppc_1860",
                document_title="Pakistan Penal Code, 1860",
                provision_type="section",
                provision_number="11F",
                title="Provision 11F",
                body=make_long_body("Section 11F", repeats=4),
                page_number=3,
            ),
        ]
    )
    retriever = FilterAwareRetriever(chunks)

    neighbors = retrieve_neighbor_documents(
        retriever=retriever,
        section_number="11EE",
        question="neighbor lookup",
        radius=1,
        top_k=5,
        enabled=True,
        document_ids=["ppc_1860"],
        provision_type="section",
    )

    assert [
        document.metadata["provision_number"]
        for document, _score in neighbors
    ] == ["11E", "11EE", "11F"]


def test_f09_none_score_behavior_remains_unchanged() -> None:
    class DummyVectorStoreNoScore:
        def similarity_search(self, query: str, k: int, filter=None):
            return [
                Document(
                    page_content="fallback doc",
                    metadata={
                        "document_id": "ppc_1860",
                        "chunk_id": "ppc_1860::section::379::part-1::chunk-1",
                    },
                )
            ]

    retriever = AdaptiveRetriever(DummyVectorStoreNoScore())
    results = retriever.search_with_scores(query="query", k=1)

    assert results[0][1] is None


def test_f08_retrieval_error_behavior_remains_unchanged() -> None:
    class DummyRetriever:
        def scroll_documents(self, metadata_filter, page_size=128):
            raise RuntimeError("qdrant down")

    with pytest.raises(MandatoryRetrievalError):
        retrieve_exact_provision_documents(
            retriever=DummyRetriever(),
            question="Section 379?",
            provision_numbers=["379"],
            provision_type="section",
            document_ids=["ppc_1860"],
            top_k=5,
        )
