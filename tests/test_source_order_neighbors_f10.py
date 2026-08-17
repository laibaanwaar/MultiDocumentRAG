from __future__ import annotations

from collections import defaultdict

import pytest
from langchain_core.documents import Document

from rag import retriever as retriever_module
from rag.retrieval_errors import OptionalRetrievalError
from rag.retriever import AdaptiveRetriever, fetch_candidates, retrieve_neighbor_documents
from rag.schemas import CandidateDocument
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


def make_long_body(label: str, repeats: int = 50) -> str:
    sentence = (
        f"{label} preserves the source order of legal provisions "
        "and keeps the context safe for retrieval."
    )
    return " ".join([sentence] * repeats)


class FilterAwareRetriever:
    def __init__(
        self,
        documents: list[Document],
        *,
        scroll_error: Exception | None = None,
        search_error: Exception | None = None,
    ) -> None:
        self.documents = documents
        self.scroll_error = scroll_error
        self.search_error = search_error

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
        if self.scroll_error is not None:
            raise self.scroll_error

        return [
            document
            for document in self.documents
            if self._matches(document, metadata_filter)
        ][:page_size]

    def search_with_scores(self, query: str, k: int, metadata_filter=None):
        if self.search_error is not None:
            raise self.search_error

        matches = [
            document
            for document in self.documents
            if self._matches(document, metadata_filter)
        ][:k]

        return [
            (document, float(index + 1))
            for index, document in enumerate(matches)
        ]


class DummyVectorStoreNoScore:
    def similarity_search(self, query: str, k: int, filter=None):
        return [
            Document(
                page_content="fallback doc",
                metadata={
                    "document_id": "ppc_1860",
                    "chunk_id": "ppc_1860::section::fallback::chunk-1",
                },
            )
        ]


def _chunks_by_provision(chunks: list[Document]) -> dict[str, list[Document]]:
    grouped: dict[str, list[Document]] = defaultdict(list)
    for chunk in chunks:
        grouped[chunk.metadata["provision_number"]].append(chunk)
    return grouped


def _section_sequence_pages(
    document_id: str,
    document_title: str,
    numbers: list[str],
    *,
    provision_type: str = "section",
) -> list[Document]:
    pages: list[Document] = []
    for index, number in enumerate(numbers, start=1):
        pages.append(
            make_page(
                document_id=document_id,
                document_title=document_title,
                provision_type=provision_type,
                provision_number=number,
                title=f"Provision {number}",
                body=make_long_body(f"Section {number}", repeats=8),
                page_number=index,
            )
        )
    return pages


def _get_provision_chunks(chunks: list[Document], provision_number: str) -> list[Document]:
    return [
        chunk
        for chunk in chunks
        if chunk.metadata.get("provision_number") == provision_number
    ]


def test_source_sequence_9_9a_10_assigns_source_order_ordinals() -> None:
    chunks = create_chunks(
        _section_sequence_pages(
            "ppc_1860",
            "Pakistan Penal Code, 1860",
            ["9", "9A", "10"],
        )
    )

    assert [
        chunk.metadata["provision_ordinal"]
        for chunk in _get_provision_chunks(chunks, "9")
    ] == [1]
    assert [
        chunk.metadata["provision_ordinal"]
        for chunk in _get_provision_chunks(chunks, "9A")
    ] == [2]
    assert [
        chunk.metadata["provision_ordinal"]
        for chunk in _get_provision_chunks(chunks, "10")
    ] == [3]

    nine_a = _get_provision_chunks(chunks, "9A")[0]
    assert nine_a.metadata["previous_provision_number"] == "9"
    assert nine_a.metadata["next_provision_number"] == "10"


def test_inserted_provisions_use_source_order_rather_than_integer_arithmetic() -> None:
    chunks = create_chunks(
        _section_sequence_pages(
            "ppc_1860",
            "Pakistan Penal Code, 1860",
            ["11E", "11EE", "11F"],
        )
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


def test_multiple_chunks_of_one_provision_share_one_provision_ordinal() -> None:
    chunks = create_chunks(
        [
            make_page(
                document_id="ppc_1860",
                document_title="Pakistan Penal Code, 1860",
                provision_type="section",
                provision_number="12",
                title="Long provision",
                body=make_long_body("Section 12", repeats=120),
                page_number=1,
            )
        ]
    )

    provision_chunks = _get_provision_chunks(chunks, "12")
    ordinals = {
        chunk.metadata["provision_ordinal"]
        for chunk in provision_chunks
    }

    assert len(provision_chunks) > 1
    assert ordinals == {1}


def test_child_subsection_chunks_inherit_parent_provision_ordinal() -> None:
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

    child_chunks = [
        chunk
        for chunk in chunks
        if chunk.metadata.get("subsection_path")
    ]
    parent_chunk = next(
        chunk
        for chunk in chunks
        if chunk.metadata["provision_number"] == "11EE"
        and not chunk.metadata.get("subsection_path")
    )

    assert child_chunks
    assert {
        chunk.metadata["provision_ordinal"]
        for chunk in child_chunks
    } == {parent_chunk.metadata["provision_ordinal"]}


def test_first_provision_has_no_previous_neighbor() -> None:
    chunks = create_chunks(
        _section_sequence_pages(
            "ppc_1860",
            "Pakistan Penal Code, 1860",
            ["9", "9A", "10"],
        )
    )

    first = _get_provision_chunks(chunks, "9")[0]
    assert first.metadata["previous_provision_number"] is None


def test_last_provision_has_no_next_neighbor() -> None:
    chunks = create_chunks(
        _section_sequence_pages(
            "ppc_1860",
            "Pakistan Penal Code, 1860",
            ["9", "9A", "10"],
        )
    )

    last = _get_provision_chunks(chunks, "10")[0]
    assert last.metadata["next_provision_number"] is None


def test_radius_two_returns_two_source_order_provisions_on_each_side() -> None:
    chunks = create_chunks(
        _section_sequence_pages(
            "ppc_1860",
            "Pakistan Penal Code, 1860",
            ["9", "9A", "10", "11", "12"],
        )
    )
    retriever = FilterAwareRetriever(chunks)

    neighbors = retrieve_neighbor_documents(
        retriever=retriever,
        section_number="10",
        question="neighbor lookup",
        radius=2,
        top_k=10,
        enabled=True,
        document_ids=["ppc_1860"],
        provision_type="section",
    )

    assert [
        document.metadata["provision_number"]
        for document, _score in neighbors
    ] == ["9", "9A", "10", "11", "12"]


def test_neighbor_expansion_cannot_cross_document_boundaries() -> None:
    chunks = create_chunks(
        _section_sequence_pages(
            "ppc_1860",
            "Pakistan Penal Code, 1860",
            ["9", "9A", "10"],
        )
        + _section_sequence_pages(
            "ata_1997",
            "Anti-Terrorism Act, 1997",
            ["9", "9A", "10"],
        )
    )
    retriever = FilterAwareRetriever(chunks)

    neighbors = retrieve_neighbor_documents(
        retriever=retriever,
        section_number="9A",
        question="neighbor lookup",
        radius=1,
        top_k=10,
        enabled=True,
        document_ids=["ppc_1860"],
        provision_type="section",
    )

    assert {document.metadata["document_id"] for document, _score in neighbors} == {
        "ppc_1860"
    }


def test_same_provision_number_in_two_documents_keeps_separate_sequences() -> None:
    chunks = create_chunks(
        _section_sequence_pages(
            "ppc_1860",
            "Pakistan Penal Code, 1860",
            ["9", "9A", "10"],
        )
        + _section_sequence_pages(
            "ata_1997",
            "Anti-Terrorism Act, 1997",
            ["9", "9A", "10"],
        )
    )

    ppc_9a = next(
        chunk
        for chunk in chunks
        if chunk.metadata["document_id"] == "ppc_1860"
        and chunk.metadata["provision_number"] == "9A"
    )
    ata_9a = next(
        chunk
        for chunk in chunks
        if chunk.metadata["document_id"] == "ata_1997"
        and chunk.metadata["provision_number"] == "9A"
    )

    assert ppc_9a.metadata["provision_ordinal"] == 2
    assert ata_9a.metadata["provision_ordinal"] == 2


def test_articles_and_sections_do_not_share_one_adjacency_sequence() -> None:
    section_chunks = create_chunks(
        _section_sequence_pages(
            "ppc_1860",
            "Pakistan Penal Code, 1860",
            ["1", "2"],
            provision_type="section",
        )
    )
    article_chunks = create_chunks(
        _section_sequence_pages(
            "constitution_1973",
            "Constitution of Pakistan, 1973",
            ["1", "2"],
            provision_type="article",
        )
    )

    assert _get_provision_chunks(section_chunks, "1")[0].metadata["provision_ordinal"] == 1
    assert _get_provision_chunks(article_chunks, "1")[0].metadata["provision_ordinal"] == 1


def test_existing_numeric_provision_behavior_still_works_through_source_order() -> None:
    chunks = create_chunks(
        _section_sequence_pages(
            "ppc_1860",
            "Pakistan Penal Code, 1860",
            ["378", "379", "380"],
        )
    )
    retriever = FilterAwareRetriever(chunks)

    neighbors = retrieve_neighbor_documents(
        retriever=retriever,
        section_number="379",
        question="neighbor lookup",
        radius=1,
        top_k=10,
        enabled=True,
        document_ids=["ppc_1860"],
        provision_type="section",
    )

    assert [
        document.metadata["provision_number"]
        for document, _score in neighbors
    ] == ["378", "379", "380"]


def test_neighbor_candidates_retain_neighbor_retrieval_method(
    monkeypatch,
) -> None:
    monkeypatch.setitem(
        retriever_module.LEGAL_CONCEPTS,
        "source_order_smoke",
        {
            "preferred_sections": {"379"},
            "preferred_documents": {"ppc_1860"},
        },
    )

    neighbor_doc = Document(
        page_content="Neighbor body.",
        metadata={
            "document_id": "ppc_1860",
            "document_name": "Pakistan Penal Code, 1860",
            "document_title": "Pakistan Penal Code, 1860",
            "document_short_name": "PPC",
            "document_type": "legal_document",
            "provision_type": "section",
            "provision_number": "379",
            "section_number": "379",
            "heading_only_chunk": False,
            "provision_body_present": True,
            "chunk_id": "ppc_1860::section::379::chunk-1",
            "provision_ordinal": 1,
            "document_chunk_number": 1,
        },
    )

    monkeypatch.setattr(
        retriever_module,
        "retrieve_neighbor_documents",
        lambda **kwargs: [(neighbor_doc, 0.8)],
    )

    candidates, _exact_documents = fetch_candidates(
        retriever=FilterAwareRetriever([]),
        question="What is Section 379?",
        queries=[],
        question_type="section_lookup",
        section_number=None,
        detected_concepts=["source_order_smoke"],
        top_k=5,
        neighbor_radius=1,
        enable_neighbor_retrieval=True,
        min_relevance_score=0.0,
        document_ids=["ppc_1860"],
        provision_type="section",
        provision_numbers=["379"],
        article_number=None,
        legal_references=None,
    )

    assert [candidate.retrieval_method for candidate in candidates] == [
        "neighbor"
    ]


def test_f08_optional_retrieval_error_behavior_remains_unchanged() -> None:
    retriever = FilterAwareRetriever(
        [],
        search_error=RuntimeError("neighbor down"),
    )

    with pytest.raises(OptionalRetrievalError):
        retrieve_neighbor_documents(
            retriever=retriever,
            section_number="379",
            question="neighbor lookup",
            radius=1,
            top_k=5,
            enabled=True,
            document_ids=["ppc_1860"],
            provision_type="section",
        )


def test_f09_none_score_behavior_remains_unchanged() -> None:
    retriever = AdaptiveRetriever(DummyVectorStoreNoScore())

    results = retriever.search_with_scores(
        query="query",
        k=1,
    )

    assert results[0][1] is None


def test_f07_document_routing_remains_unchanged(monkeypatch) -> None:
    monkeypatch.setitem(
        retriever_module.LEGAL_CONCEPTS,
        "concept_a",
        {
            "preferred_sections": {"378"},
            "preferred_documents": {"doc_a"},
        },
    )
    monkeypatch.setitem(
        retriever_module.LEGAL_CONCEPTS,
        "concept_b",
        {
            "preferred_sections": {"405"},
            "preferred_documents": {"doc_b"},
        },
    )

    calls: list[tuple[str | None, tuple[str, ...] | None]] = []

    def fake_neighbor(
        *,
        retriever,
        section_number,
        question,
        radius,
        top_k,
        enabled,
        document_ids=None,
        provision_type="section",
    ):
        calls.append(
            (
                section_number,
                tuple(document_ids or []),
            )
        )
        return []

    monkeypatch.setattr(
        retriever_module,
        "retrieve_neighbor_documents",
        fake_neighbor,
    )

    fetch_candidates(
        retriever=FilterAwareRetriever([]),
        question="neighbor route test",
        queries=[],
        question_type="general",
        section_number=None,
        detected_concepts=["concept_a", "concept_b"],
        top_k=5,
        neighbor_radius=1,
        enable_neighbor_retrieval=True,
        min_relevance_score=0.0,
        document_ids=None,
        provision_type=None,
        provision_numbers=None,
        article_number=None,
        legal_references=None,
    )

    assert calls == [
        ("378", ("doc_a",)),
        ("405", ("doc_b",)),
    ]
