from langchain_core.documents import Document

from rag.intent_router import classify_question, route_question
from rag.intent_router import (
    extract_article_numbers,
    extract_legal_references,
    extract_section_numbers,
)
from rag.retriever import (
    AdaptiveRetriever,
    fetch_candidates,
    retrieve_exact_provision_documents,
)
from rag.schemas import LegalReference


def test_bare_section_379_uses_section_lookup_classification() -> None:
    assert classify_question("What does Section 379 state?") == "section_lookup"


def test_vs_comparison_is_treated_as_a_comparison() -> None:
    assert (
        classify_question("PPC Section 379 vs CrPC Section 379")
        == "comparison"
    )


def test_extract_section_numbers_supports_numeric_single_multi_and_hyphenated() -> None:
    assert extract_section_numbers("Section 379 of PPC") == ["379"]
    assert extract_section_numbers("Section 21A applies here") == ["21A"]
    assert extract_section_numbers("Section 11EE may be relevant") == ["11EE"]
    assert extract_section_numbers("Section 21AA and section 21aa") == ["21AA"]
    assert extract_section_numbers("Sections 153-A and 298-B are cited") == [
        "153A",
        "298B",
    ]
    assert extract_section_numbers("Section 175A and section 270AA") == [
        "175A",
        "270AA",
    ]


def test_extract_article_numbers_supports_numeric_single_multi_and_hyphenated() -> None:
    assert extract_article_numbers("Article 10 of the Constitution") == ["10"]
    assert extract_article_numbers("Article 21A is invoked") == ["21A"]
    assert extract_article_numbers("Article 11EE is referenced") == ["11EE"]
    assert extract_article_numbers("Article 21AA and article 21aa") == ["21AA"]
    assert extract_article_numbers("Articles 153-A and 298-B are compared") == [
        "153A",
        "298B",
    ]
    assert extract_article_numbers("Article 175A and article 270AA") == [
        "175A",
        "270AA",
    ]


def test_extract_legal_references_supports_section_11EE() -> None:
    assert extract_legal_references("Section 11EE") == [
        LegalReference(
            provision_type="section",
            base_number="11EE",
            subsection_path=[],
            component_type=None,
            original_citation="Section 11EE",
        )
    ]


def test_extract_legal_references_supports_section_11EE_subsection() -> None:
    assert extract_legal_references("Section 11EE(2)") == [
        LegalReference(
            provision_type="section",
            base_number="11EE",
            subsection_path=["2"],
            component_type="subsection",
            original_citation="Section 11EE(2)",
        )
    ]


def test_extract_legal_references_supports_section_11EE_clause() -> None:
    assert extract_legal_references("Section 11EE(2)(b)") == [
        LegalReference(
            provision_type="section",
            base_number="11EE",
            subsection_path=["2", "b"],
            component_type="clause",
            original_citation="Section 11EE(2)(b)",
        )
    ]


def test_extract_legal_references_supports_article_10_subsection() -> None:
    assert extract_legal_references("Article 10(2)") == [
        LegalReference(
            provision_type="article",
            base_number="10",
            subsection_path=["2"],
            component_type="subsection",
            original_citation="Article 10(2)",
        )
    ]


def test_extract_legal_references_supports_alphanumeric_and_hyphenated_bases() -> None:
    references = extract_legal_references(
        "Section 21A, Section 21AA, Section 153-A, Article 298-B"
    )

    assert [
        reference.provision_type for reference in references
    ] == ["section", "section", "section", "article"]
    assert [
        reference.base_number for reference in references
    ] == ["21A", "21AA", "153A", "298B"]


def test_route_question_preserves_section_379_legacy_behavior_and_exposes_legal_reference() -> None:
    plan = route_question("What does Section 379 state?")

    assert plan.question_type == "section_lookup"
    assert plan.section_number == "379"
    assert plan.article_number is None
    assert plan.provision_numbers == ["379"]
    assert plan.legal_references == [
        LegalReference(
            provision_type="section",
            base_number="379",
            subsection_path=[],
            component_type=None,
            original_citation="Section 379",
        )
    ]


def test_theft_scenario_routes_to_ppc_and_sections_378_379() -> None:
    plan = route_question(
        "Ali dishonestly took a mobile phone without the owner's consent."
    )

    assert plan.question_type == "fact_scenario"
    assert plan.concepts == ["theft"]
    assert plan.document_ids == ["ppc_1860"]
    assert set(plan.section_hints) == {"378", "379"}
    assert set(plan.provision_numbers) == {"378", "379"}


def test_theft_scenario_keeps_exact_hits_and_still_runs_semantic_retrieval() -> None:
    exact_378 = Document(
        page_content="Section 378. Theft.",
        metadata={
            "document_id": "ppc_1860",
            "document_name": "Pakistan Penal Code 1860",
            "provision_type": "section",
            "provision_number": "378",
            "section_number": "378",
            "heading_only_chunk": False,
            "chunk_id": "ppc_1860:section:378:1",
        },
    )
    exact_379 = Document(
        page_content="Section 379. Punishment for theft.",
        metadata={
            "document_id": "ppc_1860",
            "document_name": "Pakistan Penal Code 1860",
            "provision_type": "section",
            "provision_number": "379",
            "section_number": "379",
            "heading_only_chunk": False,
            "chunk_id": "ppc_1860:section:379:1",
        },
    )
    semantic_411 = Document(
        page_content="Section 411. Dishonestly receiving stolen property.",
        metadata={
            "document_id": "ppc_1860",
            "document_name": "Pakistan Penal Code 1860",
            "provision_type": "section",
            "provision_number": "411",
            "section_number": "411",
            "heading_only_chunk": False,
            "chunk_id": "ppc_1860:section:411:1",
        },
    )

    class DummyRetriever:
        def __init__(self) -> None:
            self.queries: list[str] = []
            self.scroll_filters: list[object] = []

        def scroll_documents(self, metadata_filter, page_size=128):
            self.scroll_filters.append(metadata_filter)
            return [exact_378, exact_379]

        def search_with_scores(self, query: str, k: int, metadata_filter=None):
            self.queries.append(query)

            return [
                (semantic_411, 0.95),
            ]

    plan = route_question(
        "Ali dishonestly took a mobile phone without the owner's consent."
    )

    retriever = DummyRetriever()
    candidates, exact_documents = fetch_candidates(
        retriever=retriever,
        question=plan.original_question,
        queries=plan.retrieval_queries,
        question_type=plan.question_type,
        section_number=plan.section_number,
        detected_concepts=plan.concepts,
        top_k=5,
        neighbor_radius=2,
        enable_neighbor_retrieval=False,
        min_relevance_score=0.0,
        document_ids=plan.document_ids,
        provision_type=plan.provision_type,
        provision_numbers=plan.provision_numbers,
        article_number=plan.article_number,
    )

    assert [doc.metadata["section_number"] for doc in exact_documents] == [
        "378",
        "379",
    ]
    assert {
        candidate.document.metadata["section_number"]
        for candidate in candidates
    } == {"411"}
    assert retriever.queries == plan.retrieval_queries
    assert len(retriever.scroll_filters) == 1


def test_fact_scenario_returns_exact_hits_and_supporting_candidates() -> None:
    exact_378 = Document(
        page_content="Section 378. Theft.",
        metadata={
            "document_id": "ppc_1860",
            "document_name": "Pakistan Penal Code 1860",
            "provision_type": "section",
            "provision_number": "378",
            "section_number": "378",
            "heading_only_chunk": False,
            "chunk_id": "ppc_1860:section:378:1",
        },
    )
    exact_379 = Document(
        page_content="Section 379. Punishment for theft.",
        metadata={
            "document_id": "ppc_1860",
            "document_name": "Pakistan Penal Code 1860",
            "provision_type": "section",
            "provision_number": "379",
            "section_number": "379",
            "heading_only_chunk": False,
            "chunk_id": "ppc_1860:section:379:1",
        },
    )
    semantic_411 = Document(
        page_content="Section 411. Dishonestly receiving stolen property.",
        metadata={
            "document_id": "ppc_1860",
            "document_name": "Pakistan Penal Code 1860",
            "provision_type": "section",
            "provision_number": "411",
            "section_number": "411",
            "heading_only_chunk": False,
            "chunk_id": "ppc_1860:section:411:1",
        },
    )

    class DummyRetriever:
        def __init__(self) -> None:
            self.queries: list[str] = []
            self.scroll_filters: list[object] = []

        def scroll_documents(self, metadata_filter, page_size=128):
            self.scroll_filters.append(metadata_filter)
            return [exact_378, exact_379]

        def search_with_scores(self, query: str, k: int, metadata_filter=None):
            self.queries.append(query)

            return [
                (semantic_411, 0.95),
            ]

    plan = route_question(
        "Ali dishonestly took a mobile phone without the owner's consent."
    )

    assert plan.question_type == "fact_scenario"

    retriever = DummyRetriever()
    candidates, exact_documents = fetch_candidates(
        retriever=retriever,
        question=plan.original_question,
        queries=plan.retrieval_queries,
        question_type=plan.question_type,
        section_number=plan.section_number,
        detected_concepts=plan.concepts,
        top_k=5,
        neighbor_radius=2,
        enable_neighbor_retrieval=False,
        min_relevance_score=0.0,
        document_ids=plan.document_ids,
        provision_type=plan.provision_type,
        provision_numbers=plan.provision_numbers,
        article_number=plan.article_number,
    )

    assert [doc.metadata["section_number"] for doc in exact_documents] == [
        "378",
        "379",
    ]
    assert {
        candidate.document.metadata["section_number"]
        for candidate in candidates
    } == {"411"}
    assert retriever.queries == plan.retrieval_queries
    assert len(retriever.scroll_filters) == 1


def test_exact_provision_scroll_returns_all_valid_chunks_in_source_order() -> None:
    class DummyPoint:
        def __init__(self, point_id, payload) -> None:
            self.id = point_id
            self.payload = payload

    class DummyClient:
        def __init__(self) -> None:
            self.calls: list[dict[str, object]] = []

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
            self.calls.append(
                {
                    "collection_name": collection_name,
                    "offset": offset,
                    "limit": limit,
                    "with_payload": with_payload,
                    "with_vectors": with_vectors,
                    "scroll_filter": scroll_filter,
                }
            )

            first_page = [
                DummyPoint(
                    "late-part",
                    {
                        "page_content": "Section 379 body - later chunk.",
                        "metadata": {
                            "document_id": "ppc_1860",
                            "provision_type": "section",
                            "provision_number": "379",
                            "section_number": "379",
                            "heading_only_chunk": False,
                            "provision_body_present": True,
                            "document_chunk_number": 12,
                            "chunk_number": 12,
                            "page_start": 2,
                            "page_end": 2,
                            "chunk_id": "ppc_1860::section::379::part-2::chunk-12",
                            "provision_part_number": 2,
                        },
                    },
                ),
                DummyPoint(
                    "heading-only",
                    {
                        "page_content": "Section 379. Theft.",
                        "metadata": {
                            "document_id": "ppc_1860",
                            "provision_type": "section",
                            "provision_number": "379",
                            "section_number": "379",
                            "heading_only_chunk": True,
                            "provision_body_present": True,
                            "document_chunk_number": 10,
                            "chunk_number": 10,
                            "page_start": 1,
                            "page_end": 1,
                            "chunk_id": "ppc_1860::section::379::heading",
                            "provision_part_number": 1,
                        },
                    },
                ),
            ]

            second_page = [
                DummyPoint(
                    "other-document",
                    {
                        "page_content": "Wrong law Section 379 body.",
                        "metadata": {
                            "document_id": "crpc_1898",
                            "provision_type": "section",
                            "provision_number": "379",
                            "section_number": "379",
                            "heading_only_chunk": False,
                            "provision_body_present": True,
                            "document_chunk_number": 4,
                            "chunk_number": 4,
                            "page_start": 7,
                            "page_end": 7,
                            "chunk_id": "crpc_1898::section::379::chunk-4",
                            "provision_part_number": 1,
                        },
                    },
                ),
                DummyPoint(
                    "first-body",
                    {
                        "page_content": "Section 379 body - first chunk.",
                        "metadata": {
                            "document_id": "ppc_1860",
                            "provision_type": "section",
                            "provision_number": "379",
                            "section_number": "379",
                            "heading_only_chunk": False,
                            "provision_body_present": True,
                            "document_chunk_number": 11,
                            "chunk_number": 11,
                            "page_start": 1,
                            "page_end": 1,
                            "chunk_id": "ppc_1860::section::379::part-1::chunk-11",
                            "provision_part_number": 1,
                        },
                    },
                ),
                DummyPoint(
                    "bodyless",
                    {
                        "page_content": "Section 379 bodyless chunk.",
                        "metadata": {
                            "document_id": "ppc_1860",
                            "provision_type": "section",
                            "provision_number": "379",
                            "section_number": "379",
                            "heading_only_chunk": False,
                            "provision_body_present": False,
                            "document_chunk_number": 13,
                            "chunk_number": 13,
                            "page_start": 3,
                            "page_end": 3,
                            "chunk_id": "ppc_1860::section::379::bodyless",
                            "provision_part_number": 3,
                        },
                    },
                ),
            ]

            if offset is None:
                return first_page, "page-2"

            if offset == "page-2":
                return second_page, None

            return [], None

    class DummyVectorStore:
        def __init__(self) -> None:
            self.client = DummyClient()
            self.collection_name = "pakistan_legal_knowledge_base"
            self.content_payload_key = "page_content"
            self.metadata_payload_key = "metadata"

    retriever = AdaptiveRetriever(DummyVectorStore())

    results = retrieve_exact_provision_documents(
        retriever=retriever,
        question="What is the punishment for theft under Section 379?",
        provision_numbers=["379"],
        provision_type="section",
        document_ids=["ppc_1860"],
        top_k=4,
    )

    documents = [document for document, _score in results]

    assert [document.page_content for document in documents] == [
        "Section 379 body - first chunk.",
        "Section 379 body - later chunk.",
    ]
    assert [document.metadata["document_chunk_number"] for document in documents] == [11, 12]
    assert len(retriever.vector_store.client.calls) == 2
