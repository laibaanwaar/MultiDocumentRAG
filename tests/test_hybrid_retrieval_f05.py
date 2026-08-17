from __future__ import annotations

from langchain_core.documents import Document

from rag.ranker import rank_candidates
from rag.retriever import AdaptiveRetriever, fetch_candidates
from rag.schemas import CandidateDocument


class DummyPoint:
    def __init__(self, point_id: str, payload: dict) -> None:
        self.id = point_id
        self.payload = payload


class DummyLexicalClient:
    def __init__(self, points: list[DummyPoint]) -> None:
        self.points = points

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
        return self.points, None


class DummyVectorStore:
    def __init__(self, points: list[DummyPoint]) -> None:
        self.client = DummyLexicalClient(points)
        self.collection_name = "pakistan_legal_knowledge_base"
        self.content_payload_key = "page_content"
        self.metadata_payload_key = "metadata"


def make_point(
    *,
    point_id: str,
    document_id: str,
    chunk_id: str,
    page_content: str,
    provision_number: str | None = None,
    provision_type: str = "section",
    document_title: str = "Pakistan Penal Code 1860",
    short_name: str = "PPC",
) -> DummyPoint:
    metadata = {
        "document_id": document_id,
        "document_title": document_title,
        "document_short_name": short_name,
        "provision_type": provision_type,
        "heading_only_chunk": False,
        "provision_body_present": True,
        "chunk_id": chunk_id,
    }

    if provision_number is not None:
        metadata["provision_number"] = provision_number
        metadata["section_number"] = (
            provision_number
            if provision_type == "section"
            else None
        )
        metadata["article_number"] = (
            provision_number
            if provision_type == "article"
            else None
        )

    return DummyPoint(
        point_id=point_id,
        payload={
            "page_content": page_content,
            "metadata": metadata,
        },
    )


def make_document(
    *,
    document_id: str,
    chunk_id: str,
    page_content: str,
    provision_number: str = "379",
) -> Document:
    return Document(
        page_content=page_content,
        metadata={
            "document_id": document_id,
            "document_name": "Pakistan Penal Code 1860",
            "document_title": "Pakistan Penal Code 1860",
            "document_short_name": "PPC",
            "provision_type": "section",
            "provision_number": provision_number,
            "section_number": provision_number,
            "heading_only_chunk": False,
            "provision_body_present": True,
            "chunk_id": chunk_id,
        },
    )


def test_lexical_search_prioritizes_rare_terms_and_phrases() -> None:
    points = [
        make_point(
            point_id="rare",
            document_id="ppc_1860",
            chunk_id="ppc_1860::section::100::chunk-1",
            page_content=(
                "This clause discusses mens rea and intent "
                "in rare legal circumstances."
            ),
            provision_number="100",
        ),
        make_point(
            point_id="phrase",
            document_id="ppc_1860",
            chunk_id="ppc_1860::section::101::chunk-1",
            page_content=(
                "A person does not amount to an offence "
                "when there is lawful authority."
            ),
            provision_number="101",
        ),
        make_point(
            point_id="time",
            document_id="ppc_1860",
            chunk_id="ppc_1860::section::102::chunk-1",
            page_content=(
                "The accused shall be produced within "
                "twenty-four hours of arrest."
            ),
            provision_number="102",
        ),
    ]

    retriever = AdaptiveRetriever(
        DummyVectorStore(points)
    )

    rare_results = retriever.search_lexical(
        query="mens rea",
        k=3,
    )
    phrase_results = retriever.search_lexical(
        query="does not amount to an offence",
        k=3,
    )
    time_results = retriever.search_lexical(
        query="twenty-four hours",
        k=3,
    )

    assert rare_results[0][0].metadata["chunk_id"] == (
        "ppc_1860::section::100::chunk-1"
    )
    assert phrase_results[0][0].metadata["chunk_id"] == (
        "ppc_1860::section::101::chunk-1"
    )
    assert time_results[0][0].metadata["chunk_id"] == (
        "ppc_1860::section::102::chunk-1"
    )


def test_lexical_search_respects_document_routing() -> None:
    points = [
        make_point(
            point_id="ppc",
            document_id="ppc_1860",
            chunk_id="ppc_1860::section::200::chunk-1",
            page_content="Section 200 contains a rare legal marker.",
            provision_number="200",
        ),
        make_point(
            point_id="ata",
            document_id="ata_1997",
            chunk_id="ata_1997::section::200::chunk-1",
            page_content="Section 200 contains a rare legal marker.",
            provision_number="200",
        ),
        make_point(
            point_id="constitution",
            document_id="constitution_1973",
            chunk_id="constitution_1973::article::200::chunk-1",
            page_content="Article 200 contains a rare legal marker.",
            provision_number="200",
            provision_type="article",
            document_title="Constitution of Pakistan",
            short_name="Constitution",
        ),
    ]

    retriever = AdaptiveRetriever(
        DummyVectorStore(points)
    )

    results = retriever.search_lexical(
        query="rare legal marker",
        k=10,
        document_ids=["ppc_1860"],
    )

    assert [
        document.metadata["document_id"]
        for document, _score in results
    ] == ["ppc_1860"]


def test_fetch_candidates_emits_vector_and_lexical_candidates() -> None:
    exact_document = make_document(
        document_id="ppc_1860",
        chunk_id="ppc_1860::section::379::chunk-1",
        page_content="Section 379. Punishment for theft.",
    )
    vector_document = make_document(
        document_id="ppc_1860",
        chunk_id="ppc_1860::section::411::chunk-1",
        page_content="Section 411. Dishonestly receiving stolen property.",
        provision_number="411",
    )
    lexical_document = make_document(
        document_id="ppc_1860",
        chunk_id="ppc_1860::section::402::chunk-1",
        page_content="Section 402. Possession after the fact.",
        provision_number="402",
    )

    class DummyRetriever:
        def __init__(self) -> None:
            self.vector_queries: list[str] = []
            self.lexical_queries: list[str] = []
            self.scroll_calls: int = 0

        def scroll_documents(self, metadata_filter, page_size=128):
            self.scroll_calls += 1
            return [exact_document]

        def search_with_scores(self, query: str, k: int, metadata_filter=None):
            self.vector_queries.append(query)
            return [(vector_document, 0.88)]

        def search_lexical(
            self,
            query: str,
            k: int,
            document_ids=None,
        ):
            self.lexical_queries.append(query)
            return [(lexical_document, 4.25)]

    retriever = DummyRetriever()

    candidates, exact_documents = fetch_candidates(
        retriever=retriever,
        question="What is Section 379?",
        queries=["What is Section 379?"],
        question_type="section_lookup",
        section_number="379",
        detected_concepts=[],
        top_k=5,
        neighbor_radius=2,
        enable_neighbor_retrieval=False,
        min_relevance_score=0.5,
        document_ids=["ppc_1860"],
        provision_type="section",
        provision_numbers=["379"],
        article_number=None,
    )

    assert [doc.metadata["chunk_id"] for doc in exact_documents] == [
        "ppc_1860::section::379::chunk-1"
    ]
    assert {
        candidate.retrieval_method
        for candidate in candidates
    } == {"vector", "lexical"}
    assert retriever.vector_queries == ["What is Section 379?"]
    assert retriever.lexical_queries == ["What is Section 379?"]
    assert retriever.scroll_calls == 1


def test_rank_candidates_uses_independent_rrf_lists() -> None:
    shared_vector = Document(
        page_content="shared vector text",
        metadata={"document_id": "shared_vector", "chunk_id": "shared_vector"},
    )
    shared_lexical = Document(
        page_content="shared lexical text",
        metadata={"document_id": "shared_lexical", "chunk_id": "shared_lexical"},
    )
    vector_second = Document(
        page_content="vector second",
        metadata={"document_id": "vector_second", "chunk_id": "vector_second"},
    )
    lexical_second = Document(
        page_content="lexical second",
        metadata={"document_id": "lexical_second", "chunk_id": "lexical_second"},
    )

    items = [
        CandidateDocument(
            document=shared_vector,
            relevance_score=0.9,
            query_index=0,
            query_text="query",
            retrieval_method="vector",
        ),
        CandidateDocument(
            document=vector_second,
            relevance_score=0.8,
            query_index=0,
            query_text="query",
            retrieval_method="vector",
        ),
        CandidateDocument(
            document=shared_lexical,
            relevance_score=3.0,
            query_index=0,
            query_text="query",
            retrieval_method="lexical",
        ),
        CandidateDocument(
            document=lexical_second,
            relevance_score=2.0,
            query_index=0,
            query_text="query",
            retrieval_method="lexical",
        ),
    ]

    ranked = rank_candidates(
        question="query",
        detected_concepts=[],
        candidates=items,
        semantic_threshold=0.85,
    )

    by_id = {
        item.document_id: item
        for item in ranked
    }

    assert by_id["shared_vector"].fusion_score == by_id["shared_lexical"].fusion_score
    assert by_id["vector_second"].fusion_score == by_id["lexical_second"].fusion_score


def test_rank_candidates_keeps_bm25_out_of_semantic_similarity() -> None:
    lexical_only = Document(
        page_content="lexical only text",
        metadata={"document_id": "lexical_only", "chunk_id": "lexical_only"},
    )

    ranked = rank_candidates(
        question="lexical only text",
        detected_concepts=[],
        candidates=[
            CandidateDocument(
                document=lexical_only,
                relevance_score=999.0,
                query_index=0,
                query_text="query",
                retrieval_method="lexical",
            )
        ],
        semantic_threshold=0.85,
    )

    assert ranked[0].relevance_score is None
    assert ranked[0].fusion_score > 0.0


def test_dual_channel_reinforcement_improves_fusion_score() -> None:
    shared = Document(
        page_content="shared text",
        metadata={"document_id": "shared", "chunk_id": "shared"},
    )
    vector_only = Document(
        page_content="vector only text",
        metadata={"document_id": "vector_only", "chunk_id": "vector_only"},
    )
    lexical_only = Document(
        page_content="lexical only text",
        metadata={"document_id": "lexical_only_2", "chunk_id": "lexical_only_2"},
    )

    ranked = rank_candidates(
        question="shared text",
        detected_concepts=[],
        candidates=[
            CandidateDocument(
                document=shared,
                relevance_score=0.92,
                query_index=0,
                query_text="query",
                retrieval_method="vector",
            ),
            CandidateDocument(
                document=vector_only,
                relevance_score=0.91,
                query_index=0,
                query_text="query",
                retrieval_method="vector",
            ),
            CandidateDocument(
                document=shared,
                relevance_score=5.0,
                query_index=0,
                query_text="query",
                retrieval_method="lexical",
            ),
            CandidateDocument(
                document=lexical_only,
                relevance_score=4.0,
                query_index=0,
                query_text="query",
                retrieval_method="lexical",
            ),
        ],
        semantic_threshold=0.85,
    )

    by_id = {item.document_id: item for item in ranked}

    assert by_id["shared"].fusion_score > by_id["vector_only"].fusion_score
