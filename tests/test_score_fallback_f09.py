from __future__ import annotations

from langchain_core.documents import Document

from rag import answer_service
from rag.answer_service import build_exact_ranked_items
from rag.ranker import (
    calculate_retrieval_confidence,
    deduplicate_ranked_documents,
    rank_candidates,
)
from rag.retrieval_errors import MandatoryRetrievalError
from rag.retriever import AdaptiveRetriever, fetch_candidates
from rag.schemas import CandidateDocument, QueryPlan


class DummyVectorStoreNoScore:
    def similarity_search(self, query: str, k: int, filter=None):
        return [
            Document(
                page_content="doc one",
                metadata={"document_id": "doc1", "chunk_id": "chunk1"},
            )
        ]


class DummyRetriever:
    def __init__(
        self,
        *,
        vector_results: list[tuple[Document, float | None]] | None = None,
        lexical_results: list[tuple[Document, float]] | None = None,
    ) -> None:
        self.vector_results = vector_results or []
        self.lexical_results = lexical_results or []

    def search_with_scores(self, query: str, k: int, metadata_filter=None):
        return list(self.vector_results)

    def search_lexical(self, query: str, k: int, document_ids=None):
        return list(self.lexical_results)


class DummyChatModel:
    def __init__(self) -> None:
        self.invocations = 0

    def invoke(self, prompt):
        self.invocations += 1
        return "Grounded answer."


def make_document(
    *,
    document_id: str,
    chunk_id: str,
    page_content: str,
    provision_number: str,
    provision_type: str = "section",
    extra_metadata: dict | None = None,
) -> Document:
    metadata = {
        "document_id": document_id,
        "document_name": document_id,
        "document_title": document_id,
        "document_short_name": document_id,
        "provision_type": provision_type,
        "provision_number": provision_number,
        "section_number": (
            provision_number
            if provision_type == "section"
            else None
        ),
        "article_number": (
            provision_number
            if provision_type == "article"
            else None
        ),
        "heading_only_chunk": False,
        "provision_body_present": True,
        "chunk_id": chunk_id,
    }
    if extra_metadata:
        metadata.update(extra_metadata)

    return Document(
        page_content=page_content,
        metadata=metadata,
    )


def make_candidate(
    document: Document,
    *,
    score: float | None,
    query_index: int,
    query_text: str,
    retrieval_method: str,
    retrieval_rank: int | None = None,
) -> CandidateDocument:
    return CandidateDocument(
        document=document,
        relevance_score=score,
        query_index=query_index,
        query_text=query_text,
        retrieval_method=retrieval_method,
        retrieval_rank=retrieval_rank,
    )


def test_search_with_scores_fallback_returns_none_score() -> None:
    retriever = AdaptiveRetriever(DummyVectorStoreNoScore())

    results = retriever.search_with_scores(
        query="query",
        k=1,
    )

    assert results == [
        (
            results[0][0],
            None,
        )
    ]


def test_unscored_fallback_document_is_retained_under_positive_threshold() -> None:
    doc = make_document(
        document_id="ppc_1860",
        chunk_id="ppc_1860::section::379::chunk-1",
        page_content="Section 379 text.",
        provision_number="379",
    )
    candidates, _ = fetch_candidates(
        retriever=DummyRetriever(
            vector_results=[(doc, None)]
        ),
        question="Section 379?",
        queries=["Section 379?"],
        question_type="section_lookup",
        section_number="379",
        detected_concepts=[],
        top_k=5,
        neighbor_radius=1,
        enable_neighbor_retrieval=False,
        min_relevance_score=0.5,
        document_ids=["ppc_1860"],
        provision_type="section",
        provision_numbers=["379"],
        article_number=None,
        legal_references=None,
    )

    assert len(candidates) == 1
    assert candidates[0].relevance_score is None


def test_real_vector_score_below_threshold_is_rejected() -> None:
    doc = make_document(
        document_id="ppc_1860",
        chunk_id="ppc_1860::section::380::chunk-1",
        page_content="Section 380 text.",
        provision_number="380",
    )
    candidates, _ = fetch_candidates(
        retriever=DummyRetriever(
            vector_results=[(doc, 0.4)]
        ),
        question="Section 380?",
        queries=["Section 380?"],
        question_type="section_lookup",
        section_number="380",
        detected_concepts=[],
        top_k=5,
        neighbor_radius=1,
        enable_neighbor_retrieval=False,
        min_relevance_score=0.5,
        document_ids=["ppc_1860"],
        provision_type="section",
        provision_numbers=["380"],
        article_number=None,
        legal_references=None,
    )

    assert candidates == []


def test_real_vector_score_above_threshold_is_retained() -> None:
    doc = make_document(
        document_id="ppc_1860",
        chunk_id="ppc_1860::section::381::chunk-1",
        page_content="Section 381 text.",
        provision_number="381",
    )
    candidates, _ = fetch_candidates(
        retriever=DummyRetriever(
            vector_results=[(doc, 0.8)]
        ),
        question="Section 381?",
        queries=["Section 381?"],
        question_type="section_lookup",
        section_number="381",
        detected_concepts=[],
        top_k=5,
        neighbor_radius=1,
        enable_neighbor_retrieval=False,
        min_relevance_score=0.5,
        document_ids=["ppc_1860"],
        provision_type="section",
        provision_numbers=["381"],
        article_number=None,
        legal_references=None,
    )

    assert len(candidates) == 1
    assert candidates[0].relevance_score == 0.8


def test_candidate_document_preserves_retrieval_rank_for_fallback_docs() -> None:
    doc = make_document(
        document_id="ppc_1860",
        chunk_id="ppc_1860::section::382::chunk-1",
        page_content="Section 382 text.",
        provision_number="382",
    )

    candidates, _ = fetch_candidates(
        retriever=DummyRetriever(
            vector_results=[(doc, None)]
        ),
        question="Section 382?",
        queries=["Section 382?"],
        question_type="section_lookup",
        section_number="382",
        detected_concepts=[],
        top_k=5,
        neighbor_radius=1,
        enable_neighbor_retrieval=False,
        min_relevance_score=0.0,
        document_ids=["ppc_1860"],
        provision_type="section",
        provision_numbers=["382"],
        article_number=None,
        legal_references=None,
    )

    assert candidates[0].retrieval_rank == 1


def test_unscored_fallback_documents_preserve_backend_rank_in_rrf() -> None:
    doc_a = make_document(
        document_id="ppc_1860",
        chunk_id="ppc_1860::section::383::chunk-1",
        page_content="A",
        provision_number="383",
    )
    doc_b = make_document(
        document_id="ppc_1860",
        chunk_id="ppc_1860::section::384::chunk-1",
        page_content="B",
        provision_number="384",
    )

    ranked = rank_candidates(
        question="query",
        detected_concepts=[],
        candidates=[
            make_candidate(
                doc_b,
                score=None,
                query_index=0,
                query_text="query",
                retrieval_method="vector",
                retrieval_rank=2,
            ),
            make_candidate(
                doc_a,
                score=None,
                query_index=0,
                query_text="query",
                retrieval_method="vector",
                retrieval_rank=1,
            ),
        ],
        semantic_threshold=0.94,
    )

    assert [item.document_id for item in ranked] == [
        "ppc_1860",
        "ppc_1860",
    ]
    assert ranked[0].document.metadata["chunk_id"] == (
        "ppc_1860::section::383::chunk-1"
    )
    assert ranked[1].document.metadata["chunk_id"] == (
        "ppc_1860::section::384::chunk-1"
    )


def test_reversing_insertion_order_does_not_change_rank_when_retrieval_rank_is_supplied() -> None:
    doc_a = make_document(
        document_id="ppc_1860",
        chunk_id="ppc_1860::section::385::chunk-1",
        page_content="A",
        provision_number="385",
    )
    doc_b = make_document(
        document_id="ppc_1860",
        chunk_id="ppc_1860::section::386::chunk-1",
        page_content="B",
        provision_number="386",
    )

    candidates = [
        make_candidate(
            doc_a,
            score=None,
            query_index=0,
            query_text="query",
            retrieval_method="vector",
            retrieval_rank=1,
        ),
        make_candidate(
            doc_b,
            score=None,
            query_index=0,
            query_text="query",
            retrieval_method="vector",
            retrieval_rank=2,
        ),
    ]

    ranked_forward = rank_candidates(
        question="query",
        detected_concepts=[],
        candidates=candidates,
        semantic_threshold=0.94,
    )
    ranked_reversed = rank_candidates(
        question="query",
        detected_concepts=[],
        candidates=list(reversed(candidates)),
        semantic_threshold=0.94,
    )

    assert [
        item.document.metadata["chunk_id"]
        for item in ranked_forward
    ] == [
        item.document.metadata["chunk_id"]
        for item in ranked_reversed
    ]


def test_none_does_not_become_fake_zero_semantic_score() -> None:
    doc = make_document(
        document_id="ppc_1860",
        chunk_id="ppc_1860::section::387::chunk-1",
        page_content="Section 387 text.",
        provision_number="387",
    )

    ranked = rank_candidates(
        question="query",
        detected_concepts=[],
        candidates=[
            make_candidate(
                doc,
                score=None,
                query_index=0,
                query_text="query",
                retrieval_method="vector",
                retrieval_rank=1,
            )
        ],
        semantic_threshold=0.94,
    )

    assert ranked[0].relevance_score is None


def test_real_vector_score_overwrites_unscored_candidate_for_same_chunk() -> None:
    doc = make_document(
        document_id="ppc_1860",
        chunk_id="ppc_1860::section::388::chunk-1",
        page_content="Section 388 text.",
        provision_number="388",
    )

    ranked = rank_candidates(
        question="query",
        detected_concepts=[],
        candidates=[
            make_candidate(
                doc,
                score=None,
                query_index=0,
                query_text="query",
                retrieval_method="vector",
                retrieval_rank=1,
            ),
            make_candidate(
                doc,
                score=0.83,
                query_index=0,
                query_text="query",
                retrieval_method="vector",
                retrieval_rank=2,
            ),
        ],
        semantic_threshold=0.94,
    )

    assert ranked[0].relevance_score == 0.83


def test_lexical_scores_remain_isolated_from_semantic_similarity() -> None:
    doc = make_document(
        document_id="ppc_1860",
        chunk_id="ppc_1860::section::389::chunk-1",
        page_content="Section 389 text.",
        provision_number="389",
    )

    ranked = rank_candidates(
        question="query",
        detected_concepts=[],
        candidates=[
            make_candidate(
                doc,
                score=9.5,
                query_index=0,
                query_text="query",
                retrieval_method="lexical",
            )
        ],
        semantic_threshold=0.94,
    )

    assert ranked[0].relevance_score is None


def test_f08_retrieval_error_behavior_remains_unchanged(monkeypatch) -> None:
    plan = QueryPlan(
        original_question="What is Section 379?",
        question_type="section_lookup",
        concepts=["theft"],
        section_number="379",
        retrieval_queries=["section 379"],
        document_ids=["ppc_1860"],
        provision_numbers=["379"],
        provision_type="section",
    )

    monkeypatch.setattr(
        answer_service,
        "fetch_candidates",
        lambda **kwargs: (_ for _ in ()).throw(
            MandatoryRetrievalError(
                "broken",
                operation="scroll_exact_provision",
                route="provision:type=section;numbers=379;document_ids=ppc_1860",
                collection="legal",
                original_exception=RuntimeError("boom"),
            )
        ),
    )

    result = answer_service.answer_question(
        question=plan.original_question,
        retriever=object(),
        chat_model=DummyChatModel(),
        plan=plan,
        retrieval_k=5,
    )

    assert result["retrieval_status"] == "error"
    assert (
        result["answer"]
        != "The answer was not found in the four indexed legal documents."
    )


def test_f06_and_f05_behavior_remain_unchanged() -> None:
    shared = make_document(
        document_id="ppc_1860",
        chunk_id="ppc_1860::section::390::chunk-1",
        page_content="shared text",
        provision_number="390",
    )
    exact_shared = make_document(
        document_id="ppc_1860",
        chunk_id="ppc_1860::section::390::exact",
        page_content="shared text",
        provision_number="390",
    )

    ranked = rank_candidates(
        question="query",
        detected_concepts=[],
        candidates=[
            make_candidate(
                shared,
                score=0.9,
                query_index=0,
                query_text="query",
                retrieval_method="vector",
                retrieval_rank=1,
            ),
            make_candidate(
                shared,
                score=7.0,
                query_index=0,
                query_text="query",
                retrieval_method="lexical",
            ),
        ],
        semantic_threshold=0.94,
    )

    merged = deduplicate_ranked_documents(
        build_exact_ranked_items([exact_shared]) + ranked,
        semantic_threshold=0.94,
    )

    assert len(merged) == 1
    assert set(merged[0].retrieval_methods) == {
        "exact",
        "vector",
        "lexical",
    }
