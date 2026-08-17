from __future__ import annotations

from langchain_core.documents import Document

from rag import answer_service, retriever as retriever_module
from rag.answer_service import build_exact_ranked_items
from rag.retrieval_errors import (
    MandatoryRetrievalError,
    OptionalRetrievalError,
)
from rag.ranker import (
    deduplicate_ranked_documents,
    rank_candidates,
)
from rag.schemas import CandidateDocument, LegalReference, QueryPlan


class DummyRetriever:
    def __init__(
        self,
        *,
        scroll_error: Exception | None = None,
        search_error: Exception | None = None,
        search_results: list[tuple[Document, float]] | None = None,
    ) -> None:
        self.scroll_error = scroll_error
        self.search_error = search_error
        self.search_results = search_results or []
        self.scroll_calls = 0
        self.search_calls = 0

    def scroll_documents(self, metadata_filter, page_size=128):
        self.scroll_calls += 1

        if self.scroll_error is not None:
            raise self.scroll_error

        return []

    def search_with_scores(self, query: str, k: int, metadata_filter=None):
        self.search_calls += 1

        if self.search_error is not None:
            raise self.search_error

        return list(self.search_results)


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
    subsection_path_key: str | None = None,
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

    if subsection_path_key is not None:
        metadata["subsection_path_key"] = subsection_path_key

    return Document(
        page_content=page_content,
        metadata=metadata,
    )


def make_reference(
    *,
    base_number: str = "379",
    subsection_path: list[str] | None = None,
) -> LegalReference:
    return LegalReference(
        provision_type="section",
        base_number=base_number,
        subsection_path=subsection_path or [],
        component_type=None,
        original_citation=f"Section {base_number}",
    )


def make_candidate(
    document: Document,
    *,
    score: float,
    query_index: int,
    query_text: str,
    retrieval_method: str,
) -> CandidateDocument:
    return CandidateDocument(
        document=document,
        relevance_score=score,
        query_index=query_index,
        query_text=query_text,
        retrieval_method=retrieval_method,
    )


def test_exact_provision_qdrant_failure_raises_mandatory_error() -> None:
    retriever = DummyRetriever(scroll_error=RuntimeError("qdrant down"))

    try:
        retriever_module.retrieve_exact_provision_documents(
            retriever=retriever,
            question="Section 379?",
            provision_numbers=["379"],
            provision_type="section",
            document_ids=["ppc_1860"],
            top_k=5,
        )
    except MandatoryRetrievalError as exc:
        assert exc.operation in {
            "scroll_exact_provision",
            "build_exact_provision_filter",
        }
    else:
        raise AssertionError("MandatoryRetrievalError was not raised.")


def test_exact_child_operational_failure_does_not_trigger_parent_fallback(
    monkeypatch,
) -> None:
    retriever = DummyRetriever(scroll_error=RuntimeError("child down"))
    called = {"parent": 0}

    def fake_parent(*args, **kwargs):
        called["parent"] += 1
        return []

    monkeypatch.setattr(
        retriever_module,
        "retrieve_exact_provision_documents",
        fake_parent,
    )

    try:
        retriever_module.retrieve_exact_legal_reference_documents(
            retriever=retriever,
            question="Section 379 clause",
            legal_references=[make_reference(subsection_path=["1"])],
            document_ids=["ppc_1860"],
            top_k=5,
        )
    except MandatoryRetrievalError:
        pass
    else:
        raise AssertionError("MandatoryRetrievalError was not raised.")

    assert called["parent"] == 0


def test_child_zero_match_still_triggers_parent_fallback(monkeypatch) -> None:
    retriever = DummyRetriever()
    parent_doc = make_document(
        document_id="ppc_1860",
        chunk_id="ppc_1860::section::379::parent",
        page_content="Section 379 parent chunk.",
        provision_number="379",
    )

    monkeypatch.setattr(
        retriever_module,
        "retrieve_exact_provision_documents",
        lambda **kwargs: [(parent_doc, 1.0)],
    )

    result = retriever_module.retrieve_exact_legal_reference_documents(
        retriever=retriever,
        question="Section 379 clause",
        legal_references=[make_reference(subsection_path=["1"])],
        document_ids=["ppc_1860"],
        top_k=5,
    )

    assert [document.metadata["chunk_id"] for document in result] == [
        "ppc_1860::section::379::parent"
    ]


def test_broken_exact_filter_is_observable_and_not_converted_to_empty_list(
    monkeypatch,
) -> None:
    retriever = DummyRetriever()

    def broken_filter(*args, **kwargs):
        raise ValueError("bad filter")

    monkeypatch.setattr(
        retriever_module,
        "build_provision_filter",
        broken_filter,
    )

    try:
        retriever_module.retrieve_exact_provision_documents(
            retriever=retriever,
            question="Section 379?",
            provision_numbers=["379"],
            provision_type="section",
            document_ids=["ppc_1860"],
            top_k=5,
        )
    except MandatoryRetrievalError as exc:
        assert exc.category == "ValueError"
    else:
        raise AssertionError("MandatoryRetrievalError was not raised.")


def test_neighbor_retrieval_operational_failure_is_observable() -> None:
    retriever = DummyRetriever(search_error=RuntimeError("neighbor down"))

    try:
        retriever_module.retrieve_neighbor_documents(
            retriever=retriever,
            section_number="379",
            question="neighbor",
            radius=1,
            top_k=5,
            enabled=True,
            document_ids=["ppc_1860"],
            provision_type="section",
        )
    except OptionalRetrievalError as exc:
        assert exc.operation == "search_neighbor_documents"
    else:
        raise AssertionError("OptionalRetrievalError was not raised.")


def test_genuine_neighbor_zero_match_remains_empty_result() -> None:
    retriever = DummyRetriever(search_results=[])

    result = retriever_module.retrieve_neighbor_documents(
        retriever=retriever,
        section_number="379",
        question="neighbor",
        radius=1,
        top_k=5,
        enabled=True,
        document_ids=["ppc_1860"],
        provision_type="section",
    )

    assert result == []


def test_answer_question_does_not_invoke_llm_when_mandatory_retrieval_fails(
    monkeypatch,
) -> None:
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
    chat_model = DummyChatModel()

    def broken_fetch_candidates(**kwargs):
        raise MandatoryRetrievalError(
            "broken",
            operation="scroll_exact_provision",
            route="provision:type=section;numbers=379;document_ids=ppc_1860",
            collection="legal",
            original_exception=RuntimeError("boom"),
        )

    monkeypatch.setattr(
        answer_service,
        "fetch_candidates",
        broken_fetch_candidates,
    )

    result = answer_service.answer_question(
        question=plan.original_question,
        retriever=object(),
        chat_model=chat_model,
        plan=plan,
        retrieval_k=5,
    )

    assert chat_model.invocations == 0
    assert result["retrieval_status"] == "error"
    assert result["answer"].startswith(
        "The legal sources could not be retrieved"
    )


def test_mandatory_retrieval_failure_does_not_return_not_found_message(
    monkeypatch,
) -> None:
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


def test_legitimate_zero_result_retrieval_still_returns_empty_result(
    monkeypatch,
) -> None:
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
    chat_model = DummyChatModel()

    monkeypatch.setattr(
        answer_service,
        "fetch_candidates",
        lambda **kwargs: ([], []),
    )

    result = answer_service.answer_question(
        question=plan.original_question,
        retriever=object(),
        chat_model=chat_model,
        plan=plan,
        retrieval_k=5,
    )

    assert chat_model.invocations == 0
    assert result["retrieval_status"] == "no_match"
    assert result["answer"] == (
        "The answer was not found in the four indexed legal documents."
    )


def test_explicit_nonexistent_provision_skips_semantic_substitution(
    monkeypatch,
) -> None:
    plan = QueryPlan(
        original_question="What is the punishment under Section 11EEEEE(2)?",
        question_type="section_lookup",
        concepts=[],
        section_number="11EEEEE",
        retrieval_queries=["Section 11EEEEE(2)"],
        document_ids=["ata_1997"],
        provision_numbers=["11EEEEE"],
        provision_type="section",
    )
    chat_model = DummyChatModel()
    fallback_document = make_document(
        document_id="ata_1997",
        chunk_id="ata_1997::section::11EE",
        page_content="Section 11EE is nearby but not exact.",
        provision_number="11EE",
    )

    monkeypatch.setattr(
        answer_service,
        "fetch_candidates",
        lambda **kwargs: (
            [
                make_candidate(
                    fallback_document,
                    score=0.95,
                    query_index=0,
                    query_text="Section 11EEEEE(2)",
                    retrieval_method="vector",
                )
            ],
            [],
        ),
    )

    result = answer_service.answer_question(
        question=plan.original_question,
        retriever=object(),
        chat_model=chat_model,
        plan=plan,
        retrieval_k=5,
    )

    assert chat_model.invocations == 0
    assert result["retrieval_status"] == "no_match"
    assert result["answer"] == (
        "The answer was not found in the four indexed legal documents."
    )


def test_valid_parent_fallback_still_generates_an_answer(
    monkeypatch,
) -> None:
    plan = QueryPlan(
        original_question="What does Section 379(1) say?",
        question_type="section_lookup",
        concepts=[],
        section_number="379",
        retrieval_queries=["Section 379(1)"],
        document_ids=["ppc_1860"],
        provision_numbers=["379"],
        provision_type="section",
        legal_references=[make_reference(subsection_path=["1"])],
    )
    chat_model = DummyChatModel()
    parent_document = make_document(
        document_id="ppc_1860",
        chunk_id="ppc_1860::section::379::parent",
        page_content="Section 379 parent chunk.",
        provision_number="379",
    )

    monkeypatch.setattr(
        answer_service,
        "fetch_candidates",
        lambda **kwargs: ([], [parent_document]),
    )

    result = answer_service.answer_question(
        question=plan.original_question,
        retriever=object(),
        chat_model=chat_model,
        plan=plan,
        retrieval_k=5,
    )

    assert chat_model.invocations == 1
    assert result["answer"] == "Grounded answer."


def test_existing_f06_provenance_and_ranking_remain_unchanged() -> None:
    shared = make_document(
        document_id="ppc_1860",
        chunk_id="ppc_1860::section::379::exact",
        page_content="Section 379 theft punishment.",
        provision_number="379",
    )

    ranked = rank_candidates(
        question="theft punishment",
        detected_concepts=[],
        candidates=[
            make_candidate(
                shared,
                score=0.93,
                query_index=0,
                query_text="theft punishment",
                retrieval_method="vector",
            ),
            make_candidate(
                shared,
                score=7.0,
                query_index=0,
                query_text="theft punishment",
                retrieval_method="lexical",
            ),
        ],
        semantic_threshold=0.94,
    )

    merged = deduplicate_ranked_documents(
        build_exact_ranked_items([shared]) + ranked,
        semantic_threshold=0.94,
    )

    assert len(merged) == 1
    assert set(merged[0].retrieval_methods) == {
        "exact",
        "vector",
        "lexical",
    }
    assert merged[0].final_score == 1.0
