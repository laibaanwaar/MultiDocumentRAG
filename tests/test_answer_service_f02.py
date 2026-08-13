from langchain_core.documents import Document

from rag import answer_service
from rag.schemas import CandidateDocument, QueryPlan


class DummyChatModel:
    def invoke(self, prompt):
        return "Grounded answer."


def _make_document(
    section_number: str,
    page_content: str,
    *,
    chunk_id: str,
) -> Document:
    return Document(
        page_content=page_content,
        metadata={
            "document_id": "ppc_1860",
            "document_name": "Pakistan Penal Code 1860",
            "provision_type": "section",
            "provision_number": section_number,
            "section_number": section_number,
            "heading_only_chunk": False,
            "chunk_id": chunk_id,
        },
    )


def _make_candidate(
    document: Document,
    *,
    query_index: int = 0,
    relevance_score: float = 0.95,
) -> CandidateDocument:
    return CandidateDocument(
        document=document,
        relevance_score=relevance_score,
        query_index=query_index,
        query_text="test query",
    )


def test_section_lookup_retains_exact_and_supporting_semantic_context(
    monkeypatch,
) -> None:
    exact_378 = _make_document(
        "378",
        "Section 378. Theft.",
        chunk_id="ppc_1860:section:378:exact",
    )
    semantic_378 = _make_document(
        "378",
        "Section 378. Theft.",
        chunk_id="ppc_1860:section:378:semantic",
    )
    semantic_379 = _make_document(
        "379",
        "Section 379. Punishment for theft.",
        chunk_id="ppc_1860:section:379:semantic",
    )

    plan = QueryPlan(
        original_question="What does Section 378 say?",
        question_type="section_lookup",
        concepts=["theft"],
        section_number="378",
        retrieval_queries=["section 378"],
        document_ids=["ppc_1860"],
        provision_numbers=["378"],
        provision_type="section",
    )

    candidates = [
        _make_candidate(semantic_378, relevance_score=0.98),
        _make_candidate(semantic_379, relevance_score=0.9),
    ]

    def fake_fetch_candidates(**kwargs):
        return candidates, [exact_378]

    monkeypatch.setattr(
        answer_service,
        "fetch_candidates",
        fake_fetch_candidates,
    )
    monkeypatch.setattr(
        answer_service,
        "MAX_CONTEXT_DOCUMENTS",
        2,
    )
    monkeypatch.setattr(
        answer_service,
        "MAX_CONTEXT_PROVISIONS",
        2,
    )

    result = answer_service.answer_question(
        question=plan.original_question,
        retriever=object(),
        chat_model=DummyChatModel(),
        plan=plan,
        retrieval_k=5,
    )

    assert result["retrieved_contexts"] == [
        "Section 378. Theft.",
        "Section 379. Punishment for theft.",
    ]
    assert result["retrieved_document_count"] == 2


def test_fact_scenario_retains_exact_and_supporting_semantic_context(
    monkeypatch,
) -> None:
    exact_378 = _make_document(
        "378",
        "Section 378. Theft.",
        chunk_id="ppc_1860:section:378:exact",
    )
    semantic_379 = _make_document(
        "379",
        "Section 379. Punishment for theft.",
        chunk_id="ppc_1860:section:379:semantic",
    )

    plan = QueryPlan(
        original_question="Ali dishonestly took a mobile phone without consent.",
        question_type="fact_scenario",
        concepts=["theft"],
        section_number="378",
        retrieval_queries=["theft facts"],
        document_ids=["ppc_1860"],
        provision_numbers=["378", "379"],
        provision_type="section",
    )

    candidates = [
        _make_candidate(semantic_379, relevance_score=0.9),
    ]

    def fake_fetch_candidates(**kwargs):
        return candidates, [exact_378]

    monkeypatch.setattr(
        answer_service,
        "fetch_candidates",
        fake_fetch_candidates,
    )
    monkeypatch.setattr(
        answer_service,
        "MAX_CONTEXT_DOCUMENTS",
        2,
    )
    monkeypatch.setattr(
        answer_service,
        "MAX_CONTEXT_PROVISIONS",
        2,
    )

    result = answer_service.answer_question(
        question=plan.original_question,
        retriever=object(),
        chat_model=DummyChatModel(),
        plan=plan,
        retrieval_k=5,
    )

    assert result["retrieved_contexts"] == [
        "Section 378. Theft.",
        "Section 379. Punishment for theft.",
    ]
    assert result["retrieved_document_count"] == 2


def test_semantic_only_behavior_remains_unchanged(
    monkeypatch,
) -> None:
    semantic_378 = _make_document(
        "378",
        "Section 378. Theft.",
        chunk_id="ppc_1860:section:378:semantic",
    )
    semantic_379 = _make_document(
        "379",
        "Section 379. Punishment for theft.",
        chunk_id="ppc_1860:section:379:semantic",
    )

    plan = QueryPlan(
        original_question="What provisions are relevant to theft?",
        question_type="fact_scenario",
        concepts=["theft"],
        section_number=None,
        retrieval_queries=["theft provisions"],
        document_ids=["ppc_1860"],
        provision_numbers=[],
        provision_type="section",
    )

    candidates = [
        _make_candidate(semantic_378, relevance_score=0.98),
        _make_candidate(semantic_379, relevance_score=0.9),
    ]

    def fake_fetch_candidates(**kwargs):
        return candidates, []

    monkeypatch.setattr(
        answer_service,
        "fetch_candidates",
        fake_fetch_candidates,
    )
    monkeypatch.setattr(
        answer_service,
        "MAX_CONTEXT_DOCUMENTS",
        2,
    )
    monkeypatch.setattr(
        answer_service,
        "MAX_CONTEXT_PROVISIONS",
        2,
    )

    result = answer_service.answer_question(
        question=plan.original_question,
        retriever=object(),
        chat_model=DummyChatModel(),
        plan=plan,
        retrieval_k=5,
    )

    assert result["retrieved_contexts"] == [
        "Section 378. Theft.",
        "Section 379. Punishment for theft.",
    ]
    assert result["retrieved_document_count"] == 2
