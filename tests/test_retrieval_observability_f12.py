from __future__ import annotations

from langchain_core.documents import Document

from rag import answer_service
from rag.retrieval_errors import MandatoryRetrievalError
from rag.schemas import CandidateDocument, QueryPlan


class DummyChatModel:
    def __init__(self) -> None:
        self.invocations = 0

    def invoke(self, prompt):
        self.invocations += 1
        return "Grounded answer. [Source 1]"


def _make_document() -> Document:
    return Document(
        page_content="ATA | Section 7: Punishment for acts of terrorism.",
        metadata={
            "document_id": "ata_1997",
            "document_name": "Anti-Terrorism Act, 1997",
            "document_title": "Anti-Terrorism Act, 1997",
            "document_short_name": "ATA",
            "document_type": "special_criminal_law",
            "provision_type": "section",
            "provision_number": "7",
            "section_number": "7",
            "heading_only_chunk": False,
            "chunk_id": "ata_1997:section:7:chunk-1",
            "chunk_number": 12,
            "document_chunk_number": 12,
            "page_number": 12,
            "page_range": "12",
        },
    )


def _make_candidate() -> CandidateDocument:
    return CandidateDocument(
        document=_make_document(),
        relevance_score=0.97,
        query_index=0,
        query_text="section 7 punishment",
        retrieval_method="vector",
        retrieval_rank=1,
    )


def _make_plan() -> QueryPlan:
    return QueryPlan(
        original_question="What is the punishment under Section 7?",
        question_type="section_lookup",
        concepts=["terrorism"],
        section_number="7",
        retrieval_queries=["section 7 punishment"],
        document_ids=["ata_1997"],
        provision_numbers=["7"],
        provision_type="section",
    )


def test_answer_question_emits_optional_retrieval_trace(monkeypatch) -> None:
    plan = _make_plan()
    candidate = _make_candidate()
    chat_model = DummyChatModel()

    def fake_fetch_candidates(**kwargs):
        retrieval_trace = kwargs["retrieval_trace"]
        retrieval_trace.setdefault("events", []).append(
            {
                "channel": "vector",
                "route_id": "vector:0:section 7 punishment",
                "status": "success",
                "latency_ms": 1.5,
                "accepted_count": 1,
                "raw_count": 1,
                "query_index": 0,
                "query_text": "section 7 punishment",
            }
        )
        return [candidate], []

    monkeypatch.setattr(
        answer_service,
        "fetch_candidates",
        fake_fetch_candidates,
    )
    monkeypatch.setattr(answer_service, "MAX_CONTEXT_DOCUMENTS", 1)
    monkeypatch.setattr(answer_service, "MAX_CONTEXT_PROVISIONS", 1)

    result = answer_service.answer_question(
        question=plan.original_question,
        retriever=object(),
        chat_model=chat_model,
        plan=plan,
        retrieval_k=5,
        include_trace=True,
    )

    assert chat_model.invocations == 1
    assert result["retrieved_document_count"] == 1
    assert "retrieval_trace" in result

    trace = result["retrieval_trace"]
    assert trace["retrieval_status"] == "success"
    assert trace["question"]["normalized_type"] == "section_lookup"
    assert trace["question"]["retrieval_queries"] == [
        "section 7 punishment"
    ]
    assert trace["retrieval"]["candidate_count"] == 1
    assert trace["retrieval"]["selected_context_count"] == 1
    assert trace["retrieval"]["channels"]["vector"]["event_count"] == 1
    assert trace["retrieval"]["channels"]["vector"]["failure_count"] == 0
    assert trace["retrieval"]["candidate_chunks"][0]["retrieval_method"] == "vector"
    assert trace["retrieval"]["selected_contexts"][0]["provision_identity"] == "ata_1997::section::7"
    assert trace["cited_source_labels"] == ["Source 1"]
    assert trace["sources"][0]["document_id"] == "ata_1997"

    result_without_trace = answer_service.answer_question(
        question=plan.original_question,
        retriever=object(),
        chat_model=DummyChatModel(),
        plan=plan,
        retrieval_k=5,
    )

    assert "retrieval_trace" not in result_without_trace


def test_answer_question_emits_error_trace_for_mandatory_failure(
    monkeypatch,
) -> None:
    plan = _make_plan()
    chat_model = DummyChatModel()

    def broken_fetch_candidates(**kwargs):
        raise MandatoryRetrievalError(
            "broken",
            operation="scroll_exact_provision",
            route="provision:type=section;numbers=7;document_ids=ata_1997",
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
        include_trace=True,
    )

    assert chat_model.invocations == 0
    assert result["retrieval_status"] == "error"
    assert "retrieval_trace" in result

    trace = result["retrieval_trace"]
    assert trace["retrieval_status"] == "error"
    assert trace["retrieval"]["candidate_count"] == 0
    assert trace["retrieval"]["selected_context_count"] == 0
    assert trace["retrieval"]["events"] == []
