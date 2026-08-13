from langchain_core.documents import Document

from rag import answer_service
from rag.retriever import fetch_candidates
from rag.schemas import LegalReference, QueryPlan


def _make_document(
    *,
    document_id: str,
    provision_type: str,
    provision_number: str,
    page_content: str,
    chunk_id: str,
    base_provision_number: str | None = None,
    subsection_path_key: str | None = None,
    component_type: str | None = None,
) -> Document:
    metadata = {
        "document_id": document_id,
        "document_name": "Test Legal Document",
        "document_title": "Test Legal Document",
        "document_short_name": "TLD",
        "document_type": "legal_document",
        "provision_type": provision_type,
        "provision_number": provision_number,
        "heading_only_chunk": False,
        "provision_body_present": True,
        "chunk_id": chunk_id,
    }

    if provision_type == "section":
        metadata["section_number"] = provision_number
    else:
        metadata["article_number"] = provision_number

    if base_provision_number is not None:
        metadata["base_provision_number"] = base_provision_number

    if subsection_path_key is not None:
        metadata["subsection_path_key"] = subsection_path_key

    if component_type is not None:
        metadata["component_type"] = component_type

    return Document(
        page_content=page_content,
        metadata=metadata,
    )


class DummyRetriever:
    def __init__(
        self,
        *,
        scroll_responses: list[list[Document]],
        search_responses: list[list[tuple[Document, float]]],
    ) -> None:
        self.scroll_responses = list(scroll_responses)
        self.search_responses = list(search_responses)
        self.scroll_filters: list[object] = []
        self.search_queries: list[str] = []

    def scroll_documents(self, metadata_filter, page_size=128):
        self.scroll_filters.append(metadata_filter)

        if self.scroll_responses:
            return self.scroll_responses.pop(0)

        return []

    def search_with_scores(self, query: str, k: int, metadata_filter=None):
        self.search_queries.append(query)

        if self.search_responses:
            return self.search_responses.pop(0)

        return []


def test_child_reference_exact_match_keeps_child_only_and_semantic_candidates() -> None:
    child_doc = _make_document(
        document_id="ppc_1860",
        provision_type="section",
        provision_number="11EE",
        page_content="Section 11EE(2)(b) child text.",
        chunk_id="ppc_1860::section::11EE::child-2-b",
        base_provision_number="11EE",
        subsection_path_key="2.b",
        component_type="clause",
    )
    parent_doc = _make_document(
        document_id="ppc_1860",
        provision_type="section",
        provision_number="11EE",
        page_content="Section 11EE parent text.",
        chunk_id="ppc_1860::section::11EE::parent",
        base_provision_number="11EE",
    )
    semantic_doc = _make_document(
        document_id="ppc_1860",
        provision_type="section",
        provision_number="411",
        page_content="Section 411 semantic support.",
        chunk_id="ppc_1860::section::411::semantic",
    )

    retriever = DummyRetriever(
        scroll_responses=[[child_doc, parent_doc]],
        search_responses=[[(semantic_doc, 0.97)]],
    )

    candidates, exact_documents = fetch_candidates(
        retriever=retriever,
        question="Explain Section 11EE(2)(b).",
        queries=["section 11EE"],
        question_type="section_lookup",
        section_number="11EE",
        detected_concepts=[],
        top_k=5,
        neighbor_radius=2,
        enable_neighbor_retrieval=False,
        min_relevance_score=0.0,
        document_ids=["ppc_1860"],
        provision_type="section",
        provision_numbers=["11EE"],
        article_number=None,
        legal_references=[
            LegalReference(
                provision_type="section",
                base_number="11EE",
                subsection_path=["2", "b"],
                component_type="clause",
                original_citation="Section 11EE(2)(b)",
            )
        ],
    )

    assert [doc.metadata["subsection_path_key"] for doc in exact_documents] == [
        "2.b"
    ]
    assert [doc.metadata["provision_number"] for doc in exact_documents] == [
        "11EE"
    ]
    assert [candidate.document.metadata["provision_number"] for candidate in candidates] == [
        "411"
    ]
    assert retriever.scroll_filters
    assert len(retriever.scroll_filters) == 1
    assert retriever.search_queries == ["section 11EE"]


def test_child_reference_exact_subsection_two_matches_directly() -> None:
    child_doc = _make_document(
        document_id="ppc_1860",
        provision_type="section",
        provision_number="11EE",
        page_content="Section 11EE(2) child text.",
        chunk_id="ppc_1860::section::11EE::child-2",
        base_provision_number="11EE",
        subsection_path_key="2",
        component_type="subsection",
    )

    retriever = DummyRetriever(
        scroll_responses=[[child_doc]],
        search_responses=[[]],
    )

    candidates, exact_documents = fetch_candidates(
        retriever=retriever,
        question="Explain Section 11EE(2).",
        queries=["section 11EE"],
        question_type="section_lookup",
        section_number="11EE",
        detected_concepts=[],
        top_k=5,
        neighbor_radius=2,
        enable_neighbor_retrieval=False,
        min_relevance_score=0.0,
        document_ids=["ppc_1860"],
        provision_type="section",
        provision_numbers=["11EE"],
        article_number=None,
        legal_references=[
            LegalReference(
                provision_type="section",
                base_number="11EE",
                subsection_path=["2"],
                component_type="subsection",
                original_citation="Section 11EE(2)",
            )
        ],
    )

    assert candidates == []
    assert [doc.metadata["subsection_path_key"] for doc in exact_documents] == [
        "2"
    ]
    assert len(retriever.scroll_filters) == 1


def test_missing_child_falls_back_to_parent_and_logs(caplog) -> None:
    parent_doc = _make_document(
        document_id="ppc_1860",
        provision_type="section",
        provision_number="11EE",
        page_content="Section 11EE parent text.",
        chunk_id="ppc_1860::section::11EE::parent",
        base_provision_number="11EE",
    )

    retriever = DummyRetriever(
        scroll_responses=[[], [parent_doc]],
        search_responses=[[]],
    )

    with caplog.at_level("INFO"):
        candidates, exact_documents = fetch_candidates(
            retriever=retriever,
            question="Explain Section 11EE(99).",
            queries=["section 11EE"],
            question_type="section_lookup",
            section_number="11EE",
            detected_concepts=[],
            top_k=5,
            neighbor_radius=2,
            enable_neighbor_retrieval=False,
            min_relevance_score=0.0,
            document_ids=["ppc_1860"],
            provision_type="section",
            provision_numbers=["11EE"],
            article_number=None,
            legal_references=[
                LegalReference(
                    provision_type="section",
                    base_number="11EE",
                    subsection_path=["99"],
                    component_type="subsection",
                    original_citation="Section 11EE(99)",
                )
            ],
        )

    assert candidates == []
    assert [doc.metadata["provision_number"] for doc in exact_documents] == [
        "11EE"
    ]
    assert len(retriever.scroll_filters) == 2
    assert "falling back to parent provision" in caplog.text.lower()
    assert "11ee" in caplog.text.lower()
    assert "99" in caplog.text.lower()


def test_article_child_reference_matches_exactly() -> None:
    child_doc = _make_document(
        document_id="constitution_1973",
        provision_type="article",
        provision_number="10",
        page_content="Article 10(2) child text.",
        chunk_id="constitution_1973::article::10::child-2",
        base_provision_number="10",
        subsection_path_key="2",
        component_type="subsection",
    )

    retriever = DummyRetriever(
        scroll_responses=[[child_doc]],
        search_responses=[[]],
    )

    candidates, exact_documents = fetch_candidates(
        retriever=retriever,
        question="Explain Article 10(2).",
        queries=["article 10"],
        question_type="article_lookup",
        section_number=None,
        detected_concepts=[],
        top_k=5,
        neighbor_radius=2,
        enable_neighbor_retrieval=False,
        min_relevance_score=0.0,
        document_ids=None,
        provision_type="article",
        provision_numbers=["10"],
        article_number="10",
        legal_references=[
            LegalReference(
                provision_type="article",
                base_number="10",
                subsection_path=["2"],
                component_type="subsection",
                original_citation="Article 10(2)",
            )
        ],
    )

    assert candidates == []
    assert [doc.metadata["provision_number"] for doc in exact_documents] == [
        "10"
    ]
    assert [doc.metadata["subsection_path_key"] for doc in exact_documents] == [
        "2"
    ]


def test_plain_section_379_exact_behavior_remains_unchanged() -> None:
    exact_379 = _make_document(
        document_id="ppc_1860",
        provision_type="section",
        provision_number="379",
        page_content="Section 379 parent text.",
        chunk_id="ppc_1860::section::379::parent",
    )

    retriever = DummyRetriever(
        scroll_responses=[[exact_379]],
        search_responses=[[]],
    )

    candidates, exact_documents = fetch_candidates(
        retriever=retriever,
        question="What does Section 379 state?",
        queries=["section 379"],
        question_type="section_lookup",
        section_number="379",
        detected_concepts=[],
        top_k=5,
        neighbor_radius=2,
        enable_neighbor_retrieval=False,
        min_relevance_score=0.0,
        document_ids=["ppc_1860"],
        provision_type="section",
        provision_numbers=["379"],
        article_number=None,
    )

    assert candidates == []
    assert [doc.metadata["provision_number"] for doc in exact_documents] == [
        "379"
    ]
    assert len(retriever.scroll_filters) == 1


def test_answer_service_passes_legal_references_to_fetch_candidates(
    monkeypatch,
) -> None:
    captured: dict[str, object] = {}

    def fake_fetch_candidates(**kwargs):
        captured.update(kwargs)
        return [], []

    monkeypatch.setattr(
        answer_service,
        "fetch_candidates",
        fake_fetch_candidates,
    )

    plan = QueryPlan(
        original_question="Explain Section 11EE(2)(b).",
        question_type="section_lookup",
        concepts=[],
        section_number="11EE",
        retrieval_queries=["section 11EE"],
        legal_references=[
            LegalReference(
                provision_type="section",
                base_number="11EE",
                subsection_path=["2", "b"],
                component_type="clause",
                original_citation="Section 11EE(2)(b)",
            )
        ],
        provision_numbers=["11EE"],
        provision_type="section",
    )

    class DummyChatModel:
        def invoke(self, prompt):
            return "No answer."

    result = answer_service.answer_question(
        question=plan.original_question,
        retriever=object(),
        chat_model=DummyChatModel(),
        plan=plan,
        retrieval_k=5,
    )

    assert captured["legal_references"] == plan.legal_references
    assert result["retrieved_contexts"] == []
