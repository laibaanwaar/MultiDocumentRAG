from langchain_core.documents import Document

from rag.answer_service import (
    _build_section_lookup_clarification,
    _filter_documents_by_mentioned_laws,
    _prefer_ppc_documents,
)
from rag.intent_router import classify_question


def _make_document(document_name: str, section_number: str = "379") -> Document:
    return Document(
        page_content="Sample text",
        metadata={
            "document_name": document_name,
            "document_id": document_name.lower().replace(" ", "_"),
            "section_number": section_number,
        },
    )


def test_bare_section_379_uses_section_lookup_classification() -> None:
    assert classify_question("What does Section 379 state?") == "section_lookup"


def test_vs_comparison_is_treated_as_a_comparison() -> None:
    assert (
        classify_question("PPC Section 379 vs CrPC Section 379")
        == "comparison"
    )


def test_section_379_clarification_mentions_all_matching_laws() -> None:
    documents = [
        _make_document("Pakistan Penal Code 1860"),
        _make_document("Code of Criminal Procedure"),
    ]

    clarification = _build_section_lookup_clarification(
        section_number="379",
        documents=documents,
    )

    assert "Section 379 appears in multiple indexed laws" in clarification
    assert "Pakistan Penal Code 1860" in clarification
    assert "Code of Criminal Procedure" in clarification


def test_ppc_is_preferred_for_theft_punishment_queries() -> None:
    documents = [
        _make_document("Code of Criminal Procedure"),
        _make_document("Pakistan Penal Code 1860"),
    ]

    preferred = _prefer_ppc_documents(documents)

    assert [document.metadata["document_name"] for document in preferred] == [
        "Pakistan Penal Code 1860",
    ]


def test_mentioned_law_filter_keeps_only_the_requested_document() -> None:
    documents = [
        _make_document("Pakistan Penal Code 1860"),
        _make_document("Code of Criminal Procedure"),
    ]

    filtered = _filter_documents_by_mentioned_laws(
        documents,
        {"crpc"},
    )

    assert [document.metadata["document_name"] for document in filtered] == [
        "Code of Criminal Procedure",
    ]
