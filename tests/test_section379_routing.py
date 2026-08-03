from langchain_core.documents import Document

from rag.intent_router import classify_question, route_question
from rag.retriever import fetch_candidates


def test_bare_section_379_uses_section_lookup_classification() -> None:
    assert classify_question("What does Section 379 state?") == "section_lookup"


def test_vs_comparison_is_treated_as_a_comparison() -> None:
    assert (
        classify_question("PPC Section 379 vs CrPC Section 379")
        == "comparison"
    )


def test_theft_scenario_routes_to_ppc_and_sections_378_379() -> None:
    plan = route_question(
        "Ali dishonestly took a mobile phone without the owner's consent."
    )

    assert plan.question_type == "fact_scenario"
    assert plan.concepts == ["theft"]
    assert plan.document_ids == ["ppc_1860"]
    assert set(plan.section_hints) == {"378", "379"}
    assert set(plan.provision_numbers) == {"378", "379"}


def test_theft_scenario_uses_exact_section_hits_before_semantic_fallback() -> None:
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

        def search_with_scores(self, query: str, k: int, metadata_filter=None):
            self.queries.append(query)

            if query == "Ali dishonestly took a mobile phone without the owner's consent.":
                return [
                    (exact_378, 0.99),
                    (exact_379, 0.98),
                ]

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
    assert candidates == []
    assert retriever.queries == [
        "Ali dishonestly took a mobile phone without the owner's consent.",
        "Ali dishonestly took a mobile phone without the owner's consent.",
    ]
