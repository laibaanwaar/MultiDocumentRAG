from __future__ import annotations

from langchain_core.documents import Document

import rag.retriever as retriever_module
from rag.answer_service import build_exact_ranked_items
from rag.ranker import (
    deduplicate_ranked_documents,
    rank_candidates,
)
from rag.schemas import CandidateDocument


class DummyRetriever:
    def search_with_scores(self, query: str, k: int, metadata_filter=None):
        return []

    def search_lexical(self, query: str, k: int, document_ids=None):
        return []


def make_document(
    *,
    document_id: str,
    chunk_id: str,
    page_content: str,
    provision_number: str,
    provision_type: str = "section",
) -> Document:
    return Document(
        page_content=page_content,
        metadata={
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
        },
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


def patch_concepts(monkeypatch, concept_map: dict[str, dict]) -> None:
    for concept_name, concept_data in concept_map.items():
        monkeypatch.setitem(
            retriever_module.LEGAL_CONCEPTS,
            concept_name,
            concept_data,
        )


def run_fetch_candidates(
    *,
    monkeypatch,
    detected_concepts: list[str],
    document_ids: list[str] | None = None,
):
    calls: list[dict[str, object]] = []

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
            {
                "section_number": section_number,
                "document_ids": document_ids,
                "provision_type": provision_type,
            }
        )
        return []

    monkeypatch.setattr(
        retriever_module,
        "retrieve_neighbor_documents",
        fake_neighbor,
    )

    retriever_module.fetch_candidates(
        retriever=DummyRetriever(),
        question="neighbor route test",
        queries=["neighbor route test"],
        question_type="general",
        section_number=None,
        detected_concepts=detected_concepts,
        top_k=3,
        neighbor_radius=1,
        enable_neighbor_retrieval=True,
        min_relevance_score=0.0,
        document_ids=document_ids,
        provision_type=None,
        provision_numbers=None,
        article_number=None,
        legal_references=None,
    )

    return calls


def test_concept_a_sections_never_mix_with_concept_b_sections(monkeypatch) -> None:
    patch_concepts(
        monkeypatch,
        {
            "concept_a": {
                "preferred_sections": {"378", "379"},
                "preferred_documents": {"doc_a"},
            },
            "concept_b": {
                "preferred_sections": {"405", "408", "409"},
                "preferred_documents": {"doc_b"},
            },
        },
    )

    calls = run_fetch_candidates(
        monkeypatch=monkeypatch,
        detected_concepts=["concept_a", "concept_b"],
    )

    assert {
        (tuple(call["document_ids"] or []),
         call["section_number"])
        for call in calls
    } == {
        (("doc_a",), "378"),
        (("doc_a",), "379"),
        (("doc_b",), "405"),
        (("doc_b",), "408"),
        (("doc_b",), "409"),
    }


def test_disjoint_preferred_documents_are_not_exchanged(monkeypatch) -> None:
    patch_concepts(
        monkeypatch,
        {
            "concept_a": {
                "preferred_sections": {"378"},
                "preferred_documents": {"doc_a"},
            },
            "concept_b": {
                "preferred_sections": {"405"},
                "preferred_documents": {"doc_b"},
            },
        },
    )

    calls = run_fetch_candidates(
        monkeypatch=monkeypatch,
        detected_concepts=["concept_a", "concept_b"],
    )

    assert {
        (tuple(call["document_ids"] or []), call["section_number"])
        for call in calls
    } == {
        (("doc_a",), "378"),
        (("doc_b",), "405"),
    }


def test_identical_legal_route_executes_only_once(monkeypatch) -> None:
    patch_concepts(
        monkeypatch,
        {
            "concept_a": {
                "preferred_sections": {"378", "379"},
                "preferred_documents": {"doc_a"},
            },
            "concept_b": {
                "preferred_sections": {"378", "379"},
                "preferred_documents": {"doc_a"},
            },
        },
    )

    calls = run_fetch_candidates(
        monkeypatch=monkeypatch,
        detected_concepts=["concept_a", "concept_b"],
    )

    assert [
        (
            tuple(call["document_ids"] or []),
            call["section_number"],
        )
        for call in calls
    ] == [
        (("doc_a",), "378"),
        (("doc_a",), "379"),
    ]


def test_explicit_query_document_routing_is_respected(monkeypatch) -> None:
    patch_concepts(
        monkeypatch,
        {
            "concept_a": {
                "preferred_sections": {"378"},
            }
        },
    )

    calls = run_fetch_candidates(
        monkeypatch=monkeypatch,
        detected_concepts=["concept_a"],
        document_ids=["user_doc"],
    )

    assert calls == [
        {
            "section_number": "378",
            "document_ids": ["user_doc"],
            "provision_type": "section",
        }
    ]


def test_incompatible_explicit_route_skips_concept_route(monkeypatch) -> None:
    patch_concepts(
        monkeypatch,
        {
            "concept_a": {
                "preferred_sections": {"378"},
                "preferred_documents": {"doc_a"},
            }
        },
    )

    calls = run_fetch_candidates(
        monkeypatch=monkeypatch,
        detected_concepts=["concept_a"],
        document_ids=["user_doc"],
    )

    assert calls == []


def test_concept_without_preferred_sections_produces_no_neighbor_route(
    monkeypatch,
) -> None:
    patch_concepts(
        monkeypatch,
        {
            "concept_a": {
                "preferred_sections": set(),
                "preferred_documents": {"doc_a"},
            }
        },
    )

    calls = run_fetch_candidates(
        monkeypatch=monkeypatch,
        detected_concepts=["concept_a"],
    )

    assert calls == []


def test_neighbor_candidates_keep_neighbor_retrieval_method(monkeypatch) -> None:
    patch_concepts(
        monkeypatch,
        {
            "concept_a": {
                "preferred_sections": {"378"},
                "preferred_documents": {"doc_a"},
            }
        },
    )

    neighbor_doc = make_document(
        document_id="doc_a",
        chunk_id="doc_a::section::378::chunk-1",
        page_content="Neighbor content.",
        provision_number="378",
    )

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
        return [(neighbor_doc, 0.8)]

    monkeypatch.setattr(
        retriever_module,
        "retrieve_neighbor_documents",
        fake_neighbor,
    )

    candidates, _exact_documents = retriever_module.fetch_candidates(
        retriever=DummyRetriever(),
        question="neighbor route test",
        queries=["neighbor route test"],
        question_type="general",
        section_number=None,
        detected_concepts=["concept_a"],
        top_k=3,
        neighbor_radius=1,
        enable_neighbor_retrieval=True,
        min_relevance_score=0.0,
        document_ids=None,
        provision_type=None,
        provision_numbers=None,
        article_number=None,
        legal_references=None,
    )

    assert [candidate.retrieval_method for candidate in candidates] == [
        "neighbor"
    ]


def test_f06_provenance_and_ranking_remain_unchanged() -> None:
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
            make_candidate(
                shared,
                score=0.8,
                query_index=1,
                query_text="neighbor:378",
                retrieval_method="neighbor",
            ),
        ],
        semantic_threshold=0.94,
    )

    exact_ranked = build_exact_ranked_items([shared])
    merged = deduplicate_ranked_documents(
        exact_ranked + ranked,
        semantic_threshold=0.94,
    )

    assert len(merged) == 1
    assert set(merged[0].retrieval_methods) == {
        "exact",
        "vector",
        "lexical",
        "neighbor",
    }
    assert merged[0].final_score == 1.0
