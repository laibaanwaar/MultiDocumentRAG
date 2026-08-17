from __future__ import annotations

from langchain_core.documents import Document

from rag.answer_service import build_exact_ranked_items
from rag.ranker import (
    deduplicate_ranked_documents,
    rank_candidates,
    select_context_documents,
)
from rag.schemas import CandidateDocument


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


def test_same_chunk_vector_and_lexical_fuse_into_one_ranked_document() -> None:
    shared = make_document(
        document_id="ppc_1860",
        chunk_id="ppc_1860::section::379::chunk-1",
        page_content="Section 379 theft punishment.",
        provision_number="379",
    )

    ranked = rank_candidates(
        question="What is theft punishment?",
        detected_concepts=[],
        candidates=[
            make_candidate(
                shared,
                score=0.91,
                query_index=0,
                query_text="What is theft punishment?",
                retrieval_method="vector",
            ),
            make_candidate(
                shared,
                score=7.0,
                query_index=0,
                query_text="What is theft punishment?",
                retrieval_method="lexical",
            ),
        ],
        semantic_threshold=0.94,
    )

    assert len(ranked) == 1
    assert set(ranked[0].retrieval_methods) == {
        "vector",
        "lexical",
    }
    assert ranked[0].matched_queries == 2
    assert len(ranked[0].retrieval_routes) == 2


def test_same_route_duplicate_does_not_inflate_fusion_or_matches() -> None:
    shared = make_document(
        document_id="ppc_1860",
        chunk_id="ppc_1860::section::411::chunk-1",
        page_content="Section 411 stolen property.",
        provision_number="411",
    )

    single = rank_candidates(
        question="stolen property",
        detected_concepts=[],
        candidates=[
            make_candidate(
                shared,
                score=0.81,
                query_index=0,
                query_text="stolen property",
                retrieval_method="vector",
            )
        ],
        semantic_threshold=0.94,
    )[0]

    duplicated = rank_candidates(
        question="stolen property",
        detected_concepts=[],
        candidates=[
            make_candidate(
                shared,
                score=0.81,
                query_index=0,
                query_text="stolen property",
                retrieval_method="vector",
            ),
            make_candidate(
                shared,
                score=0.81,
                query_index=0,
                query_text="stolen property",
                retrieval_method="vector",
            ),
        ],
        semantic_threshold=0.94,
    )[0]

    assert duplicated.fusion_score == single.fusion_score
    assert duplicated.matched_queries == single.matched_queries == 1


def test_neighbor_candidate_preserves_neighbor_provenance() -> None:
    neighbor_doc = make_document(
        document_id="ppc_1860",
        chunk_id="ppc_1860::section::410::chunk-1",
        page_content="Neighbor section text.",
        provision_number="410",
    )

    ranked = rank_candidates(
        question="neighbor lookup",
        detected_concepts=[],
        candidates=[
            make_candidate(
                neighbor_doc,
                score=0.71,
                query_index=2,
                query_text="neighbor:409",
                retrieval_method="neighbor",
            )
        ],
        semantic_threshold=0.94,
    )

    assert ranked[0].retrieval_methods == ["neighbor"]
    assert ranked[0].matched_query_indices == [2]
    assert ranked[0].retrieval_routes == [
        "neighbor|2|neighbor:409"
    ]


def test_candidate_insertion_order_does_not_change_final_ranking() -> None:
    doc_a = make_document(
        document_id="ppc_1860",
        chunk_id="ppc_1860::section::379::chunk-1",
        page_content="Theft punishment text.",
        provision_number="379",
    )
    doc_b = make_document(
        document_id="ppc_1860",
        chunk_id="ppc_1860::section::380::chunk-1",
        page_content="Different section text.",
        provision_number="380",
    )
    doc_c = make_document(
        document_id="ppc_1860",
        chunk_id="ppc_1860::section::381::chunk-1",
        page_content="Third section text.",
        provision_number="381",
    )

    candidates = [
        make_candidate(
            doc_a,
            score=0.93,
            query_index=0,
            query_text="theft",
            retrieval_method="vector",
        ),
        make_candidate(
            doc_b,
            score=6.0,
            query_index=0,
            query_text="theft",
            retrieval_method="lexical",
        ),
        make_candidate(
            doc_c,
            score=0.88,
            query_index=1,
            query_text="property",
            retrieval_method="vector",
        ),
        make_candidate(
            doc_a,
            score=5.0,
            query_index=1,
            query_text="property",
            retrieval_method="lexical",
        ),
    ]

    ranked_forward = rank_candidates(
        question="theft and property",
        detected_concepts=[],
        candidates=candidates,
        semantic_threshold=0.94,
    )
    ranked_reversed = rank_candidates(
        question="theft and property",
        detected_concepts=[],
        candidates=list(reversed(candidates)),
        semantic_threshold=0.94,
    )

    assert [
        item.document_id
        for item in ranked_forward
    ] == [
        item.document_id
        for item in ranked_reversed
    ]
    assert [
        item.provision_number
        for item in ranked_forward
    ] == [
        item.provision_number
        for item in ranked_reversed
    ]


def test_exact_vector_and_lexical_same_chunk_merges_to_one_result() -> None:
    shared = make_document(
        document_id="ppc_1860",
        chunk_id="ppc_1860::section::379::exact",
        page_content="Section 379 theft punishment.",
        provision_number="379",
    )

    exact_items = build_exact_ranked_items([shared])
    ranked_items = rank_candidates(
        question="theft punishment",
        detected_concepts=[],
        candidates=[
            make_candidate(
                shared,
                score=0.92,
                query_index=0,
                query_text="theft punishment",
                retrieval_method="vector",
            ),
            make_candidate(
                shared,
                score=9.5,
                query_index=0,
                query_text="theft punishment",
                retrieval_method="lexical",
            ),
        ],
        semantic_threshold=0.94,
    )

    merged = deduplicate_ranked_documents(
        exact_items + ranked_items,
        semantic_threshold=0.94,
    )

    assert len(merged) == 1
    assert set(merged[0].retrieval_methods) == {
        "exact",
        "vector",
        "lexical",
    }
    assert merged[0].matched_queries == 3
    assert merged[0].fusion_score == 1.0
    assert merged[0].relevance_score == 1.0
    assert merged[0].final_score == 1.0


def test_final_selected_context_has_no_duplicate_chunk() -> None:
    shared = make_document(
        document_id="ppc_1860",
        chunk_id="ppc_1860::section::379::chunk-1",
        page_content="Section 379 theft punishment.",
        provision_number="379",
    )
    support = make_document(
        document_id="ppc_1860",
        chunk_id="ppc_1860::section::380::chunk-1",
        page_content="Section 380 different support.",
        provision_number="380",
    )

    ranked = deduplicate_ranked_documents(
        build_exact_ranked_items([shared])
        + rank_candidates(
            question="theft punishment",
            detected_concepts=[],
            candidates=[
                make_candidate(
                    shared,
                    score=0.91,
                    query_index=0,
                    query_text="theft punishment",
                    retrieval_method="vector",
                ),
                make_candidate(
                    shared,
                    score=8.0,
                    query_index=0,
                    query_text="theft punishment",
                    retrieval_method="lexical",
                ),
                make_candidate(
                    support,
                    score=0.80,
                    query_index=0,
                    query_text="theft punishment",
                    retrieval_method="vector",
                ),
            ],
            semantic_threshold=0.94,
        ),
        semantic_threshold=0.94,
    )

    selected = select_context_documents(
        ranked_items=ranked,
        question_type="section_lookup",
        maximum_documents=10,
        max_context_sections=10,
        final_k=10,
    )

    chunk_ids = [
        document.metadata["chunk_id"]
        for document in selected
    ]
    assert len(chunk_ids) == len(set(chunk_ids))


def test_different_provisions_and_documents_remain_distinct() -> None:
    same_doc_10 = make_document(
        document_id="ppc_1860",
        chunk_id="ppc_1860::section::10::chunk-1",
        page_content="Shared text.",
        provision_number="10",
    )
    same_doc_11 = make_document(
        document_id="ppc_1860",
        chunk_id="ppc_1860::section::11::chunk-1",
        page_content="Shared text.",
        provision_number="11",
    )
    other_doc_10 = make_document(
        document_id="ata_1997",
        chunk_id="ata_1997::section::10::chunk-1",
        page_content="Shared text.",
        provision_number="10",
    )

    ranked = deduplicate_ranked_documents(
        [
            *build_exact_ranked_items([same_doc_10]),
            *build_exact_ranked_items([same_doc_11]),
            *build_exact_ranked_items([other_doc_10]),
        ],
        semantic_threshold=0.94,
    )

    assert {
        item.document.metadata["chunk_id"]
        for item in ranked
    } == {
        "ppc_1860::section::10::chunk-1",
        "ppc_1860::section::11::chunk-1",
        "ata_1997::section::10::chunk-1",
    }


def test_exact_citation_is_pinned_at_highest_priority() -> None:
    exact_doc = make_document(
        document_id="ppc_1860",
        chunk_id="ppc_1860::section::379::exact",
        page_content="Exact citation text.",
        provision_number="379",
    )
    support_doc = make_document(
        document_id="ppc_1860",
        chunk_id="ppc_1860::section::380::chunk-1",
        page_content="Support text.",
        provision_number="380",
    )

    ranked = deduplicate_ranked_documents(
        build_exact_ranked_items([exact_doc])
        + rank_candidates(
            question="exact citation",
            detected_concepts=[],
            candidates=[
                make_candidate(
                    exact_doc,
                    score=0.90,
                    query_index=0,
                    query_text="exact citation",
                    retrieval_method="vector",
                ),
                make_candidate(
                    support_doc,
                    score=0.89,
                    query_index=0,
                    query_text="exact citation",
                    retrieval_method="vector",
                ),
            ],
            semantic_threshold=0.94,
        ),
        semantic_threshold=0.94,
    )

    assert ranked[0].document.metadata["chunk_id"] == (
        "ppc_1860::section::379::exact"
    )
    assert ranked[0].final_score == 1.0
