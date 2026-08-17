from __future__ import annotations

import json
import logging
import os
import re
import time
from typing import Any

from dotenv import load_dotenv
from langchain_groq import ChatGroq

from rag.context_builder import create_sources, format_context
from rag.intent_router import get_retrieval_k, route_question
from rag.prompt_builder import build_grounded_prompt
from rag.retrieval_errors import MandatoryRetrievalError
from rag.ranker import (
    calculate_retrieval_confidence,
    deduplicate_ranked_documents,
    merge_provision_parts,
    rank_candidates,
    select_context_documents,
)
from rag.retriever import AdaptiveRetriever, fetch_candidates
from rag.schemas import QueryPlan, RankedDocument
from rag.vector_store import create_vector_store


load_dotenv()

logger = logging.getLogger(__name__)


GROQ_MODEL = os.getenv(
    "GROQ_MODEL",
    "openai/gpt-oss-120b",
)

TOP_K = int(os.getenv("TOP_K", "6"))
MAX_CONTEXT_DOCUMENTS = int(
    os.getenv("MAX_CONTEXT_DOCUMENTS", "10")
)
MAX_CONTEXT_PROVISIONS = int(
    os.getenv("MAX_CONTEXT_PROVISIONS", "4")
)
MIN_RELEVANCE_SCORE = float(
    os.getenv("MIN_RELEVANCE_SCORE", "0.0")
)
NEIGHBOR_RADIUS = int(
    os.getenv("NEIGHBOR_SECTION_RADIUS", "2")
)
SEMANTIC_DEDUP_THRESHOLD = float(
    os.getenv("SEMANTIC_DEDUP_THRESHOLD", "0.94")
)

ENABLE_NEIGHBOR_RETRIEVAL = (
    os.getenv(
        "ENABLE_NEIGHBOR_RETRIEVAL",
        "True",
    ).strip().lower()
    in {"1", "true", "yes", "on"}
)

QUESTION_MIN_LENGTH = 3
QUESTION_MAX_LENGTH = 4000

FINAL_K_BY_QUESTION_TYPE = {
    "punishment": 3,
    "definition": 3,
    "section_lookup": 4,
    "article_lookup": 4,
    "fact_scenario": 8,
    "comparison": 8,
}


def get_chat_model() -> ChatGroq:
    """Create the Groq chat model."""

    if not os.getenv("GROQ_API_KEY"):
        raise ValueError(
            "GROQ_API_KEY is missing from the .env file."
        )

    return ChatGroq(
        model=GROQ_MODEL,
        temperature=0,
    )


def create_rag_components():
    """Open Qdrant and create the retriever and chat model."""

    vector_store, client = create_vector_store(
        reset=False
    )

    return (
        AdaptiveRetriever(vector_store),
        get_chat_model(),
        client,
    )


def extract_response_text(
    response: Any,
) -> str:
    """Extract plain text from a LangChain chat response."""

    content = getattr(
        response,
        "content",
        response,
    )

    if isinstance(content, str):
        return content.strip()

    if isinstance(content, list):
        parts: list[str] = []

        for item in content:
            if isinstance(item, dict):
                text = item.get("text")
            else:
                text = getattr(
                    item,
                    "text",
                    None,
                )

            if isinstance(text, str) and text.strip():
                parts.append(text.strip())

        return "\n".join(parts)

    return str(content).strip()


def filter_sources_used_in_answer(
    answer: str,
    sources: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Return only source records cited in the generated answer."""

    used_numbers = {
        int(number)
        for number in re.findall(
            r"\[Source\s+(\d+)\]",
            answer,
            flags=re.IGNORECASE,
        )
    }

    if not used_numbers:
        return sources

    return [
        source
        for index, source in enumerate(
            sources,
            start=1,
        )
        if index in used_numbers
    ]


def build_exact_ranked_items(
    documents: list[Any],
) -> list[RankedDocument]:
    """Represent exact Section or Article hits as top-ranked items."""

    return [
        RankedDocument(
            document=document,
            fusion_score=1.0,
            relevance_score=1.0,
            matched_queries=1,
            retrieval_methods=["exact"],
            matched_query_indices=[],
            retrieval_routes=["exact"],
            final_score=1.0,
        )
        for document in documents
    ]


def _has_explicit_provision_request(plan: QueryPlan) -> bool:
    return bool(
        plan.legal_references
        or plan.section_number
        or plan.article_number
    )


def _candidate_matches_explicit_provision(
    candidates: list[Any],
    plan: QueryPlan,
) -> bool:
    explicit_pairs: set[tuple[str, str]] = set()

    if plan.section_number:
        explicit_pairs.add(("section", plan.section_number.strip().upper()))

    if plan.article_number:
        explicit_pairs.add(("article", plan.article_number.strip().upper()))

    for reference in plan.legal_references:
        base_number = str(reference.base_number).strip().upper()
        if base_number:
            explicit_pairs.add(
                (
                    str(reference.provision_type).strip().lower(),
                    base_number,
                )
            )

    if not explicit_pairs:
        return False

    explicit_document_ids = {
        str(document_id).strip().lower()
        for document_id in plan.document_ids
        if str(document_id).strip()
    }

    for candidate in candidates:
        metadata = getattr(candidate.document, "metadata", {})
        candidate_document_id = str(
            metadata.get("document_id") or ""
        ).strip().lower()
        candidate_provision_type = str(
            metadata.get("provision_type") or ""
        ).strip().lower()
        candidate_provision_number = str(
            metadata.get("provision_number")
            or metadata.get("section_number")
            or metadata.get("article_number")
            or ""
        ).strip().upper()

        if not candidate_provision_type or not candidate_provision_number:
            continue

        if (
            explicit_document_ids
            and candidate_document_id not in explicit_document_ids
        ):
            continue

        if (
            candidate_provision_type,
            candidate_provision_number,
        ) in explicit_pairs:
            return True

    return False


def _approx_token_count(text: str) -> int:
    return len(
        re.findall(
            r"\S+",
            text,
        )
    )


def _document_chunk_id(document: Any) -> str:
    metadata = getattr(
        document,
        "metadata",
        {},
    )
    return str(
        metadata.get("chunk_id")
        or metadata.get("document_chunk_id")
        or metadata.get("chunk_number")
        or metadata.get("provision_number")
        or ""
    ).strip()


def _document_provision_identity(document: Any) -> str:
    metadata = getattr(
        document,
        "metadata",
        {},
    )
    document_id = str(
        metadata.get("document_id") or ""
    ).strip().lower()
    provision_type = str(
        metadata.get("provision_type") or ""
    ).strip().lower()
    provision_number = str(
        metadata.get("provision_number")
        or metadata.get("section_number")
        or metadata.get("article_number")
        or ""
    ).strip().upper()

    if not document_id or not provision_type or not provision_number:
        return ""

    return "::".join(
        [
            document_id,
            provision_type,
            provision_number,
        ]
    )


def _document_trace_summary(document: Any) -> dict[str, Any]:
    metadata = getattr(
        document,
        "metadata",
        {},
    )
    page_content = str(
        getattr(
            document,
            "page_content",
            "",
        )
        or ""
    )

    return {
        "chunk_id": _document_chunk_id(document),
        "provision_identity": _document_provision_identity(
            document
        ),
        "document_id": str(
            metadata.get("document_id") or ""
        ).strip(),
        "provision_type": str(
            metadata.get("provision_type") or ""
        ).strip(),
        "provision_number": str(
            metadata.get("provision_number")
            or metadata.get("section_number")
            or metadata.get("article_number")
            or ""
        ).strip(),
        "chunk_number": metadata.get("chunk_number"),
        "document_chunk_number": metadata.get(
            "document_chunk_number"
        ),
        "page_number": metadata.get("page_number"),
        "page_range": metadata.get("page_range"),
        "estimated_token_count": _approx_token_count(
            page_content
        ),
    }


def _candidate_trace_summary(
    candidate: Any,
) -> dict[str, Any]:
    return {
        **_document_trace_summary(
            candidate.document
        ),
        "retrieval_method": str(
            candidate.retrieval_method or ""
        ).strip(),
        "query_index": candidate.query_index,
        "query_text": candidate.query_text,
        "retrieval_rank": candidate.retrieval_rank,
        "relevance_score": candidate.relevance_score,
    }


def _ranked_trace_summary(
    item: RankedDocument,
) -> dict[str, Any]:
    return {
        **_document_trace_summary(
            item.document
        ),
        "fusion_score": item.fusion_score,
        "relevance_score": item.relevance_score,
        "matched_queries": item.matched_queries,
        "retrieval_methods": list(
            item.retrieval_methods
        ),
        "matched_query_indices": list(
            item.matched_query_indices
        ),
        "retrieval_routes": list(
            item.retrieval_routes
        ),
        "final_score": item.final_score,
    }


def _source_trace_summary(
    source: dict[str, Any],
) -> dict[str, Any]:
    return {
        "label": source.get("label"),
        "document_id": source.get("document_id"),
        "provision_type": source.get("provision_type"),
        "provision_number": source.get("provision_number"),
        "chunk_number": source.get("chunk_number"),
        "page_start": source.get("page_start"),
        "page_end": source.get("page_end"),
        "page_number": source.get("page_number"),
        "quality_status": source.get("quality_status"),
    }


def _serialize_legal_reference(
    reference: Any,
) -> dict[str, Any]:
    return {
        "provision_type": getattr(
            reference,
            "provision_type",
            None,
        ),
        "base_number": getattr(
            reference,
            "base_number",
            None,
        ),
        "subsection_path": list(
            getattr(
                reference,
                "subsection_path",
                [],
            )
        ),
        "component_type": getattr(
            reference,
            "component_type",
            None,
        ),
        "original_citation": getattr(
            reference,
            "original_citation",
            "",
        ),
    }


def _parse_cited_source_labels(
    answer: str,
) -> list[str]:
    return [
        f"Source {number}"
        for number in sorted(
            {
                int(number)
                for number in re.findall(
                    r"\[Source\s+(\d+)\]",
                    answer,
                    flags=re.IGNORECASE,
                )
            }
        )
    ]


def _build_retrieval_trace(
    *,
    question: str,
    plan: QueryPlan,
    retrieval_k: int,
    retrieval_status: str,
    candidates: list[Any],
    exact_documents: list[Any],
    ranked_items: list[RankedDocument],
    retrieved_documents: list[Any],
    confidence: Any | None,
    sources: list[dict[str, Any]],
    answer: str | None,
    retrieval_issues: list[Any],
    retrieval_trace: dict[str, Any],
    timings_ms: dict[str, float],
    generation_started_at: float | None,
    generation_finished_at: float | None,
) -> dict[str, Any]:
    retrieval_events = list(
        retrieval_trace.get(
            "events",
            [],
        )
    )

    channel_metrics: dict[str, dict[str, Any]] = {}
    for event in retrieval_events:
        channel = str(
            event.get("channel") or ""
        ).strip() or "unknown"
        channel_entry = channel_metrics.setdefault(
            channel,
            {
                "event_count": 0,
                "failure_count": 0,
                "latency_ms_total": 0.0,
                "accepted_count": 0,
                "raw_count": 0,
            },
        )
        channel_entry["event_count"] += 1
        channel_entry["latency_ms_total"] += float(
            event.get("latency_ms") or 0.0
        )
        channel_entry["accepted_count"] += int(
            event.get("accepted_count") or 0
        )
        channel_entry["raw_count"] += int(
            event.get("raw_count") or 0
        )
        if str(event.get("status") or "") != "success":
            channel_entry["failure_count"] += 1

    retrieved_context_summaries = [
        _document_trace_summary(document)
        for document in retrieved_documents
    ]

    trace: dict[str, Any] = {
        "schema_version": "f12",
        "retrieval_status": retrieval_status,
        "question": {
            "original": question,
            "normalized_type": plan.question_type,
            "retrieval_k": retrieval_k,
            "detected_concepts": list(plan.concepts),
            "extracted_citations": [
                _serialize_legal_reference(reference)
                for reference in getattr(
                    plan,
                    "legal_references",
                    [],
                )
            ],
            "retrieval_queries": list(
                plan.retrieval_queries
            ),
            "routing": {
                "document_ids": list(
                    getattr(
                        plan,
                        "document_ids",
                        [],
                    )
                ),
                "document_hints": list(
                    getattr(
                        plan,
                        "document_hints",
                        [],
                    )
                ),
                "provision_type": getattr(
                    plan,
                    "provision_type",
                    None,
                ),
                "provision_numbers": list(
                    getattr(
                        plan,
                        "provision_numbers",
                        [],
                    )
                ),
                "section_number": plan.section_number,
                "article_number": plan.article_number,
            },
        },
        "retrieval": {
            "events": retrieval_events,
            "channels": channel_metrics,
            "candidate_count": len(candidates),
            "exact_document_count": len(exact_documents),
            "ranked_item_count": len(ranked_items),
            "selected_context_count": len(
                retrieved_documents
            ),
            "candidate_chunks": [
                _candidate_trace_summary(candidate)
                for candidate in candidates
            ],
            "exact_chunks": [
                _document_trace_summary(document)
                for document in exact_documents
            ],
            "ranked_chunks": [
                _ranked_trace_summary(item)
                for item in ranked_items
            ],
            "selected_contexts": retrieved_context_summaries,
            "retrieval_issues": [
                {
                    "operation": getattr(
                        issue,
                        "operation",
                        None,
                    ),
                    "route": getattr(
                        issue,
                        "route",
                        None,
                    ),
                    "collection": getattr(
                        issue,
                        "collection",
                        None,
                    ),
                    "category": getattr(
                        issue,
                        "category",
                        None,
                    ),
                }
                for issue in retrieval_issues
            ],
        },
        "sources": [
            _source_trace_summary(source)
            for source in sources
        ],
        "cited_source_labels": (
            _parse_cited_source_labels(answer)
            if answer
            else []
        ),
        "timings_ms": timings_ms,
    }

    if confidence is not None:
        trace["confidence"] = {
            "label": confidence.label,
            "score": confidence.score,
            "top_similarity": confidence.top_similarity,
            "average_similarity": confidence.average_similarity,
            "section_count": confidence.section_count,
            "concept_coverage": confidence.concept_coverage,
            "document_count": getattr(
                confidence,
                "document_count",
                0,
            ),
            "document_coverage": getattr(
                confidence,
                "document_coverage",
                0.0,
            ),
            "exact_document_match": getattr(
                confidence,
                "exact_document_match",
                False,
            ),
            "exact_provision_match": getattr(
                confidence,
                "exact_provision_match",
                False,
            ),
        }

    if generation_started_at is not None and generation_finished_at is not None:
        trace["timings_ms"]["generation"] = (
            (generation_finished_at - generation_started_at) * 1000.0
        )

    trace["timings_ms"].setdefault("generation", 0.0)
    trace["timings_ms"].setdefault("retrieval", 0.0)
    trace["timings_ms"].setdefault("ranking", 0.0)
    trace["timings_ms"].setdefault("selection", 0.0)
    trace["timings_ms"].setdefault("total", 0.0)
    trace["timings_ms"].setdefault("exact", 0.0)
    trace["timings_ms"].setdefault("vector", 0.0)
    trace["timings_ms"].setdefault("lexical", 0.0)
    trace["timings_ms"].setdefault("neighbor", 0.0)

    return trace


def empty_result(
    question_type: str,
    concepts: list[str],
    retrieval_trace: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Return a consistent response when retrieval finds no context."""

    return {
        "answer": (
            "The answer was not found in the four indexed "
            "legal documents."
        ),
        "sources": [],
        "retrieved_contexts": [],
        "question_type": question_type,
        "detected_concepts": concepts,
        "retrieved_document_count": 0,
        "retrieval_status": "no_match",
        "confidence": {
            "label": "Low",
            "score": 0.0,
            "top_similarity": 0.0,
            "average_similarity": 0.0,
            "section_count": 0,
            "concept_coverage": 0.0,
        },
    }

    if retrieval_trace is not None:
        result["retrieval_trace"] = retrieval_trace

    return result


def retrieval_error_result(
    question_type: str,
    concepts: list[str],
    retrieval_trace: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Return a controlled response when mandatory retrieval fails."""

    result = {
        "answer": (
            "The legal sources could not be retrieved because the "
            "retrieval service encountered an error."
        ),
        "sources": [],
        "retrieved_contexts": [],
        "question_type": question_type,
        "detected_concepts": concepts,
        "retrieved_document_count": 0,
        "retrieval_status": "error",
        "confidence": {
            "label": "Low",
            "score": 0.0,
            "top_similarity": 0.0,
            "average_similarity": 0.0,
            "section_count": 0,
            "concept_coverage": 0.0,
        },
    }

    if retrieval_trace is not None:
        result["retrieval_trace"] = retrieval_trace

    return result


def get_final_k(question_type: str) -> int:
    return FINAL_K_BY_QUESTION_TYPE.get(question_type, 5)


def answer_question(
    question: str,
    retriever: AdaptiveRetriever,
    chat_model: Any,
    *,
    plan: QueryPlan | None = None,
    retrieval_k: int | None = None,
    include_trace: bool = False,
) -> dict[str, Any]:
    """Retrieve legal context and generate a grounded answer."""

    overall_started_at = time.perf_counter()
    question = question.strip()

    if len(question) < QUESTION_MIN_LENGTH:
        raise ValueError(
            "The question must contain at least "
            f"{QUESTION_MIN_LENGTH} characters."
        )

    if len(question) > QUESTION_MAX_LENGTH:
        raise ValueError(
            "The question must not exceed "
            f"{QUESTION_MAX_LENGTH} characters."
        )

    plan = plan or route_question(question)

    if retrieval_k is None:
        retrieval_k = get_retrieval_k(
            plan.question_type,
            TOP_K,
        )

    retrieval_issues: list[Any] = []
    retrieval_trace_data: dict[str, Any] = {}
    retrieval_started_at = time.perf_counter()

    try:
        candidates, exact_documents = fetch_candidates(
            retriever=retriever,
            question=plan.original_question,
            queries=plan.retrieval_queries,
            question_type=plan.question_type,
            section_number=plan.section_number,
            detected_concepts=plan.concepts,
            top_k=retrieval_k,
            neighbor_radius=NEIGHBOR_RADIUS,
            enable_neighbor_retrieval=(
                ENABLE_NEIGHBOR_RETRIEVAL
            ),
            min_relevance_score=MIN_RELEVANCE_SCORE,
            document_ids=getattr(
                plan,
                "document_ids",
                [],
            ),
            provision_type=getattr(
                plan,
                "provision_type",
                None,
            ),
            provision_numbers=getattr(
                plan,
                "provision_numbers",
                [],
            ),
            article_number=getattr(
                plan,
                "article_number",
                None,
            ),
            legal_references=getattr(
                plan,
                "legal_references",
                [],
            ),
            retrieval_issues=retrieval_issues,
            retrieval_trace=retrieval_trace_data,
        )
    except MandatoryRetrievalError:
        timings_ms = {
            "retrieval": (time.perf_counter() - retrieval_started_at)
            * 1000.0,
            "ranking": 0.0,
            "selection": 0.0,
            "generation": 0.0,
            "total": (time.perf_counter() - overall_started_at)
            * 1000.0,
        }
        trace = _build_retrieval_trace(
            question=plan.original_question,
            plan=plan,
            retrieval_k=retrieval_k,
            retrieval_status="error",
            candidates=[],
            exact_documents=[],
            ranked_items=[],
            retrieved_documents=[],
            confidence=None,
            sources=[],
            answer=None,
            retrieval_issues=retrieval_issues,
            retrieval_trace=retrieval_trace_data,
            timings_ms=timings_ms,
            generation_started_at=None,
            generation_finished_at=None,
        )
        logger.info(
            "retrieval_trace_summary=%s",
            json.dumps(
                {
                    "status": trace["retrieval_status"],
                    "question_type": trace["question"]["normalized_type"],
                    "candidate_count": trace["retrieval"]["candidate_count"],
                    "exact_document_count": trace["retrieval"]["exact_document_count"],
                    "selected_context_count": trace["retrieval"]["selected_context_count"],
                    "timings_ms": trace["timings_ms"],
                },
                ensure_ascii=False,
                default=str,
            ),
        )
        return retrieval_error_result(
            question_type=plan.question_type,
            concepts=plan.concepts,
            retrieval_trace=trace if include_trace else None,
        )

    if (
        _has_explicit_provision_request(plan)
        and not exact_documents
        and not _candidate_matches_explicit_provision(
            candidates,
            plan,
        )
    ):
        timings_ms = {
            "retrieval": (
                time.perf_counter() - retrieval_started_at
            )
            * 1000.0,
            "ranking": 0.0,
            "selection": 0.0,
            "generation": 0.0,
            "total": (
                time.perf_counter() - overall_started_at
            )
            * 1000.0,
        }
        trace = _build_retrieval_trace(
            question=plan.original_question,
            plan=plan,
            retrieval_k=retrieval_k,
            retrieval_status="no_match",
            candidates=candidates,
            exact_documents=exact_documents,
            ranked_items=[],
            retrieved_documents=[],
            confidence=None,
            sources=[],
            answer=None,
            retrieval_issues=retrieval_issues,
            retrieval_trace=retrieval_trace_data,
            timings_ms=timings_ms,
            generation_started_at=None,
            generation_finished_at=None,
        )
        logger.info(
            "retrieval_trace_summary=%s",
            json.dumps(
                {
                    "status": trace["retrieval_status"],
                    "question_type": trace["question"]["normalized_type"],
                    "candidate_count": trace["retrieval"]["candidate_count"],
                    "exact_document_count": trace["retrieval"]["exact_document_count"],
                    "selected_context_count": trace["retrieval"]["selected_context_count"],
                    "timings_ms": trace["timings_ms"],
                },
                ensure_ascii=False,
                default=str,
            ),
        )
        return empty_result(
            question_type=plan.question_type,
            concepts=plan.concepts,
            retrieval_trace=trace if include_trace else None,
        )

    ranking_started_at = time.perf_counter()
    ranked_items = rank_candidates(
        question=plan.original_question,
        detected_concepts=plan.concepts,
        candidates=candidates,
        semantic_threshold=(
            SEMANTIC_DEDUP_THRESHOLD
        ),
    )
    ranking_finished_at = time.perf_counter()

    if exact_documents:
        exact_ranked_items = build_exact_ranked_items(
            merge_provision_parts(
                exact_documents
            )
        )
        ranked_items = deduplicate_ranked_documents(
            exact_ranked_items + ranked_items,
            semantic_threshold=(
                SEMANTIC_DEDUP_THRESHOLD
            ),
        )

    selection_started_at = time.perf_counter()
    retrieved_documents = select_context_documents(
        ranked_items=ranked_items,
        question_type=plan.question_type,
        maximum_documents=(
            MAX_CONTEXT_DOCUMENTS
        ),
        max_context_sections=(
            MAX_CONTEXT_PROVISIONS
        ),
        final_k=get_final_k(plan.question_type),
    )
    selection_finished_at = time.perf_counter()

    if not retrieved_documents:
        timings_ms = {
            "retrieval": (
                ranking_started_at - retrieval_started_at
            )
            * 1000.0,
            "ranking": (
                ranking_finished_at - ranking_started_at
            )
            * 1000.0,
            "selection": (
                selection_finished_at - selection_started_at
            )
            * 1000.0,
            "generation": 0.0,
            "total": (
                time.perf_counter() - overall_started_at
            )
            * 1000.0,
        }
        trace = _build_retrieval_trace(
            question=plan.original_question,
            plan=plan,
            retrieval_k=retrieval_k,
            retrieval_status="no_match",
            candidates=candidates,
            exact_documents=exact_documents,
            ranked_items=ranked_items,
            retrieved_documents=[],
            confidence=None,
            sources=[],
            answer=None,
            retrieval_issues=retrieval_issues,
            retrieval_trace=retrieval_trace_data,
            timings_ms=timings_ms,
            generation_started_at=None,
            generation_finished_at=None,
        )
        logger.info(
            "retrieval_trace_summary=%s",
            json.dumps(
                {
                    "status": trace["retrieval_status"],
                    "question_type": trace["question"]["normalized_type"],
                    "candidate_count": trace["retrieval"]["candidate_count"],
                    "exact_document_count": trace["retrieval"]["exact_document_count"],
                    "selected_context_count": trace["retrieval"]["selected_context_count"],
                    "timings_ms": trace["timings_ms"],
                },
                ensure_ascii=False,
                default=str,
            ),
        )
        return empty_result(
            question_type=plan.question_type,
            concepts=plan.concepts,
            retrieval_trace=trace if include_trace else None,
        )

    generation_started_at = time.perf_counter()
    confidence = calculate_retrieval_confidence(
        ranked_items=ranked_items,
        selected_documents=retrieved_documents,
        detected_concepts=plan.concepts,
    )

    prompt = build_grounded_prompt(
        question=plan.original_question,
        question_type=plan.question_type,
        context=format_context(
            retrieved_documents
        ),
        confidence=confidence,
    )

    answer = extract_response_text(
        chat_model.invoke(prompt)
    )
    generation_finished_at = time.perf_counter()

    sources = filter_sources_used_in_answer(
        answer=answer,
        sources=create_sources(
            retrieved_documents
        ),
    )

    timings_ms = {
        "retrieval": (
            ranking_started_at - retrieval_started_at
        )
        * 1000.0,
        "ranking": (
            ranking_finished_at - ranking_started_at
        )
        * 1000.0,
        "selection": (
            selection_finished_at - selection_started_at
        )
        * 1000.0,
        "generation": (
            generation_finished_at - generation_started_at
        )
        * 1000.0,
        "total": (
            generation_finished_at - overall_started_at
        )
        * 1000.0,
    }

    trace = _build_retrieval_trace(
        question=plan.original_question,
        plan=plan,
        retrieval_k=retrieval_k,
        retrieval_status="success",
        candidates=candidates,
        exact_documents=exact_documents,
        ranked_items=ranked_items,
        retrieved_documents=retrieved_documents,
        confidence=confidence,
        sources=sources,
        answer=answer,
        retrieval_issues=retrieval_issues,
        retrieval_trace=retrieval_trace_data,
        timings_ms=timings_ms,
        generation_started_at=generation_started_at,
        generation_finished_at=generation_finished_at,
    )

    logger.info(
        "retrieval_trace_summary=%s",
        json.dumps(
            {
                "status": trace["retrieval_status"],
                "question_type": trace["question"]["normalized_type"],
                "candidate_count": trace["retrieval"]["candidate_count"],
                "exact_document_count": trace["retrieval"]["exact_document_count"],
                "selected_context_count": trace["retrieval"]["selected_context_count"],
                "cited_source_labels": trace["cited_source_labels"],
                "timings_ms": trace["timings_ms"],
            },
            ensure_ascii=False,
            default=str,
        ),
    )

    result = {
        "answer": answer,
        "sources": sources,
        "retrieved_contexts": [
            document.page_content
            for document in retrieved_documents
        ],
        "question_type": plan.question_type,
        "detected_concepts": plan.concepts,
        "retrieved_document_count": len(
            retrieved_documents
        ),
        "confidence": {
            "label": confidence.label,
            "score": confidence.score,
            "top_similarity": (
                confidence.top_similarity
            ),
            "average_similarity": (
                confidence.average_similarity
            ),
            "section_count": (
                confidence.section_count
            ),
            "concept_coverage": (
                confidence.concept_coverage
            ),
        },
    }

    if include_trace:
        result["retrieval_trace"] = trace

    return result
