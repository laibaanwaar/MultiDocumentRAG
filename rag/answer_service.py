from __future__ import annotations

import os
import re
from typing import Any

from dotenv import load_dotenv
from langchain_groq import ChatGroq

from rag.context_builder import create_sources, format_context
from rag.intent_router import get_retrieval_k, route_question
from rag.prompt_builder import build_grounded_prompt
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


GROQ_MODEL = os.getenv(
    "GROQ_MODEL",
    "llama-3.3-70b-versatile",
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
            final_score=1.0,
        )
        for document in documents
    ]


def empty_result(
    question_type: str,
    concepts: list[str],
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
        "confidence": {
            "label": "Low",
            "score": 0.0,
            "top_similarity": 0.0,
            "average_similarity": 0.0,
            "section_count": 0,
            "concept_coverage": 0.0,
        },
    }


def get_final_k(question_type: str) -> int:
    return FINAL_K_BY_QUESTION_TYPE.get(question_type, 5)


def answer_question(
    question: str,
    retriever: AdaptiveRetriever,
    chat_model: Any,
    *,
    plan: QueryPlan | None = None,
    retrieval_k: int | None = None,
) -> dict[str, Any]:
    """Retrieve legal context and generate a grounded answer."""

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
    )

    ranked_items = rank_candidates(
        question=plan.original_question,
        detected_concepts=plan.concepts,
        candidates=candidates,
        semantic_threshold=(
            SEMANTIC_DEDUP_THRESHOLD
        ),
    )

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

    if not retrieved_documents:
        return empty_result(
            question_type=plan.question_type,
            concepts=plan.concepts,
        )

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

    sources = filter_sources_used_in_answer(
        answer=answer,
        sources=create_sources(
            retrieved_documents
        ),
    )

    return {
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
