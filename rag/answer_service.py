import os
import re
import time
from typing import Any

from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

from rag.context_builder import create_sources, format_context
from rag.intent_router import get_retrieval_k, normalize_text, route_question
from rag.prompt_builder import build_grounded_prompt
from rag.ranker import (
    calculate_retrieval_confidence,
    merge_section_parts,
    rank_candidates,
    select_context_documents,
)
from rag.retriever import AdaptiveRetriever, fetch_candidates
from rag.schemas import RankedDocument, RetrievalConfidence
from rag.schemas import QueryPlan
from rag.vector_store import create_vector_store


load_dotenv()


CHAT_MODEL = os.getenv("CHAT_MODEL", "gemini-3.6-flash")
TOP_K = int(os.getenv("TOP_K", "6"))
MAX_CONTEXT_DOCUMENTS = int(os.getenv("MAX_CONTEXT_DOCUMENTS", "10"))
MAX_CONTEXT_SECTIONS = int(os.getenv("MAX_CONTEXT_SECTIONS", "4"))
MIN_RELEVANCE_SCORE = float(os.getenv("MIN_RELEVANCE_SCORE", "0.0"))
ENABLE_NEIGHBOR_RETRIEVAL = os.getenv("ENABLE_NEIGHBOR_RETRIEVAL", "True").strip().lower() in {
    "1",
    "true",
    "yes",
    "on",
}
NEIGHBOR_SECTION_RADIUS = int(os.getenv("NEIGHBOR_SECTION_RADIUS", "2"))
SEMANTIC_DEDUP_THRESHOLD = float(os.getenv("SEMANTIC_DEDUP_THRESHOLD", "0.94"))
GOOGLE_API_MAX_RETRIES = int(os.getenv("GOOGLE_API_MAX_RETRIES", "4"))
GOOGLE_API_RETRY_DELAY = float(os.getenv("GOOGLE_API_RETRY_DELAY", "5"))

QUESTION_MIN_LENGTH = 3
QUESTION_MAX_LENGTH = 4000

THEFT_SECTION_NUMBER = "379"
LAW_NAME_ALIASES: dict[str, tuple[str, ...]] = {
    "ppc": (
        "ppc",
        "pakistan penal code",
        "pakistan penal code 1860",
    ),
    "crpc": (
        "crpc",
        "code of criminal procedure",
        "criminal procedure code",
    ),
}


def get_chat_model() -> ChatGoogleGenerativeAI:
    api_key = os.getenv("GOOGLE_API_KEY")

    if not api_key:
        raise ValueError("GOOGLE_API_KEY is missing from the .env file.")

    return ChatGoogleGenerativeAI(
        model=CHAT_MODEL,
        google_api_key=api_key,
    )


def create_rag_components():
    vector_store, client = create_vector_store(reset=False)
    return AdaptiveRetriever(vector_store), get_chat_model(), client


def extract_response_text(response: Any) -> str:
    response_text = getattr(response, "text", None)

    if isinstance(response_text, str) and response_text.strip():
        return response_text.strip()

    content = getattr(response, "content", "")

    if isinstance(content, str):
        return content.strip()

    if isinstance(content, list):
        text_parts: list[str] = []

        for item in content:
            item_text = item.get("text") if isinstance(item, dict) else getattr(item, "text", None)
            if isinstance(item_text, str) and item_text.strip():
                text_parts.append(item_text.strip())

        return "\n".join(text_parts).strip()

    return str(content).strip()


def invoke_chat_model_with_retry(
    chat_model: ChatGoogleGenerativeAI,
    prompt: str,
):
    retry_markers = {
        "429",
        "resource_exhausted",
        "quota",
        "rate limit",
        "temporarily unavailable",
        "503",
        "502",
        "504",
        "timeout",
    }

    for attempt in range(1, GOOGLE_API_MAX_RETRIES + 1):
        try:
            return chat_model.invoke(prompt)
        except Exception as error:
            error_text = str(error).lower()
            is_retryable = any(marker in error_text for marker in retry_markers)

            if not is_retryable or attempt == GOOGLE_API_MAX_RETRIES:
                raise

            delay = GOOGLE_API_RETRY_DELAY * (2 ** (attempt - 1))
            print(
                "Gemini request temporarily failed. "
                f"Retrying in {delay:.0f} seconds ({attempt}/{GOOGLE_API_MAX_RETRIES})..."
            )
            time.sleep(delay)

    raise RuntimeError("Gemini request failed after all retries.")


def filter_sources_used_in_answer(
    answer: str,
    sources: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    used_labels = {
        int(number)
        for number in re.findall(r"\[Source\s+(\d+)\]", answer, re.IGNORECASE)
    }

    if not used_labels:
        return sources

    return [
        source
        for index, source in enumerate(sources, start=1)
        if index in used_labels
    ]


def _build_ranked_items_for_exact_documents(documents: list[Any]) -> list[RankedDocument]:
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


def _document_text(document: Any) -> str:
    metadata = getattr(document, "metadata", {}) or {}
    return normalize_text(
        f"{metadata.get('document_name', '')} {metadata.get('document_id', '')}"
    )


def _document_display_name(document: Any) -> str:
    metadata = getattr(document, "metadata", {}) or {}
    return str(
        metadata.get("document_name")
        or metadata.get("document_id")
        or "Unknown document"
    )


def _detect_law_mentions(question: str) -> set[str]:
    normalized = normalize_text(question)
    mentioned: set[str] = set()

    for canonical_name, aliases in LAW_NAME_ALIASES.items():
        if any(alias in normalized for alias in aliases):
            mentioned.add(canonical_name)

    return mentioned


def _document_matches_law(document: Any, law_name: str) -> bool:
    aliases = LAW_NAME_ALIASES.get(law_name, ())
    document_text = _document_text(document)
    return any(alias in document_text for alias in aliases)


def _filter_documents_by_mentioned_laws(
    documents: list[Any],
    mentioned_laws: set[str],
) -> list[Any]:
    if not mentioned_laws:
        return documents

    filtered = [
        document
        for document in documents
        if any(_document_matches_law(document, law_name) for law_name in mentioned_laws)
    ]

    return filtered or documents


def _prefer_ppc_documents(documents: list[Any]) -> list[Any]:
    preferred = [
        document
        for document in documents
        if _document_matches_law(document, "ppc")
    ]

    return preferred or documents


def _build_section_lookup_clarification(
    section_number: str,
    documents: list[Any],
) -> str:
    document_names: list[str] = []
    seen_names: set[str] = set()

    for document in documents:
        display_name = _document_display_name(document)

        if display_name in seen_names:
            continue

        seen_names.add(display_name)
        document_names.append(display_name)

    if len(document_names) == 1:
        return ""

    if len(document_names) == 2:
        law_list = f"{document_names[0]} and {document_names[1]}"
    else:
        law_list = ", ".join(document_names[:-1]) + f", and {document_names[-1]}"

    return (
        f"Section {section_number} appears in multiple indexed laws: "
        f"{law_list}. Which law did you mean?"
    )


def _build_clarification_result(
    question_type: str,
    clarification: str,
) -> dict[str, Any]:
    return {
        "answer": clarification,
        "sources": [],
        "retrieved_contexts": [],
        "question_type": question_type,
        "confidence": {
            "label": "Low",
            "score": 0.0,
            "top_similarity": 0.0,
            "average_similarity": 0.0,
            "section_count": 0,
            "concept_coverage": 0.0,
        },
        "needs_clarification": True,
    }


def answer_question(
    question: str,
    retriever: AdaptiveRetriever,
    chat_model: ChatGoogleGenerativeAI,
    *,
    plan: QueryPlan | None = None,
    retrieval_k: int | None = None,
) -> dict[str, Any]:
    if not question or len(question.strip()) < QUESTION_MIN_LENGTH:
        raise ValueError(
            "The question must contain at least "
            f"{QUESTION_MIN_LENGTH} characters."
        )

    original_question = question.strip()

    if len(original_question) > QUESTION_MAX_LENGTH:
        raise ValueError(
            "The question must not exceed "
            f"{QUESTION_MAX_LENGTH} characters."
        )

    if plan is None:
        plan = route_question(original_question)

    if retrieval_k is None:
        top_k = TOP_K
        retrieval_k = get_retrieval_k(plan.question_type, top_k)

    retrieval_section_number = plan.section_number
    mentioned_laws = _detect_law_mentions(plan.original_question)

    if (
        plan.question_type == "punishment"
        and plan.section_number is None
        and plan.concepts == ["theft"]
    ):
        retrieval_section_number = THEFT_SECTION_NUMBER

    candidate_documents, exact_documents = fetch_candidates(
        retriever=retriever,
        question=plan.original_question,
        queries=plan.retrieval_queries,
        question_type=plan.question_type,
        section_number=retrieval_section_number,
        detected_concepts=plan.concepts,
        top_k=retrieval_k,
        neighbor_radius=NEIGHBOR_SECTION_RADIUS,
        enable_neighbor_retrieval=ENABLE_NEIGHBOR_RETRIEVAL,
        min_relevance_score=MIN_RELEVANCE_SCORE,
    )

    if exact_documents:
        exact_documents = _filter_documents_by_mentioned_laws(
            exact_documents,
            mentioned_laws,
        )

        if (
            plan.question_type == "punishment"
            and plan.section_number is None
            and plan.concepts == ["theft"]
            and not mentioned_laws
        ):
            exact_documents = _prefer_ppc_documents(exact_documents)

        if (
            plan.question_type == "section_lookup"
            and retrieval_section_number
            and len({
                _document_display_name(document)
                for document in exact_documents
            }) > 1
            and not mentioned_laws
        ):
            clarification = _build_section_lookup_clarification(
                section_number=retrieval_section_number,
                documents=exact_documents,
            )

            if clarification:
                return _build_clarification_result(
                    question_type=plan.question_type,
                    clarification=clarification,
                )

        retrieved_documents = merge_section_parts(exact_documents)[:MAX_CONTEXT_DOCUMENTS]
        ranked_items = _build_ranked_items_for_exact_documents(retrieved_documents)
    else:
        ranked_items = rank_candidates(
            question=plan.original_question,
            detected_concepts=plan.concepts,
            candidates=candidate_documents,
            semantic_threshold=SEMANTIC_DEDUP_THRESHOLD,
        )
        retrieved_documents = select_context_documents(
            ranked_items=ranked_items,
            question_type=plan.question_type,
            maximum_documents=MAX_CONTEXT_DOCUMENTS,
            max_context_sections=MAX_CONTEXT_SECTIONS,
            final_k=8 if plan.question_type in {"fact_scenario", "comparison"} else 5,
        )

    if not retrieved_documents:
        return {
            "answer": "The answer was not found in the indexed Pakistan Penal Code document.",
            "sources": [],
            "retrieved_contexts": [],
            "question_type": plan.question_type,
            "confidence": {
                "label": "Low",
                "score": 0.0,
            },
        }

    confidence = calculate_retrieval_confidence(
        ranked_items=ranked_items,
        selected_documents=retrieved_documents,
        detected_concepts=plan.concepts,
    )

    context = format_context(retrieved_documents)
    prompt = build_grounded_prompt(
        question=plan.original_question,
        question_type=plan.question_type,
        context=context,
        confidence=confidence,
    )

    response = invoke_chat_model_with_retry(
        chat_model=chat_model,
        prompt=prompt,
    )

    answer = extract_response_text(response)
    all_sources = create_sources(retrieved_documents)
    used_sources = filter_sources_used_in_answer(answer, all_sources)

    return {
        "answer": answer,
        "sources": used_sources,
        "retrieved_contexts": [document.page_content for document in retrieved_documents],
        "question_type": plan.question_type,
        "detected_concepts": plan.concepts,
        "retrieved_document_count": len(retrieved_documents),
        "confidence": {
            "label": confidence.label,
            "score": confidence.score,
            "top_similarity": confidence.top_similarity,
            "average_similarity": confidence.average_similarity,
            "section_count": confidence.section_count,
            "concept_coverage": confidence.concept_coverage,
        },
    }
