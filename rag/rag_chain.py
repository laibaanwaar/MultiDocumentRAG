"""Compatibility wrapper for the modular RAG pipeline."""

from typing import Any

from rag.schemas import QueryPlan
from rag.answer_service import (
    answer_question as run_answer_service,
    create_rag_components,
)
from rag.intent_router import get_retrieval_k, route_question


def answer_question(
    question: str,
    retriever: Any,
    chat_model: Any,
    top_k: int = 5,
    plan: QueryPlan | None = None,
) -> dict[str, Any]:
    """Route a question and pass it to the answer service."""

    plan = plan or route_question(question)

    return run_answer_service(
        question=question,
        retriever=retriever,
        chat_model=chat_model,
        plan=plan,
        retrieval_k=get_retrieval_k(
            plan.question_type,
            top_k,
        ),
    )
