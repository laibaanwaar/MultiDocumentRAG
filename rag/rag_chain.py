"""Backward-compatible chain wrapper for the modular RAG architecture."""

from rag.answer_service import answer_question as _answer_question
from rag.answer_service import create_rag_components
from rag.intent_router import get_retrieval_k, route_question


def answer_question(
    question: str,
    retriever,
    chat_model,
    top_k: int = 5,
):
    plan = route_question(question)
    retrieval_k = get_retrieval_k(
        plan.question_type,
        top_k=top_k,
    )

    return _answer_question(
        question=question,
        retriever=retriever,
        chat_model=chat_model,
        plan=plan,
        retrieval_k=retrieval_k,
    )
