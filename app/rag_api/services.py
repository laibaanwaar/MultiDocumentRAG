from __future__ import annotations

import os
import logging
import sys
from datetime import timedelta
from pathlib import Path
from typing import Any

from django.conf import settings
from django.db import DatabaseError, IntegrityError, transaction
from django.utils import timezone
from rest_framework.exceptions import APIException

CURRENT_FILE = Path(__file__).resolve()
REPO_ROOT = CURRENT_FILE.parents[2]

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

os.environ.setdefault(
    "QDRANT_PATH",
    str(REPO_ROOT / "qdrant_storage"),
)
os.environ.setdefault(
    "QDRANT_COLLECTION",
    "pakistan_legal_knowledge_base",
)

from documents.models import DocumentCategory
from rag.intent_router import DOCUMENT_ROUTES, normalize_text, route_question
from rag.rag_chain import answer_question as run_rag_question
from rag.rag_chain import create_rag_components
from rag.schemas import QueryPlan

from billing.models import Subscription
from rag_api.exceptions import (
    RagCategoryInactiveError,
    RagCategoryNotFoundError,
    RagInternalError,
    RagQueryLimitExceededError,
    RagServiceUnavailableError,
    RagSubscriptionRequiredError,
)
from rag_api.models import QueryHistory


logger = logging.getLogger(__name__)

DEFAULT_USAGE_RESET_DAYS = 30


def _normalize_text(value: str) -> str:
    return normalize_text(value)


def _category_to_document_ids(category: DocumentCategory) -> list[str]:
    category_tokens = {
        _normalize_text(category.code),
        _normalize_text(category.name),
    }

    document_ids: list[str] = []

    for route in DOCUMENT_ROUTES:
        route_tokens = {
            _normalize_text(route.document_id),
            _normalize_text(route.short_name),
            _normalize_text(route.full_name),
        }
        route_tokens.update(
            _normalize_text(alias)
            for alias in route.aliases
        )

        if category_tokens & route_tokens:
            document_ids.append(route.document_id)

    if not document_ids:
        raise RagCategoryNotFoundError()

    return document_ids


def _load_category_or_404(category_id: int) -> DocumentCategory:
    try:
        category = DocumentCategory.objects.get(pk=category_id)
    except DocumentCategory.DoesNotExist as exception:
        raise RagCategoryNotFoundError() from exception

    if not category.is_active:
        raise RagCategoryInactiveError()

    return category


def _build_query_plan(question: str, category: DocumentCategory | None) -> QueryPlan:
    plan = route_question(question)

    if category is not None:
        plan.document_ids = _category_to_document_ids(category)
        plan.document_hints = [
            category.code,
            category.name,
        ]

    return plan


def _load_active_subscription(user) -> Subscription:
    try:
        subscription = (
            Subscription.objects.select_related("plan")
            .get(user=user)
        )
    except Subscription.DoesNotExist as exception:
        raise RagSubscriptionRequiredError() from exception

    if not subscription.plan.is_active:
        raise RagSubscriptionRequiredError()

    if subscription.status != Subscription.Status.ACTIVE:
        raise RagSubscriptionRequiredError()

    if subscription.is_expired:
        raise RagSubscriptionRequiredError()

    return subscription


def _reset_usage_if_needed(subscription: Subscription) -> bool:
    reset_at = subscription.usage_reset_at

    if reset_at is None:
        return False

    if timezone.now() < reset_at:
        return False

    reset_days = DEFAULT_USAGE_RESET_DAYS

    if subscription.plan.billing_period == subscription.plan.BillingPeriod.YEARLY:
        reset_days = 365

    subscription.queries_used = 0
    subscription.usage_reset_at = timezone.now() + timedelta(days=reset_days)
    subscription.save(
        update_fields=[
            "queries_used",
            "usage_reset_at",
            "updated_at",
        ]
    )

    return True


def _classify_rag_failure(error: Exception) -> Exception:
    message = str(error).lower()

    service_unavailable_markers = (
        "qdrant",
        "groq",
        "embedding",
        "sentence-transformers",
        "connection",
        "timeout",
        "temporarily unavailable",
        "database is locked",
        "resource temporarily unavailable",
    )

    if any(marker in message for marker in service_unavailable_markers):
        return RagServiceUnavailableError()

    return RagInternalError()


def _finalize_usage_and_history(
    *,
    user,
    question: str,
    category: DocumentCategory | None,
    rag_result: dict[str, Any],
    subscription: Subscription,
) -> QueryHistory:
    history = QueryHistory.objects.create(
        user=user,
        question=question,
        answer=rag_result["answer"],
        category=category,
        sources=rag_result.get("sources", []),
    )

    subscription.queries_used += 1
    subscription.save(update_fields=["queries_used", "updated_at"])

    return history


def _has_relevant_context(rag_result: dict[str, Any]) -> bool:
    try:
        return int(rag_result.get("retrieved_document_count", 0)) > 0
    except (TypeError, ValueError):
        return False


def run_query(
    *,
    user,
    question: str,
    category_id: int | None = None,
) -> dict[str, Any]:
    question = question.strip()
    client = None

    category = None
    if category_id is not None:
        category = _load_category_or_404(category_id)

    plan = _build_query_plan(question, category)

    try:
        retriever, chat_model, client = create_rag_components()
    except Exception as error:
        logger.exception("Failed to initialize the current RAG pipeline.")
        raise _classify_rag_failure(error) from error

    try:
        with transaction.atomic():
            subscription = _load_active_subscription(user)

            # Re-check usage under the row lock so concurrent requests
            # cannot exceed the plan limit.
            subscription = (
                Subscription.objects.select_for_update()
                .select_related("plan")
                .get(pk=subscription.pk)
            )

            if not subscription.plan.is_active:
                raise RagSubscriptionRequiredError()

            if subscription.status != Subscription.Status.ACTIVE:
                raise RagSubscriptionRequiredError()

            if subscription.is_expired:
                raise RagSubscriptionRequiredError()

            _reset_usage_if_needed(subscription)

            if subscription.queries_used >= subscription.plan.query_limit:
                raise RagQueryLimitExceededError()

            try:
                rag_result = run_rag_question(
                    question=question,
                    retriever=retriever,
                    chat_model=chat_model,
                    plan=plan,
                )
            except Exception as error:
                raise _classify_rag_failure(error) from error

            history = None

            if _has_relevant_context(rag_result):
                history = _finalize_usage_and_history(
                    user=user,
                    question=question,
                    category=category,
                    rag_result=rag_result,
                    subscription=subscription,
                )

            return {
                "history": history,
                "rag_result": rag_result,
                "queries_remaining": subscription.queries_remaining,
                "category": category,
            }

    except (RagCategoryNotFoundError, RagCategoryInactiveError, RagSubscriptionRequiredError, RagQueryLimitExceededError):
        raise
    except APIException:
        raise
    except (IntegrityError, DatabaseError) as error:
        logger.exception("Database error while saving the legal query.")
        raise RagInternalError() from error
    except Exception as error:
        logger.exception("Unexpected legal query failure.")
        raise _classify_rag_failure(error) from error
    finally:
        try:
            client.close()
        except Exception:
            logger.debug("Failed to close the Qdrant client cleanly.", exc_info=True)


def get_query_history_queryset(user):
    return (
        QueryHistory.objects
        .filter(user=user)
        .select_related("category")
        .order_by("-created_at", "-id")
    )
