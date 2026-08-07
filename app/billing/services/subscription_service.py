import logging
from dataclasses import dataclass
from datetime import timedelta
from typing import Any

from django.db import DatabaseError, IntegrityError, transaction
from django.utils import timezone

from billing.exceptions import (
    BillingConfigurationError,
    DuplicateSubscriptionError,
    PlanInactiveError,
    SubscriptionExpiredError,
    SubscriptionInactiveError,
    SubscriptionNotFoundError,
)
from billing.models import Plan, Subscription


logger = logging.getLogger(__name__)

FREE_PLAN_CODE = "FREE"
FREE_PLAN_NAME = "Free"
FREE_PLAN_PRICE = "0.00"
FREE_PLAN_BILLING_PERIOD = Plan.BillingPeriod.MONTHLY
FREE_PLAN_DOCUMENT_LIMIT = 3
FREE_PLAN_QUERY_LIMIT = 30
FREE_PLAN_STATUS = Subscription.Status.ACTIVE


@dataclass(frozen=True)
class SubscriptionResult:
    subscription: Any
    queries_remaining: int


def _default_usage_reset_at():
    return timezone.now() + timedelta(days=30)


def get_active_plans():
    return Plan.objects.filter(is_active=True).order_by("id")


def _get_free_plan():
    plan, created = Plan.objects.get_or_create(
        code=FREE_PLAN_CODE,
        defaults={
            "name": FREE_PLAN_NAME,
            "price": FREE_PLAN_PRICE,
            "billing_period": FREE_PLAN_BILLING_PERIOD,
            "document_limit": FREE_PLAN_DOCUMENT_LIMIT,
            "query_limit": FREE_PLAN_QUERY_LIMIT,
            "is_active": True,
        },
    )

    if not plan.is_active:
        raise BillingConfigurationError()

    if created:
        logger.info("Created default free billing plan.")

    return plan


def assign_free_plan_to_user(user) -> Subscription:
    try:
        with transaction.atomic():
            plan = _get_free_plan()
            subscription, _created = Subscription.objects.get_or_create(
                user=user,
                defaults={
                    "plan": plan,
                    "status": FREE_PLAN_STATUS,
                    "queries_used": 0,
                    "usage_reset_at": _default_usage_reset_at(),
                    "started_at": timezone.now(),
                },
            )
    except BillingConfigurationError:
        raise
    except Subscription.MultipleObjectsReturned as exception:
        logger.error(
            "Duplicate subscription records found while assigning free plan for user_id=%s",
            getattr(user, "id", None),
        )
        raise DuplicateSubscriptionError() from exception
    except IntegrityError as exception:
        logger.error(
            "Subscription integrity error while assigning free plan for user_id=%s",
            getattr(user, "id", None),
        )
        raise DuplicateSubscriptionError() from exception
    except DatabaseError:
        logger.error(
            "Database error while assigning free plan for user_id=%s",
            getattr(user, "id", None),
        )
        raise

    return subscription


def _get_user_subscription(user) -> Subscription:
    try:
        subscription = (
            Subscription.objects.select_related("plan")
            .get(user=user)
        )
    except Subscription.DoesNotExist as exception:
        raise SubscriptionNotFoundError() from exception
    except Subscription.MultipleObjectsReturned as exception:
        raise DuplicateSubscriptionError() from exception

    return subscription


def get_subscription_for_user(user) -> SubscriptionResult:
    subscription = _get_user_subscription(user)

    if not subscription.plan.is_active:
        raise PlanInactiveError()

    if subscription.status == Subscription.Status.CANCELED:
        raise SubscriptionInactiveError()

    if subscription.is_expired:
        raise SubscriptionExpiredError()

    return SubscriptionResult(
        subscription=subscription,
        queries_remaining=subscription.queries_remaining,
    )
