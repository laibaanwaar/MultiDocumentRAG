from __future__ import annotations

import logging
from dataclasses import dataclass

from django.contrib.auth import get_user_model
from django.core.exceptions import ObjectDoesNotExist
from django.db import DatabaseError
from django.db.models import Count, Q, QuerySet

from billing.models import Plan, Subscription
from accounts.exceptions import AdminUserNotFoundError
from rag_api.models import QueryHistory


logger = logging.getLogger(__name__)

User = get_user_model()


@dataclass(frozen=True)
class AdminDashboardData:
    summary: dict
    recent_users: list
    subscription_overview: list


@dataclass(frozen=True)
class AdminUserDetailData:
    user: User
    subscription: Subscription | None
    recent_queries: list[QueryHistory]
    total_queries: int


def _normal_user_queryset() -> QuerySet:
    return User.objects.filter(
        is_staff=False,
        is_superuser=False,
    )


def _get_user_subscription(user: User) -> Subscription | None:
    try:
        return user.subscription
    except ObjectDoesNotExist:
        return None


def get_admin_dashboard_data() -> AdminDashboardData:
    normal_users = _normal_user_queryset()

    summary = {
        "total_users": normal_users.count(),
        "paid_subscribers": Subscription.objects.filter(
            user__is_staff=False,
            user__is_superuser=False,
            status=Subscription.Status.ACTIVE,
            plan__price__gt=0,
        ).count(),
        "active_subscribers": Subscription.objects.filter(
            user__is_staff=False,
            user__is_superuser=False,
            status=Subscription.Status.ACTIVE,
        ).count(),
    }

    recent_users = list(
        normal_users.select_related("subscription__plan")
        .order_by("-date_joined", "-id")[:5]
    )

    plans = list(
        Plan.objects.annotate(
            subscribers_count=Count(
                "subscriptions",
                filter=Q(
                    subscriptions__user__is_staff=False,
                    subscriptions__user__is_superuser=False,
                ),
            )
        ).order_by("id")
    )

    return AdminDashboardData(
        summary=summary,
        recent_users=recent_users,
        subscription_overview=plans,
    )


def get_admin_users_queryset(*, filters: dict) -> QuerySet:
    queryset = _normal_user_queryset().select_related("subscription__plan")

    status = filters.get("status")
    if status:
        queryset = queryset.filter(
            is_active=(status == "active")
        )

    plan_code = filters.get("plan")
    if plan_code:
        queryset = queryset.filter(
            subscription__plan__code__iexact=plan_code
        )

    search = filters.get("search")
    if search:
        queryset = queryset.filter(
            Q(username__icontains=search)
            | Q(email__icontains=search)
        )

    return queryset.order_by("-date_joined", "-id")


def get_admin_user_detail(*, user_id: int) -> AdminUserDetailData:
    user = (
        _normal_user_queryset()
        .select_related("subscription__plan")
        .filter(pk=user_id)
        .first()
    )

    if user is None:
        raise AdminUserNotFoundError()

    subscription = _get_user_subscription(user)

    recent_queries = list(
        QueryHistory.objects.filter(user=user)
        .order_by("-created_at", "-id")[:5]
    )

    total_queries = QueryHistory.objects.filter(user=user).count()

    return AdminUserDetailData(
        user=user,
        subscription=subscription,
        recent_queries=recent_queries,
        total_queries=total_queries,
    )
