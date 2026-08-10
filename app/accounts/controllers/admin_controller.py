import logging

from django.core.exceptions import ObjectDoesNotExist
from django.db import DatabaseError
from rest_framework.exceptions import (
    APIException,
    AuthenticationFailed,
    NotAuthenticated,
    NotFound,
    PermissionDenied,
)
from rest_framework.pagination import PageNumberPagination
from rest_framework.permissions import IsAuthenticated
from rest_framework.response import Response
from rest_framework.views import APIView
from rest_framework_simplejwt.authentication import JWTAuthentication
from rest_framework_simplejwt.exceptions import InvalidToken

from accounts.admin_serializers import (
    AdminDashboardResponseSerializer,
    AdminUserDetailResponseSerializer,
    AdminUserListSerializer,
    AdminUsersFilterSerializer,
)
from accounts.exceptions import AdminUserNotFoundError
from accounts.services.admin_service import (
    get_admin_dashboard_data,
    get_admin_user_detail,
    get_admin_users_queryset,
)
from billing.permissions import IsStaffUser


logger = logging.getLogger(__name__)


def _flatten_detail(detail):
    if isinstance(detail, dict):
        for value in detail.values():
            yield from _flatten_detail(value)
    elif isinstance(detail, (list, tuple)):
        for item in detail:
            yield from _flatten_detail(item)
    elif detail is not None:
        yield str(detail)


def _error_response(code: str, message: str, status_code: int):
    return Response(
        {
            "code": code,
            "message": message,
        },
        status=status_code,
    )


def _build_subscription_payload(subscription):
    if subscription is None:
        return None

    try:
        plan = subscription.plan
    except ObjectDoesNotExist:
        plan = None

    return {
        "id": subscription.id,
        "plan_id": None if plan is None else plan.id,
        "plan_name": None if plan is None else plan.name,
        "plan_code": None if plan is None else plan.code,
        "plan_price": None if plan is None else plan.price,
        "status": subscription.status,
        "query_limit": None if plan is None else plan.query_limit,
        "queries_used": subscription.queries_used,
        "queries_remaining": (
            0 if plan is None else subscription.queries_remaining
        ),
        "usage_reset_at": subscription.usage_reset_at,
        "started_at": subscription.started_at,
        "expires_at": subscription.expires_at,
        "cancelled_at": subscription.cancelled_at,
    }


class AdminUsersPagination(PageNumberPagination):
    page_size = 5
    page_query_param = "page"
    page_size_query_param = None

    def get_paginated_response(self, data):
        return Response(
            {
                "count": self.page.paginator.count,
                "next": self.get_next_link(),
                "previous": self.get_previous_link(),
                "results": data,
            }
        )


class AdminBaseController(APIView):
    authentication_classes = [JWTAuthentication]
    permission_classes = [IsAuthenticated, IsStaffUser]

    def handle_exception(self, exc):
        if isinstance(exc, NotAuthenticated):
            return _error_response(
                code="AUTHENTICATION_REQUIRED",
                message="Authentication credentials were not provided.",
                status_code=401,
            )

        if isinstance(exc, (AuthenticationFailed, InvalidToken)):
            detail_text = " ".join(_flatten_detail(getattr(exc, "detail", exc)))
            detail_lower = detail_text.lower()

            if "expired" in detail_lower:
                return _error_response(
                    code="TOKEN_EXPIRED",
                    message="The access token has expired.",
                    status_code=401,
                )

            return _error_response(
                code="TOKEN_INVALID",
                message="The access token is invalid.",
                status_code=401,
            )

        if isinstance(exc, PermissionDenied):
            return _error_response(
                code="FORBIDDEN",
                message="Admin access is required.",
                status_code=403,
            )

        if isinstance(exc, NotFound):
            return _error_response(
                code="PAGE_NOT_FOUND",
                message="The requested page does not exist.",
                status_code=404,
            )

        return super().handle_exception(exc)


class AdminDashboardController(AdminBaseController):
    def get(self, request):
        try:
            result = get_admin_dashboard_data()
        except APIException:
            raise
        except DatabaseError:
            logger.exception("Database error while building admin dashboard.")
            return Response(
                {
                    "code": "INTERNAL_ERROR",
                    "message": "The admin dashboard could not be retrieved right now.",
                },
                status=500,
            )
        except Exception:
            logger.exception("Unexpected admin dashboard failure.")
            return Response(
                {
                    "code": "INTERNAL_ERROR",
                    "message": "The admin dashboard could not be retrieved because of an unexpected error.",
                },
                status=500,
            )

        response_data = AdminDashboardResponseSerializer(
            {
                "summary": result.summary,
                "recent_users": result.recent_users,
                "subscription_overview": result.subscription_overview,
            }
        ).data

        return Response(
            {
                "message": "Admin dashboard retrieved successfully.",
                "data": response_data,
            },
            status=200,
        )


class AdminUsersController(AdminBaseController):
    def get(self, request):
        serializer = AdminUsersFilterSerializer(data=request.query_params)

        if not serializer.is_valid():
            return Response(
                {
                    "code": "INVALID_FILTER",
                    "message": "One or more filters are invalid.",
                    "errors": serializer.errors,
                },
                status=400,
            )

        try:
            queryset = get_admin_users_queryset(filters=serializer.validated_data)
            paginator = AdminUsersPagination()
            page = paginator.paginate_queryset(queryset, request, view=self)
            response_data = AdminUserListSerializer(page, many=True).data
        except APIException:
            raise
        except DatabaseError:
            logger.exception("Database error while listing admin users.")
            return Response(
                {
                    "code": "INTERNAL_ERROR",
                    "message": "The users could not be retrieved right now.",
                },
                status=500,
            )
        except Exception:
            logger.exception("Unexpected admin users listing failure.")
            return Response(
                {
                    "code": "INTERNAL_ERROR",
                    "message": "The users could not be retrieved because of an unexpected error.",
                },
                status=500,
            )

        return paginator.get_paginated_response(response_data)


class AdminUserDetailController(AdminBaseController):
    def get(self, request, user_id: int):
        try:
            result = get_admin_user_detail(user_id=user_id)
        except AdminUserNotFoundError:
            return Response(
                {
                    "code": "USER_NOT_FOUND",
                    "message": "The user was not found.",
                },
                status=404,
            )
        except APIException:
            raise
        except DatabaseError:
            logger.exception("Database error while retrieving admin user id=%s", user_id)
            return Response(
                {
                    "code": "INTERNAL_ERROR",
                    "message": "The user could not be retrieved right now.",
                },
                status=500,
            )
        except Exception:
            logger.exception("Unexpected admin user detail failure.")
            return Response(
                {
                    "code": "INTERNAL_ERROR",
                    "message": "The user could not be retrieved because of an unexpected error.",
                },
                status=500,
            )

        response_data = AdminUserDetailResponseSerializer(
            {
                "user": result.user,
                "subscription": _build_subscription_payload(result.subscription),
                "rag_usage": {
                    "total_queries": result.total_queries,
                    "recent_queries": result.recent_queries,
                },
            }
        ).data

        return Response(
            {
                "message": "Admin user retrieved successfully.",
                "data": response_data,
            },
            status=200,
        )
