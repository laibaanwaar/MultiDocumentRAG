import logging

from django.db import DatabaseError
from rest_framework.exceptions import (
    APIException,
    AuthenticationFailed,
    NotAuthenticated,
    NotFound,
    PermissionDenied,
)
from rest_framework.permissions import IsAuthenticated
from rest_framework.response import Response
from rest_framework.pagination import PageNumberPagination
from rest_framework.throttling import ScopedRateThrottle
from rest_framework.views import APIView
from rest_framework_simplejwt.authentication import JWTAuthentication
from rest_framework_simplejwt.exceptions import InvalidToken

from billing.admin_serializers import (
    AdminPlanCreateSerializer,
    AdminPlanResponseSerializer,
    AdminPlanUpdateSerializer,
    AdminSubscriptionFilterSerializer,
    AdminSubscriptionListSerializer,
)
from billing.exceptions import EmptyUpdateError, ImmutableFieldError
from billing.permissions import IsStaffUser
from billing.services.admin_billing_service import (
    create_plan,
    get_admin_subscriptions,
    update_plan,
)


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


class AdminBillingBaseController(APIView):
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


class AdminBillingPlansController(AdminBillingBaseController):
    throttle_classes = [ScopedRateThrottle]
    throttle_scope = "billing_admin_plans"

    def post(self, request):
        serializer = AdminPlanCreateSerializer(data=request.data)

        if not serializer.is_valid():
            return Response(
                {
                    "code": "INVALID_REQUEST",
                    "message": "The plan payload is invalid.",
                    "errors": serializer.errors,
                },
                status=400,
            )

        try:
            plan = create_plan(validated_data=serializer.validated_data)
        except APIException:
            raise
        except DatabaseError:
            logger.exception("Database error while creating billing plan.")
            return Response(
                {
                    "code": "SERVICE_UNAVAILABLE",
                    "message": "The plan could not be created right now.",
                },
                status=503,
            )
        except Exception:
            logger.exception("Unexpected billing plan creation failure.")
            return Response(
                {
                    "code": "INTERNAL_SERVER_ERROR",
                    "message": (
                        "The plan could not be created because of an unexpected error."
                    ),
                },
                status=500,
            )

        response_data = AdminPlanResponseSerializer(plan).data

        return Response(
            {
                "message": "Plan created successfully.",
                "data": response_data,
            },
            status=201,
        )

    def patch(self, request, plan_id):
        if not request.data:
            raise EmptyUpdateError()

        if "code" in request.data:
            raise ImmutableFieldError()

        serializer = AdminPlanUpdateSerializer(data=request.data, partial=True)

        if not serializer.is_valid():
            return Response(
                {
                    "code": "INVALID_REQUEST",
                    "message": "The plan update payload is invalid.",
                    "errors": serializer.errors,
                },
                status=400,
            )

        try:
            plan = update_plan(
                plan_id=plan_id,
                validated_data=serializer.validated_data,
            )
        except APIException:
            raise
        except DatabaseError:
            logger.exception("Database error while updating billing plan id=%s", plan_id)
            return Response(
                {
                    "code": "SERVICE_UNAVAILABLE",
                    "message": "The plan could not be updated right now.",
                },
                status=503,
            )
        except Exception:
            logger.exception("Unexpected billing plan update failure.")
            return Response(
                {
                    "code": "INTERNAL_SERVER_ERROR",
                    "message": (
                        "The plan could not be updated because of an unexpected error."
                    ),
                },
                status=500,
            )

        response_data = AdminPlanResponseSerializer(plan).data

        return Response(
            {
                "message": "Plan updated successfully.",
                "data": response_data,
            },
            status=200,
        )


class AdminSubscriptionPagination(PageNumberPagination):
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


class AdminSubscriptionsController(AdminBillingBaseController):
    throttle_classes = [ScopedRateThrottle]
    throttle_scope = "billing_admin_subscriptions"

    def get(self, request):
        serializer = AdminSubscriptionFilterSerializer(data=request.query_params)

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
            queryset = get_admin_subscriptions(filters=serializer.validated_data)
            paginator = AdminSubscriptionPagination()
            page = paginator.paginate_queryset(queryset, request, view=self)
            response_data = AdminSubscriptionListSerializer(page, many=True).data
        except DatabaseError:
            logger.exception("Database error while listing admin subscriptions.")
            return Response(
                {
                    "code": "SERVICE_UNAVAILABLE",
                    "message": "The subscriptions could not be retrieved right now.",
                },
                status=503,
            )
        except Exception:
            logger.exception("Unexpected admin subscription listing failure.")
            return Response(
                {
                    "code": "INTERNAL_SERVER_ERROR",
                    "message": (
                        "The subscriptions could not be retrieved because of an unexpected error."
                    ),
                },
                status=500,
            )

        return paginator.get_paginated_response(response_data)
