import logging

from django.db import DatabaseError
from rest_framework import status
from rest_framework.exceptions import APIException, AuthenticationFailed, NotAuthenticated
from rest_framework.permissions import AllowAny, IsAuthenticated
from rest_framework.response import Response
from rest_framework.throttling import ScopedRateThrottle
from rest_framework.views import APIView
from rest_framework_simplejwt.authentication import JWTAuthentication
from rest_framework_simplejwt.exceptions import InvalidToken

from billing.serializers import PlanSerializer, SubscriptionSerializer
from billing.services.subscription_service import (
    get_active_plans,
    get_subscription_for_user,
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


class PlansController(APIView):
    permission_classes = [AllowAny]
    authentication_classes = []
    throttle_classes = [ScopedRateThrottle]
    throttle_scope = "billing_plans"

    def get(self, request):
        try:
            plans = get_active_plans()
            serializer = PlanSerializer(plans, many=True)
        except DatabaseError:
            logger.error("Database error while listing billing plans.")
            return Response(
                {
                    "code": "SERVICE_UNAVAILABLE",
                    "message": (
                        "The billing plans could not be retrieved right now."
                    ),
                },
                status=status.HTTP_503_SERVICE_UNAVAILABLE,
            )
        except Exception:
            logger.error("Unexpected billing plan listing failure.")
            return Response(
                {
                    "code": "INTERNAL_SERVER_ERROR",
                    "message": (
                        "The billing plans could not be retrieved because of "
                        "an unexpected error."
                    ),
                },
                status=status.HTTP_500_INTERNAL_SERVER_ERROR,
            )

        return Response(
            {
                "message": "Billing plans retrieved successfully.",
                "data": serializer.data,
            },
            status=status.HTTP_200_OK,
        )


class SubscriptionController(APIView):
    authentication_classes = [JWTAuthentication]
    permission_classes = [IsAuthenticated]
    throttle_classes = [ScopedRateThrottle]
    throttle_scope = "billing_subscription"

    def handle_exception(self, exc):
        if isinstance(exc, NotAuthenticated):
            return _error_response(
                code="AUTHENTICATION_REQUIRED",
                message="Authentication credentials were not provided.",
                status_code=status.HTTP_401_UNAUTHORIZED,
            )

        if isinstance(exc, (AuthenticationFailed, InvalidToken)):
            detail_text = " ".join(_flatten_detail(getattr(exc, "detail", exc)))
            detail_lower = detail_text.lower()

            if "expired" in detail_lower:
                return _error_response(
                    code="TOKEN_EXPIRED",
                    message="The access token has expired.",
                    status_code=status.HTTP_401_UNAUTHORIZED,
                )

            return _error_response(
                code="TOKEN_INVALID",
                message="The access token is invalid.",
                status_code=status.HTTP_401_UNAUTHORIZED,
            )

        return super().handle_exception(exc)

    def get(self, request):
        if not request.user.is_active:
            return Response(
                {
                    "code": "ACCOUNT_INACTIVE",
                    "message": "This account is inactive.",
                },
                status=status.HTTP_403_FORBIDDEN,
            )

        try:
            result = get_subscription_for_user(request.user)
            serializer = SubscriptionSerializer(result.subscription)
            payload = dict(serializer.data)
            payload["queries_remaining"] = result.queries_remaining
        except APIException:
            raise
        except DatabaseError:
            logger.error(
                "Database error while retrieving subscription for user_id=%s",
                getattr(request.user, "id", None),
            )
            return Response(
                {
                    "code": "SERVICE_UNAVAILABLE",
                    "message": (
                        "The subscription could not be retrieved right now."
                    ),
                },
                status=status.HTTP_503_SERVICE_UNAVAILABLE,
            )
        except Exception:
            logger.error("Unexpected subscription retrieval failure.")
            return Response(
                {
                    "code": "INTERNAL_SERVER_ERROR",
                    "message": (
                        "The subscription could not be retrieved because of "
                        "an unexpected error."
                    ),
                },
                status=status.HTTP_500_INTERNAL_SERVER_ERROR,
            )

        return Response(
            {
                "message": "Subscription retrieved successfully.",
                "data": payload,
            },
            status=status.HTTP_200_OK,
        )
