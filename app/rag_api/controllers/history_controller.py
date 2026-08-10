import logging

from django.db import DatabaseError
from rest_framework.exceptions import AuthenticationFailed, NotAuthenticated, ParseError
from rest_framework.pagination import PageNumberPagination
from rest_framework.permissions import IsAuthenticated
from rest_framework.response import Response
from rest_framework.views import APIView
from rest_framework_simplejwt.authentication import JWTAuthentication
from rest_framework_simplejwt.exceptions import InvalidToken

from rag_api.serializers import QueryHistorySerializer
from rag_api.services import get_query_history_queryset


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


class RagHistoryPagination(PageNumberPagination):
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


class RagHistoryController(APIView):
    authentication_classes = [JWTAuthentication]
    permission_classes = [IsAuthenticated]

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

        if isinstance(exc, ParseError):
            return _error_response(
                code="INVALID_REQUEST",
                message="The request payload could not be parsed.",
                status_code=400,
            )

        return super().handle_exception(exc)

    def get(self, request):
        if not request.user.is_active:
            return Response(
                {
                    "code": "USER_INACTIVE",
                    "message": "This account is inactive.",
                },
                status=403,
            )

        try:
            queryset = get_query_history_queryset(request.user)
            paginator = RagHistoryPagination()
            page = paginator.paginate_queryset(queryset, request, view=self)
            payload = QueryHistorySerializer(page, many=True).data
        except DatabaseError:
            logger.exception("Database error while listing legal query history.")
            return Response(
                {
                    "code": "INTERNAL_ERROR",
                    "message": (
                        "The query history could not be retrieved because of an unexpected error."
                    ),
                },
                status=500,
            )
        except Exception:
            logger.exception("Unexpected legal query history listing failure.")
            return Response(
                {
                    "code": "INTERNAL_ERROR",
                    "message": (
                        "The query history could not be retrieved because of an unexpected error."
                    ),
                },
                status=500,
            )

        return paginator.get_paginated_response(payload)
