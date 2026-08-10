import logging

from django.db import DatabaseError
from rest_framework.exceptions import (
    AuthenticationFailed,
    APIException,
    NotAuthenticated,
    ParseError,
)
from rest_framework.permissions import IsAuthenticated
from rest_framework.response import Response
from rest_framework.views import APIView
from rest_framework_simplejwt.authentication import JWTAuthentication
from rest_framework_simplejwt.exceptions import InvalidToken

from rag_api.exceptions import (
    RagCategoryInactiveError,
    RagCategoryNotFoundError,
    RagInternalError,
    RagInvalidQuestionError,
    RagInvalidRequestError,
    RagQueryLimitExceededError,
    RagServiceUnavailableError,
    RagSubscriptionRequiredError,
    RagUserInactiveError,
)
from rag_api.serializers import (
    RagQueryRequestSerializer,
    QueryHistorySerializer,
)
from rag_api.services import run_query


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


def _question_error_only(errors: dict) -> bool:
    if set(errors.keys()) != {"question"}:
        return False
    return True


class RagQueryController(APIView):
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

    def post(self, request):
        if not request.user.is_active:
            return Response(
                {
                    "code": "USER_INACTIVE",
                    "message": "This account is inactive.",
                },
                status=403,
            )

        serializer = RagQueryRequestSerializer(data=request.data)

        if not serializer.is_valid():
            errors = serializer.errors

            if _question_error_only(errors):
                return Response(
                    {
                        "code": "INVALID_QUESTION",
                        "message": "The question is invalid.",
                        "errors": errors,
                    },
                    status=400,
                )

            return Response(
                {
                    "code": "INVALID_REQUEST",
                    "message": "The request payload is invalid.",
                    "errors": errors,
                },
                status=400,
            )

        try:
            result = run_query(
                user=request.user,
                question=serializer.validated_data["question"],
                category_id=serializer.validated_data.get("category_id"),
            )
        except RagSubscriptionRequiredError:
            return Response(
                {
                    "code": "SUBSCRIPTION_REQUIRED",
                    "message": "An active subscription is required.",
                },
                status=403,
            )
        except RagQueryLimitExceededError:
            return Response(
                {
                    "code": "QUERY_LIMIT_EXCEEDED",
                    "message": "Your plan query limit has been reached.",
                },
                status=429,
            )
        except RagServiceUnavailableError:
            return Response(
                {
                    "code": "RAG_SERVICE_UNAVAILABLE",
                    "message": (
                        "The legal question service is unavailable right now."
                    ),
                },
                status=503,
            )
        except RagCategoryNotFoundError:
            return Response(
                {
                    "code": "CATEGORY_NOT_FOUND",
                    "message": "The category was not found.",
                },
                status=404,
            )
        except RagCategoryInactiveError:
            return Response(
                {
                    "code": "CATEGORY_INACTIVE",
                    "message": "The category is inactive.",
                },
                status=400,
            )
        except RagInvalidRequestError as error:
            return Response(
                {
                    "code": "INVALID_REQUEST",
                    "message": error.detail.get("message", "The request payload is invalid."),
                },
                status=400,
            )
        except RagInvalidQuestionError as error:
            return Response(
                {
                    "code": "INVALID_QUESTION",
                    "message": error.detail.get("message", "The question is invalid."),
                },
                status=400,
            )
        except RagInternalError:
            return Response(
                {
                    "code": "INTERNAL_ERROR",
                    "message": (
                        "The legal question could not be processed because of an unexpected error."
                    ),
                },
                status=500,
            )
        except DatabaseError:
            logger.exception("Database error while processing legal query.")
            return Response(
                {
                    "code": "INTERNAL_ERROR",
                    "message": (
                        "The legal question could not be processed because of an unexpected error."
                    ),
                },
                status=500,
            )
        except APIException:
            raise
        except Exception:
            logger.exception("Unexpected legal query failure.")
            return Response(
                {
                    "code": "INTERNAL_ERROR",
                    "message": (
                        "The legal question could not be processed because of an unexpected error."
                    ),
                },
                status=500,
            )

        history = result["history"]
        rag_result = result["rag_result"]

        response_payload = {
            "history_id": None if history is None else history.id,
            "question": (
                serializer.validated_data["question"]
                if history is None
                else history.question
            ),
            "answer": rag_result["answer"],
            "sources": rag_result.get("sources", []),
            "category": (
                None
                if history is None or history.category is None
                else {
                    "id": history.category.id,
                    "name": history.category.name,
                    "code": history.category.code,
                }
            ),
            "queries_remaining": result["queries_remaining"],
        }

        return Response(
            {
                "message": (
                    "Legal question answered successfully."
                    if history is not None
                    else "No relevant context was found."
                ),
                "data": response_payload,
            },
            status=200,
        )
