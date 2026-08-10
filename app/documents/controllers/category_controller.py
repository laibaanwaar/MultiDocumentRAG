import logging

from django.db import DatabaseError
from rest_framework.exceptions import (
    APIException,
    AuthenticationFailed,
    NotAuthenticated,
    PermissionDenied,
)
from rest_framework.permissions import IsAuthenticated
from rest_framework.response import Response
from rest_framework.views import APIView
from rest_framework_simplejwt.authentication import JWTAuthentication
from rest_framework_simplejwt.exceptions import InvalidToken

from billing.permissions import IsStaffUser
from documents.serializers import (
    DocumentCategoryCreateSerializer,
    DocumentCategoryListSerializer,
    DocumentCategoryResponseSerializer,
    DocumentCategoryUpdateSerializer,
)
from documents.services.category_service import (
    create_category,
    delete_category,
    get_active_categories,
    update_category,
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


class DocumentCategoryController(APIView):
    authentication_classes = [JWTAuthentication]

    def get_permissions(self):
        if self.request.method in {"POST", "PATCH", "DELETE"}:
            return [IsAuthenticated(), IsStaffUser()]
        return [IsAuthenticated()]

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

        return super().handle_exception(exc)

    def post(self, request):
        serializer = DocumentCategoryCreateSerializer(data=request.data)

        if not serializer.is_valid():
            return Response(
                {
                    "code": "INVALID_REQUEST",
                    "message": "The document category payload is invalid.",
                    "errors": serializer.errors,
                },
                status=400,
            )

        try:
            result = create_category(validated_data=serializer.validated_data)
        except APIException:
            raise
        except DatabaseError:
            logger.exception("Database error while creating document category.")
            return Response(
                {
                    "code": "SERVICE_UNAVAILABLE",
                    "message": (
                        "The document category could not be created right now."
                    ),
                },
                status=503,
            )
        except Exception:
            logger.exception("Unexpected document category creation failure.")
            return Response(
                {
                    "code": "INTERNAL_SERVER_ERROR",
                    "message": (
                        "The document category could not be created because of an unexpected error."
                    ),
                },
                status=500,
            )

        response_data = DocumentCategoryResponseSerializer(result.category).data

        return Response(
            {
                "message": "Document category created successfully.",
                "data": response_data,
            },
            status=201,
        )

    def get(self, request):
        try:
            categories = get_active_categories()
            serialized = DocumentCategoryListSerializer(categories, many=True).data
        except DatabaseError:
            logger.exception("Database error while listing document categories.")
            return Response(
                {
                    "code": "SERVICE_UNAVAILABLE",
                    "message": (
                        "The document categories could not be retrieved right now."
                    ),
                },
                status=503,
            )
        except Exception:
            logger.exception("Unexpected document category listing failure.")
            return Response(
                {
                    "code": "INTERNAL_SERVER_ERROR",
                    "message": (
                        "The document categories could not be retrieved because of an unexpected error."
                    ),
                },
                status=500,
            )

        return Response(
            {
                "count": len(serialized),
                "results": serialized,
            },
            status=200,
        )

    def patch(self, request, category_id):
        if not request.data:
            return Response(
                {
                    "code": "EMPTY_UPDATE",
                    "message": "The update request must include at least one field.",
                },
                status=400,
            )

        if "code" in request.data:
            return Response(
                {
                    "code": "IMMUTABLE_FIELD",
                    "message": "The code field cannot be updated.",
                },
                status=400,
            )

        serializer = DocumentCategoryUpdateSerializer(data=request.data, partial=True)

        if not serializer.is_valid():
            return Response(
                {
                    "code": "INVALID_REQUEST",
                    "message": "The document category update payload is invalid.",
                    "errors": serializer.errors,
                },
                status=400,
            )

        try:
            result = update_category(
                category_id=category_id,
                validated_data=serializer.validated_data,
            )
        except APIException:
            raise
        except DatabaseError:
            logger.exception("Database error while updating document category id=%s", category_id)
            return Response(
                {
                    "code": "SERVICE_UNAVAILABLE",
                    "message": (
                        "The document category could not be updated right now."
                    ),
                },
                status=503,
            )
        except Exception:
            logger.exception("Unexpected document category update failure.")
            return Response(
                {
                    "code": "INTERNAL_SERVER_ERROR",
                    "message": (
                        "The document category could not be updated because of an unexpected error."
                    ),
                },
                status=500,
            )

        response_data = DocumentCategoryResponseSerializer(result.category).data

        return Response(
            {
                "message": "Document category updated successfully.",
                "data": response_data,
            },
            status=200,
        )

    def delete(self, request, category_id):
        try:
            delete_category(category_id=category_id)
        except APIException:
            raise
        except DatabaseError:
            logger.exception("Database error while deleting document category id=%s", category_id)
            return Response(
                {
                    "code": "SERVICE_UNAVAILABLE",
                    "message": (
                        "The document category could not be deleted right now."
                    ),
                },
                status=503,
            )
        except Exception:
            logger.exception("Unexpected document category delete failure.")
            return Response(
                {
                    "code": "INTERNAL_SERVER_ERROR",
                    "message": (
                        "The document category could not be deleted because of an unexpected error."
                    ),
                },
                status=500,
            )

        return Response(status=204)
