import logging

from django.db import DatabaseError
from rest_framework.exceptions import (
    APIException,
    AuthenticationFailed,
    NotAuthenticated,
    NotFound,
    PermissionDenied,
)
from rest_framework.pagination import PageNumberPagination
from rest_framework.parsers import FormParser, JSONParser, MultiPartParser
from rest_framework.permissions import IsAuthenticated
from rest_framework.response import Response
from rest_framework.views import APIView
from rest_framework_simplejwt.authentication import JWTAuthentication
from rest_framework_simplejwt.exceptions import InvalidToken

from billing.permissions import IsStaffUser
from documents.legal_document_exceptions import (
    LegalDocumentImmutableFieldError,
)
from documents.legal_document_serializers import (
    LegalDocumentCreateSerializer,
    LegalDocumentDetailSerializer,
    LegalDocumentFilterSerializer,
    LegalDocumentListSerializer,
    LegalDocumentResponseSerializer,
    LegalDocumentUpdateSerializer,
)
from documents.services.legal_document_service import (
    archive_legal_document,
    create_legal_document,
    get_legal_document_or_404,
    get_legal_documents_queryset,
    update_legal_document,
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


class LegalDocumentPagination(PageNumberPagination):
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


class LegalDocumentController(APIView):
    authentication_classes = [JWTAuthentication]
    permission_classes = [IsAuthenticated, IsStaffUser]
    parser_classes = [MultiPartParser, FormParser, JSONParser]

    protected_fields = {
        "file",
        "checksum_sha256",
        "content_type",
        "file_size",
        "original_filename",
        "status",
        "uploaded_by",
        "ingestion_error",
        "created_at",
        "updated_at",
    }

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

    def _contains_protected_field(self, data, *, allow_file: bool = False) -> bool:
        protected_fields = set(self.protected_fields)

        if allow_file:
            protected_fields.discard("file")

        return any(field in data for field in protected_fields)

    def post(self, request):
        if self._contains_protected_field(request.data, allow_file=True):
            raise LegalDocumentImmutableFieldError()

        serializer = LegalDocumentCreateSerializer(data=request.data)

        if not serializer.is_valid():
            return Response(
                {
                    "code": "INVALID_REQUEST",
                    "message": "The legal document payload is invalid.",
                    "errors": serializer.errors,
                },
                status=400,
            )

        try:
            document = create_legal_document(
                validated_data=serializer.validated_data,
                uploaded_by=request.user,
            )
        except APIException:
            raise
        except DatabaseError:
            logger.exception("Database error while creating legal document.")
            return Response(
                {
                    "code": "SERVICE_UNAVAILABLE",
                    "message": "The legal document could not be created right now.",
                },
                status=503,
            )
        except Exception:
            logger.exception("Unexpected legal document creation failure.")
            return Response(
                {
                    "code": "INTERNAL_SERVER_ERROR",
                    "message": (
                        "The legal document could not be created because of an unexpected error."
                    ),
                },
                status=500,
            )

        response_data = LegalDocumentResponseSerializer(document).data

        return Response(
            {
                "message": "Legal document created successfully.",
                "data": response_data,
            },
            status=201,
        )

    def get(self, request, document_id=None):
        if document_id is None:
            serializer = LegalDocumentFilterSerializer(
                data=request.query_params
            )

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
                queryset = get_legal_documents_queryset(
                    filters=serializer.validated_data
                )
                paginator = LegalDocumentPagination()
                page = paginator.paginate_queryset(
                    queryset,
                    request,
                    view=self,
                )
                response_data = LegalDocumentListSerializer(
                    page,
                    many=True,
                ).data
            except DatabaseError:
                logger.exception("Database error while listing legal documents.")
                return Response(
                    {
                        "code": "SERVICE_UNAVAILABLE",
                        "message": "The legal documents could not be retrieved right now.",
                    },
                    status=503,
                )
            except Exception:
                logger.exception("Unexpected legal document listing failure.")
                return Response(
                    {
                        "code": "INTERNAL_SERVER_ERROR",
                        "message": (
                            "The legal documents could not be retrieved because of an unexpected error."
                        ),
                    },
                    status=500,
                )

            return paginator.get_paginated_response(response_data)

        try:
            document = get_legal_document_or_404(document_id=document_id)
        except APIException:
            raise
        except DatabaseError:
            logger.exception(
                "Database error while retrieving legal document id=%s",
                document_id,
            )
            return Response(
                {
                    "code": "SERVICE_UNAVAILABLE",
                    "message": "The legal document could not be retrieved right now.",
                },
                status=503,
            )
        except Exception:
            logger.exception(
                "Unexpected legal document retrieval failure id=%s",
                document_id,
            )
            return Response(
                {
                    "code": "INTERNAL_SERVER_ERROR",
                    "message": (
                        "The legal document could not be retrieved because of an unexpected error."
                    ),
                },
                status=500,
            )

        response_data = LegalDocumentDetailSerializer(document).data

        return Response(
            {
                "data": response_data,
            },
            status=200,
        )

    def patch(self, request, document_id):
        if not request.data:
            return Response(
                {
                    "code": "EMPTY_UPDATE",
                    "message": "The update request must include at least one field.",
                },
                status=400,
            )

        if self._contains_protected_field(request.data):
            raise LegalDocumentImmutableFieldError()

        serializer = LegalDocumentUpdateSerializer(
            data=request.data,
            partial=True,
        )

        if not serializer.is_valid():
            return Response(
                {
                    "code": "INVALID_REQUEST",
                    "message": "The legal document update payload is invalid.",
                    "errors": serializer.errors,
                },
                status=400,
            )

        try:
            document = update_legal_document(
                document_id=document_id,
                validated_data=serializer.validated_data,
            )
        except APIException:
            raise
        except DatabaseError:
            logger.exception(
                "Database error while updating legal document id=%s",
                document_id,
            )
            return Response(
                {
                    "code": "SERVICE_UNAVAILABLE",
                    "message": "The legal document could not be updated right now.",
                },
                status=503,
            )
        except Exception:
            logger.exception(
                "Unexpected legal document update failure id=%s",
                document_id,
            )
            return Response(
                {
                    "code": "INTERNAL_SERVER_ERROR",
                    "message": (
                        "The legal document could not be updated because of an unexpected error."
                    ),
                },
                status=500,
            )

        response_data = LegalDocumentResponseSerializer(document).data

        return Response(
            {
                "message": "Legal document updated successfully.",
                "data": response_data,
            },
            status=200,
        )

    def delete(self, request, document_id):
        try:
            archive_legal_document(document_id=document_id)
        except APIException:
            raise
        except DatabaseError:
            logger.exception(
                "Database error while archiving legal document id=%s",
                document_id,
            )
            return Response(
                {
                    "code": "SERVICE_UNAVAILABLE",
                    "message": "The legal document could not be archived right now.",
                },
                status=503,
            )
        except Exception:
            logger.exception(
                "Unexpected legal document archive failure id=%s",
                document_id,
            )
            return Response(
                {
                    "code": "INTERNAL_SERVER_ERROR",
                    "message": (
                        "The legal document could not be archived because of an unexpected error."
                    ),
                },
                status=500,
            )

        return Response(status=204)
