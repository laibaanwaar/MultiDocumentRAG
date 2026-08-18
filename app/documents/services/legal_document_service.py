from __future__ import annotations

import hashlib
import logging
import shutil
from pathlib import Path

from django.conf import settings
from django.core.files.uploadedfile import UploadedFile
from django.db import DatabaseError, IntegrityError, transaction
from django.db.models import Q
from qdrant_client import QdrantClient
from qdrant_client.models import FieldCondition, Filter, MatchValue

from documents.exceptions import DocumentCategoryNotFoundError
from documents.legal_document_exceptions import (
    LegalDocumentAlreadyExistsError,
    LegalDocumentCategoryInactiveError,
    LegalDocumentFileTooLargeError,
    LegalDocumentInvalidFileTypeError,
    LegalDocumentInvalidPdfError,
    LegalDocumentNotFoundError,
)
from documents.models import DocumentCategory, LegalDocument
from documents.services.document_ingestion_service import (
    ingest_uploaded_document,
)


logger = logging.getLogger(__name__)

ALLOWED_PDF_CONTENT_TYPES = {
    "application/pdf",
    "application/x-pdf",
}
QDRANT_COLLECTION_NAME = "pakistan_legal_knowledge_base"
QDRANT_STORAGE_PATH = (
    Path(settings.BASE_DIR).parent / "qdrant_storage"
)


def _max_upload_size() -> int:
    value = getattr(
        settings,
        "LEGAL_DOCUMENT_UPLOAD_MAX_SIZE",
        20 * 1024 * 1024,
    )

    return int(value)


def _archive_subdir() -> str:
    value = getattr(
        settings,
        "LEGAL_DOCUMENT_ARCHIVE_SUBDIR",
        "archived",
    )

    return str(value).strip("/") or "archived"


def _compute_sha256(uploaded_file: UploadedFile) -> str:
    hasher = hashlib.sha256()
    uploaded_file.seek(0)

    for chunk in uploaded_file.chunks():
        hasher.update(chunk)

    uploaded_file.seek(0)
    return hasher.hexdigest()


def _validate_uploaded_pdf(uploaded_file: UploadedFile) -> None:
    original_name = Path(uploaded_file.name).name

    if Path(original_name).suffix.lower() != ".pdf":
        raise LegalDocumentInvalidFileTypeError()

    content_type = str(
        getattr(uploaded_file, "content_type", "")
    ).strip().lower()

    if content_type not in ALLOWED_PDF_CONTENT_TYPES:
        raise LegalDocumentInvalidFileTypeError()

    file_size = getattr(uploaded_file, "size", None)

    if file_size is not None and int(file_size) > _max_upload_size():
        raise LegalDocumentFileTooLargeError()

    uploaded_file.seek(0)
    header = uploaded_file.read(5)
    uploaded_file.seek(0)

    if header != b"%PDF-":
        raise LegalDocumentInvalidPdfError()


def _cleanup_file_field(document: LegalDocument) -> None:
    file_name = getattr(document.file, "name", "")

    if not file_name:
        return

    storage = document.file.storage

    try:
        if storage.exists(file_name):
            storage.delete(file_name)
    except Exception:
        logger.exception(
            "Failed to clean up file storage for document id=%s",
            document.pk,
        )


def _archive_document_file(document: LegalDocument) -> str | None:
    file_path = Path(document.file.path)

    if not file_path.exists():
        return None

    archived_dir = (
        Path(settings.MEDIA_ROOT)
        / _archive_subdir()
    )
    archived_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    archived_path = archived_dir / file_path.name

    if file_path.resolve() == archived_path.resolve():
        return str(
            Path(_archive_subdir()) / file_path.name
        ).replace("\\", "/")

    shutil.move(
        str(file_path),
        str(archived_path),
    )

    archived_name = str(
        Path(_archive_subdir()) / file_path.name
    ).replace("\\", "/")

    document.file.name = archived_name
    return archived_name


def _delete_qdrant_points(document_id: str) -> None:
    storage_path = Path(QDRANT_STORAGE_PATH).expanduser().resolve()

    if not storage_path.exists():
        return

    try:
        client = QdrantClient(path=str(storage_path))
    except Exception:
        logger.exception(
            "Failed to open Qdrant storage for document id=%s",
            document_id,
        )
        return

    try:
        if not client.collection_exists(
            collection_name=QDRANT_COLLECTION_NAME
        ):
            return

        client.delete(
            collection_name=QDRANT_COLLECTION_NAME,
            points_selector=Filter(
                must=[
                    FieldCondition(
                        key="metadata.document_id",
                        match=MatchValue(
                            value=document_id
                        ),
                    )
                ]
            ),
        )
    except Exception:
        logger.exception(
            "Failed to delete Qdrant points for document id=%s",
            document_id,
        )
    finally:
        client.close()


def build_legal_document_rag_metadata(
    document: LegalDocument,
) -> dict[str, object]:
    return {
        "document_id": str(document.pk),
        "document_title": document.title,
        "category_id": document.category_id,
        "category_code": document.category.code,
        "section_number": None,
        "document_type": "legal_document",
        "original_filename": document.original_filename,
        "file_size": document.file_size,
        "content_type": document.content_type,
        "checksum_sha256": document.checksum_sha256,
        "status": document.status,
        "uploaded_by_id": document.uploaded_by_id,
        "ingestion_error": document.ingestion_error,
    }


def _infer_uploaded_document_provision_type(
    document: LegalDocument,
) -> str:
    category_text = " ".join(
        [
            str(document.category.code or ""),
            str(document.category.name or ""),
            str(document.title or ""),
            str(document.original_filename or ""),
        ]
    ).lower()

    if "constitution" in category_text:
        return "article"

    return "section"


def _build_uploaded_document_ingestion_metadata(
    document: LegalDocument,
) -> dict[str, object]:
    metadata = build_legal_document_rag_metadata(document)

    metadata.update(
        {
            "document_name": document.original_filename,
            "document_short_name": Path(
                document.original_filename
            ).stem
            or document.title,
            "document_type": "legal_document",
            "provision_type": _infer_uploaded_document_provision_type(
                document
            ),
            "source_path": str(document.file.path),
            "source_file_name": document.file.name,
        }
    )

    return metadata


def _set_document_ingestion_state(
    *,
    document_id: int,
    status: str,
    ingestion_error: str | None,
) -> LegalDocument:
    with transaction.atomic():
        document = (
            LegalDocument.objects
            .select_for_update()
            .select_related(
                "category",
                "uploaded_by",
            )
            .get(pk=document_id)
        )

        document.status = status
        document.ingestion_error = ingestion_error
        document.save(
            update_fields=[
                "status",
                "ingestion_error",
                "updated_at",
            ]
        )

    return document


def _format_ingestion_error(error: Exception) -> str:
    message = f"{error.__class__.__name__}: {error}"

    return message[:4000]


def process_legal_document_ingestion(
    *,
    document_id: int,
) -> LegalDocument:
    document = _set_document_ingestion_state(
        document_id=document_id,
        status=LegalDocument.Status.PROCESSING,
        ingestion_error=None,
    )

    ingestion_metadata = _build_uploaded_document_ingestion_metadata(
        document
    )

    try:
        ingest_uploaded_document(
            file_path=document.file.path,
            document_id=str(document.pk),
            document_name=ingestion_metadata["document_name"],
            document_title=document.title,
            document_short_name=ingestion_metadata["document_short_name"],
            document_type=str(
                ingestion_metadata["document_type"]
            ),
            provision_type=str(
                ingestion_metadata["provision_type"]
            ),
            metadata=ingestion_metadata,
        )
    except Exception as error:
        logger.exception(
            "Uploaded legal document ingestion failed id=%s",
            document_id,
        )

        try:
            return _set_document_ingestion_state(
                document_id=document_id,
                status=LegalDocument.Status.FAILED,
                ingestion_error=_format_ingestion_error(error),
            )
        except DatabaseError:
            logger.exception(
                "Failed to persist failed ingestion status for legal document id=%s",
                document_id,
            )
            raise

    return _set_document_ingestion_state(
        document_id=document_id,
        status=LegalDocument.Status.READY,
        ingestion_error=None,
    )


def get_legal_documents_queryset(
    filters: dict[str, object],
):
    queryset = (
        LegalDocument.objects
        .select_related(
            "category",
            "uploaded_by",
        )
        .all()
    )

    category_code = filters.get("category")
    if category_code:
        queryset = queryset.filter(
            category__code__iexact=str(category_code).strip()
        )

    status = filters.get("status")
    if status:
        queryset = queryset.filter(
            status=str(status).strip().upper()
        )

    search = filters.get("search")
    if search:
        queryset = queryset.filter(
            Q(title__icontains=str(search).strip())
            | Q(
                original_filename__icontains=str(search).strip()
            )
        )

    return queryset.order_by(
        "-created_at",
        "-id",
    )


def get_legal_document_or_404(
    document_id: int,
) -> LegalDocument:
    try:
        return (
            LegalDocument.objects
            .select_related(
                "category",
                "uploaded_by",
            )
            .get(pk=document_id)
        )
    except LegalDocument.DoesNotExist as exception:
        raise LegalDocumentNotFoundError() from exception


def _resolve_category(
    category_id: int,
) -> DocumentCategory:
    try:
        category = DocumentCategory.objects.get(
            pk=category_id
        )
    except DocumentCategory.DoesNotExist as exception:
        raise DocumentCategoryNotFoundError() from exception

    if not category.is_active:
        raise LegalDocumentCategoryInactiveError()

    return category


def create_legal_document(
    *,
    validated_data: dict,
    uploaded_by,
) -> LegalDocument:
    title = validated_data["title"].strip()
    category = _resolve_category(
        int(validated_data["category_id"])
    )
    uploaded_file: UploadedFile = validated_data["file"]
    original_filename = Path(uploaded_file.name).name

    _validate_uploaded_pdf(uploaded_file)

    checksum_sha256 = _compute_sha256(uploaded_file)

    if LegalDocument.objects.filter(
        checksum_sha256=checksum_sha256
    ).exists():
        raise LegalDocumentAlreadyExistsError()

    document = LegalDocument(
        title=title,
        category=category,
        original_filename=original_filename,
        file_size=int(getattr(uploaded_file, "size", 0) or 0),
        content_type=str(
            getattr(uploaded_file, "content_type", "")
        ).strip(),
        checksum_sha256=checksum_sha256,
        status=LegalDocument.Status.PENDING,
        uploaded_by=uploaded_by,
        ingestion_error=None,
    )

    try:
        with transaction.atomic():
            document.file = uploaded_file
            document.save()
    except (LegalDocumentAlreadyExistsError, LegalDocumentCategoryInactiveError):
        raise
    except IntegrityError as exception:
        _cleanup_file_field(document)
        logger.warning(
            "Integrity error while creating legal document title=%s",
            title,
        )

        if "checksum_sha256" in str(exception).lower():
            raise LegalDocumentAlreadyExistsError() from exception

        raise
    except DatabaseError:
        _cleanup_file_field(document)
        logger.exception(
            "Database error while creating legal document title=%s",
            title,
        )
        raise
    except Exception:
        _cleanup_file_field(document)
        logger.exception(
            "Unexpected legal document creation failure title=%s",
            title,
        )
        raise

    return document


def update_legal_document(
    *,
    document_id: int,
    validated_data: dict,
) -> LegalDocument:
    try:
        with transaction.atomic():
            try:
                document = (
                    LegalDocument.objects
                    .select_for_update()
                    .select_related(
                        "category",
                        "uploaded_by",
                    )
                    .get(pk=document_id)
                )
            except LegalDocument.DoesNotExist as exception:
                raise LegalDocumentNotFoundError() from exception

            should_requeue = False

            if "title" in validated_data:
                normalized_title = validated_data["title"].strip()
                if normalized_title != document.title:
                    should_requeue = (
                        document.status
                        == LegalDocument.Status.READY
                    )
                    document.title = normalized_title

            if "category_id" in validated_data:
                category = _resolve_category(
                    int(validated_data["category_id"])
                )
                if category.pk != document.category_id:
                    should_requeue = (
                        should_requeue
                        or document.status
                        == LegalDocument.Status.READY
                    )
                    document.category = category

            if should_requeue:
                document.status = LegalDocument.Status.PENDING
                document.ingestion_error = None

            document.save()
    except LegalDocumentNotFoundError:
        raise
    except DatabaseError:
        logger.exception(
            "Database error while updating legal document id=%s",
            document_id,
        )
        raise

    return document


def archive_legal_document(
    *,
    document_id: int,
) -> LegalDocument:
    try:
        with transaction.atomic():
            try:
                document = (
                    LegalDocument.objects
                    .select_for_update()
                    .select_related(
                        "category",
                        "uploaded_by",
                    )
                    .get(pk=document_id)
                )
            except LegalDocument.DoesNotExist as exception:
                raise LegalDocumentNotFoundError() from exception

            if document.status != LegalDocument.Status.ARCHIVED:
                document.status = LegalDocument.Status.ARCHIVED
                document.ingestion_error = None
                document.save(
                    update_fields=[
                        "status",
                        "ingestion_error",
                        "updated_at",
                    ]
                )
    except LegalDocumentNotFoundError:
        raise
    except DatabaseError:
        logger.exception(
            "Database error while archiving legal document id=%s",
            document_id,
        )
        raise

    _delete_qdrant_points(str(document.pk))

    try:
        archived_name = _archive_document_file(document)
        if archived_name:
            LegalDocument.objects.filter(
                pk=document.pk
            ).update(
                file=archived_name,
            )
    except Exception:
        logger.exception(
            "Failed to archive file for legal document id=%s",
            document.pk,
        )

    return document
