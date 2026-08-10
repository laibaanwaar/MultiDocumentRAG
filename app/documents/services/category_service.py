import logging
from dataclasses import dataclass

from django.db import DatabaseError, IntegrityError, transaction

from documents.exceptions import (
    DocumentCategoryCodeExistsError,
    DocumentCategoryInUseError,
    DocumentCategoryNameExistsError,
    DocumentCategoryNotFoundError,
)
from documents.models import DocumentCategory


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class CategoryCreateResult:
    category: DocumentCategory


@dataclass(frozen=True)
class CategoryUpdateResult:
    category: DocumentCategory


def _resolve_integrity_error(exception: IntegrityError):
    cause = getattr(exception, "__cause__", None)
    diag = getattr(cause, "diag", None)
    constraint_name = getattr(diag, "constraint_name", None)
    message = str(exception).lower()

    if constraint_name == "document_category_code_ci_unique":
        return DocumentCategoryCodeExistsError()

    if constraint_name == "document_category_name_ci_unique":
        return DocumentCategoryNameExistsError()

    if "document_category_code_ci_unique" in message or "code" in message:
        return DocumentCategoryCodeExistsError()

    if "document_category_name_ci_unique" in message or "name" in message:
        return DocumentCategoryNameExistsError()

    return None


def create_category(*, validated_data: dict) -> CategoryCreateResult:
    name = validated_data["name"].strip()
    code = validated_data["code"].strip().upper()
    description = validated_data.get("description", "") or ""
    is_active = validated_data.get("is_active", True)

    try:
        with transaction.atomic():
            if DocumentCategory.objects.filter(name__iexact=name).exists():
                raise DocumentCategoryNameExistsError()

            if DocumentCategory.objects.filter(code__iexact=code).exists():
                raise DocumentCategoryCodeExistsError()

            category = DocumentCategory.objects.create(
                name=name,
                code=code,
                description=description,
                is_active=is_active,
            )
    except (DocumentCategoryNameExistsError, DocumentCategoryCodeExistsError):
        raise
    except IntegrityError as exception:
        logger.warning(
            "Integrity error while creating document category code=%s name=%s",
            code,
            name,
        )
        resolved_error = _resolve_integrity_error(exception)
        if resolved_error is not None:
            raise resolved_error from exception
        raise
    except DatabaseError:
        logger.exception(
            "Database error while creating document category code=%s name=%s",
            code,
            name,
        )
        raise

    return CategoryCreateResult(category=category)


def get_category_or_404(category_id: int) -> DocumentCategory:
    try:
        return DocumentCategory.objects.get(pk=category_id)
    except DocumentCategory.DoesNotExist as exception:
        raise DocumentCategoryNotFoundError() from exception


def update_category(*, category_id: int, validated_data: dict) -> CategoryUpdateResult:
    try:
        with transaction.atomic():
            try:
                category = DocumentCategory.objects.select_for_update().get(
                    pk=category_id
                )
            except DocumentCategory.DoesNotExist as exception:
                raise DocumentCategoryNotFoundError() from exception

            if "name" in validated_data:
                name = validated_data["name"].strip()
                duplicate_name = (
                    DocumentCategory.objects.filter(name__iexact=name)
                    .exclude(pk=category_id)
                    .exists()
                )
                if duplicate_name:
                    raise DocumentCategoryNameExistsError()
                category.name = name

            if "description" in validated_data:
                category.description = validated_data["description"] or ""

            if "is_active" in validated_data:
                category.is_active = validated_data["is_active"]

            try:
                category.save()
            except IntegrityError as exception:
                logger.warning(
                    "Integrity error while updating document category id=%s",
                    category_id,
                )
                resolved_error = _resolve_integrity_error(exception)
                if resolved_error is not None:
                    raise resolved_error from exception
                raise
    except (DocumentCategoryNameExistsError, DocumentCategoryNotFoundError):
        raise
    except DatabaseError:
        logger.exception(
            "Database error while updating document category id=%s",
            category_id,
        )
        raise

    return CategoryUpdateResult(category=category)


def _category_has_related_documents(category: DocumentCategory) -> bool:
    related_manager = getattr(category, "documents", None)
    if related_manager is None:
        return False

    try:
        return related_manager.exists()
    except Exception:
        return False


def delete_category(*, category_id: int) -> None:
    try:
        with transaction.atomic():
            try:
                category = DocumentCategory.objects.select_for_update().get(
                    pk=category_id
                )
            except DocumentCategory.DoesNotExist as exception:
                raise DocumentCategoryNotFoundError() from exception

            if _category_has_related_documents(category):
                raise DocumentCategoryInUseError()

            category.delete()
    except (DocumentCategoryInUseError, DocumentCategoryNotFoundError):
        raise
    except DatabaseError:
        logger.exception(
            "Database error while deleting document category id=%s",
            category_id,
        )
        raise


def get_active_categories():
    return DocumentCategory.objects.filter(is_active=True).order_by("name", "id")
