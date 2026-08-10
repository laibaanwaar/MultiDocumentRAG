import re
from pathlib import Path
from uuid import uuid4

from django.conf import settings
from django.db import models
from django.db.models import Q
from django.db.models.functions import Lower


class DocumentCategory(models.Model):
    name = models.CharField(max_length=120)
    code = models.CharField(max_length=50)
    description = models.TextField(blank=True, default="")
    is_active = models.BooleanField(default=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ["name", "id"]
        constraints = [
            models.UniqueConstraint(
                Lower("name"),
                name="document_category_name_ci_unique",
            ),
            models.UniqueConstraint(
                Lower("code"),
                name="document_category_code_ci_unique",
            ),
            models.CheckConstraint(
                condition=~Q(name=""),
                name="document_category_name_not_blank",
            ),
            models.CheckConstraint(
                condition=~Q(code=""),
                name="document_category_code_not_blank",
            ),
        ]

    def __str__(self) -> str:
        return f"{self.name} ({self.code})"


def _sanitize_filename_component(value: str) -> str:
    stem = Path(value).stem.strip()
    stem = re.sub(r"[\s_]+", "-", stem)
    stem = re.sub(r"[^A-Za-z0-9.-]+", "-", stem)
    stem = re.sub(r"-+", "-", stem).strip("-.")
    return stem or "document"


def legal_document_upload_to(instance, filename: str) -> str:
    original_name = Path(filename).name
    safe_stem = _sanitize_filename_component(original_name)
    suffix = Path(original_name).suffix.lower()

    if suffix != ".pdf":
        suffix = ".pdf"

    return f"{uuid4().hex}_{safe_stem}{suffix}"


class LegalDocument(models.Model):
    class Status(models.TextChoices):
        PENDING = "PENDING", "Pending"
        PROCESSING = "PROCESSING", "Processing"
        READY = "READY", "Ready"
        FAILED = "FAILED", "Failed"
        ARCHIVED = "ARCHIVED", "Archived"

    title = models.CharField(max_length=255)
    category = models.ForeignKey(
        DocumentCategory,
        on_delete=models.PROTECT,
        related_name="documents",
    )
    file = models.FileField(
        upload_to=legal_document_upload_to,
        max_length=255,
    )
    original_filename = models.CharField(max_length=255)
    file_size = models.PositiveBigIntegerField()
    content_type = models.CharField(max_length=100)
    checksum_sha256 = models.CharField(max_length=64, unique=True)
    status = models.CharField(
        max_length=20,
        choices=Status.choices,
        default=Status.PENDING,
    )
    uploaded_by = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.PROTECT,
        related_name="uploaded_legal_documents",
    )
    ingestion_error = models.TextField(
        blank=True,
        null=True,
        default=None,
    )
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ["-created_at", "-id"]
        indexes = [
            models.Index(fields=["status", "id"]),
            models.Index(fields=["category", "id"]),
        ]

    def __str__(self) -> str:
        return f"{self.title} ({self.original_filename})"
