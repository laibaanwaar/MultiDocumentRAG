from django.conf import settings
from django.db import models

from documents.models import DocumentCategory


class QueryHistory(models.Model):
    user = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.CASCADE,
        related_name="legal_queries",
    )
    question = models.TextField()
    answer = models.TextField()
    category = models.ForeignKey(
        DocumentCategory,
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        related_name="query_histories",
    )
    sources = models.JSONField(default=list)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ["-created_at", "-id"]

    def __str__(self) -> str:
        return f"QueryHistory(user_id={self.user_id}, id={self.id})"
