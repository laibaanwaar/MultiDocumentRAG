from django.conf import settings
from django.db import models
from django.db.models import PROTECT
from django.utils import timezone


class Plan(models.Model):
    class BillingPeriod(models.TextChoices):
        MONTHLY = "monthly", "Monthly"
        YEARLY = "yearly", "Yearly"

    name = models.CharField(max_length=120)
    code = models.CharField(max_length=50, unique=True)
    price = models.DecimalField(max_digits=10, decimal_places=2)
    billing_period = models.CharField(
        max_length=20,
        choices=BillingPeriod.choices,
    )
    document_limit = models.PositiveIntegerField()
    query_limit = models.PositiveIntegerField()
    is_active = models.BooleanField(default=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ["id"]

    def __str__(self) -> str:
        return f"{self.name} ({self.code})"


class Subscription(models.Model):
    class Status(models.TextChoices):
        ACTIVE = "active", "Active"
        CANCELED = "canceled", "Canceled"
        EXPIRED = "expired", "Expired"

    user = models.OneToOneField(
        settings.AUTH_USER_MODEL,
        on_delete=models.CASCADE,
        related_name="subscription",
    )
    plan = models.ForeignKey(
        Plan,
        on_delete=PROTECT,
        related_name="subscriptions",
    )
    status = models.CharField(
        max_length=20,
        choices=Status.choices,
        default=Status.ACTIVE,
    )
    queries_used = models.PositiveIntegerField(default=0)
    usage_reset_at = models.DateTimeField(null=True, blank=True)
    started_at = models.DateTimeField(default=timezone.now)
    expires_at = models.DateTimeField(null=True, blank=True)
    cancelled_at = models.DateTimeField(null=True, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ["-created_at"]

    @property
    def queries_remaining(self) -> int:
        remaining = self.plan.query_limit - self.queries_used
        return max(0, remaining)

    @property
    def is_expired(self) -> bool:
        if self.status == self.Status.EXPIRED:
            return True

        if self.expires_at is None:
            return False

        return timezone.now() >= self.expires_at

    def __str__(self) -> str:
        return f"Subscription for user={self.user_id}"
