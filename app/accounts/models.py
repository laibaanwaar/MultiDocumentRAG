from django.conf import settings
from django.db import models
from django.db.models import Q
from django.utils import timezone


class EmailOTP(models.Model):
    class Purpose(models.TextChoices):
        EMAIL_VERIFICATION = (
            "email_verification",
            "Email verification",
        )

    user = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.CASCADE,
        related_name="email_otps",
    )

    purpose = models.CharField(
        max_length=32,
        choices=Purpose.choices,
        default=Purpose.EMAIL_VERIFICATION,
    )

    # Never store the plain OTP.
    code_hash = models.CharField(max_length=255)

    expires_at = models.DateTimeField()

    attempts = models.PositiveSmallIntegerField(default=0)

    is_used = models.BooleanField(default=False)

    created_at = models.DateTimeField(auto_now_add=True)

    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ["-created_at"]

        constraints = [
            models.UniqueConstraint(
                fields=["user", "purpose"],
                condition=Q(is_used=False),
                name="accounts_one_active_email_otp",
            )
        ]

        indexes = [
            models.Index(
                fields=[
                    "user",
                    "purpose",
                    "is_used",
                ]
            ),
            models.Index(fields=["expires_at"]),
        ]

    @property
    def is_expired(self) -> bool:
        return timezone.now() >= self.expires_at

    def __str__(self) -> str:
        return (
            f"Email OTP for user={self.user_id}, "
            f"purpose={self.purpose}"
        )