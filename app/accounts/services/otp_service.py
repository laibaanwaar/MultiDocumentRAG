import secrets
from datetime import timedelta

from django.conf import settings
from django.contrib.auth.hashers import check_password, make_password
from django.utils import timezone

from accounts.models import EmailOTP


OTP_EXPIRY_MINUTES = 30
OTP_MAX_ATTEMPTS = 5
OTP_RESEND_COOLDOWN_SECONDS = 60
OTP_LOOKBACK_LIMIT = 10


def generate_six_digit_otp() -> str:
    """
    Generate a six-digit code from 100000 to 999999.
    """

    return str(
        secrets.randbelow(900_000) + 100_000
    )


def _email_verification_purpose() -> str:
    return EmailOTP.Purpose.EMAIL_VERIFICATION


def _verification_code_expiry_minutes() -> int:
    return getattr(
        settings,
        "EMAIL_OTP_EXPIRY_MINUTES",
        OTP_EXPIRY_MINUTES,
    )


def _resend_cooldown_seconds() -> int:
    return getattr(
        settings,
        "EMAIL_OTP_RESEND_COOLDOWN_SECONDS",
        OTP_RESEND_COOLDOWN_SECONDS,
    )


def get_resend_cooldown_seconds() -> int:
    return _resend_cooldown_seconds()


def create_email_verification_otp(user):
    """
    Invalidate previous unused OTPs and create a new OTP.
    Returns the plain OTP temporarily for email delivery.
    """

    now = timezone.now()

    EmailOTP.objects.filter(
        user=user,
        purpose=_email_verification_purpose(),
        is_used=False,
    ).update(
        is_used=True,
        updated_at=now,
    )

    otp_code = generate_six_digit_otp()

    otp_record = EmailOTP.objects.create(
        user=user,
        purpose=_email_verification_purpose(),
        code_hash=make_password(otp_code),
        expires_at=now
        + timedelta(minutes=_verification_code_expiry_minutes()),
    )

    return otp_code, otp_record


def get_latest_email_otp(user):
    return (
        EmailOTP.objects.filter(
            user=user,
            purpose=_email_verification_purpose(),
        )
        .order_by("-created_at")
        .first()
    )


def get_active_email_otp(user):
    return (
        EmailOTP.objects.filter(
            user=user,
            purpose=_email_verification_purpose(),
            is_used=False,
        )
        .order_by("-created_at")
        .first()
    )


def get_email_otp_history(user):
    return list(
        EmailOTP.objects.filter(
            user=user,
            purpose=_email_verification_purpose(),
        )
        .order_by("-created_at")[:OTP_LOOKBACK_LIMIT]
    )


def is_otp_within_resend_cooldown(otp_record) -> bool:
    cooldown = timedelta(
        seconds=_resend_cooldown_seconds()
    )
    return (
        timezone.now() - otp_record.created_at
    ) < cooldown


def mark_otp_used(otp_record) -> None:
    otp_record.is_used = True
    otp_record.save(update_fields=["is_used", "updated_at"])


def increment_otp_attempts(otp_record) -> None:
    otp_record.attempts += 1
    otp_record.save(
        update_fields=["attempts", "updated_at"]
    )


def check_otp_code(otp_record, otp_code: str) -> bool:
    return check_password(
        otp_code,
        otp_record.code_hash,
    )
