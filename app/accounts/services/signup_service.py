import hashlib
import logging
from dataclasses import dataclass
from datetime import timedelta
from typing import Any

from django.contrib.auth import get_user_model
from django.db import connection, transaction
from django.utils import timezone

from accounts.models import EmailOTP
from accounts.exceptions import (
    EmailAlreadyVerifiedError,
    EmailVerificationRequiredError,
    OTPDeliveryError,
    OTPResendCooldownError,
    OTPTooManyAttemptsError,
    OTPExpiredError,
    OTPInvalidError,
    OTPReusedError,
    SignupConflictError,
)
from accounts.services.email_service import (
    send_signup_otp,
)
from accounts.services.otp_service import (
    OTP_MAX_ATTEMPTS,
    OTP_LOOKBACK_LIMIT,
    check_otp_code,
    create_email_verification_otp,
    increment_otp_attempts,
    get_resend_cooldown_seconds,
    is_otp_within_resend_cooldown,
    mark_otp_used,
)


logger = logging.getLogger(__name__)
User = get_user_model()


@dataclass(frozen=True)
class SignupResult:
    user: Any
    otp_expires_at: Any


@dataclass(frozen=True)
class ResendOTPResult:
    user: Any
    otp_expires_at: Any


@dataclass(frozen=True)
class VerifyEmailResult:
    user: Any


def _postgres_transaction_lock(value: str) -> None:
    """
    Serialize concurrent signup attempts for the same
    normalized username or email.

    The project uses PostgreSQL. For another database,
    this function safely becomes a no-op.
    """

    if connection.vendor != "postgresql":
        return

    digest = hashlib.blake2b(
        value.encode("utf-8"),
        digest_size=8,
    ).digest()

    lock_id = int.from_bytes(
        digest,
        byteorder="big",
        signed=True,
    )

    with connection.cursor() as cursor:
        cursor.execute(
            "SELECT pg_advisory_xact_lock(%s)",
            [lock_id],
        )


def _create_pending_user(validated_data) -> tuple:
    username = validated_data["username"]
    email = validated_data["email"]

    # Always obtain locks in the same order to prevent
    # deadlocks between simultaneous requests.
    _postgres_transaction_lock(
        f"signup-email:{email}"
    )
    _postgres_transaction_lock(
        f"signup-username:{username}"
    )

    if User.objects.filter(
        email__iexact=email,
        is_active=True,
    ).exists():
        raise SignupConflictError(
            code="EMAIL_ALREADY_REGISTERED",
            message=(
                "An active account with this email "
                "already exists."
            ),
            next_action="login",
        )

    if User.objects.filter(
        email__iexact=email,
        is_active=False,
    ).exists():
        raise SignupConflictError(
            code="EMAIL_VERIFICATION_PENDING",
            message=(
                "An account with this email already exists "
                "but has not been verified."
            ),
            next_action="resend_otp",
        )

    if User.objects.filter(
        username__iexact=username
    ).exists():
        raise SignupConflictError(
            code="USERNAME_ALREADY_REGISTERED",
            message=(
                "This username is already in use."
            ),
        )

    password = validated_data["password"]

    user = User.objects.create_user(
        username=username,
        email=email,
        password=password,
        first_name=validated_data.get(
            "first_name",
            "",
        ),
        last_name=validated_data.get(
            "last_name",
            "",
        ),

        # Public signup can only create normal users.
        is_active=False,
        is_staff=False,
        is_superuser=False,
    )

    otp_code, otp_record = (
        create_email_verification_otp(user)
    )

    return user, otp_code, otp_record


def _get_user_for_email_verification(email: str, *, for_update: bool = False):
    if User.objects.filter(
        email__iexact=email,
        is_active=True,
    ).exists():
        raise EmailAlreadyVerifiedError()

    users = User.objects.filter(
        email__iexact=email,
        is_active=False,
    ).order_by("id")
    if for_update:
        users = users.select_for_update()

    user = users.first()

    if user is None:
        raise EmailVerificationRequiredError(
            code="EMAIL_NOT_FOUND",
            message=(
                "No account was found for this email address."
            ),
        )

    return user


def resend_email_verification_otp(*, email: str) -> ResendOTPResult:
    with transaction.atomic():
        user = _get_user_for_email_verification(
            email,
            for_update=True,
        )

        active_otp = (
            EmailOTP.objects.select_for_update()
            .filter(
                user=user,
                purpose=EmailOTP.Purpose.EMAIL_VERIFICATION,
                is_used=False,
            )
            .order_by("-created_at")
            .first()
        )
        if active_otp is not None and (
            is_otp_within_resend_cooldown(active_otp)
        ):
            retry_after = max(
                1,
                int(
                    (
                        active_otp.created_at
                        + timedelta(
                            seconds=get_resend_cooldown_seconds()
                        )
                    )
                    - timezone.now()
                ).total_seconds(),
            )
            raise OTPResendCooldownError(
                retry_after_seconds=retry_after,
            )

        otp_code, otp_record = create_email_verification_otp(user)

    try:
        send_signup_otp(
            recipient_email=user.email,
            otp_code=otp_code,
        )
    except Exception as exception:
        logger.exception(
            "OTP resend email delivery failed for user_id=%s",
            user.id,
        )

        raise OTPDeliveryError() from exception

    return ResendOTPResult(
        user=user,
        otp_expires_at=otp_record.expires_at,
    )


def verify_email_otp(*, email: str, otp_code: str) -> VerifyEmailResult:
    with transaction.atomic():
        user = _get_user_for_email_verification(
            email,
            for_update=True,
        )

        history = list(
            EmailOTP.objects.select_for_update()
            .filter(
                user=user,
                purpose=EmailOTP.Purpose.EMAIL_VERIFICATION,
            )
            .order_by("-created_at")[:OTP_LOOKBACK_LIMIT]
        )
        if not history:
            raise EmailVerificationRequiredError(
                code="NO_OTP_FOUND",
                message=(
                    "No verification code was found for this account."
                ),
            )

        active_otp = next(
            (
                otp_record
                for otp_record in history
                if not otp_record.is_used
            ),
            None,
        )

        if active_otp is not None and active_otp.is_expired:
            mark_otp_used(active_otp)
            raise OTPExpiredError()

        matched_otp = None
        for otp_record in history:
            if check_otp_code(otp_record, otp_code):
                matched_otp = otp_record
                break

        if matched_otp is None:
            if active_otp is not None:
                increment_otp_attempts(active_otp)

                if active_otp.attempts >= OTP_MAX_ATTEMPTS:
                    mark_otp_used(active_otp)
                    raise OTPTooManyAttemptsError()

            raise OTPInvalidError()

        if matched_otp.is_used:
            raise OTPReusedError()

        if matched_otp.is_expired:
            mark_otp_used(matched_otp)
            raise OTPExpiredError()

        mark_otp_used(matched_otp)
        user.is_active = True
        user.save(update_fields=["is_active"])

        from billing.services.subscription_service import (
            assign_free_plan_to_user,
        )

        assign_free_plan_to_user(user)

    return VerifyEmailResult(user=user)


def signup_user(validated_data) -> SignupResult:
    """
    Create an inactive normal user and send an email OTP.
    """

    # Copy data so the serializer's validated_data is not
    # unexpectedly changed.
    signup_data = dict(validated_data)

    signup_data.pop(
        "password_confirm",
        None,
    )

    with transaction.atomic():
        user, otp_code, otp_record = (
            _create_pending_user(signup_data)
        )

    # Send only after PostgreSQL successfully commits.
    try:
        send_signup_otp(
            recipient_email=user.email,
            otp_code=otp_code,
        )
    except Exception as exception:
        # Never log the OTP or raw password.
        logger.exception(
            "OTP email delivery failed for user_id=%s",
            user.id,
        )

        raise OTPDeliveryError() from exception

    return SignupResult(
        user=user,
        otp_expires_at=otp_record.expires_at,
    )
