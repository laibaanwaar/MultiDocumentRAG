from rest_framework import status
from rest_framework.exceptions import APIException


class SignupConflictError(APIException):
    status_code = status.HTTP_409_CONFLICT
    default_code = "signup_conflict"

    def __init__(
        self,
        *,
        code: str,
        message: str,
        **extra,
    ):
        detail = {
            "code": code,
            "message": message,
        }
        detail.update(extra)

        super().__init__(detail=detail)


class OTPDeliveryError(APIException):
    status_code = status.HTTP_503_SERVICE_UNAVAILABLE
    default_code = "otp_delivery_failed"

    def __init__(self):
        super().__init__(
            detail={
                "code": "OTP_DELIVERY_FAILED",
                "message": (
                    "The account was created, but the "
                    "verification email could not be sent."
                ),
                "next_action": "resend_otp",
            }
        )


class EmailVerificationRequiredError(APIException):
    status_code = status.HTTP_400_BAD_REQUEST
    default_code = "email_verification_required"

    def __init__(self, *, code: str, message: str, **extra):
        detail = {
            "code": code,
            "message": message,
        }
        detail.update(extra)

        super().__init__(detail=detail)


class EmailAlreadyVerifiedError(APIException):
    status_code = status.HTTP_409_CONFLICT
    default_code = "email_already_verified"

    def __init__(self):
        super().__init__(
            detail={
                "code": "EMAIL_ALREADY_VERIFIED",
                "message": (
                    "This email address has already been verified."
                ),
                "next_action": "login",
            }
        )


class OTPResendCooldownError(APIException):
    status_code = status.HTTP_429_TOO_MANY_REQUESTS
    default_code = "otp_resend_cooldown"

    def __init__(self, *, retry_after_seconds: int):
        super().__init__(
            detail={
                "code": "OTP_RESEND_COOLDOWN",
                "message": (
                    "Please wait before requesting another "
                    "verification code."
                ),
                "next_action": "resend_otp",
                "retry_after_seconds": retry_after_seconds,
            }
        )


class OTPInvalidError(APIException):
    status_code = status.HTTP_400_BAD_REQUEST
    default_code = "invalid_otp"

    def __init__(self):
        super().__init__(
            detail={
                "code": "INVALID_OTP",
                "message": "The verification code is invalid.",
            }
        )


class OTPExpiredError(APIException):
    status_code = status.HTTP_400_BAD_REQUEST
    default_code = "otp_expired"

    def __init__(self):
        super().__init__(
            detail={
                "code": "OTP_EXPIRED",
                "message": (
                    "The verification code has expired. "
                    "Please request a new one."
                ),
                "next_action": "resend_otp",
            }
        )


class OTPReusedError(APIException):
    status_code = status.HTTP_400_BAD_REQUEST
    default_code = "otp_reused"

    def __init__(self):
        super().__init__(
            detail={
                "code": "OTP_ALREADY_USED",
                "message": (
                    "This verification code has already been used."
                ),
                "next_action": "resend_otp",
            }
        )


class OTPTooManyAttemptsError(APIException):
    status_code = status.HTTP_429_TOO_MANY_REQUESTS
    default_code = "otp_too_many_attempts"

    def __init__(self):
        super().__init__(
            detail={
                "code": "TOO_MANY_OTP_ATTEMPTS",
                "message": (
                    "Too many incorrect attempts. "
                    "Please request a new code."
                ),
                "next_action": "resend_otp",
            }
        )


class InvalidCredentialsError(APIException):
    status_code = status.HTTP_401_UNAUTHORIZED
    default_code = "invalid_credentials"

    def __init__(self):
        super().__init__(
            detail={
                "code": "INVALID_CREDENTIALS",
                "message": "Invalid username/email or password.",
            }
        )


class EmailNotVerifiedError(APIException):
    status_code = status.HTTP_403_FORBIDDEN
    default_code = "email_not_verified"

    def __init__(self):
        super().__init__(
            detail={
                "code": "EMAIL_NOT_VERIFIED",
                "message": (
                    "This email address has not been verified yet."
                ),
            }
        )


class TokenInvalidError(APIException):
    status_code = status.HTTP_401_UNAUTHORIZED
    default_code = "token_invalid"

    def __init__(self):
        super().__init__(
            detail={
                "code": "TOKEN_INVALID",
                "message": "The refresh token is invalid or expired.",
            }
        )


class LogoutRefreshTokenRequiredError(APIException):
    status_code = status.HTTP_400_BAD_REQUEST
    default_code = "refresh_token_required"

    def __init__(self):
        super().__init__(
            detail={
                "code": "REFRESH_TOKEN_REQUIRED",
                "message": "The refresh token is required.",
            }
        )


class LogoutInvalidRefreshTokenError(APIException):
    status_code = status.HTTP_400_BAD_REQUEST
    default_code = "invalid_refresh_token"

    def __init__(self, *, message: str = "The refresh token is invalid."):
        super().__init__(
            detail={
                "code": "INVALID_REFRESH_TOKEN",
                "message": message,
            }
        )


class LogoutRefreshTokenExpiredError(APIException):
    status_code = status.HTTP_401_UNAUTHORIZED
    default_code = "refresh_token_expired"

    def __init__(self):
        super().__init__(
            detail={
                "code": "INVALID_REFRESH_TOKEN",
                "message": "The refresh token is invalid or expired.",
            }
        )


class LogoutTokenUserMismatchError(APIException):
    status_code = status.HTTP_403_FORBIDDEN
    default_code = "token_user_mismatch"

    def __init__(self):
        super().__init__(
            detail={
                "code": "TOKEN_USER_MISMATCH",
                "message": (
                    "The refresh token does not belong to the authenticated user."
                ),
            }
        )


class AccountInactiveError(APIException):
    status_code = status.HTTP_403_FORBIDDEN
    default_code = "account_inactive"

    def __init__(self):
        super().__init__(
            detail={
                "code": "ACCOUNT_INACTIVE",
                "message": "This account is inactive.",
            }
        )


class AdminUserNotFoundError(APIException):
    status_code = status.HTTP_404_NOT_FOUND
    default_code = "admin_user_not_found"

    def __init__(self):
        super().__init__(
            detail={
                "code": "USER_NOT_FOUND",
                "message": "The user was not found.",
            }
        )
