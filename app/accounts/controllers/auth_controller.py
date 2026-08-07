import logging

from django.contrib.auth import get_user_model
from django.db import DatabaseError, IntegrityError
from django.utils import timezone
from rest_framework import status
from rest_framework.exceptions import APIException, AuthenticationFailed, NotAuthenticated
from rest_framework.permissions import AllowAny, IsAuthenticated
from rest_framework.response import Response
from rest_framework.throttling import ScopedRateThrottle
from rest_framework.views import APIView
from rest_framework_simplejwt.authentication import JWTAuthentication
from rest_framework_simplejwt.exceptions import InvalidToken

from accounts.serializers import (
    LoginSerializer,
    ResendOTPSerializer,
    RefreshTokenSerializer,
    SignupSerializer,
    VerifyEmailSerializer,
)
from accounts.services.refresh_service import (
    refresh_access_token,
)
from accounts.services.login_service import (
    authenticate_login,
    build_login_user_payload,
)
from accounts.services.profile_service import (
    get_user_profile,
)
from accounts.services.signup_service import (
    resend_email_verification_otp,
    signup_user,
    verify_email_otp,
)


logger = logging.getLogger(__name__)
User = get_user_model()


def mask_email(email: str) -> str:
    local_part, domain = email.split("@", 1)

    if len(local_part) <= 2:
        masked_local = local_part[0] + "*"
    else:
        masked_local = (
            local_part[:2]
            + ("*" * min(len(local_part) - 2, 6))
        )

    return f"{masked_local}@{domain}"


def otp_seconds_remaining(expires_at) -> int:
    remaining = int((expires_at - timezone.now()).total_seconds())
    return max(0, remaining)


def _flatten_detail(detail):
    if isinstance(detail, dict):
        for value in detail.values():
            yield from _flatten_detail(value)
    elif isinstance(detail, (list, tuple)):
        for item in detail:
            yield from _flatten_detail(item)
    elif detail is not None:
        yield str(detail)


def _auth_error_response(code: str, message: str, status_code: int):
    return Response(
        {
            "code": code,
            "message": message,
        },
        status=status_code,
    )


class SignupController(APIView):
    """
    Create a normal inactive user and email an OTP.

    No JWT is returned before email verification.
    """

    permission_classes = [AllowAny]
    authentication_classes = []
    throttle_classes = [ScopedRateThrottle]
    throttle_scope = "auth_signup"

    def post(self, request):
        serializer = SignupSerializer(data=request.data)

        if not serializer.is_valid():
            return Response(
                {
                    "code": "VALIDATION_ERROR",
                    "message": "The signup information is invalid.",
                    "errors": serializer.errors,
                },
                status=status.HTTP_400_BAD_REQUEST,
            )

        try:
            result = signup_user(serializer.validated_data)
        except APIException:
            raise
        except (IntegrityError, DatabaseError):
            logger.exception("Database error during signup.")
            return Response(
                {
                    "code": "DATABASE_ERROR",
                    "message": (
                        "The account could not be created right now."
                    ),
                },
                status=status.HTTP_503_SERVICE_UNAVAILABLE,
            )
        except Exception:
            logger.exception("Unexpected signup failure.")
            return Response(
                {
                    "code": "SIGNUP_FAILED",
                    "message": (
                        "The account could not be created because "
                        "of an unexpected error."
                    ),
                },
                status=status.HTTP_500_INTERNAL_SERVER_ERROR,
            )

        return Response(
            {
                "message": (
                    "Account created. A verification code has been "
                    "sent to your email."
                ),
                "data": {
                    "username": result.user.username,
                    "email": mask_email(result.user.email),
                    "verification_required": True,
                    "otp_expires_in_seconds": otp_seconds_remaining(
                        result.otp_expires_at
                    ),
                },
            },
            status=status.HTTP_201_CREATED,
        )


class ResendOTPController(APIView):
    permission_classes = [AllowAny]
    authentication_classes = []
    throttle_classes = [ScopedRateThrottle]
    throttle_scope = "auth_resend_otp"

    def post(self, request):
        serializer = ResendOTPSerializer(data=request.data)

        if not serializer.is_valid():
            return Response(
                {
                    "code": "VALIDATION_ERROR",
                    "message": "The resend request is invalid.",
                    "errors": serializer.errors,
                },
                status=status.HTTP_400_BAD_REQUEST,
            )

        try:
            result = resend_email_verification_otp(
                email=serializer.validated_data["email"],
            )
        except APIException:
            raise
        except (IntegrityError, DatabaseError):
            logger.exception("Database error while resending OTP.")
            return Response(
                {
                    "code": "DATABASE_ERROR",
                    "message": (
                        "The verification code could not be resent right now."
                    ),
                },
                status=status.HTTP_503_SERVICE_UNAVAILABLE,
            )
        except Exception:
            logger.exception("Unexpected resend OTP failure.")
            return Response(
                {
                    "code": "RESEND_OTP_FAILED",
                    "message": (
                        "The verification code could not be resent because "
                        "of an unexpected error."
                    ),
                },
                status=status.HTTP_500_INTERNAL_SERVER_ERROR,
            )

        return Response(
            {
                "message": "A new verification code has been sent to your email.",
                "data": {
                    "email": mask_email(result.user.email),
                    "verification_required": True,
                    "otp_expires_in_seconds": otp_seconds_remaining(
                        result.otp_expires_at
                    ),
                },
            },
            status=status.HTTP_200_OK,
        )


class VerifyEmailController(APIView):
    permission_classes = [AllowAny]
    authentication_classes = []
    throttle_classes = [ScopedRateThrottle]
    throttle_scope = "auth_verify_email"

    def post(self, request):
        serializer = VerifyEmailSerializer(data=request.data)

        if not serializer.is_valid():
            return Response(
                {
                    "code": "VALIDATION_ERROR",
                    "message": "The verification request is invalid.",
                    "errors": serializer.errors,
                },
                status=status.HTTP_400_BAD_REQUEST,
            )

        try:
            result = verify_email_otp(
                email=serializer.validated_data["email"],
                otp_code=serializer.validated_data["otp"],
            )
        except APIException:
            raise
        except (IntegrityError, DatabaseError):
            logger.exception("Database error while verifying email.")
            return Response(
                {
                    "code": "DATABASE_ERROR",
                    "message": (
                        "The email verification could not be completed right now."
                    ),
                },
                status=status.HTTP_503_SERVICE_UNAVAILABLE,
            )
        except Exception:
            logger.exception("Unexpected email verification failure.")
            return Response(
                {
                    "code": "EMAIL_VERIFICATION_FAILED",
                    "message": (
                        "The email could not be verified because of an "
                        "unexpected error."
                    ),
                },
                status=status.HTTP_500_INTERNAL_SERVER_ERROR,
            )

        return Response(
            {
                "message": "Email verified successfully.",
                "data": {
                    "email": mask_email(result.user.email),
                    "verified": True,
                },
            },
            status=status.HTTP_200_OK,
        )


class LoginController(APIView):
    permission_classes = [AllowAny]
    authentication_classes = []
    throttle_classes = [ScopedRateThrottle]
    throttle_scope = "auth_login"

    def post(self, request):
        serializer = LoginSerializer(data=request.data)

        if not serializer.is_valid():
            return Response(
                {
                    "code": "VALIDATION_ERROR",
                    "message": "The login information is invalid.",
                    "errors": serializer.errors,
                },
                status=status.HTTP_400_BAD_REQUEST,
            )

        try:
            result = authenticate_login(
                identifier=serializer.validated_data["identifier"],
                password=serializer.validated_data["password"],
            )
        except APIException:
            raise
        except (IntegrityError, DatabaseError):
            logger.error("Database error during login.")
            return Response(
                {
                    "code": "DATABASE_ERROR",
                    "message": (
                        "The login could not be completed right now."
                    ),
                },
                status=status.HTTP_503_SERVICE_UNAVAILABLE,
            )
        except Exception:
            logger.error("Unexpected login failure.")
            return Response(
                {
                    "code": "LOGIN_FAILED",
                    "message": (
                        "The login could not be completed because of "
                        "an unexpected error."
                    ),
                },
                status=status.HTTP_500_INTERNAL_SERVER_ERROR,
            )

        return Response(
            {
                "message": "Login successful.",
                "data": {
                    "access": result.access,
                    "refresh": result.refresh,
                    "user": build_login_user_payload(
                        result.user
                    ),
                },
            },
            status=status.HTTP_200_OK,
        )


class RefreshTokenController(APIView):
    permission_classes = [AllowAny]
    authentication_classes = []
    throttle_classes = [ScopedRateThrottle]
    throttle_scope = "auth_refresh"

    def post(self, request):
        serializer = RefreshTokenSerializer(data=request.data)

        if not serializer.is_valid():
            return Response(
                {
                    "code": "VALIDATION_ERROR",
                    "message": "The refresh request is invalid.",
                    "errors": serializer.errors,
                },
                status=status.HTTP_400_BAD_REQUEST,
            )

        try:
            result = refresh_access_token(
                refresh_token=serializer.validated_data["refresh"],
            )
        except APIException:
            raise
        except (IntegrityError, DatabaseError):
            logger.warning("Database error during token refresh.")
            return Response(
                {
                    "code": "DATABASE_ERROR",
                    "message": (
                        "The access token could not be refreshed right now."
                    ),
                },
                status=status.HTTP_503_SERVICE_UNAVAILABLE,
            )
        except Exception:
            logger.warning("Unexpected token refresh failure.")
            return Response(
                {
                    "code": "REFRESH_FAILED",
                    "message": (
                        "The access token could not be refreshed because of "
                        "an unexpected error."
                    ),
                },
                status=status.HTTP_500_INTERNAL_SERVER_ERROR,
            )

        return Response(
            {
                "message": "Access token refreshed successfully.",
                "data": {
                    "access": result.access,
                },
            },
            status=status.HTTP_200_OK,
        )


class MeController(APIView):
    authentication_classes = [JWTAuthentication]
    permission_classes = [IsAuthenticated]
    throttle_classes = [ScopedRateThrottle]
    throttle_scope = "auth_me"

    def handle_exception(self, exc):
        if isinstance(exc, NotAuthenticated):
            return _auth_error_response(
                code="AUTHENTICATION_REQUIRED",
                message="Authentication credentials were not provided.",
                status_code=status.HTTP_401_UNAUTHORIZED,
            )

        if isinstance(exc, (AuthenticationFailed, InvalidToken)):
            detail_text = " ".join(_flatten_detail(getattr(exc, "detail", exc)))
            detail_lower = detail_text.lower()

            if "expired" in detail_lower:
                return _auth_error_response(
                    code="TOKEN_EXPIRED",
                    message="The access token has expired.",
                    status_code=status.HTTP_401_UNAUTHORIZED,
                )

            return _auth_error_response(
                code="TOKEN_INVALID",
                message="The access token is invalid.",
                status_code=status.HTTP_401_UNAUTHORIZED,
            )

        return super().handle_exception(exc)

    def get(self, request):
        if not request.user.is_active:
            return Response(
                {
                    "code": "ACCOUNT_INACTIVE",
                    "message": "This account is inactive.",
                },
                status=status.HTTP_403_FORBIDDEN,
            )

        try:
            result = get_user_profile(user_id=request.user.id)
        except User.DoesNotExist:
            return _auth_error_response(
                code="TOKEN_INVALID",
                message="The access token is invalid.",
                status_code=status.HTTP_401_UNAUTHORIZED,
            )
        except DatabaseError:
            logger.warning(
                "Database error while retrieving authenticated user profile."
            )
            return Response(
                {
                    "code": "SERVICE_UNAVAILABLE",
                    "message": (
                        "The user profile could not be retrieved right now."
                    ),
                },
                status=status.HTTP_503_SERVICE_UNAVAILABLE,
            )
        except Exception:
            logger.error("Unexpected user profile retrieval failure.")
            return Response(
                {
                    "code": "INTERNAL_SERVER_ERROR",
                    "message": (
                        "The user profile could not be retrieved because "
                        "of an unexpected error."
                    ),
                },
                status=status.HTTP_500_INTERNAL_SERVER_ERROR,
            )

        return Response(
            {
                "message": "User profile retrieved successfully.",
                "data": {
                    "id": result.user.id,
                    "username": result.user.username,
                    "email": result.user.email,
                    "first_name": result.user.first_name,
                    "last_name": result.user.last_name,
                    "role": result.role,
                },
            },
            status=status.HTTP_200_OK,
        )
