import logging
from dataclasses import dataclass
from typing import Any

from django.contrib.auth.hashers import check_password
from django.contrib.auth import get_user_model
from django.db import DatabaseError
from rest_framework_simplejwt.tokens import RefreshToken

from accounts.exceptions import (
    EmailNotVerifiedError,
    InvalidCredentialsError,
)


logger = logging.getLogger(__name__)
User = get_user_model()


@dataclass(frozen=True)
class LoginResult:
    access: str
    refresh: str
    user: Any


def _normalize_identifier(identifier: str) -> str:
    return identifier.strip()


def _get_user_by_identifier(identifier: str):
    normalized = _normalize_identifier(identifier)

    if "@" in normalized:
        queryset = User.objects.filter(email__iexact=normalized)
    else:
        queryset = User.objects.filter(username__iexact=normalized)

    return queryset.order_by("id").first()


def _user_role(user) -> str:
    return "admin" if user.is_staff or user.is_superuser else "user"


def authenticate_login(*, identifier: str, password: str) -> LoginResult:
    try:
        user = _get_user_by_identifier(identifier)
    except DatabaseError:
        raise

    if user is None:
        raise InvalidCredentialsError()

    if not user.is_active:
        raise EmailNotVerifiedError()

    if not check_password(password, user.password):
        raise InvalidCredentialsError()

    refresh = RefreshToken.for_user(user)
    access = str(refresh.access_token)

    return LoginResult(
        access=access,
        refresh=str(refresh),
        user=user,
    )


def build_login_user_payload(user) -> dict[str, Any]:
    return {
        "id": user.id,
        "username": user.username,
        "email": user.email,
        "first_name": user.first_name,
        "last_name": user.last_name,
        "role": _user_role(user),
    }
