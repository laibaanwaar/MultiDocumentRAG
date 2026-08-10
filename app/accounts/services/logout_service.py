from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from django.db import DatabaseError
from rest_framework_simplejwt.exceptions import ExpiredTokenError, TokenError
from rest_framework_simplejwt.settings import api_settings
from rest_framework_simplejwt.token_blacklist.models import (
    BlacklistedToken,
)
from rest_framework_simplejwt.tokens import RefreshToken, UntypedToken

from accounts.exceptions import (
    LogoutInvalidRefreshTokenError,
    LogoutRefreshTokenExpiredError,
    LogoutTokenUserMismatchError,
)


@dataclass(frozen=True)
class LogoutResult:
    logged_out: bool
    already_blacklisted: bool = False


def _validate_refresh_token_payload(refresh_token: str) -> UntypedToken:
    try:
        token = UntypedToken(refresh_token)
    except ExpiredTokenError as exception:
        raise LogoutRefreshTokenExpiredError() from exception
    except TokenError as exception:
        raise LogoutInvalidRefreshTokenError() from exception

    token_type = token.get(api_settings.TOKEN_TYPE_CLAIM)

    if token_type != RefreshToken.token_type:
        raise LogoutInvalidRefreshTokenError(
            message="The refresh token is invalid."
        )

    return token


def _assert_token_belongs_to_user(*, payload: dict[str, Any], user) -> None:
    token_user_id = payload.get(api_settings.USER_ID_CLAIM)

    if token_user_id is None or str(token_user_id) != str(user.id):
        raise LogoutTokenUserMismatchError()


def logout_refresh_token(*, user, refresh_token: str) -> LogoutResult:
    token = _validate_refresh_token_payload(refresh_token)
    _assert_token_belongs_to_user(
        payload=token.payload,
        user=user,
    )

    # Re-create the refresh token object only after the token has been
    # validated and ownership has been confirmed. This keeps blacklist
    # behavior idempotent without skipping token validation.
    refresh = RefreshToken(refresh_token, verify=False)

    try:
        blacklist_result = refresh.blacklist()
    except DatabaseError:
        raise

    already_blacklisted = False

    if isinstance(blacklist_result, tuple):
        blacklisted_token, created = blacklist_result
        already_blacklisted = not created
    else:
        already_blacklisted = BlacklistedToken.objects.filter(
            token__jti=token.payload[api_settings.JTI_CLAIM]
        ).exists()

    return LogoutResult(
        logged_out=True,
        already_blacklisted=already_blacklisted,
    )
