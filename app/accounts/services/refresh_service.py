import logging
from dataclasses import dataclass
from typing import Any

from rest_framework_simplejwt.exceptions import TokenError
from rest_framework_simplejwt.tokens import RefreshToken

from accounts.exceptions import TokenInvalidError


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RefreshResult:
    access: str
    user_id: Any


def refresh_access_token(*, refresh_token: str) -> RefreshResult:
    try:
        token = RefreshToken(refresh_token)
    except TokenError as exception:
        raise TokenInvalidError() from exception
    except Exception:
        raise

    return RefreshResult(
        access=str(token.access_token),
        user_id=token.get("user_id"),
    )
