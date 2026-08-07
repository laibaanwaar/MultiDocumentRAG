import logging
from dataclasses import dataclass
from typing import Any

from django.contrib.auth import get_user_model


logger = logging.getLogger(__name__)
User = get_user_model()


@dataclass(frozen=True)
class UserProfileResult:
    user: Any
    role: str


def _user_role(user) -> str:
    return "admin" if user.is_staff or user.is_superuser else "user"


def get_user_profile(*, user_id: int) -> UserProfileResult:
    user = User.objects.get(pk=user_id)

    return UserProfileResult(
        user=user,
        role=_user_role(user),
    )
