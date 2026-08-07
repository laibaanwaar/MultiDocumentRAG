from django.contrib.auth.password_validation import (
    validate_password,
)
from django.core.exceptions import (
    ValidationError as DjangoValidationError,
)
from rest_framework import serializers


def validate_strong_password(
    password: str,
    *,
    user,
) -> None:
    errors: list[str] = []

    # Run Django's configured password validators.
    try:
        validate_password(
            password=password,
            user=user,
        )
    except DjangoValidationError as exception:
        errors.extend(exception.messages)

    if len(password) < 12:
        errors.append(
            "Password must contain at least 12 characters."
        )

    if not any(character.isupper() for character in password):
        errors.append(
            "Password must contain at least one uppercase letter."
        )

    if not any(character.islower() for character in password):
        errors.append(
            "Password must contain at least one lowercase letter."
        )

    if not any(character.isdigit() for character in password):
        errors.append(
            "Password must contain at least one number."
        )

    if not any(
        not character.isalnum()
        and not character.isspace()
        for character in password
    ):
        errors.append(
            "Password must contain at least one special character."
        )

    if any(character.isspace() for character in password):
        errors.append(
            "Password must not contain spaces."
        )

    if errors:
        # Remove duplicate messages while preserving order.
        unique_errors = list(dict.fromkeys(errors))

        raise serializers.ValidationError(unique_errors)