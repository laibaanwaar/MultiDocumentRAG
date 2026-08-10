from django.contrib.auth import get_user_model
from django.contrib.auth.validators import (
    UnicodeUsernameValidator,
)
from rest_framework import serializers

from accounts.validators import validate_strong_password


User = get_user_model()


class StrictInputSerializer(serializers.Serializer):
    """
    Reject fields that were not explicitly defined.

    This prevents clients from submitting fields such as
    is_staff, is_superuser, role, or is_active.
    """

    def to_internal_value(self, data):
        if hasattr(data, "keys"):
            unknown_fields = set(data.keys()) - set(
                self.fields.keys()
            )

            if unknown_fields:
                raise serializers.ValidationError(
                    {
                        field: ["This field is not allowed."]
                        for field in sorted(unknown_fields)
                    }
                )

        return super().to_internal_value(data)


class SignupSerializer(StrictInputSerializer):
    username = serializers.CharField(
        min_length=3,
        max_length=150,
        validators=[UnicodeUsernameValidator()],
    )

    email = serializers.EmailField(
        max_length=254,
    )

    first_name = serializers.CharField(
        max_length=150,
        required=False,
        allow_blank=True,
    )

    last_name = serializers.CharField(
        max_length=150,
        required=False,
        allow_blank=True,
    )

    password = serializers.CharField(
        write_only=True,
        min_length=12,
        max_length=128,
        trim_whitespace=False,
        style={
            "input_type": "password",
        },
    )

    password_confirm = serializers.CharField(
        write_only=True,
        max_length=128,
        trim_whitespace=False,
        style={
            "input_type": "password",
        },
    )

    def validate_username(self, value: str) -> str:
        # Lowercase normalization prevents User and user
        # from becoming separate usernames.
        normalized_username = value.strip().lower()

        if not normalized_username:
            raise serializers.ValidationError(
                "Username is required."
            )

        return normalized_username

    def validate_email(self, value: str) -> str:
        return value.strip().lower()

    def validate_first_name(self, value: str) -> str:
        return value.strip()

    def validate_last_name(self, value: str) -> str:
        return value.strip()

    def validate(self, attrs):
        password = attrs["password"]
        password_confirm = attrs["password_confirm"]

        if password != password_confirm:
            raise serializers.ValidationError(
                {
                    "password_confirm": [
                        "Passwords do not match."
                    ]
                }
            )

        # Unsaved user object lets Django check whether the
        # password is similar to personal information.
        candidate_user = User(
            username=attrs["username"],
            email=attrs["email"],
            first_name=attrs.get("first_name", ""),
            last_name=attrs.get("last_name", ""),
        )

        validate_strong_password(
            password,
            user=candidate_user,
        )

        return attrs


class ResendOTPSerializer(StrictInputSerializer):
    email = serializers.EmailField(
        max_length=254,
    )

    def validate_email(self, value: str) -> str:
        return value.strip().lower()


class VerifyEmailSerializer(StrictInputSerializer):
    email = serializers.EmailField(
        max_length=254,
    )
    otp = serializers.CharField(
        min_length=6,
        max_length=6,
        trim_whitespace=True,
    )

    def validate_email(self, value: str) -> str:
        return value.strip().lower()

    def validate_otp(self, value: str) -> str:
        if not value.isdigit():
            raise serializers.ValidationError(
                "OTP must contain exactly 6 digits."
            )

        if len(value) != 6:
            raise serializers.ValidationError(
                "OTP must contain exactly 6 digits."
            )

        return value


class LoginSerializer(StrictInputSerializer):
    identifier = serializers.CharField(
        max_length=254,
        trim_whitespace=True,
    )
    password = serializers.CharField(
        write_only=True,
        max_length=128,
        trim_whitespace=False,
        style={
            "input_type": "password",
        },
    )

    def validate_identifier(self, value: str) -> str:
        normalized = value.strip()

        if not normalized:
            raise serializers.ValidationError(
                "Identifier is required."
            )

        return normalized

    def validate_password(self, value: str) -> str:
        if not value:
            raise serializers.ValidationError(
                "Password is required."
            )

        return value


class RefreshTokenSerializer(StrictInputSerializer):
    refresh = serializers.CharField(
        write_only=True,
        trim_whitespace=False,
    )

    def validate_refresh(self, value: str) -> str:
        if not value:
            raise serializers.ValidationError(
                "Refresh token is required."
            )

        return value


class LogoutSerializer(StrictInputSerializer):
    refresh = serializers.CharField(
        write_only=True,
        trim_whitespace=False,
    )

    def validate_refresh(self, value: str) -> str:
        normalized = value.strip()

        if not normalized:
            raise serializers.ValidationError(
                "Refresh token is required."
            )

        return normalized
