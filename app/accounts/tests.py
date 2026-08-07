import re
from datetime import timedelta
from unittest.mock import patch

from django.contrib.auth import get_user_model
from django.db import DatabaseError
from django.core import mail
from django.test import override_settings
from django.utils import timezone
from rest_framework.test import APITestCase
from rest_framework_simplejwt.tokens import AccessToken, RefreshToken

from accounts.models import EmailOTP


User = get_user_model()


@override_settings(
    EMAIL_BACKEND="django.core.mail.backends.locmem.EmailBackend",
)
class EmailOTPAuthAPITests(APITestCase):
    def _signup(self, *, username="testuser", email="user@example.com"):
        with patch(
            "accounts.services.otp_service.generate_six_digit_otp",
            return_value="123456",
        ):
            response = self.client.post(
                "/api/v1/auth/signup/",
                {
                    "username": username,
                    "email": email,
                    "first_name": "Test",
                    "last_name": "User",
                    "password": "StrongPass!234",
                    "password_confirm": "StrongPass!234",
                },
                format="json",
            )

        self.assertEqual(response.status_code, 201)

        self.assertEqual(len(mail.outbox), 1)
        match = re.search(
            r"verification code is:\s*(\d{6})",
            mail.outbox[-1].body,
        )
        self.assertIsNotNone(match)

        return response, match.group(1)

    def test_signup_creates_inactive_user_and_sends_otp(self):
        response, otp = self._signup()

        self.assertEqual(response.data["data"]["verification_required"], True)
        self.assertGreaterEqual(
            response.data["data"]["otp_expires_in_seconds"],
            1790,
        )
        self.assertLessEqual(
            response.data["data"]["otp_expires_in_seconds"],
            1800,
        )

        user = User.objects.get(email="user@example.com")
        self.assertFalse(user.is_active)

        otp_record = EmailOTP.objects.get(user=user)
        self.assertFalse(otp_record.is_used)
        self.assertFalse(otp_record.is_expired)
        self.assertTrue(otp_record.code_hash)
        self.assertNotEqual(otp_record.code_hash, otp)

    def test_duplicate_active_email_signup_returns_409(self):
        self._signup()

        response = self.client.post(
            "/api/v1/auth/signup/",
            {
                "username": "anotheruser",
                "email": "user@example.com",
                "password": "StrongPass!234",
                "password_confirm": "StrongPass!234",
            },
            format="json",
        )

        self.assertEqual(response.status_code, 409)
        self.assertEqual(response.data["code"], "EMAIL_ALREADY_REGISTERED")

    def test_resend_otp_enforces_cooldown(self):
        self._signup()

        with patch(
            "accounts.services.otp_service.generate_six_digit_otp",
            return_value="654321",
        ):
            response = self.client.post(
                "/api/v1/auth/resend-otp/",
                {"email": "user@example.com"},
                format="json",
            )

        self.assertEqual(response.status_code, 429)
        self.assertEqual(response.data["code"], "OTP_RESEND_COOLDOWN")

    def test_resend_otp_replaces_old_code_after_cooldown(self):
        self._signup()
        user = User.objects.get(email="user@example.com")
        otp_record = EmailOTP.objects.get(user=user, is_used=False)
        otp_record.created_at = timezone.now() - timedelta(seconds=120)
        otp_record.save(update_fields=["created_at"])

        with patch(
            "accounts.services.otp_service.generate_six_digit_otp",
            return_value="654321",
        ):
            response = self.client.post(
                "/api/v1/auth/resend-otp/",
                {"email": "user@example.com"},
                format="json",
            )

        self.assertEqual(response.status_code, 200)
        self.assertGreaterEqual(
            response.data["data"]["otp_expires_in_seconds"],
            1790,
        )
        self.assertLessEqual(
            response.data["data"]["otp_expires_in_seconds"],
            1800,
        )
        self.assertEqual(len(mail.outbox), 2)

        user.refresh_from_db()
        self.assertFalse(user.is_active)

        old_record = EmailOTP.objects.filter(user=user, code_hash__isnull=False).earliest("created_at")
        new_record = EmailOTP.objects.filter(user=user, is_used=False).latest("created_at")

        self.assertTrue(old_record.is_used)
        self.assertFalse(new_record.is_used)
        self.assertEqual(
            re.search(r"verification code is:\s*(\d{6})", mail.outbox[-1].body).group(1),
            "654321",
        )

    def test_verify_email_activates_user(self):
        _, otp = self._signup()

        response = self.client.post(
            "/api/v1/auth/verify-email/",
            {
                "email": "user@example.com",
                "otp": otp,
            },
            format="json",
        )

        self.assertEqual(response.status_code, 200)
        self.assertTrue(response.data["data"]["verified"])

        user = User.objects.get(email="user@example.com")
        self.assertTrue(user.is_active)

        otp_record = EmailOTP.objects.get(user=user)
        self.assertTrue(otp_record.is_used)

    def test_verify_email_counts_wrong_attempts_and_blocks_after_limit(self):
        _, otp = self._signup()

        for attempt in range(4):
            response = self.client.post(
                "/api/v1/auth/verify-email/",
                {
                    "email": "user@example.com",
                    "otp": "000000",
                },
                format="json",
            )
            self.assertEqual(response.status_code, 400)
            self.assertEqual(response.data["code"], "INVALID_OTP")

        response = self.client.post(
            "/api/v1/auth/verify-email/",
            {
                "email": "user@example.com",
                "otp": "000000",
            },
            format="json",
        )
        self.assertEqual(response.status_code, 429)
        self.assertEqual(
            response.data["code"],
            "TOO_MANY_OTP_ATTEMPTS",
        )

        final_response = self.client.post(
            "/api/v1/auth/verify-email/",
            {
                "email": "user@example.com",
                "otp": otp,
            },
            format="json",
        )
        self.assertEqual(final_response.status_code, 400)
        self.assertEqual(final_response.data["code"], "OTP_ALREADY_USED")

    def test_verify_expired_otp_returns_400(self):
        _, otp = self._signup()
        user = User.objects.get(email="user@example.com")
        otp_record = EmailOTP.objects.get(user=user)
        otp_record.expires_at = timezone.now() - timedelta(seconds=1)
        otp_record.save(update_fields=["expires_at"])

        response = self.client.post(
            "/api/v1/auth/verify-email/",
            {
                "email": "user@example.com",
                "otp": otp,
            },
            format="json",
        )

        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.data["code"], "OTP_EXPIRED")

    def test_verify_reused_otp_after_resend_returns_400(self):
        _, first_otp = self._signup()
        user = User.objects.get(email="user@example.com")
        first_record = EmailOTP.objects.get(user=user)
        first_record.created_at = timezone.now() - timedelta(seconds=120)
        first_record.save(update_fields=["created_at"])

        with patch(
            "accounts.services.otp_service.generate_six_digit_otp",
            return_value="654321",
        ):
            resend_response = self.client.post(
                "/api/v1/auth/resend-otp/",
                {"email": "user@example.com"},
                format="json",
            )
        self.assertEqual(resend_response.status_code, 200)
        second_otp = re.search(
            r"verification code is:\s*(\d{6})",
            mail.outbox[-1].body,
        ).group(1)

        reuse_response = self.client.post(
            "/api/v1/auth/verify-email/",
            {
                "email": "user@example.com",
                "otp": first_otp,
            },
            format="json",
        )

        self.assertEqual(reuse_response.status_code, 400)
        self.assertEqual(
            reuse_response.data["code"],
            "OTP_ALREADY_USED",
        )

        success_response = self.client.post(
            "/api/v1/auth/verify-email/",
            {
                "email": "user@example.com",
                "otp": second_otp,
            },
            format="json",
        )
        self.assertEqual(success_response.status_code, 200)


class LoginAPITests(APITestCase):
    def _create_user(
        self,
        *,
        username="loginuser",
        email="loginuser@example.com",
        password="StrongPass!234",
        is_active=True,
        is_staff=False,
        is_superuser=False,
    ):
        user = User.objects.create_user(
            username=username,
            email=email,
            password=password,
            first_name="Login",
            last_name="User",
            is_active=is_active,
            is_staff=is_staff,
            is_superuser=is_superuser,
        )
        return user

    def test_username_login_success(self):
        user = self._create_user()

        response = self.client.post(
            "/api/v1/auth/login/",
            {
                "identifier": "LOGINUSER",
                "password": "StrongPass!234",
            },
            format="json",
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.data["message"], "Login successful.")
        self.assertTrue(response.data["data"]["access"])
        self.assertTrue(response.data["data"]["refresh"])
        self.assertEqual(response.data["data"]["user"]["id"], user.id)
        self.assertEqual(response.data["data"]["user"]["username"], user.username)
        self.assertEqual(response.data["data"]["user"]["email"], user.email)
        self.assertEqual(response.data["data"]["user"]["role"], "user")

    def test_email_login_success(self):
        user = self._create_user(email="Person@Example.com")

        response = self.client.post(
            "/api/v1/auth/login/",
            {
                "identifier": "PERSON@example.com",
                "password": "StrongPass!234",
            },
            format="json",
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.data["data"]["user"]["id"], user.id)

    def test_wrong_password_returns_401(self):
        self._create_user()

        response = self.client.post(
            "/api/v1/auth/login/",
            {
                "identifier": "loginuser",
                "password": "WrongPass!234",
            },
            format="json",
        )

        self.assertEqual(response.status_code, 401)
        self.assertEqual(response.data["code"], "INVALID_CREDENTIALS")

    def test_nonexistent_user_returns_401(self):
        response = self.client.post(
            "/api/v1/auth/login/",
            {
                "identifier": "missing",
                "password": "StrongPass!234",
            },
            format="json",
        )

        self.assertEqual(response.status_code, 401)
        self.assertEqual(response.data["code"], "INVALID_CREDENTIALS")

    def test_missing_fields_return_400(self):
        response = self.client.post(
            "/api/v1/auth/login/",
            {
                "identifier": "loginuser",
            },
            format="json",
        )

        self.assertEqual(response.status_code, 400)
        self.assertIn("password", response.data["errors"])

    def test_unknown_fields_return_400(self):
        response = self.client.post(
            "/api/v1/auth/login/",
            {
                "identifier": "loginuser",
                "password": "StrongPass!234",
                "role": "admin",
            },
            format="json",
        )

        self.assertEqual(response.status_code, 400)
        self.assertIn("role", response.data["errors"])

    def test_inactive_user_returns_403(self):
        self._create_user(is_active=False)

        response = self.client.post(
            "/api/v1/auth/login/",
            {
                "identifier": "loginuser",
                "password": "StrongPass!234",
            },
            format="json",
        )

        self.assertEqual(response.status_code, 403)
        self.assertEqual(response.data["code"], "EMAIL_NOT_VERIFIED")

    def test_admin_login_returns_admin_role(self):
        user = self._create_user(is_staff=True)

        response = self.client.post(
            "/api/v1/auth/login/",
            {
                "identifier": "loginuser",
                "password": "StrongPass!234",
            },
            format="json",
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.data["data"]["user"]["role"], "admin")
        self.assertEqual(response.data["data"]["user"]["id"], user.id)

    def test_database_failure_returns_503(self):
        with patch(
            "accounts.controllers.auth_controller.authenticate_login",
            side_effect=DatabaseError("db down"),
        ):
            response = self.client.post(
                "/api/v1/auth/login/",
                {
                    "identifier": "loginuser",
                    "password": "StrongPass!234",
                },
                format="json",
            )

        self.assertEqual(response.status_code, 503)
        self.assertEqual(response.data["code"], "DATABASE_ERROR")


class RefreshAPITests(APITestCase):
    def _create_user(self):
        return User.objects.create_user(
            username="refreshuser",
            email="refreshuser@example.com",
            password="StrongPass!234",
            first_name="Refresh",
            last_name="User",
            is_active=True,
        )

    def _login(self, identifier="refreshuser"):
        self._create_user()
        response = self.client.post(
            "/api/v1/auth/login/",
            {
                "identifier": identifier,
                "password": "StrongPass!234",
            },
            format="json",
        )
        self.assertEqual(response.status_code, 200)
        return response.data["data"]["access"], response.data["data"]["refresh"]

    def test_token_lifetimes_are_explicit(self):
        access_token, refresh_token = self._login()

        access = AccessToken(access_token)
        refresh = RefreshToken(refresh_token)

        self.assertEqual(access["exp"] - access["iat"], 3600)
        self.assertEqual(refresh["exp"] - refresh["iat"], 86400)

    def test_refresh_returns_new_access_token(self):
        old_access, refresh_token = self._login()

        response = self.client.post(
            "/api/v1/auth/refresh/",
            {"refresh": refresh_token},
            format="json",
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.data["message"], "Access token refreshed successfully.")
        self.assertTrue(response.data["data"]["access"])
        self.assertNotEqual(response.data["data"]["access"], old_access)

    def test_missing_refresh_token_returns_400(self):
        response = self.client.post(
            "/api/v1/auth/refresh/",
            {},
            format="json",
        )

        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.data["code"], "VALIDATION_ERROR")
        self.assertIn("refresh", response.data["errors"])

    def test_unknown_field_returns_400(self):
        response = self.client.post(
            "/api/v1/auth/refresh/",
            {
                "refresh": "abc",
                "extra": "value",
            },
            format="json",
        )

        self.assertEqual(response.status_code, 400)
        self.assertIn("extra", response.data["errors"])

    def test_malformed_refresh_token_returns_401(self):
        response = self.client.post(
            "/api/v1/auth/refresh/",
            {"refresh": "not-a-jwt"},
            format="json",
        )

        self.assertEqual(response.status_code, 401)
        self.assertEqual(response.data["code"], "TOKEN_INVALID")


class MeAPITests(APITestCase):
    def _create_user(
        self,
        *,
        username="meuser",
        email="meuser@example.com",
        password="StrongPass!234",
        is_active=True,
        is_staff=False,
        is_superuser=False,
    ):
        return User.objects.create_user(
            username=username,
            email=email,
            password=password,
            first_name="Me",
            last_name="User",
            is_active=is_active,
            is_staff=is_staff,
            is_superuser=is_superuser,
        )

    def _login_access(self, **user_kwargs):
        user = self._create_user(**user_kwargs)
        response = self.client.post(
            "/api/v1/auth/login/",
            {
                "identifier": user.username,
                "password": "StrongPass!234",
            },
            format="json",
        )
        self.assertEqual(response.status_code, 200)
        return user, response.data["data"]["access"]

    def test_valid_user_token_returns_profile(self):
        user, access = self._login_access()

        response = self.client.get(
            "/api/v1/auth/me/",
            HTTP_AUTHORIZATION=f"Bearer {access}",
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(
            response.data,
            {
                "message": "User profile retrieved successfully.",
                "data": {
                    "id": user.id,
                    "username": user.username,
                    "email": user.email,
                    "first_name": user.first_name,
                    "last_name": user.last_name,
                    "role": "user",
                },
            },
        )

    def test_admin_token_returns_admin_role(self):
        user, access = self._login_access(is_staff=True)

        response = self.client.get(
            "/api/v1/auth/me/",
            HTTP_AUTHORIZATION=f"Bearer {access}",
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.data["data"]["role"], "admin")
        self.assertEqual(response.data["data"]["id"], user.id)

    def test_missing_token_returns_401(self):
        response = self.client.get("/api/v1/auth/me/")

        self.assertEqual(response.status_code, 401)
        self.assertEqual(response.data["code"], "AUTHENTICATION_REQUIRED")

    def test_malformed_header_returns_401(self):
        response = self.client.get(
            "/api/v1/auth/me/",
            HTTP_AUTHORIZATION="Bearer",
        )

        self.assertEqual(response.status_code, 401)
        self.assertEqual(response.data["code"], "TOKEN_INVALID")

    def test_invalid_token_returns_401(self):
        response = self.client.get(
            "/api/v1/auth/me/",
            HTTP_AUTHORIZATION="Bearer not-a-jwt",
        )

        self.assertEqual(response.status_code, 401)
        self.assertEqual(response.data["code"], "TOKEN_INVALID")

    def test_expired_access_token_returns_401(self):
        user = self._create_user()
        access = AccessToken.for_user(user)
        access.set_exp(
            from_time=timezone.now() - timedelta(hours=2),
            lifetime=timedelta(hours=1),
        )

        response = self.client.get(
            "/api/v1/auth/me/",
            HTTP_AUTHORIZATION=f"Bearer {str(access)}",
        )

        self.assertEqual(response.status_code, 401)
        self.assertEqual(response.data["code"], "TOKEN_EXPIRED")

    def test_refresh_token_as_access_token_returns_401(self):
        user = self._create_user()
        refresh = RefreshToken.for_user(user)

        response = self.client.get(
            "/api/v1/auth/me/",
            HTTP_AUTHORIZATION=f"Bearer {str(refresh)}",
        )

        self.assertEqual(response.status_code, 401)
        self.assertEqual(response.data["code"], "TOKEN_INVALID")

    def test_inactive_user_returns_403(self):
        user = self._create_user(is_active=False)
        access = str(AccessToken.for_user(user))

        response = self.client.get(
            "/api/v1/auth/me/",
            HTTP_AUTHORIZATION=f"Bearer {access}",
        )

        self.assertEqual(response.status_code, 403)
        self.assertEqual(response.data["code"], "ACCOUNT_INACTIVE")

    def test_deleted_user_returns_401(self):
        user = self._create_user()
        access = str(AccessToken.for_user(user))
        user.delete()

        response = self.client.get(
            "/api/v1/auth/me/",
            HTTP_AUTHORIZATION=f"Bearer {access}",
        )

        self.assertEqual(response.status_code, 401)
        self.assertEqual(response.data["code"], "TOKEN_INVALID")

    def test_database_failure_returns_503(self):
        _, access = self._login_access()

        with patch(
            "accounts.controllers.auth_controller.get_user_profile",
            side_effect=DatabaseError("db down"),
        ):
            response = self.client.get(
                "/api/v1/auth/me/",
                HTTP_AUTHORIZATION=f"Bearer {access}",
            )

        self.assertEqual(response.status_code, 503)
        self.assertEqual(response.data["code"], "SERVICE_UNAVAILABLE")

    def test_expired_refresh_token_returns_401(self):
        user = self._create_user()
        refresh = RefreshToken.for_user(user)
        refresh.set_exp(
            from_time=timezone.now() - timedelta(days=2),
            lifetime=timedelta(days=1),
        )

        response = self.client.post(
            "/api/v1/auth/refresh/",
            {"refresh": str(refresh)},
            format="json",
        )

        self.assertEqual(response.status_code, 401)
        self.assertEqual(response.data["code"], "TOKEN_INVALID")

    def test_wrong_token_type_returns_401(self):
        access, _ = self._login()

        response = self.client.post(
            "/api/v1/auth/refresh/",
            {"refresh": access},
            format="json",
        )

        self.assertEqual(response.status_code, 401)
        self.assertEqual(response.data["code"], "TOKEN_INVALID")

    def test_invalid_signature_returns_401(self):
        _, refresh_token = self._login()
        corrupted = refresh_token[:-1] + (
            "a" if refresh_token[-1] != "a" else "b"
        )

        response = self.client.post(
            "/api/v1/auth/refresh/",
            {"refresh": corrupted},
            format="json",
        )

        self.assertEqual(response.status_code, 401)
        self.assertEqual(response.data["code"], "TOKEN_INVALID")

    def test_blacklisted_refresh_token_returns_401(self):
        user = self._create_user()
        refresh = RefreshToken.for_user(user)
        refresh_str = str(refresh)
        refresh.blacklist()

        response = self.client.post(
            "/api/v1/auth/refresh/",
            {"refresh": refresh_str},
            format="json",
        )

        self.assertEqual(response.status_code, 401)
        self.assertEqual(response.data["code"], "TOKEN_INVALID")
