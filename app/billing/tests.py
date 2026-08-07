import re
from datetime import timedelta
from types import SimpleNamespace
from unittest.mock import patch

from django.contrib.auth import get_user_model
from django.core import mail
from django.db import DatabaseError
from django.test import override_settings
from django.utils import timezone
from rest_framework.test import APITestCase
from rest_framework_simplejwt.tokens import AccessToken, RefreshToken

from billing.models import Plan, Subscription
from billing.services.subscription_service import (
    assign_free_plan_to_user,
)


User = get_user_model()


@override_settings(
    EMAIL_BACKEND="django.core.mail.backends.locmem.EmailBackend",
)
class BillingAPITests(APITestCase):
    def _signup(self, *, username="billinguser", email="billing@example.com"):
        with patch(
            "accounts.services.otp_service.generate_six_digit_otp",
            return_value="123456",
        ):
            response = self.client.post(
                "/api/v1/auth/signup/",
                {
                    "username": username,
                    "email": email,
                    "first_name": "Billing",
                    "last_name": "User",
                    "password": "StrongPass!234",
                    "password_confirm": "StrongPass!234",
                },
                format="json",
            )

        self.assertEqual(response.status_code, 201)
        self.assertEqual(len(mail.outbox), 1)
        otp = re.search(
            r"verification code is:\s*(\d{6})",
            mail.outbox[-1].body,
        ).group(1)
        return response, otp

    def _signup_and_verify(self, *, username="billinguser", email="billing@example.com"):
        _, otp = self._signup(username=username, email=email)
        response = self.client.post(
            "/api/v1/auth/verify-email/",
            {
                "email": email,
                "otp": otp,
            },
            format="json",
        )
        self.assertEqual(response.status_code, 200)
        return User.objects.get(email=email)

    def _create_active_user(self, *, username="billinguser", email="billing@example.com", is_active=True, is_staff=False, is_superuser=False):
        return User.objects.create_user(
            username=username,
            email=email,
            password="StrongPass!234",
            first_name="Billing",
            last_name="User",
            is_active=is_active,
            is_staff=is_staff,
            is_superuser=is_superuser,
        )

    def _login_access(self, user):
        response = self.client.post(
            "/api/v1/auth/login/",
            {
                "identifier": user.username,
                "password": "StrongPass!234",
            },
            format="json",
        )
        self.assertEqual(response.status_code, 200)
        return response.data["data"]["access"]

    def test_verify_email_assigns_free_plan_subscription(self):
        user = self._signup_and_verify()

        subscription = Subscription.objects.get(user=user)
        self.assertEqual(subscription.plan.code, "FREE")
        self.assertTrue(subscription.plan.is_active)
        self.assertEqual(subscription.status, Subscription.Status.ACTIVE)
        self.assertEqual(subscription.queries_used, 0)
        self.assertEqual(subscription.queries_remaining, 30)

    def test_assign_free_plan_does_not_duplicate_subscriptions(self):
        user = self._signup_and_verify()

        first = assign_free_plan_to_user(user)
        second = assign_free_plan_to_user(user)

        self.assertEqual(first.pk, second.pk)
        self.assertEqual(Subscription.objects.filter(user=user).count(), 1)

    def test_plans_endpoint_returns_only_active_plans(self):
        Plan.objects.create(
            name="Starter",
            code="STARTER",
            price="9.99",
            billing_period=Plan.BillingPeriod.MONTHLY,
            document_limit=10,
            query_limit=100,
            is_active=True,
        )
        Plan.objects.create(
            name="Hidden",
            code="HIDDEN",
            price="19.99",
            billing_period=Plan.BillingPeriod.YEARLY,
            document_limit=50,
            query_limit=500,
            is_active=False,
        )

        response = self.client.get("/api/v1/billing/plans/")

        self.assertEqual(response.status_code, 200)
        codes = [plan["code"] for plan in response.data["data"]]
        self.assertIn("FREE", codes)
        self.assertIn("STARTER", codes)
        self.assertNotIn("HIDDEN", codes)

    def test_subscription_endpoint_returns_current_subscription(self):
        user = self._signup_and_verify()
        access = self._login_access(user)

        response = self.client.get(
            "/api/v1/billing/subscription/",
            HTTP_AUTHORIZATION=f"Bearer {access}",
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.data["message"], "Subscription retrieved successfully.")
        self.assertEqual(response.data["data"]["plan"]["code"], "FREE")
        self.assertEqual(response.data["data"]["queries_remaining"], 30)

    def test_subscription_endpoint_calculates_queries_remaining(self):
        user = self._signup_and_verify(email="remaining@example.com", username="remaininguser")
        subscription = Subscription.objects.get(user=user)
        subscription.queries_used = 7
        subscription.save(update_fields=["queries_used"])

        access = self._login_access(user)
        response = self.client.get(
            "/api/v1/billing/subscription/",
            HTTP_AUTHORIZATION=f"Bearer {access}",
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.data["data"]["queries_remaining"], 23)

    def test_subscription_endpoint_rejects_inactive_plan(self):
        user = self._signup_and_verify(email="inactiveplan@example.com", username="inactiveplan")
        subscription = Subscription.objects.get(user=user)
        subscription.plan.is_active = False
        subscription.plan.save(update_fields=["is_active"])

        access = self._login_access(user)
        response = self.client.get(
            "/api/v1/billing/subscription/",
            HTTP_AUTHORIZATION=f"Bearer {access}",
        )

        self.assertEqual(response.status_code, 409)
        self.assertEqual(response.data["code"], "PLAN_INACTIVE")

    def test_subscription_endpoint_rejects_expired_subscription(self):
        user = self._signup_and_verify(email="expired@example.com", username="expireduser")
        subscription = Subscription.objects.get(user=user)
        subscription.expires_at = timezone.now() - timedelta(days=1)
        subscription.save(update_fields=["expires_at"])

        access = self._login_access(user)
        response = self.client.get(
            "/api/v1/billing/subscription/",
            HTTP_AUTHORIZATION=f"Bearer {access}",
        )

        self.assertEqual(response.status_code, 409)
        self.assertEqual(response.data["code"], "SUBSCRIPTION_EXPIRED")

    def test_subscription_endpoint_rejects_missing_subscription(self):
        user = self._create_active_user(email="nosub@example.com", username="nosub")
        access = self._login_access(user)

        response = self.client.get(
            "/api/v1/billing/subscription/",
            HTTP_AUTHORIZATION=f"Bearer {access}",
        )

        self.assertEqual(response.status_code, 404)
        self.assertEqual(response.data["code"], "SUBSCRIPTION_NOT_FOUND")

    def test_subscription_endpoint_rejects_missing_token(self):
        response = self.client.get("/api/v1/billing/subscription/")

        self.assertEqual(response.status_code, 401)
        self.assertEqual(response.data["code"], "AUTHENTICATION_REQUIRED")

    def test_subscription_endpoint_rejects_malformed_header(self):
        response = self.client.get(
            "/api/v1/billing/subscription/",
            HTTP_AUTHORIZATION="Bearer",
        )

        self.assertEqual(response.status_code, 401)
        self.assertEqual(response.data["code"], "TOKEN_INVALID")

    def test_subscription_endpoint_rejects_invalid_token(self):
        response = self.client.get(
            "/api/v1/billing/subscription/",
            HTTP_AUTHORIZATION="Bearer not-a-jwt",
        )

        self.assertEqual(response.status_code, 401)
        self.assertEqual(response.data["code"], "TOKEN_INVALID")

    def test_subscription_endpoint_rejects_expired_token(self):
        user = self._create_active_user(email="expiredtoken@example.com", username="expiredtoken")
        token = AccessToken.for_user(user)
        token.set_exp(
            from_time=timezone.now() - timedelta(hours=2),
            lifetime=timedelta(hours=1),
        )

        response = self.client.get(
            "/api/v1/billing/subscription/",
            HTTP_AUTHORIZATION=f"Bearer {str(token)}",
        )

        self.assertEqual(response.status_code, 401)
        self.assertEqual(response.data["code"], "TOKEN_EXPIRED")

    def test_subscription_endpoint_rejects_refresh_token(self):
        user = self._create_active_user(email="refreshtoken@example.com", username="refreshtoken")
        refresh = RefreshToken.for_user(user)

        response = self.client.get(
            "/api/v1/billing/subscription/",
            HTTP_AUTHORIZATION=f"Bearer {str(refresh)}",
        )

        self.assertEqual(response.status_code, 401)
        self.assertEqual(response.data["code"], "TOKEN_INVALID")

    def test_subscription_endpoint_rejects_deleted_user(self):
        user = self._create_active_user(email="deleted@example.com", username="deleted")
        access = str(AccessToken.for_user(user))
        user.delete()

        response = self.client.get(
            "/api/v1/billing/subscription/",
            HTTP_AUTHORIZATION=f"Bearer {access}",
        )

        self.assertEqual(response.status_code, 401)
        self.assertEqual(response.data["code"], "TOKEN_INVALID")

    def test_subscription_endpoint_rejects_inactive_user(self):
        user = self._create_active_user(email="inactive@example.com", username="inactive", is_active=False)
        access = str(AccessToken.for_user(user))

        response = self.client.get(
            "/api/v1/billing/subscription/",
            HTTP_AUTHORIZATION=f"Bearer {access}",
        )

        self.assertEqual(response.status_code, 403)
        self.assertEqual(response.data["code"], "ACCOUNT_INACTIVE")

    def test_plans_database_failure_returns_503(self):
        with patch(
            "billing.controllers.billing_controller.get_active_plans",
            side_effect=DatabaseError("db down"),
        ):
            response = self.client.get("/api/v1/billing/plans/")

        self.assertEqual(response.status_code, 503)
        self.assertEqual(response.data["code"], "SERVICE_UNAVAILABLE")

    def test_subscription_database_failure_returns_503(self):
        user = self._signup_and_verify(email="dbfail@example.com", username="dbfail")
        access = self._login_access(user)

        with patch(
            "billing.controllers.billing_controller.get_subscription_for_user",
            side_effect=DatabaseError("db down"),
        ):
            response = self.client.get(
                "/api/v1/billing/subscription/",
                HTTP_AUTHORIZATION=f"Bearer {access}",
            )

        self.assertEqual(response.status_code, 503)
        self.assertEqual(response.data["code"], "SERVICE_UNAVAILABLE")

    def test_verify_email_free_plan_configuration_error_returns_500(self):
        self._signup(username="confuser", email="conf@example.com")
        otp = re.search(
            r"verification code is:\s*(\d{6})",
            mail.outbox[-1].body,
        ).group(1)

        with patch(
            "billing.services.subscription_service._get_free_plan",
            return_value=SimpleNamespace(is_active=False),
        ):
            response = self.client.post(
                "/api/v1/auth/verify-email/",
                {
                    "email": "conf@example.com",
                    "otp": otp,
                },
                format="json",
            )

        self.assertEqual(response.status_code, 500)
        self.assertEqual(response.data["code"], "BILLING_CONFIGURATION_ERROR")
