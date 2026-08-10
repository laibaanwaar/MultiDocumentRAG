from __future__ import annotations

from datetime import timedelta
from uuid import uuid4

from django.contrib.auth import get_user_model
from django.test import override_settings
from django.utils import timezone
from rest_framework_simplejwt.tokens import AccessToken
from rest_framework.test import APITestCase

from billing.models import Plan, Subscription
from rag_api.models import QueryHistory


User = get_user_model()


@override_settings(
    EMAIL_BACKEND="django.core.mail.backends.locmem.EmailBackend",
)
class AdminDashboardAPITests(APITestCase):
    def _create_user(
        self,
        *,
        username: str,
        email: str,
        is_active: bool = True,
        is_staff: bool = False,
        is_superuser: bool = False,
        joined_offset_minutes: int = 0,
    ):
        user = User.objects.create_user(
            username=username,
            email=email,
            password="StrongPass!234",
            first_name="Admin",
            last_name="User",
            is_active=is_active,
            is_staff=is_staff,
            is_superuser=is_superuser,
        )
        if joined_offset_minutes:
            User.objects.filter(pk=user.pk).update(
                date_joined=timezone.now() + timedelta(minutes=joined_offset_minutes)
            )
            user.refresh_from_db()
        return user

    def _staff_headers(self, user):
        return {
            "HTTP_AUTHORIZATION": f"Bearer {str(AccessToken.for_user(user))}",
        }

    def _create_plan(self, **kwargs):
        defaults = {
            "name": "Free",
            "code": f"PLAN-{uuid4().hex[:6].upper()}",
            "price": "0.00",
            "billing_period": Plan.BillingPeriod.MONTHLY,
            "document_limit": 3,
            "query_limit": 30,
            "is_active": True,
        }
        defaults.update(kwargs)
        return Plan.objects.create(**defaults)

    def _create_subscription(self, *, user, plan, status=Subscription.Status.ACTIVE, queries_used=0):
        return Subscription.objects.create(
            user=user,
            plan=plan,
            status=status,
            queries_used=queries_used,
        )

    def test_admin_dashboard_requires_jwt(self):
        response = self.client.get("/api/v1/admin/dashboard/")

        self.assertEqual(response.status_code, 401)
        self.assertEqual(response.data["code"], "AUTHENTICATION_REQUIRED")

    def test_admin_dashboard_rejects_normal_user(self):
        user = self._create_user(username="normal", email="normal@example.com")

        response = self.client.get(
            "/api/v1/admin/dashboard/",
            **self._staff_headers(user),
        )

        self.assertEqual(response.status_code, 403)
        self.assertEqual(response.data["code"], "FORBIDDEN")

    def test_admin_dashboard_zero_data_response(self):
        User.objects.all().delete()
        Plan.objects.all().delete()

        staff = self._create_user(
            username="admin",
            email="admin@example.com",
            is_staff=True,
            is_superuser=True,
        )
        response = self.client.get(
            "/api/v1/admin/dashboard/",
            **self._staff_headers(staff),
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.data["data"]["summary"], {
            "total_users": 0,
            "paid_subscribers": 0,
            "active_subscribers": 0,
        })
        self.assertEqual(response.data["data"]["recent_users"], [])
        self.assertEqual(response.data["data"]["subscription_overview"], [])

    def test_admin_dashboard_counts_recent_users_and_plan_overview(self):
        free = self._create_plan(code="FREE", name="Free", price="0.00", is_active=True)
        pro = self._create_plan(code="PRO", name="Pro", price="49.99", is_active=True)
        hidden = self._create_plan(code="HIDDEN", name="Hidden", price="99.99", is_active=False)

        staff = self._create_user(
            username="admin",
            email="admin@example.com",
            is_staff=True,
            is_superuser=True,
            joined_offset_minutes=30,
        )

        users = []
        for index in range(6):
            user = self._create_user(
                username=f"user{index}",
                email=f"user{index}@example.com",
                joined_offset_minutes=index,
            )
            users.append(user)

        self._create_subscription(user=users[0], plan=free, status=Subscription.Status.ACTIVE)
        self._create_subscription(user=users[1], plan=pro, status=Subscription.Status.ACTIVE)
        self._create_subscription(user=users[2], plan=hidden, status=Subscription.Status.ACTIVE)
        self._create_subscription(user=users[3], plan=pro, status=Subscription.Status.CANCELED)
        self._create_subscription(user=users[4], plan=free, status=Subscription.Status.ACTIVE)

        response = self.client.get(
            "/api/v1/admin/dashboard/",
            **self._staff_headers(staff),
        )

        self.assertEqual(response.status_code, 200)
        summary = response.data["data"]["summary"]
        self.assertEqual(summary["total_users"], 6)
        self.assertEqual(summary["paid_subscribers"], 2)
        self.assertEqual(summary["active_subscribers"], 4)

        recent_users = response.data["data"]["recent_users"]
        self.assertEqual(len(recent_users), 5)
        self.assertNotIn("admin", [item["username"] for item in recent_users])

        overview = {item["code"]: item for item in response.data["data"]["subscription_overview"]}
        self.assertEqual(overview["FREE"]["subscribers_count"], 2)
        self.assertEqual(overview["PRO"]["subscribers_count"], 2)
        self.assertFalse(overview["HIDDEN"]["is_active"])

    def test_dashboard_user_without_subscription_does_not_crash(self):
        staff = self._create_user(
            username="admin",
            email="admin@example.com",
            is_staff=True,
            is_superuser=True,
        )
        self._create_plan(code="FREE", name="Free", price="0.00", is_active=True)
        self._create_user(username="nosub", email="nosub@example.com")

        response = self.client.get(
            "/api/v1/admin/dashboard/",
            **self._staff_headers(staff),
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(len(response.data["data"]["recent_users"]), 1)
        self.assertIsNone(response.data["data"]["recent_users"][0]["subscription"])


@override_settings(
    EMAIL_BACKEND="django.core.mail.backends.locmem.EmailBackend",
)
class AdminUsersAPITests(APITestCase):
    def _create_user(
        self,
        *,
        username: str,
        email: str,
        is_active: bool = True,
        is_staff: bool = False,
        is_superuser: bool = False,
        joined_offset_minutes: int = 0,
    ):
        user = User.objects.create_user(
            username=username,
            email=email,
            password="StrongPass!234",
            first_name="Admin",
            last_name="User",
            is_active=is_active,
            is_staff=is_staff,
            is_superuser=is_superuser,
        )
        if joined_offset_minutes:
            User.objects.filter(pk=user.pk).update(
                date_joined=timezone.now() + timedelta(minutes=joined_offset_minutes)
            )
            user.refresh_from_db()
        return user

    def _staff_headers(self, user):
        return {
            "HTTP_AUTHORIZATION": f"Bearer {str(AccessToken.for_user(user))}",
        }

    def _create_plan(self, **kwargs):
        defaults = {
            "name": "Free",
            "code": f"PLAN-{uuid4().hex[:6].upper()}",
            "price": "0.00",
            "billing_period": Plan.BillingPeriod.MONTHLY,
            "document_limit": 3,
            "query_limit": 30,
            "is_active": True,
        }
        defaults.update(kwargs)
        return Plan.objects.create(**defaults)

    def _create_subscription(self, *, user, plan, status=Subscription.Status.ACTIVE, queries_used=0):
        return Subscription.objects.create(
            user=user,
            plan=plan,
            status=status,
            queries_used=queries_used,
        )

    def test_users_list_requires_jwt(self):
        response = self.client.get("/api/v1/admin/users/")

        self.assertEqual(response.status_code, 401)
        self.assertEqual(response.data["code"], "AUTHENTICATION_REQUIRED")

    def test_users_list_rejects_normal_user(self):
        user = self._create_user(username="normal", email="normal@example.com")

        response = self.client.get(
            "/api/v1/admin/users/",
            **self._staff_headers(user),
        )

        self.assertEqual(response.status_code, 403)
        self.assertEqual(response.data["code"], "FORBIDDEN")

    def test_users_list_returns_five_per_page(self):
        staff = self._create_user(username="admin", email="admin@example.com", is_staff=True, is_superuser=True)
        plan = self._create_plan(code="FREE", name="Free", price="0.00")
        for index in range(6):
            user = self._create_user(
                username=f"user{index}",
                email=f"user{index}@example.com",
                joined_offset_minutes=index,
            )
            self._create_subscription(user=user, plan=plan, queries_used=index)

        response = self.client.get(
            "/api/v1/admin/users/",
            **self._staff_headers(staff),
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.data["count"], 6)
        self.assertEqual(len(response.data["results"]), 5)
        self.assertIsNotNone(response.data["next"])
        self.assertIsNone(response.data["previous"])

    def test_users_list_page_two(self):
        staff = self._create_user(username="admin", email="admin@example.com", is_staff=True, is_superuser=True)
        plan = self._create_plan(code="FREE", name="Free", price="0.00")
        created_ids = []
        for index in range(6):
            user = self._create_user(
                username=f"user{index}",
                email=f"user{index}@example.com",
                joined_offset_minutes=index,
            )
            self._create_subscription(user=user, plan=plan, queries_used=index)
            created_ids.append(user.id)

        response = self.client.get(
            "/api/v1/admin/users/?page=2",
            **self._staff_headers(staff),
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.data["count"], 6)
        self.assertEqual(len(response.data["results"]), 1)
        self.assertEqual(response.data["results"][0]["id"], created_ids[0])

    def test_users_list_newest_first_deterministic(self):
        staff = self._create_user(username="admin", email="admin@example.com", is_staff=True, is_superuser=True)
        plan = self._create_plan(code="FREE", name="Free", price="0.00")
        first = self._create_user(username="alpha", email="alpha@example.com", joined_offset_minutes=-3)
        second = self._create_user(username="beta", email="beta@example.com", joined_offset_minutes=-2)
        third = self._create_user(username="gamma", email="gamma@example.com", joined_offset_minutes=-1)
        for user in [first, second, third]:
            self._create_subscription(user=user, plan=plan)

        response = self.client.get(
            "/api/v1/admin/users/",
            **self._staff_headers(staff),
        )

        self.assertEqual(
            [item["username"] for item in response.data["results"]],
            ["gamma", "beta", "alpha"],
        )

    def test_users_list_search_username(self):
        staff = self._create_user(username="admin", email="admin@example.com", is_staff=True, is_superuser=True)
        plan = self._create_plan(code="FREE", name="Free", price="0.00")
        ali = self._create_user(username="ali", email="ali@example.com")
        other = self._create_user(username="other", email="other@example.com")
        self._create_subscription(user=ali, plan=plan)
        self._create_subscription(user=other, plan=plan)

        response = self.client.get(
            "/api/v1/admin/users/?search=ALI",
            **self._staff_headers(staff),
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.data["count"], 1)
        self.assertEqual(response.data["results"][0]["username"], "ali")

    def test_users_list_search_email(self):
        staff = self._create_user(username="admin", email="admin@example.com", is_staff=True, is_superuser=True)
        plan = self._create_plan(code="FREE", name="Free", price="0.00")
        user = self._create_user(username="emailuser", email="ali@example.com")
        self._create_subscription(user=user, plan=plan)

        response = self.client.get(
            "/api/v1/admin/users/?search=ALI@EXAMPLE.COM",
            **self._staff_headers(staff),
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.data["count"], 1)
        self.assertEqual(response.data["results"][0]["email"], "ali@example.com")

    def test_users_list_plan_filter(self):
        staff = self._create_user(username="admin", email="admin@example.com", is_staff=True, is_superuser=True)
        free = self._create_plan(code="FREE", name="Free", price="0.00")
        pro = self._create_plan(code="PRO", name="Pro", price="49.99")
        user_free = self._create_user(username="free", email="free@example.com")
        user_pro = self._create_user(username="pro", email="pro@example.com")
        self._create_subscription(user=user_free, plan=free)
        self._create_subscription(user=user_pro, plan=pro)

        response = self.client.get(
            "/api/v1/admin/users/?plan=pro",
            **self._staff_headers(staff),
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.data["count"], 1)
        self.assertEqual(response.data["results"][0]["username"], "pro")

    def test_users_list_status_filter_active_and_inactive(self):
        staff = self._create_user(username="admin", email="admin@example.com", is_staff=True, is_superuser=True)
        active = self._create_user(username="active", email="active@example.com", is_active=True)
        inactive = self._create_user(username="inactive", email="inactive@example.com", is_active=False)

        response = self.client.get(
            "/api/v1/admin/users/?status=inactive",
            **self._staff_headers(staff),
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.data["count"], 1)
        self.assertEqual(response.data["results"][0]["username"], "inactive")

        response = self.client.get(
            "/api/v1/admin/users/?status=active",
            **self._staff_headers(staff),
        )

        self.assertEqual(response.status_code, 200)
        self.assertTrue(any(item["username"] == "active" for item in response.data["results"]))

    def test_users_list_combined_filters(self):
        staff = self._create_user(username="admin", email="admin@example.com", is_staff=True, is_superuser=True)
        pro = self._create_plan(code="PRO", name="Pro", price="49.99")
        free = self._create_plan(code="FREE", name="Free", price="0.00")
        target = self._create_user(username="ali", email="ali@example.com", is_active=True)
        other = self._create_user(username="bob", email="bob@example.com", is_active=True)
        self._create_subscription(user=target, plan=pro)
        self._create_subscription(user=other, plan=free)

        response = self.client.get(
            "/api/v1/admin/users/?search=ali&plan=PRO&status=active",
            **self._staff_headers(staff),
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.data["count"], 1)
        self.assertEqual(response.data["results"][0]["username"], "ali")

    def test_users_list_no_results(self):
        staff = self._create_user(username="admin", email="admin@example.com", is_staff=True, is_superuser=True)
        self._create_user(username="alpha", email="alpha@example.com")

        response = self.client.get(
            "/api/v1/admin/users/?search=missing",
            **self._staff_headers(staff),
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.data["count"], 0)
        self.assertEqual(response.data["results"], [])

    def test_users_list_invalid_filter(self):
        staff = self._create_user(username="admin", email="admin@example.com", is_staff=True, is_superuser=True)

        response = self.client.get(
            "/api/v1/admin/users/?status=broken",
            **self._staff_headers(staff),
        )

        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.data["code"], "INVALID_FILTER")

    def test_users_list_invalid_page(self):
        staff = self._create_user(username="admin", email="admin@example.com", is_staff=True, is_superuser=True)

        response = self.client.get(
            "/api/v1/admin/users/?page=999",
            **self._staff_headers(staff),
        )

        self.assertEqual(response.status_code, 404)
        self.assertEqual(response.data["code"], "PAGE_NOT_FOUND")

    def test_users_without_subscription_return_null_subscription(self):
        staff = self._create_user(username="admin", email="admin@example.com", is_staff=True, is_superuser=True)
        self._create_user(username="nosub", email="nosub@example.com")

        response = self.client.get(
            "/api/v1/admin/users/",
            **self._staff_headers(staff),
        )

        self.assertIsNone(response.data["results"][0]["subscription"])

    def test_queries_remaining_never_negative(self):
        staff = self._create_user(username="admin", email="admin@example.com", is_staff=True, is_superuser=True)
        plan = self._create_plan(code="LOW", name="Low", price="0.00", query_limit=5)
        user = self._create_user(username="over", email="over@example.com")
        self._create_subscription(user=user, plan=plan, queries_used=12)

        response = self.client.get(
            "/api/v1/admin/users/",
            **self._staff_headers(staff),
        )

        self.assertEqual(response.data["results"][0]["subscription"]["queries_remaining"], 0)


@override_settings(
    EMAIL_BACKEND="django.core.mail.backends.locmem.EmailBackend",
)
class AdminUserDetailAPITests(APITestCase):
    def _create_user(
        self,
        *,
        username: str,
        email: str,
        is_active: bool = True,
        is_staff: bool = False,
        is_superuser: bool = False,
    ):
        return User.objects.create_user(
            username=username,
            email=email,
            password="StrongPass!234",
            first_name="Admin",
            last_name="User",
            is_active=is_active,
            is_staff=is_staff,
            is_superuser=is_superuser,
        )

    def _staff_headers(self, user):
        return {
            "HTTP_AUTHORIZATION": f"Bearer {str(AccessToken.for_user(user))}",
        }

    def _create_plan(self, **kwargs):
        defaults = {
            "name": "Free",
            "code": f"PLAN-{uuid4().hex[:6].upper()}",
            "price": "0.00",
            "billing_period": Plan.BillingPeriod.MONTHLY,
            "document_limit": 3,
            "query_limit": 30,
            "is_active": True,
        }
        defaults.update(kwargs)
        return Plan.objects.create(**defaults)

    def _create_subscription(self, *, user, plan, status=Subscription.Status.ACTIVE, queries_used=0):
        return Subscription.objects.create(
            user=user,
            plan=plan,
            status=status,
            queries_used=queries_used,
        )

    def _create_history(self, *, user, question, answer, minutes_ago=0):
        history = QueryHistory.objects.create(
            user=user,
            question=question,
            answer=answer,
            sources=[],
        )
        if minutes_ago:
            QueryHistory.objects.filter(pk=history.pk).update(
                created_at=timezone.now() - timedelta(minutes=minutes_ago)
            )
            history.refresh_from_db()
        return history

    def test_user_detail_requires_jwt(self):
        response = self.client.get("/api/v1/admin/users/1/")

        self.assertEqual(response.status_code, 401)
        self.assertEqual(response.data["code"], "AUTHENTICATION_REQUIRED")

    def test_user_detail_rejects_normal_user(self):
        staff = self._create_user(username="admin", email="admin@example.com", is_staff=True, is_superuser=True)
        response = self.client.get(
            "/api/v1/admin/users/1/",
            **self._staff_headers(staff),
        )

        self.assertEqual(response.status_code, 404)
        self.assertEqual(response.data["code"], "USER_NOT_FOUND")

    def test_user_detail_missing_user_returns_404(self):
        staff = self._create_user(username="admin", email="admin@example.com", is_staff=True, is_superuser=True)
        response = self.client.get(
            "/api/v1/admin/users/999999/",
            **self._staff_headers(staff),
        )

        self.assertEqual(response.status_code, 404)
        self.assertEqual(response.data["code"], "USER_NOT_FOUND")

    def test_user_detail_without_subscription_and_history(self):
        staff = self._create_user(username="admin", email="admin@example.com", is_staff=True, is_superuser=True)
        user = self._create_user(username="plain", email="plain@example.com")

        response = self.client.get(
            f"/api/v1/admin/users/{user.id}/",
            **self._staff_headers(staff),
        )

        self.assertEqual(response.status_code, 200)
        self.assertIsNone(response.data["data"]["subscription"])
        self.assertEqual(response.data["data"]["rag_usage"]["total_queries"], 0)
        self.assertEqual(response.data["data"]["rag_usage"]["recent_queries"], [])

    def test_user_detail_returns_subscription_and_rag_usage(self):
        staff = self._create_user(username="admin", email="admin@example.com", is_staff=True, is_superuser=True)
        plan = self._create_plan(code="PRO", name="Pro", price="49.99")
        user = self._create_user(username="detail", email="detail@example.com")
        subscription = self._create_subscription(user=user, plan=plan, queries_used=3)

        for index in range(6):
            self._create_history(
                user=user,
                question=f"Question {index + 1}",
                answer="A very long answer " + ("x" * 200),
                minutes_ago=index,
            )

        response = self.client.get(
            f"/api/v1/admin/users/{user.id}/",
            **self._staff_headers(staff),
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.data["data"]["user"]["username"], "detail")
        self.assertEqual(response.data["data"]["subscription"]["plan_code"], "PRO")
        self.assertEqual(response.data["data"]["subscription"]["queries_remaining"], 27)
        self.assertEqual(response.data["data"]["rag_usage"]["total_queries"], 6)
        self.assertEqual(len(response.data["data"]["rag_usage"]["recent_queries"]), 5)
        self.assertTrue(
            response.data["data"]["rag_usage"]["recent_queries"][0]["question"].startswith("Question 1")
        )
        self.assertLess(
            len(response.data["data"]["rag_usage"]["recent_queries"][0]["answer_preview"]),
            len("A very long answer " + ("x" * 200)),
        )

    def test_staff_user_detail_returns_404(self):
        staff = self._create_user(username="admin", email="admin@example.com", is_staff=True, is_superuser=True)
        target = self._create_user(username="targetstaff", email="targetstaff@example.com", is_staff=True)

        response = self.client.get(
            f"/api/v1/admin/users/{target.id}/",
            **self._staff_headers(staff),
        )

        self.assertEqual(response.status_code, 404)
        self.assertEqual(response.data["code"], "USER_NOT_FOUND")

    def test_user_detail_no_sensitive_fields_exposed(self):
        staff = self._create_user(username="admin", email="admin@example.com", is_staff=True, is_superuser=True)
        user = self._create_user(username="safe", email="safe@example.com")

        response = self.client.get(
            f"/api/v1/admin/users/{user.id}/",
            **self._staff_headers(staff),
        )

        payload = response.data["data"]
        self.assertNotIn("password", payload["user"])
        self.assertNotIn("refresh", response.data)
        self.assertNotIn("otp", response.data)
