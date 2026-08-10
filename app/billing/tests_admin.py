from django.contrib.auth import get_user_model
from django.db import connection
from rest_framework_simplejwt.tokens import AccessToken
from rest_framework.test import APITestCase
from django.test.utils import CaptureQueriesContext

from billing.models import Plan, Subscription


User = get_user_model()


class AdminBillingAPITests(APITestCase):
    def _create_user(
        self,
        *,
        username="regularuser",
        email="regular@example.com",
        is_staff=False,
    ):
        return User.objects.create_user(
            username=username,
            email=email,
            password="StrongPass!234",
            first_name="Regular",
            last_name="User",
            is_active=True,
            is_staff=is_staff,
        )

    def _staff_access_headers(self, *, username="adminuser", email="admin@example.com"):
        user = self._create_user(
            username=username,
            email=email,
            is_staff=True,
        )
        return {
            "HTTP_AUTHORIZATION": f"Bearer {str(AccessToken.for_user(user))}",
        }

    def _user_access_headers(self, *, username="regularuser", email="regular@example.com"):
        user = self._create_user(
            username=username,
            email=email,
            is_staff=False,
        )
        return {
            "HTTP_AUTHORIZATION": f"Bearer {str(AccessToken.for_user(user))}",
        }

    def _create_plan(self, **kwargs):
        defaults = {
            "name": "Free",
            "code": "FREE",
            "price": "0.00",
            "billing_period": Plan.BillingPeriod.MONTHLY,
            "document_limit": 3,
            "query_limit": 30,
            "is_active": True,
        }
        defaults.update(kwargs)
        return Plan.objects.create(**defaults)

    def test_admin_can_create_valid_plan(self):
        response = self.client.post(
            "/api/v1/admin/billing/plans/",
            {
                "name": "Pro",
                "code": "pro",
                "price": "2500.00",
                "billing_period": "monthly",
                "document_limit": 0,
                "query_limit": 500,
                "is_active": True,
            },
            format="json",
            **self._staff_access_headers(),
        )

        self.assertEqual(response.status_code, 201)
        self.assertEqual(response.data["message"], "Plan created successfully.")
        self.assertEqual(response.data["data"]["code"], "PRO")
        self.assertTrue(Plan.objects.filter(code="PRO").exists())

    def test_admin_create_plan_requires_jwt(self):
        response = self.client.post(
            "/api/v1/admin/billing/plans/",
            {
                "name": "Pro",
                "code": "PRO",
                "price": "2500.00",
                "billing_period": "monthly",
                "document_limit": 0,
                "query_limit": 500,
                "is_active": True,
            },
            format="json",
        )

        self.assertEqual(response.status_code, 401)
        self.assertEqual(response.data["code"], "AUTHENTICATION_REQUIRED")

    def test_admin_create_plan_rejects_normal_user(self):
        response = self.client.post(
            "/api/v1/admin/billing/plans/",
            {
                "name": "Pro",
                "code": "PRO",
                "price": "2500.00",
                "billing_period": "monthly",
                "document_limit": 0,
                "query_limit": 500,
                "is_active": True,
            },
            format="json",
            **self._user_access_headers(),
        )

        self.assertEqual(response.status_code, 403)
        self.assertEqual(response.data["code"], "FORBIDDEN")

    def test_admin_create_plan_rejects_duplicate_code(self):
        self._create_plan(code="PRO", name="Existing Pro", price="100.00")

        response = self.client.post(
            "/api/v1/admin/billing/plans/",
            {
                "name": "Another Pro",
                "code": "pro",
                "price": "2500.00",
                "billing_period": "monthly",
                "document_limit": 0,
                "query_limit": 500,
                "is_active": True,
            },
            format="json",
            **self._staff_access_headers(email="duplicate-admin@example.com"),
        )

        self.assertEqual(response.status_code, 409)
        self.assertEqual(response.data["code"], "PLAN_CODE_EXISTS")

    def test_admin_create_plan_rejects_negative_price(self):
        response = self.client.post(
            "/api/v1/admin/billing/plans/",
            {
                "name": "Pro",
                "code": "PRO",
                "price": "-1.00",
                "billing_period": "monthly",
                "document_limit": 0,
                "query_limit": 500,
                "is_active": True,
            },
            format="json",
            **self._staff_access_headers(email="negative-price@example.com"),
        )

        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.data["code"], "INVALID_REQUEST")
        self.assertIn("price", response.data["errors"])

    def test_admin_create_plan_rejects_negative_document_limit(self):
        response = self.client.post(
            "/api/v1/admin/billing/plans/",
            {
                "name": "Pro",
                "code": "PRO",
                "price": "2500.00",
                "billing_period": "monthly",
                "document_limit": -1,
                "query_limit": 500,
                "is_active": True,
            },
            format="json",
            **self._staff_access_headers(email="negative-doc@example.com"),
        )

        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.data["code"], "INVALID_REQUEST")
        self.assertIn("document_limit", response.data["errors"])

    def test_admin_create_plan_rejects_negative_query_limit(self):
        response = self.client.post(
            "/api/v1/admin/billing/plans/",
            {
                "name": "Pro",
                "code": "PRO",
                "price": "2500.00",
                "billing_period": "monthly",
                "document_limit": 0,
                "query_limit": -1,
                "is_active": True,
            },
            format="json",
            **self._staff_access_headers(email="negative-query@example.com"),
        )

        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.data["code"], "INVALID_REQUEST")
        self.assertIn("query_limit", response.data["errors"])

    def test_admin_create_plan_rejects_invalid_billing_period(self):
        response = self.client.post(
            "/api/v1/admin/billing/plans/",
            {
                "name": "Pro",
                "code": "PRO",
                "price": "2500.00",
                "billing_period": "weekly",
                "document_limit": 0,
                "query_limit": 500,
                "is_active": True,
            },
            format="json",
            **self._staff_access_headers(email="invalid-period@example.com"),
        )

        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.data["code"], "INVALID_REQUEST")
        self.assertIn("billing_period", response.data["errors"])

    def test_admin_create_plan_rejects_unknown_fields(self):
        response = self.client.post(
            "/api/v1/admin/billing/plans/",
            {
                "name": "Pro",
                "code": "PRO",
                "price": "2500.00",
                "billing_period": "monthly",
                "document_limit": 0,
                "query_limit": 500,
                "is_active": True,
                "unexpected": "value",
            },
            format="json",
            **self._staff_access_headers(email="unknown-field@example.com"),
        )

        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.data["code"], "INVALID_REQUEST")
        self.assertIn("unexpected", response.data["errors"])

    def test_admin_patch_plan_updates_plan(self):
        plan = self._create_plan(
            name="Pro",
            code="PRO",
            price="2500.00",
            billing_period=Plan.BillingPeriod.MONTHLY,
            document_limit=0,
            query_limit=500,
            is_active=True,
        )

        response = self.client.patch(
            f"/api/v1/admin/billing/plans/{plan.id}/",
            {
                "name": "Pro Plus",
                "price": "2750.00",
                "billing_period": "yearly",
                "document_limit": 10,
                "query_limit": 750,
                "is_active": False,
            },
            format="json",
            **self._staff_access_headers(email="patch-admin@example.com"),
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.data["message"], "Plan updated successfully.")
        self.assertEqual(response.data["data"]["name"], "Pro Plus")
        self.assertEqual(response.data["data"]["billing_period"], "yearly")
        plan.refresh_from_db()
        self.assertEqual(plan.name, "Pro Plus")
        self.assertFalse(plan.is_active)

    def test_admin_patch_plan_requires_jwt(self):
        plan = self._create_plan(
            name="Pro",
            code="PRO",
            price="2500.00",
            billing_period=Plan.BillingPeriod.MONTHLY,
            document_limit=0,
            query_limit=500,
            is_active=True,
        )

        response = self.client.patch(
            f"/api/v1/admin/billing/plans/{plan.id}/",
            {"name": "Pro Plus"},
            format="json",
        )

        self.assertEqual(response.status_code, 401)
        self.assertEqual(response.data["code"], "AUTHENTICATION_REQUIRED")

    def test_admin_patch_plan_rejects_normal_user(self):
        plan = self._create_plan(
            name="Pro",
            code="PRO",
            price="2500.00",
            billing_period=Plan.BillingPeriod.MONTHLY,
            document_limit=0,
            query_limit=500,
            is_active=True,
        )

        response = self.client.patch(
            f"/api/v1/admin/billing/plans/{plan.id}/",
            {"name": "Pro Plus"},
            format="json",
            **self._user_access_headers(email="patch-user@example.com"),
        )

        self.assertEqual(response.status_code, 403)
        self.assertEqual(response.data["code"], "FORBIDDEN")

    def test_admin_patch_missing_plan_returns_404(self):
        response = self.client.patch(
            "/api/v1/admin/billing/plans/999999/",
            {"name": "Pro Plus"},
            format="json",
            **self._staff_access_headers(email="missing-plan@example.com"),
        )

        self.assertEqual(response.status_code, 404)
        self.assertEqual(response.data["code"], "PLAN_NOT_FOUND")

    def test_admin_patch_empty_body_rejected(self):
        plan = self._create_plan(
            name="Pro",
            code="PRO",
            price="2500.00",
            billing_period=Plan.BillingPeriod.MONTHLY,
            document_limit=0,
            query_limit=500,
            is_active=True,
        )

        response = self.client.patch(
            f"/api/v1/admin/billing/plans/{plan.id}/",
            {},
            format="json",
            **self._staff_access_headers(email="empty-patch@example.com"),
        )

        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.data["code"], "EMPTY_UPDATE")

    def test_admin_patch_code_is_immutable(self):
        plan = self._create_plan(
            name="Pro",
            code="PRO",
            price="2500.00",
            billing_period=Plan.BillingPeriod.MONTHLY,
            document_limit=0,
            query_limit=500,
            is_active=True,
        )

        response = self.client.patch(
            f"/api/v1/admin/billing/plans/{plan.id}/",
            {"code": "ENTERPRISE"},
            format="json",
            **self._staff_access_headers(email="immutable-code@example.com"),
        )

        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.data["code"], "IMMUTABLE_FIELD")

    def test_admin_patch_plan_keeps_subscription_relation_valid(self):
        plan = Plan.objects.get(code="FREE")
        user = self._create_user(
            username="subscriber",
            email="subscriber@example.com",
            is_staff=False,
        )
        subscription = Subscription.objects.create(
            user=user,
            plan=plan,
            status=Subscription.Status.ACTIVE,
            queries_used=0,
        )

        response = self.client.patch(
            f"/api/v1/admin/billing/plans/{plan.id}/",
            {"name": "Free Plus", "query_limit": 40},
            format="json",
            **self._staff_access_headers(email="relation-admin@example.com"),
        )

        self.assertEqual(response.status_code, 200)
        subscription.refresh_from_db()
        self.assertEqual(subscription.plan_id, plan.id)
        self.assertEqual(subscription.plan.name, "Free Plus")

    def test_admin_can_retrieve_subscriptions(self):
        plan = self._create_plan(
            code="PRO",
            name="Pro",
            price="2500.00",
            billing_period=Plan.BillingPeriod.MONTHLY,
            document_limit=0,
            query_limit=500,
            is_active=True,
        )
        user = self._create_user(username="ali", email="ali@example.com")
        Subscription.objects.create(
            user=user,
            plan=plan,
            status=Subscription.Status.ACTIVE,
            queries_used=100,
        )

        response = self.client.get(
            "/api/v1/admin/subscriptions/",
            **self._staff_access_headers(email="list-admin@example.com"),
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.data["count"], 1)
        self.assertEqual(len(response.data["results"]), 1)
        self.assertEqual(response.data["results"][0]["user"]["username"], "ali")
        self.assertEqual(response.data["results"][0]["plan"]["code"], "PRO")

    def test_admin_subscriptions_requires_jwt(self):
        response = self.client.get("/api/v1/admin/subscriptions/")

        self.assertEqual(response.status_code, 401)
        self.assertEqual(response.data["code"], "AUTHENTICATION_REQUIRED")

    def test_admin_subscriptions_rejects_normal_user(self):
        response = self.client.get(
            "/api/v1/admin/subscriptions/",
            **self._user_access_headers(email="user-list@example.com"),
        )

        self.assertEqual(response.status_code, 403)
        self.assertEqual(response.data["code"], "FORBIDDEN")

    def test_admin_subscriptions_returns_five_per_page(self):
        plan = self._create_plan(
            code="PRO",
            name="Pro",
            price="2500.00",
            billing_period=Plan.BillingPeriod.MONTHLY,
            document_limit=0,
            query_limit=500,
            is_active=True,
        )
        for index in range(6):
            user = self._create_user(
                username=f"user{index}",
                email=f"user{index}@example.com",
            )
            Subscription.objects.create(
                user=user,
                plan=plan,
                status=Subscription.Status.ACTIVE,
                queries_used=index,
            )

        response = self.client.get(
            "/api/v1/admin/subscriptions/",
            **self._staff_access_headers(email="page-admin@example.com"),
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.data["count"], 6)
        self.assertEqual(len(response.data["results"]), 5)
        self.assertIsNone(response.data["previous"])
        self.assertIsNotNone(response.data["next"])

    def test_admin_subscriptions_page_two_works(self):
        plan = self._create_plan(
            code="PRO",
            name="Pro",
            price="2500.00",
            billing_period=Plan.BillingPeriod.MONTHLY,
            document_limit=0,
            query_limit=500,
            is_active=True,
        )
        created_ids = []
        for index in range(6):
            user = self._create_user(
                username=f"paged{index}",
                email=f"paged{index}@example.com",
            )
            subscription = Subscription.objects.create(
                user=user,
                plan=plan,
                status=Subscription.Status.ACTIVE,
                queries_used=index,
            )
            created_ids.append(subscription.id)

        response = self.client.get(
            "/api/v1/admin/subscriptions/?page=2",
            **self._staff_access_headers(email="page2-admin@example.com"),
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.data["count"], 6)
        self.assertEqual(len(response.data["results"]), 1)
        self.assertEqual(response.data["results"][0]["id"], created_ids[0])

    def test_admin_subscriptions_newest_first(self):
        plan = self._create_plan(
            code="PRO",
            name="Pro",
            price="2500.00",
            billing_period=Plan.BillingPeriod.MONTHLY,
            document_limit=0,
            query_limit=500,
            is_active=True,
        )
        old_user = self._create_user(username="old", email="old@example.com")
        old_sub = Subscription.objects.create(
            user=old_user,
            plan=plan,
            status=Subscription.Status.ACTIVE,
            queries_used=1,
        )
        new_user = self._create_user(username="new", email="new@example.com")
        new_sub = Subscription.objects.create(
            user=new_user,
            plan=plan,
            status=Subscription.Status.ACTIVE,
            queries_used=2,
        )

        response = self.client.get(
            "/api/v1/admin/subscriptions/",
            **self._staff_access_headers(email="order-admin@example.com"),
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.data["results"][0]["id"], new_sub.id)
        self.assertEqual(response.data["results"][1]["id"], old_sub.id)

    def test_admin_subscriptions_empty_list_returns_200(self):
        response = self.client.get(
            "/api/v1/admin/subscriptions/",
            **self._staff_access_headers(email="empty-admin@example.com"),
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.data["count"], 0)
        self.assertEqual(response.data["results"], [])

    def test_admin_subscriptions_status_filter_works(self):
        plan = self._create_plan(
            code="PRO",
            name="Pro",
            price="2500.00",
            billing_period=Plan.BillingPeriod.MONTHLY,
            document_limit=0,
            query_limit=500,
            is_active=True,
        )
        active_user = self._create_user(username="active", email="active@example.com")
        canceled_user = self._create_user(username="canceled", email="canceled@example.com")
        Subscription.objects.create(
            user=active_user,
            plan=plan,
            status=Subscription.Status.ACTIVE,
        )
        Subscription.objects.create(
            user=canceled_user,
            plan=plan,
            status=Subscription.Status.CANCELED,
        )

        response = self.client.get(
            "/api/v1/admin/subscriptions/?status=active",
            **self._staff_access_headers(email="status-admin@example.com"),
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.data["count"], 1)
        self.assertEqual(response.data["results"][0]["status"], "active")

    def test_admin_subscriptions_plan_filter_is_case_insensitive(self):
        plan = self._create_plan(
            code="PRO",
            name="Pro",
            price="2500.00",
            billing_period=Plan.BillingPeriod.MONTHLY,
            document_limit=0,
            query_limit=500,
            is_active=True,
        )
        user = self._create_user(username="planuser", email="planuser@example.com")
        Subscription.objects.create(
            user=user,
            plan=plan,
            status=Subscription.Status.ACTIVE,
        )

        response = self.client.get(
            "/api/v1/admin/subscriptions/?plan=pro",
            **self._staff_access_headers(email="plan-admin@example.com"),
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.data["count"], 1)
        self.assertEqual(response.data["results"][0]["plan"]["code"], "PRO")

    def test_admin_subscriptions_username_search_works(self):
        plan = self._create_plan(
            code="PRO",
            name="Pro",
            price="2500.00",
            billing_period=Plan.BillingPeriod.MONTHLY,
            document_limit=0,
            query_limit=500,
            is_active=True,
        )
        ali_user = self._create_user(username="ali", email="ali@example.com")
        other_user = self._create_user(username="other", email="other@example.com")
        Subscription.objects.create(
            user=ali_user,
            plan=plan,
            status=Subscription.Status.ACTIVE,
        )
        Subscription.objects.create(
            user=other_user,
            plan=plan,
            status=Subscription.Status.ACTIVE,
        )

        response = self.client.get(
            "/api/v1/admin/subscriptions/?search=ali",
            **self._staff_access_headers(email="search-admin@example.com"),
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.data["count"], 1)
        self.assertEqual(response.data["results"][0]["user"]["username"], "ali")

    def test_admin_subscriptions_email_search_works(self):
        plan = self._create_plan(
            code="PRO",
            name="Pro",
            price="2500.00",
            billing_period=Plan.BillingPeriod.MONTHLY,
            document_limit=0,
            query_limit=500,
            is_active=True,
        )
        email_user = self._create_user(username="emailuser", email="ali@example.com")
        Subscription.objects.create(
            user=email_user,
            plan=plan,
            status=Subscription.Status.ACTIVE,
        )

        response = self.client.get(
            "/api/v1/admin/subscriptions/?search=ALI@EXAMPLE.COM",
            **self._staff_access_headers(email="email-search-admin@example.com"),
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.data["count"], 1)
        self.assertEqual(response.data["results"][0]["user"]["email"], "ali@example.com")

    def test_admin_subscriptions_combined_filters_work(self):
        pro_plan = self._create_plan(
            code="PRO",
            name="Pro",
            price="2500.00",
            billing_period=Plan.BillingPeriod.MONTHLY,
            document_limit=0,
            query_limit=500,
            is_active=True,
        )
        basic_plan = self._create_plan(
            code="BASIC",
            name="Basic",
            price="1000.00",
            billing_period=Plan.BillingPeriod.MONTHLY,
            document_limit=1,
            query_limit=100,
            is_active=True,
        )
        ali_user = self._create_user(username="ali", email="ali@example.com")
        bob_user = self._create_user(username="bob", email="bob@example.com")
        Subscription.objects.create(
            user=ali_user,
            plan=pro_plan,
            status=Subscription.Status.ACTIVE,
        )
        Subscription.objects.create(
            user=bob_user,
            plan=basic_plan,
            status=Subscription.Status.ACTIVE,
        )

        response = self.client.get(
            "/api/v1/admin/subscriptions/?status=active&plan=PRO&search=ali",
            **self._staff_access_headers(email="combined-admin@example.com"),
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.data["count"], 1)
        self.assertEqual(response.data["results"][0]["user"]["username"], "ali")
        self.assertEqual(response.data["results"][0]["plan"]["code"], "PRO")

    def test_admin_subscriptions_invalid_status_rejected(self):
        response = self.client.get(
            "/api/v1/admin/subscriptions/?status=random",
            **self._staff_access_headers(email="invalid-status-admin@example.com"),
        )

        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.data["code"], "INVALID_FILTER")

    def test_admin_subscriptions_includes_canceled_expired_and_inactive_plan(self):
        inactive_plan = self._create_plan(
            code="HIDDEN",
            name="Hidden",
            price="3000.00",
            billing_period=Plan.BillingPeriod.YEARLY,
            document_limit=2,
            query_limit=200,
            is_active=False,
        )
        canceled_user = self._create_user(username="canceled", email="canceled@example.com")
        expired_user = self._create_user(username="expired", email="expired@example.com")
        hidden_user = self._create_user(username="hidden", email="hidden@example.com")
        Subscription.objects.create(
            user=canceled_user,
            plan=Plan.objects.get(code="FREE"),
            status=Subscription.Status.CANCELED,
        )
        Subscription.objects.create(
            user=expired_user,
            plan=Plan.objects.get(code="FREE"),
            status=Subscription.Status.EXPIRED,
        )
        Subscription.objects.create(
            user=hidden_user,
            plan=inactive_plan,
            status=Subscription.Status.ACTIVE,
        )

        response = self.client.get(
            "/api/v1/admin/subscriptions/",
            **self._staff_access_headers(email="status-admin2@example.com"),
        )

        self.assertEqual(response.status_code, 200)
        statuses = [item["status"] for item in response.data["results"]]
        plan_codes = [item["plan"]["code"] for item in response.data["results"]]
        self.assertIn("canceled", statuses)
        self.assertIn("expired", statuses)
        self.assertIn("HIDDEN", plan_codes)

    def test_admin_subscriptions_queries_remaining_never_negative(self):
        plan = self._create_plan(
            code="PRO",
            name="Pro",
            price="2500.00",
            billing_period=Plan.BillingPeriod.MONTHLY,
            document_limit=0,
            query_limit=5,
            is_active=True,
        )
        user = self._create_user(username="over", email="over@example.com")
        Subscription.objects.create(
            user=user,
            plan=plan,
            status=Subscription.Status.ACTIVE,
            queries_used=12,
        )

        response = self.client.get(
            "/api/v1/admin/subscriptions/",
            **self._staff_access_headers(email="remaining-admin@example.com"),
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.data["results"][0]["queries_remaining"], 0)

    def test_admin_subscriptions_avoids_n_plus_one_queries(self):
        plan = self._create_plan(
            code="PRO",
            name="Pro",
            price="2500.00",
            billing_period=Plan.BillingPeriod.MONTHLY,
            document_limit=0,
            query_limit=500,
            is_active=True,
        )
        for index in range(5):
            user = self._create_user(
                username=f"nplus{index}",
                email=f"nplus{index}@example.com",
            )
            Subscription.objects.create(
                user=user,
                plan=plan,
                status=Subscription.Status.ACTIVE,
            )

        headers = self._staff_access_headers(email="nplus-admin@example.com")
        with CaptureQueriesContext(connection) as context:
            response = self.client.get(
                "/api/v1/admin/subscriptions/",
                **headers,
            )

        self.assertEqual(response.status_code, 200)
        self.assertLessEqual(len(context.captured_queries), 4)

    def test_admin_subscriptions_invalid_page_returns_404(self):
        response = self.client.get(
            "/api/v1/admin/subscriptions/?page=999",
            **self._staff_access_headers(email="bad-page-admin@example.com"),
        )

        self.assertEqual(response.status_code, 404)
        self.assertEqual(response.data["code"], "PAGE_NOT_FOUND")
