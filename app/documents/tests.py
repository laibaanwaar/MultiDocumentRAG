from django.contrib.auth import get_user_model
from django.test import override_settings
from rest_framework_simplejwt.tokens import AccessToken
from rest_framework.test import APITestCase
from unittest.mock import patch

from documents.models import DocumentCategory


User = get_user_model()


@override_settings(
    EMAIL_BACKEND="django.core.mail.backends.locmem.EmailBackend",
)
class DocumentCategoryAPITests(APITestCase):
    def _create_user(self, *, username="user", email="user@example.com", is_staff=False):
        return User.objects.create_user(
            username=username,
            email=email,
            password="StrongPass!234",
            first_name="Doc",
            last_name="User",
            is_active=True,
            is_staff=is_staff,
        )

    def _staff_headers(self, *, username="admin", email="admin@example.com"):
        user = self._create_user(username=username, email=email, is_staff=True)
        return {
            "HTTP_AUTHORIZATION": f"Bearer {str(AccessToken.for_user(user))}",
        }

    def _user_headers(self, *, username="user", email="user@example.com"):
        user = self._create_user(username=username, email=email, is_staff=False)
        return {
            "HTTP_AUTHORIZATION": f"Bearer {str(AccessToken.for_user(user))}",
        }

    def _create_category(self, **kwargs):
        defaults = {
            "name": "Pakistan Penal Code",
            "code": "PPC",
            "description": "Pakistan Penal Code 1860",
            "is_active": True,
        }
        defaults.update(kwargs)
        return DocumentCategory.objects.create(**defaults)

    def test_admin_successfully_creates_category(self):
        response = self.client.post(
            "/api/v1/document-categories/",
            {
                "name": "Pakistan Penal Code",
                "code": "PPC",
                "description": "Pakistan Penal Code 1860",
                "is_active": True,
            },
            format="json",
            **self._staff_headers(),
        )

        self.assertEqual(response.status_code, 201)
        self.assertEqual(response.data["message"], "Document category created successfully.")
        self.assertEqual(response.data["data"]["code"], "PPC")
        self.assertTrue(DocumentCategory.objects.filter(code="PPC").exists())

    def test_normal_user_gets_403_on_create(self):
        response = self.client.post(
            "/api/v1/document-categories/",
            {
                "name": "Pakistan Penal Code",
                "code": "PPC",
                "description": "Pakistan Penal Code 1860",
                "is_active": True,
            },
            format="json",
            **self._user_headers(),
        )

        self.assertEqual(response.status_code, 403)
        self.assertEqual(response.data["code"], "FORBIDDEN")

    def test_unauthenticated_user_gets_401_on_create(self):
        response = self.client.post(
            "/api/v1/document-categories/",
            {
                "name": "Pakistan Penal Code",
                "code": "PPC",
                "description": "Pakistan Penal Code 1860",
                "is_active": True,
            },
            format="json",
        )

        self.assertEqual(response.status_code, 401)
        self.assertEqual(response.data["code"], "AUTHENTICATION_REQUIRED")

    def test_lowercase_code_becomes_uppercase(self):
        response = self.client.post(
            "/api/v1/document-categories/",
            {
                "name": "Anti-Money Laundering Act",
                "code": "amla",
                "description": "Anti-Money Laundering Act, 2010",
                "is_active": True,
            },
            format="json",
            **self._staff_headers(email="lowercase@example.com"),
        )

        self.assertEqual(response.status_code, 201)
        self.assertEqual(response.data["data"]["code"], "AMLA")

    def test_duplicate_code_rejected(self):
        self._create_category(code="PPC", name="Pakistan Penal Code")

        response = self.client.post(
            "/api/v1/document-categories/",
            {
                "name": "Another Law",
                "code": "ppc",
                "description": "",
                "is_active": True,
            },
            format="json",
            **self._staff_headers(email="duplicate-code@example.com"),
        )

        self.assertEqual(response.status_code, 409)
        self.assertEqual(response.data["code"], "CATEGORY_CODE_EXISTS")

    def test_duplicate_name_rejected(self):
        self._create_category(code="PPC", name="Pakistan Penal Code")

        response = self.client.post(
            "/api/v1/document-categories/",
            {
                "name": "pakistan penal code",
                "code": "PPC2",
                "description": "",
                "is_active": True,
            },
            format="json",
            **self._staff_headers(email="duplicate-name@example.com"),
        )

        self.assertEqual(response.status_code, 409)
        self.assertEqual(response.data["code"], "CATEGORY_NAME_EXISTS")

    def test_blank_name_rejected(self):
        response = self.client.post(
            "/api/v1/document-categories/",
            {
                "name": "   ",
                "code": "PPC",
                "description": "",
                "is_active": True,
            },
            format="json",
            **self._staff_headers(email="blank-name@example.com"),
        )

        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.data["code"], "INVALID_REQUEST")
        self.assertIn("name", response.data["errors"])

    def test_blank_code_rejected(self):
        response = self.client.post(
            "/api/v1/document-categories/",
            {
                "name": "Pakistan Penal Code",
                "code": "   ",
                "description": "",
                "is_active": True,
            },
            format="json",
            **self._staff_headers(email="blank-code@example.com"),
        )

        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.data["code"], "INVALID_REQUEST")
        self.assertIn("code", response.data["errors"])

    def test_invalid_boolean_rejected(self):
        response = self.client.post(
            "/api/v1/document-categories/",
            {
                "name": "Pakistan Penal Code",
                "code": "PPC",
                "description": "",
                "is_active": "maybe",
            },
            format="json",
            **self._staff_headers(email="invalid-bool@example.com"),
        )

        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.data["code"], "INVALID_REQUEST")
        self.assertIn("is_active", response.data["errors"])

    def test_unknown_fields_rejected(self):
        response = self.client.post(
            "/api/v1/document-categories/",
            {
                "name": "Pakistan Penal Code",
                "code": "PPC",
                "description": "",
                "is_active": True,
                "unexpected": "value",
            },
            format="json",
            **self._staff_headers(email="unknown-field@example.com"),
        )

        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.data["code"], "INVALID_REQUEST")
        self.assertIn("unexpected", response.data["errors"])

    def test_authenticated_normal_user_can_list_categories(self):
        self._create_category(code="PPC", name="Pakistan Penal Code")

        response = self.client.get(
            "/api/v1/document-categories/",
            **self._user_headers(email="list-user@example.com"),
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.data["count"], 1)
        self.assertEqual(response.data["results"][0]["code"], "PPC")

    def test_admin_can_list_categories(self):
        self._create_category(code="PPC", name="Pakistan Penal Code")

        response = self.client.get(
            "/api/v1/document-categories/",
            **self._staff_headers(email="list-admin@example.com"),
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.data["count"], 1)
        self.assertEqual(response.data["results"][0]["code"], "PPC")

    def test_unauthenticated_user_gets_401_on_list(self):
        response = self.client.get("/api/v1/document-categories/")

        self.assertEqual(response.status_code, 401)
        self.assertEqual(response.data["code"], "AUTHENTICATION_REQUIRED")

    def test_invalid_token_gets_401_on_list(self):
        response = self.client.get(
            "/api/v1/document-categories/",
            HTTP_AUTHORIZATION="Bearer not-a-jwt",
        )

        self.assertEqual(response.status_code, 401)
        self.assertEqual(response.data["code"], "TOKEN_INVALID")

    def test_only_active_categories_returned(self):
        self._create_category(code="PPC", name="Pakistan Penal Code", is_active=True)
        self._create_category(code="AMLA", name="Anti-Money Laundering Act", is_active=False)

        response = self.client.get(
            "/api/v1/document-categories/",
            **self._staff_headers(email="active-only@example.com"),
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.data["count"], 1)
        self.assertEqual(response.data["results"][0]["code"], "PPC")

    def test_inactive_category_excluded(self):
        self._create_category(code="AMLA", name="Anti-Money Laundering Act", is_active=False)

        response = self.client.get(
            "/api/v1/document-categories/",
            **self._staff_headers(email="inactive@example.com"),
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.data["count"], 0)
        self.assertEqual(response.data["results"], [])

    def test_empty_database_returns_empty_results(self):
        response = self.client.get(
            "/api/v1/document-categories/",
            **self._staff_headers(email="empty@example.com"),
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.data["count"], 0)
        self.assertEqual(response.data["results"], [])

    def test_categories_ordered_alphabetically(self):
        self._create_category(code="ZZZ", name="Zed Law")
        self._create_category(code="AAA", name="Alpha Law")
        self._create_category(code="MMM", name="Middle Law")

        response = self.client.get(
            "/api/v1/document-categories/",
            **self._staff_headers(email="sorted@example.com"),
        )

        self.assertEqual(response.status_code, 200)
        names = [category["name"] for category in response.data["results"]]
        self.assertEqual(names, ["Alpha Law", "Middle Law", "Zed Law"])

    def test_admin_successfully_patches_category(self):
        category = self._create_category(code="PPC", name="Pakistan Penal Code")

        response = self.client.patch(
            f"/api/v1/document-categories/{category.id}/",
            {
                "name": "Pakistan Penal Code Updated",
                "description": "Updated description",
                "is_active": False,
            },
            format="json",
            **self._staff_headers(email="patch-admin@example.com"),
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.data["message"], "Document category updated successfully.")
        self.assertEqual(response.data["data"]["name"], "Pakistan Penal Code Updated")
        category.refresh_from_db()
        self.assertEqual(category.name, "Pakistan Penal Code Updated")
        self.assertFalse(category.is_active)

    def test_patch_requires_admin_jwt(self):
        category = self._create_category(code="PPC", name="Pakistan Penal Code")

        response = self.client.patch(
            f"/api/v1/document-categories/{category.id}/",
            {"name": "Updated"},
            format="json",
        )

        self.assertEqual(response.status_code, 401)
        self.assertEqual(response.data["code"], "AUTHENTICATION_REQUIRED")

    def test_patch_rejects_normal_user(self):
        category = self._create_category(code="PPC", name="Pakistan Penal Code")

        response = self.client.patch(
            f"/api/v1/document-categories/{category.id}/",
            {"name": "Updated"},
            format="json",
            **self._user_headers(email="patch-user@example.com"),
        )

        self.assertEqual(response.status_code, 403)
        self.assertEqual(response.data["code"], "FORBIDDEN")

    def test_patch_missing_category_returns_404(self):
        response = self.client.patch(
            "/api/v1/document-categories/999999/",
            {"name": "Updated"},
            format="json",
            **self._staff_headers(email="missing-patch@example.com"),
        )

        self.assertEqual(response.status_code, 404)
        self.assertEqual(response.data["code"], "CATEGORY_NOT_FOUND")

    def test_patch_duplicate_name_rejected(self):
        category = self._create_category(code="PPC", name="Pakistan Penal Code")
        self._create_category(code="AMLA", name="Anti-Money Laundering Act")

        response = self.client.patch(
            f"/api/v1/document-categories/{category.id}/",
            {"name": "anti-money laundering act"},
            format="json",
            **self._staff_headers(email="dup-name-patch@example.com"),
        )

        self.assertEqual(response.status_code, 409)
        self.assertEqual(response.data["code"], "CATEGORY_NAME_EXISTS")

    def test_patch_rejects_immutable_code(self):
        category = self._create_category(code="PPC", name="Pakistan Penal Code")

        response = self.client.patch(
            f"/api/v1/document-categories/{category.id}/",
            {"code": "AMLA"},
            format="json",
            **self._staff_headers(email="immutable-code@example.com"),
        )

        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.data["code"], "IMMUTABLE_FIELD")

    def test_patch_rejects_empty_body(self):
        category = self._create_category(code="PPC", name="Pakistan Penal Code")

        response = self.client.patch(
            f"/api/v1/document-categories/{category.id}/",
            {},
            format="json",
            **self._staff_headers(email="empty-patch@example.com"),
        )

        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.data["code"], "EMPTY_UPDATE")

    def test_admin_successfully_deletes_category(self):
        category = self._create_category(code="PPC", name="Pakistan Penal Code")

        response = self.client.delete(
            f"/api/v1/document-categories/{category.id}/",
            **self._staff_headers(email="delete-admin@example.com"),
        )

        self.assertEqual(response.status_code, 204)
        self.assertFalse(DocumentCategory.objects.filter(id=category.id).exists())

    def test_delete_requires_admin_jwt(self):
        category = self._create_category(code="PPC", name="Pakistan Penal Code")

        response = self.client.delete(f"/api/v1/document-categories/{category.id}/")

        self.assertEqual(response.status_code, 401)
        self.assertEqual(response.data["code"], "AUTHENTICATION_REQUIRED")

    def test_delete_rejects_normal_user(self):
        category = self._create_category(code="PPC", name="Pakistan Penal Code")

        response = self.client.delete(
            f"/api/v1/document-categories/{category.id}/",
            **self._user_headers(email="delete-user@example.com"),
        )

        self.assertEqual(response.status_code, 403)
        self.assertEqual(response.data["code"], "FORBIDDEN")

    def test_delete_missing_category_returns_404(self):
        response = self.client.delete(
            "/api/v1/document-categories/999999/",
            **self._staff_headers(email="missing-delete@example.com"),
        )

        self.assertEqual(response.status_code, 404)
        self.assertEqual(response.data["code"], "CATEGORY_NOT_FOUND")

    def test_protected_delete_returns_409(self):
        category = self._create_category(code="PPC", name="Pakistan Penal Code")

        with patch(
            "documents.services.category_service._category_has_related_documents",
            return_value=True,
        ):
            response = self.client.delete(
                f"/api/v1/document-categories/{category.id}/",
                **self._staff_headers(email="protected-delete@example.com"),
            )

        self.assertEqual(response.status_code, 409)
        self.assertEqual(response.data["code"], "CATEGORY_IN_USE")
        self.assertIn("deactivate it", response.data["message"].lower())
