import tempfile
from pathlib import Path
from unittest.mock import patch
from uuid import uuid4

from django.contrib.auth import get_user_model
from django.test import override_settings
from django.core.files.uploadedfile import SimpleUploadedFile
from rest_framework_simplejwt.tokens import AccessToken
from rest_framework.test import APITestCase

from documents.models import DocumentCategory, LegalDocument


User = get_user_model()


def _pdf_bytes(label: str) -> bytes:
    return (
        b"%PDF-1.4\n"
        b"1 0 obj\n"
        b"<< /Type /Catalog >>\n"
        b"endobj\n"
        b"trailer\n"
        b"<< /Label ("
        + label.encode("utf-8")
        + b") >>\n"
        b"%%EOF"
    )


class LegalDocumentAPITests(APITestCase):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls._media_root = tempfile.TemporaryDirectory()
        cls._media_override = override_settings(
            MEDIA_ROOT=cls._media_root.name,
            LEGAL_DOCUMENT_UPLOAD_MAX_SIZE=1024 * 1024,
        )
        cls._media_override.enable()

    @classmethod
    def tearDownClass(cls):
        cls._media_override.disable()
        cls._media_root.cleanup()
        super().tearDownClass()

    def _create_user(
        self,
        *,
        username="user",
        email="user@example.com",
        is_staff=False,
    ):
        return User.objects.create_user(
            username=username,
            email=email,
            password="StrongPass!234",
            first_name="Doc",
            last_name="User",
            is_active=True,
            is_staff=is_staff,
        )

    def _staff_headers(self, *, username=None, email="admin@example.com"):
        unique_username = username or f"admin-{uuid4().hex[:8]}"
        user = self._create_user(
            username=unique_username,
            email=email,
            is_staff=True,
        )
        return {
            "HTTP_AUTHORIZATION": f"Bearer {str(AccessToken.for_user(user))}",
        }

    def _user_headers(self, *, username=None, email="user@example.com"):
        unique_username = username or f"user-{uuid4().hex[:8]}"
        user = self._create_user(
            username=unique_username,
            email=email,
            is_staff=False,
        )
        return {
            "HTTP_AUTHORIZATION": f"Bearer {str(AccessToken.for_user(user))}",
        }

    def _create_category(self, *, name="Criminal Law", code="CRIM", is_active=True):
        return DocumentCategory.objects.create(
            name=name,
            code=code,
            description=f"{name} category",
            is_active=is_active,
        )

    def _pdf_upload(self, *, name="document.pdf", label="one"):
        return SimpleUploadedFile(
            name,
            _pdf_bytes(label),
            content_type="application/pdf",
        )

    def _create_document(self, *, title, category, name="document.pdf", label="one"):
        response = self.client.post(
            "/api/v1/documents/",
            {
                "title": title,
                "category_id": category.id,
                "file": self._pdf_upload(name=name, label=label),
            },
            format="multipart",
            **self._staff_headers(email=f"{label}@example.com"),
        )
        self.assertEqual(response.status_code, 201)
        return response

    def test_admin_can_create_legal_document(self):
        category = self._create_category()

        response = self.client.post(
            "/api/v1/documents/",
            {
                "title": "Pakistan Penal Code",
                "category_id": category.id,
                "file": self._pdf_upload(name="Pakistan Penal Code.pdf", label="create"),
            },
            format="multipart",
            **self._staff_headers(email="create-admin@example.com"),
        )

        self.assertEqual(response.status_code, 201)
        self.assertEqual(response.data["message"], "Legal document created successfully.")
        self.assertEqual(response.data["data"]["title"], "Pakistan Penal Code")
        self.assertEqual(response.data["data"]["status"], "PENDING")
        self.assertTrue(LegalDocument.objects.filter(title="Pakistan Penal Code").exists())

    def test_unauthenticated_user_gets_401_on_create(self):
        category = self._create_category()

        response = self.client.post(
            "/api/v1/documents/",
            {
                "title": "Pakistan Penal Code",
                "category_id": category.id,
                "file": self._pdf_upload(label="unauth"),
            },
            format="multipart",
        )

        self.assertEqual(response.status_code, 401)
        self.assertEqual(response.data["code"], "AUTHENTICATION_REQUIRED")

    def test_normal_user_gets_403_on_create(self):
        category = self._create_category()

        response = self.client.post(
            "/api/v1/documents/",
            {
                "title": "Pakistan Penal Code",
                "category_id": category.id,
                "file": self._pdf_upload(label="forbidden"),
            },
            format="multipart",
            **self._user_headers(email="user-create@example.com"),
        )

        self.assertEqual(response.status_code, 403)
        self.assertEqual(response.data["code"], "FORBIDDEN")

    def test_missing_category_returns_404(self):
        response = self.client.post(
            "/api/v1/documents/",
            {
                "title": "Pakistan Penal Code",
                "category_id": 999999,
                "file": self._pdf_upload(label="missing-category"),
            },
            format="multipart",
            **self._staff_headers(email="missing-category@example.com"),
        )

        self.assertEqual(response.status_code, 404)
        self.assertEqual(response.data["code"], "CATEGORY_NOT_FOUND")

    def test_inactive_category_rejected(self):
        category = self._create_category(is_active=False)

        response = self.client.post(
            "/api/v1/documents/",
            {
                "title": "Pakistan Penal Code",
                "category_id": category.id,
                "file": self._pdf_upload(label="inactive-category"),
            },
            format="multipart",
            **self._staff_headers(email="inactive-category@example.com"),
        )

        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.data["code"], "CATEGORY_INACTIVE")

    def test_missing_file_rejected(self):
        category = self._create_category()

        response = self.client.post(
            "/api/v1/documents/",
            {
                "title": "Pakistan Penal Code",
                "category_id": category.id,
            },
            format="multipart",
            **self._staff_headers(email="missing-file@example.com"),
        )

        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.data["code"], "INVALID_REQUEST")
        self.assertIn("file", response.data["errors"])

    def test_invalid_file_type_rejected(self):
        category = self._create_category()

        response = self.client.post(
            "/api/v1/documents/",
            {
                "title": "Pakistan Penal Code",
                "category_id": category.id,
                "file": SimpleUploadedFile(
                    "notes.txt",
                    b"plain text",
                    content_type="text/plain",
                ),
            },
            format="multipart",
            **self._staff_headers(email="invalid-type@example.com"),
        )

        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.data["code"], "INVALID_FILE_TYPE")

    def test_invalid_pdf_rejected(self):
        category = self._create_category()

        response = self.client.post(
            "/api/v1/documents/",
            {
                "title": "Pakistan Penal Code",
                "category_id": category.id,
                "file": SimpleUploadedFile(
                    "fake.pdf",
                    b"not a pdf",
                    content_type="application/pdf",
                ),
            },
            format="multipart",
            **self._staff_headers(email="invalid-pdf@example.com"),
        )

        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.data["code"], "INVALID_PDF")

    def test_oversized_pdf_rejected(self):
        category = self._create_category()

        with override_settings(LEGAL_DOCUMENT_UPLOAD_MAX_SIZE=10):
            response = self.client.post(
                "/api/v1/documents/",
                {
                    "title": "Pakistan Penal Code",
                    "category_id": category.id,
                    "file": self._pdf_upload(label="oversized"),
                },
                format="multipart",
                **self._staff_headers(email="oversized@example.com"),
            )

        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.data["code"], "FILE_TOO_LARGE")

    def test_duplicate_checksum_rejected(self):
        category = self._create_category()
        payload = {
            "title": "Pakistan Penal Code",
            "category_id": category.id,
            "file": self._pdf_upload(name="first.pdf", label="duplicate"),
        }
        first = self.client.post(
            "/api/v1/documents/",
            payload,
            format="multipart",
            **self._staff_headers(email="dup-first@example.com"),
        )
        self.assertEqual(first.status_code, 201)

        second = self.client.post(
            "/api/v1/documents/",
            {
                "title": "Pakistan Penal Code Copy",
                "category_id": category.id,
                "file": self._pdf_upload(name="second.pdf", label="duplicate"),
            },
            format="multipart",
            **self._staff_headers(email="dup-second@example.com"),
        )

        self.assertEqual(second.status_code, 409)
        self.assertEqual(second.data["code"], "DOCUMENT_ALREADY_EXISTS")

    def test_filename_is_sanitized(self):
        category = self._create_category()

        response = self.client.post(
            "/api/v1/documents/",
            {
                "title": "Path Traversal Test",
                "category_id": category.id,
                "file": self._pdf_upload(name="../../evil.pdf", label="traversal"),
            },
            format="multipart",
            **self._staff_headers(email="traversal@example.com"),
        )

        self.assertEqual(response.status_code, 201)
        document = LegalDocument.objects.get(pk=response.data["data"]["id"])
        self.assertEqual(document.original_filename, "evil.pdf")
        self.assertNotIn("..", document.file.name)

    def test_admin_can_list_documents_with_pagination_and_filters(self):
        category_one = self._create_category(name="Criminal Law", code="CRIM")
        category_two = self._create_category(name="Special Law", code="SPEC")

        for index in range(6):
            self._create_document(
                title=f"Document {index + 1}",
                category=category_one if index < 4 else category_two,
                name=f"doc-{index + 1}.pdf",
                label=f"list-{index + 1}",
            )

        response = self.client.get(
            "/api/v1/documents/?page=1&category=CRIM&search=Document",
            **self._staff_headers(email="list-admin@example.com"),
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.data["count"], 4)
        self.assertEqual(len(response.data["results"]), 4)
        self.assertEqual(response.data["results"][0]["title"], "Document 4")

    def test_unauthenticated_user_gets_401_on_list(self):
        response = self.client.get("/api/v1/documents/")

        self.assertEqual(response.status_code, 401)
        self.assertEqual(response.data["code"], "AUTHENTICATION_REQUIRED")

    def test_normal_user_gets_403_on_list(self):
        response = self.client.get(
            "/api/v1/documents/",
            **self._user_headers(email="list-user@example.com"),
        )

        self.assertEqual(response.status_code, 403)
        self.assertEqual(response.data["code"], "FORBIDDEN")

    def test_admin_can_retrieve_document_detail(self):
        category = self._create_category()
        create_response = self._create_document(
            title="Detail Document",
            category=category,
            name="detail.pdf",
            label="detail",
        )
        document_id = create_response.data["data"]["id"]

        response = self.client.get(
            f"/api/v1/documents/{document_id}/",
            **self._staff_headers(email="detail-admin@example.com"),
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.data["data"]["id"], document_id)
        self.assertIn("checksum_sha256", response.data["data"])

    def test_missing_document_returns_404(self):
        response = self.client.get(
            "/api/v1/documents/999999/",
            **self._staff_headers(email="missing-detail@example.com"),
        )

        self.assertEqual(response.status_code, 404)
        self.assertEqual(response.data["code"], "DOCUMENT_NOT_FOUND")

    def test_patch_rejects_empty_body(self):
        category = self._create_category()
        create_response = self._create_document(
            title="Patch Target",
            category=category,
            name="patch-target.pdf",
            label="patch-empty",
        )

        response = self.client.patch(
            f"/api/v1/documents/{create_response.data['data']['id']}/",
            {},
            format="json",
            **self._staff_headers(email="patch-empty@example.com"),
        )

        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.data["code"], "EMPTY_UPDATE")

    def test_patch_rejects_blank_title(self):
        category = self._create_category()
        create_response = self._create_document(
            title="Patch Target",
            category=category,
            name="patch-blank.pdf",
            label="patch-blank",
        )

        response = self.client.patch(
            f"/api/v1/documents/{create_response.data['data']['id']}/",
            {"title": "   "},
            format="json",
            **self._staff_headers(email="patch-blank@example.com"),
        )

        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.data["code"], "INVALID_REQUEST")
        self.assertIn("title", response.data["errors"])

    def test_patch_rejects_immutable_fields(self):
        category = self._create_category()
        create_response = self._create_document(
            title="Patch Target",
            category=category,
            name="patch-immutable.pdf",
            label="patch-immutable",
        )

        response = self.client.patch(
            f"/api/v1/documents/{create_response.data['data']['id']}/",
            {"file": self._pdf_upload(name="new.pdf", label="new")},
            format="multipart",
            **self._staff_headers(email="patch-immutable@example.com"),
        )

        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.data["code"], "IMMUTABLE_FIELD")

    def test_patch_rejects_unknown_fields(self):
        category = self._create_category()
        create_response = self._create_document(
            title="Patch Target",
            category=category,
            name="patch-unknown.pdf",
            label="patch-unknown",
        )

        response = self.client.patch(
            f"/api/v1/documents/{create_response.data['data']['id']}/",
            {"title": "Updated Title", "unexpected": "value"},
            format="json",
            **self._staff_headers(email="patch-unknown@example.com"),
        )

        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.data["code"], "INVALID_REQUEST")
        self.assertIn("unexpected", response.data["errors"])

    def test_patch_ready_document_requeues_to_pending(self):
        category = self._create_category()
        create_response = self._create_document(
            title="Ready Target",
            category=category,
            name="ready.pdf",
            label="ready",
        )
        document = LegalDocument.objects.get(pk=create_response.data["data"]["id"])
        document.status = LegalDocument.Status.READY
        document.save(update_fields=["status"])

        response = self.client.patch(
            f"/api/v1/documents/{document.id}/",
            {"title": "Ready Target Updated"},
            format="json",
            **self._staff_headers(email="patch-ready@example.com"),
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.data["data"]["status"], "PENDING")

    def test_delete_archives_document(self):
        category = self._create_category()
        create_response = self._create_document(
            title="Delete Target",
            category=category,
            name="delete.pdf",
            label="delete",
        )
        document_id = create_response.data["data"]["id"]

        with patch(
            "documents.services.legal_document_service._delete_qdrant_points",
            return_value=None,
        ):
            response = self.client.delete(
                f"/api/v1/documents/{document_id}/",
                **self._staff_headers(email="delete-admin@example.com"),
            )

        self.assertEqual(response.status_code, 204)
        document = LegalDocument.objects.get(pk=document_id)
        self.assertEqual(document.status, LegalDocument.Status.ARCHIVED)
        self.assertTrue(document.file.name.startswith("archived/"))

    def test_delete_is_idempotent_for_already_archived_documents(self):
        category = self._create_category()
        create_response = self._create_document(
            title="Delete Target",
            category=category,
            name="delete-archived.pdf",
            label="delete-archived",
        )
        document = LegalDocument.objects.get(pk=create_response.data["data"]["id"])
        document.status = LegalDocument.Status.ARCHIVED
        document.save(update_fields=["status"])

        with patch(
            "documents.services.legal_document_service._delete_qdrant_points",
            return_value=None,
        ):
            response = self.client.delete(
                f"/api/v1/documents/{document.id}/",
                **self._staff_headers(email="delete-archived@example.com"),
            )

        self.assertEqual(response.status_code, 204)
        document.refresh_from_db()
        self.assertEqual(document.status, LegalDocument.Status.ARCHIVED)
