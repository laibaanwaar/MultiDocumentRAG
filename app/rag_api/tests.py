from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch
from uuid import uuid4

from django.contrib.auth import get_user_model
from django.test import override_settings
from rest_framework_simplejwt.tokens import AccessToken
from rest_framework.test import APITestCase

from billing.models import Plan, Subscription
from documents.models import DocumentCategory
from rag.intent_router import route_question
from rag_api.models import QueryHistory
from rag_api import services


User = get_user_model()


class _DummyClient:
    def __init__(self):
        self.closed = False

    def close(self):
        self.closed = True


@override_settings(
    EMAIL_BACKEND="django.core.mail.backends.locmem.EmailBackend",
)
class RagQueryAndHistoryAPITests(APITestCase):
    def _create_user(
        self,
        *,
        username: str | None = None,
        email: str | None = None,
        is_active: bool = True,
    ):
        unique_id = uuid4().hex[:8]
        return User.objects.create_user(
            username=username or f"user-{unique_id}",
            email=email or f"user-{unique_id}@example.com",
            password="StrongPass!234",
            first_name="Rag",
            last_name="User",
            is_active=is_active,
        )

    def _auth_headers(self, user):
        return {
            "HTTP_AUTHORIZATION": f"Bearer {str(AccessToken.for_user(user))}",
        }

    def _create_plan(self, *, query_limit: int = 30):
        return Plan.objects.create(
            name="Starter",
            code=f"STARTER-{uuid4().hex[:6].upper()}",
            price="0.00",
            billing_period=Plan.BillingPeriod.MONTHLY,
            document_limit=0,
            query_limit=query_limit,
            is_active=True,
        )

    def _create_subscription(
        self,
        *,
        user,
        plan=None,
        status=Subscription.Status.ACTIVE,
        queries_used: int = 0,
    ):
        plan = plan or self._create_plan()
        return Subscription.objects.create(
            user=user,
            plan=plan,
            status=status,
            queries_used=queries_used,
        )

    def _create_category(
        self,
        *,
        name="Pakistan Penal Code",
        code="PPC",
        is_active=True,
    ):
        return DocumentCategory.objects.create(
            name=name,
            code=code,
            description=f"{name} category",
            is_active=is_active,
        )

    def _mocked_rag_result(self, *, answer="Under Section 379, theft is punishable.", sources=None):
        return {
            "answer": answer,
            "sources": sources if sources is not None else [
                {
                    "label": "Source 1",
                    "document_id": "ppc_1860",
                    "document_name": "Pakistan Penal Code, 1860",
                    "document_title": "Pakistan Penal Code, 1860",
                    "document_short_name": "PPC",
                    "document_type": "legal_document",
                    "provision_type": "section",
                    "provision_number": "379",
                    "provision_title": "Theft",
                    "section_number": "379",
                    "section_title": "Theft",
                    "article_number": None,
                    "article_title": None,
                    "page_start": 1,
                    "page_end": 2,
                    "page_number": 1,
                    "page_range": "1-2",
                    "chunk_number": 1,
                    "chunk_numbers": [1, 2],
                    "merged_provision_parts": False,
                    "source_pages": [1, 2],
                    "quality_status": "acceptable",
                }
            ],
            "retrieved_contexts": ["Theft provision text."],
            "question_type": "section_lookup",
            "detected_concepts": ["theft"],
            "retrieved_document_count": 1,
            "confidence": {
                "label": "High",
                "score": 0.91,
                "top_similarity": 0.95,
                "average_similarity": 0.9,
                "section_count": 1,
                "concept_coverage": 1.0,
            },
        }

    def test_query_requires_jwt(self):
        response = self.client.post(
            "/api/v1/rag/query/",
            {"question": "What is Section 379 of PPC?"},
            format="json",
        )

        self.assertEqual(response.status_code, 401)
        self.assertEqual(response.data["code"], "AUTHENTICATION_REQUIRED")

    def test_query_rejects_inactive_user(self):
        user = self._create_user(is_active=False)

        response = self.client.post(
            "/api/v1/rag/query/",
            {"question": "What is Section 379 of PPC?"},
            format="json",
            **self._auth_headers(user),
        )

        self.assertEqual(response.status_code, 403)
        self.assertEqual(response.data["code"], "USER_INACTIVE")

    def test_query_rejects_missing_subscription(self):
        user = self._create_user()

        response = self.client.post(
            "/api/v1/rag/query/",
            {"question": "What is Section 379 of PPC?"},
            format="json",
            **self._auth_headers(user),
        )

        self.assertEqual(response.status_code, 403)
        self.assertEqual(response.data["code"], "SUBSCRIPTION_REQUIRED")

    def test_query_rejects_empty_body(self):
        user = self._create_user()
        self._create_subscription(user=user)

        response = self.client.post(
            "/api/v1/rag/query/",
            {},
            format="json",
            **self._auth_headers(user),
        )

        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.data["code"], "INVALID_QUESTION")

    def test_query_rejects_unknown_fields(self):
        user = self._create_user()
        self._create_subscription(user=user)

        response = self.client.post(
            "/api/v1/rag/query/",
            {
                "question": "What is Section 379 of PPC?",
                "unexpected": "value",
            },
            format="json",
            **self._auth_headers(user),
        )

        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.data["code"], "INVALID_REQUEST")
        self.assertIn("unexpected", response.data["errors"])

    def test_query_rejects_blank_question(self):
        user = self._create_user()
        self._create_subscription(user=user)

        response = self.client.post(
            "/api/v1/rag/query/",
            {"question": "   "},
            format="json",
            **self._auth_headers(user),
        )

        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.data["code"], "INVALID_QUESTION")

    def test_query_rejects_unknown_category(self):
        user = self._create_user()
        self._create_subscription(user=user)

        response = self.client.post(
            "/api/v1/rag/query/",
            {
                "question": "What is Section 379 of PPC?",
                "category_id": 999999,
            },
            format="json",
            **self._auth_headers(user),
        )

        self.assertEqual(response.status_code, 404)
        self.assertEqual(response.data["code"], "CATEGORY_NOT_FOUND")

    def test_query_rejects_inactive_category(self):
        user = self._create_user()
        self._create_subscription(user=user)
        category = self._create_category(is_active=False)

        response = self.client.post(
            "/api/v1/rag/query/",
            {
                "question": "What is Section 379 of PPC?",
                "category_id": category.id,
            },
            format="json",
            **self._auth_headers(user),
        )

        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.data["code"], "CATEGORY_INACTIVE")

    def test_query_rejects_query_limit_exceeded(self):
        user = self._create_user()
        plan = self._create_plan(query_limit=1)
        self._create_subscription(
            user=user,
            plan=plan,
            queries_used=1,
        )

        response = self.client.post(
            "/api/v1/rag/query/",
            {"question": "What is Section 379 of PPC?"},
            format="json",
            **self._auth_headers(user),
        )

        self.assertEqual(response.status_code, 429)
        self.assertEqual(response.data["code"], "QUERY_LIMIT_EXCEEDED")
        self.assertEqual(QueryHistory.objects.count(), 0)

    def test_query_success_saves_history_and_uses_category_scope(self):
        user = self._create_user()
        plan = self._create_plan(query_limit=5)
        subscription = self._create_subscription(user=user, plan=plan)
        category = self._create_category(code="PPC", name="Pakistan Penal Code")

        dummy_client = _DummyClient()

        with patch(
            "rag_api.services.create_rag_components",
            return_value=(SimpleNamespace(), SimpleNamespace(), dummy_client),
        ) as create_components, patch(
            "rag_api.services.run_rag_question",
            return_value=self._mocked_rag_result(),
        ) as run_rag:
            response = self.client.post(
                "/api/v1/rag/query/",
                {
                    "question": "What is Section 379 of PPC?",
                    "category_id": category.id,
                },
                format="json",
                **self._auth_headers(user),
            )

        self.assertTrue(dummy_client.closed)
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.data["message"], "Legal question answered successfully.")
        self.assertEqual(response.data["data"]["question"], "What is Section 379 of PPC?")
        self.assertEqual(response.data["data"]["category"]["code"], "PPC")
        self.assertEqual(response.data["data"]["queries_remaining"], 4)
        self.assertEqual(QueryHistory.objects.count(), 1)
        history = QueryHistory.objects.get()
        self.assertEqual(history.user_id, user.id)
        self.assertEqual(history.category_id, category.id)
        self.assertEqual(history.answer, response.data["data"]["answer"])
        self.assertEqual(history.sources, response.data["data"]["sources"])
        subscription.refresh_from_db()
        self.assertEqual(subscription.queries_used, 1)
        self.assertTrue(create_components.called)
        self.assertTrue(run_rag.called)
        self.assertEqual(
            run_rag.call_args.kwargs["plan"].document_ids,
            ["ppc_1860"],
        )

    def test_query_without_category_uses_cli_route_plan(self):
        user = self._create_user()
        plan = self._create_plan(query_limit=5)
        self._create_subscription(user=user, plan=plan)
        question = "What is the punishment for theft under Section 379 of the Pakistan Penal Code?"
        expected_plan = route_question(question)

        dummy_client = _DummyClient()

        with patch(
            "rag_api.services.create_rag_components",
            return_value=(SimpleNamespace(), SimpleNamespace(), dummy_client),
        ), patch(
            "rag_api.services.run_rag_question",
            return_value=self._mocked_rag_result(),
        ) as run_rag:
            response = self.client.post(
                "/api/v1/rag/query/",
                {"question": question},
                format="json",
                **self._auth_headers(user),
            )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.data["message"], "Legal question answered successfully.")
        self.assertEqual(run_rag.call_args.kwargs["plan"].document_ids, expected_plan.document_ids)
        self.assertEqual(run_rag.call_args.kwargs["plan"].document_hints, expected_plan.document_hints)

    def test_query_service_unavailable_is_mapped_to_503(self):
        user = self._create_user()
        plan = self._create_plan(query_limit=5)
        self._create_subscription(user=user, plan=plan)

        with patch(
            "rag_api.services.create_rag_components",
            return_value=(SimpleNamespace(), SimpleNamespace(), _DummyClient()),
        ), patch(
            "rag_api.services.run_rag_question",
            side_effect=RuntimeError("qdrant unavailable"),
        ):
            response = self.client.post(
                "/api/v1/rag/query/",
                {"question": "What is Section 379 of PPC?"},
                format="json",
                **self._auth_headers(user),
            )

        self.assertEqual(response.status_code, 503)
        self.assertEqual(response.data["code"], "RAG_SERVICE_UNAVAILABLE")
        self.assertEqual(QueryHistory.objects.count(), 0)

    def test_query_no_context_does_not_consume_quota_or_store_history(self):
        user = self._create_user()
        plan = self._create_plan(query_limit=5)
        subscription = self._create_subscription(user=user, plan=plan)

        empty_result = self._mocked_rag_result(
            answer="The answer was not found in the four indexed legal documents.",
            sources=[],
        )
        empty_result["retrieved_contexts"] = []
        empty_result["retrieved_document_count"] = 0

        with patch(
            "rag_api.services.create_rag_components",
            return_value=(SimpleNamespace(), SimpleNamespace(), _DummyClient()),
        ), patch(
            "rag_api.services.run_rag_question",
            return_value=empty_result,
        ):
            response = self.client.post(
                "/api/v1/rag/query/",
                {"question": "What is the punishment for theft under Section 379 of the Pakistan Penal Code?"},
                format="json",
                **self._auth_headers(user),
            )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.data["message"], "No relevant context was found.")
        self.assertIsNone(response.data["data"]["history_id"])
        self.assertEqual(response.data["data"]["queries_remaining"], 5)
        self.assertEqual(QueryHistory.objects.count(), 0)
        subscription.refresh_from_db()
        self.assertEqual(subscription.queries_used, 0)

    def test_django_qdrant_path_defaults_to_repo_root_storage(self):
        expected_path = services.REPO_ROOT / "qdrant_storage"
        self.assertEqual(
            services.os.environ["QDRANT_PATH"],
            str(expected_path),
        )
        self.assertEqual(
            services.os.environ["QDRANT_COLLECTION"],
            "pakistan_legal_knowledge_base",
        )

    def test_history_requires_jwt(self):
        response = self.client.get("/api/v1/rag/history/")

        self.assertEqual(response.status_code, 401)
        self.assertEqual(response.data["code"], "AUTHENTICATION_REQUIRED")

    def test_history_rejects_inactive_user(self):
        user = self._create_user(is_active=False)

        response = self.client.get(
            "/api/v1/rag/history/",
            **self._auth_headers(user),
        )

        self.assertEqual(response.status_code, 403)
        self.assertEqual(response.data["code"], "USER_INACTIVE")

    def test_history_returns_only_current_users_items(self):
        user_one = self._create_user(username="alpha", email="alpha@example.com")
        user_two = self._create_user(username="beta", email="beta@example.com")
        category = self._create_category()
        self._create_subscription(user=user_one)
        self._create_subscription(user=user_two)

        QueryHistory.objects.create(
            user=user_one,
            question="Question one",
            answer="Answer one",
            category=category,
            sources=[],
        )
        QueryHistory.objects.create(
            user=user_two,
            question="Question two",
            answer="Answer two",
            category=category,
            sources=[],
        )

        response = self.client.get(
            "/api/v1/rag/history/",
            **self._auth_headers(user_one),
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.data["count"], 1)
        self.assertEqual(len(response.data["results"]), 1)
        self.assertEqual(response.data["results"][0]["question"], "Question one")

    def test_history_paginates_and_orders_newest_first(self):
        user = self._create_user()
        self._create_subscription(user=user)
        category = self._create_category()

        history_ids = []
        for index in range(6):
            history = QueryHistory.objects.create(
                user=user,
                question=f"Question {index + 1}",
                answer=f"Answer {index + 1}",
                category=category,
                sources=[],
            )
            history_ids.append(history.id)

        response = self.client.get(
            "/api/v1/rag/history/?page=2",
            **self._auth_headers(user),
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.data["count"], 6)
        self.assertEqual(len(response.data["results"]), 1)
        self.assertEqual(response.data["results"][0]["id"], history_ids[0])

    def test_history_empty_list_returns_200(self):
        user = self._create_user()
        self._create_subscription(user=user)

        response = self.client.get(
            "/api/v1/rag/history/",
            **self._auth_headers(user),
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.data["count"], 0)
        self.assertEqual(response.data["results"], [])
