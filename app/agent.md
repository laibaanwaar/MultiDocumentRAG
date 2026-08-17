# Legal RAG Django Backend — Codex Instructions

## Goal

Build a Django REST Framework backend around the existing multi-document legal RAG system.

Django handles authentication, users, admin operations, documents, subscriptions, usage limits, query history, and APIs. Do not rewrite the existing `../rag/` retrieval and generation pipeline.

## Architecture

```text
React / Flutter
  -> Django REST API
  -> JWT + permission + subscription checks
  -> existing Legal RAG service
  -> Qdrant retrieval
  -> Groq answer generation
  -> PostgreSQL history and citations
```

RAG configuration:

- Qdrant collection: `pakistan_legal_knowledge_base`
- Embeddings: `sentence-transformers/all-MiniLM-L6-v2`
- Vector size: `384`
- Answer model: `openai/gpt-oss-120b`
- Existing RAG code: `../rag/`

## Roles

Only two roles exist.

### User

Can register, log in, ask legal questions, view own history/citations, manage own profile/subscription, and submit feedback.

Cannot upload documents or access admin management APIs.

### Admin

Uses `is_staff=True`

Can manage users, documents, ingestion, subscriptions, plans, payments, all queries, analytics, and audit logs.

## Django Apps

```text
accounts/   custom user, JWT auth, profile
documents/  PDFs, versions, ingestion jobs, chunk manifests
rag_api/    queries, answers, citations, feedback, RAG adapter
billing/    plans, subscriptions, usage, payments, webhooks
audit/      admin and important system actions
config/     settings, URLs, Celery
```

## Main Models

Use models close to these names:

```text
User
Plan
Subscription
Usage
Payment
WebhookEvent
LegalDocument
DocumentVersion
IngestionJob
ChunkManifest
RagQuery
RagQuerySource
RagFeedback
AuditLog
```

PostgreSQL stores application records. Qdrant stores chunk vectors, text, sections, pages, and document metadata.

## Main Routes

```text
/api/v1/auth/
  signup, resend-otp, verify-email, login, refresh, me

/api/v1/documents/
  GET active documents and document details

/api/v1/rag/
  POST query
  GET own query history/detail
  DELETE own query
  POST retry/feedback
  GET usage

/api/v1/billing/
  GET plans
  GET current subscription

/api/v1/auth/verify-email/
  activates the user and assigns the Free plan automatically

/api/v1/admin/
  users, documents, ingestion, plans, subscriptions, payments
  all queries, analytics, audit logs, system health
```

All `/api/v1/admin/` routes require:

```python
permission_classes = [IsAuthenticated, IsSystemAdmin]
```

## User Query Flow

```text
1. Validate JWT and active user.
2. Check verified email when enabled.
3. Load current subscription from PostgreSQL.
4. Check subscription status and daily/monthly quota.
5. Create RagQuery with processing status.
6. Call the existing RAG pipeline through rag_api/services.py.
7. Save answer, intent, model, response time, and status.
8. Save source snapshots in RagQuerySource.
9. Increment usage only after success.
10. Return answer, citations, and remaining quota.
```

Responses:

```text
401 unauthenticated
403 inactive user/subscription or forbidden action
429 daily/monthly quota exceeded
```

## Admin Document Flow

```text
1. Admin uploads a PDF.
2. Validate MIME type, extension, size, and SHA-256 hash.
3. Reject duplicates.
4. Create LegalDocument and queued IngestionJob.
5. Send ingestion to Celery; return 202 Accepted.
6. Run existing cleaning and section-aware chunking.
7. Generate 384-dimensional MiniLM embeddings.
8. Store vectors and legal metadata in Qdrant.
9. Save tracking data in ChunkManifest.
10. Mark document indexed or failed.
```

Document statuses:

```text
uploaded, queued, processing, indexed, failed, archived
```

Normal users must receive `403` for every document write route.

## Existing RAG Integration

Create a thin adapter in `rag_api/services.py`.

```python
from rag.answer_service import LegalAnswerService


class DjangoLegalRAGService:
    def __init__(self):
        self.service = LegalAnswerService()

    def answer(self, question: str, document_ids=None):
        # Inspect the existing method signature before final integration.
        return self.service.answer(
            question=question,
            document_ids=document_ids or [],
        )
```

Do not place retrieval, ranking, prompt, or Groq logic inside Django views.

Existing RAG flow remains:

```text
intent routing
-> query expansion
-> query embedding
-> Qdrant retrieval
-> RRF
-> MMR/ranking
-> context building
-> prompt building
-> Groq answer
-> citations
```

## Subscription Rules

- New users automatically receive a Free plan.
- Current Free plan defaults: `FREE`, monthly, 3 documents, 30 queries.
- Check current subscription from PostgreSQL on every RAG request.
- Do not rely only on JWT claims for plan status or quota.
- Verify payment webhook signatures.
- Make webhooks idempotent with a unique provider event ID.

## Coding Rules

- Create the custom User model before the first migration.
- Keep views thin; use serializers and service classes.
- Use database transactions for related writes.
- Keep secrets in `.env`.
- Never expose API keys or local storage paths.
- Archive indexed documents instead of hard deleting by default.
- Prevent an admin from deleting/deactivating their own account.
- Log admin changes.
- Preserve existing `query_cli.py` and `ingest.py`.
- Inspect existing classes and function signatures before modifying integration code.

## Implementation Order

```text
1. Custom User and JWT
2. Admin permission
3. Document models and admin upload
4. Celery/Redis ingestion
5. RAG query/history/source APIs
6. Free subscription and quota
7. Paid billing webhooks
8. Admin monitoring and audit
```

## Acceptance Checks

```text
User cannot manage documents or admin resources.
Admin can upload a valid PDF and receives 202.
Invalid/duplicate PDFs are rejected.
Indexed documents are searchable.
Answers and source snapshots are saved.
Users see only their own history.
Admin sees all users and queries.
Inactive subscriptions return 403.
Exceeded quotas return 429.
Existing CLI ingestion and querying still work.
```
