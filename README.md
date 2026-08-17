# Multi-Document Legal RAG System

A production-oriented Retrieval-Augmented Generation (RAG) system for answering legal questions from PDF documents using Large Language Models, vector search, and semantic retrieval.

The project now works as a multi-document legal knowledge base for the current Pakistan law set in this repository:

- Pakistan Penal Code, 1860
- Constitution of Pakistan
- The Anti-Terrorism Act, 1997
- Anti-Money Laundering Act, 2010

The system remains grounded in the same core idea: ingest the PDFs, clean and split the legal text, embed the chunks, store them in Qdrant, and answer questions with retrieved context and citations.

---

## What Stays the Same

- PDF documents are still the source of truth.
- Retrieval still happens through Qdrant.
- Answers are still generated from retrieved legal context.
- Source citations are still part of the response flow.
- The CLI is still the easiest way to query the system.

---

## Current Runtime Setup

- Vector database: Qdrant
- Collection name: `pakistan_legal_knowledge_base`
- Embedding model: `sentence-transformers/all-MiniLM-L6-v2`
- Embedding dimension: `384`
- Main answer model: `openai/gpt-oss-120b` via Groq
- RAGAS model: `llama-3.1-8b-instant`
- Qdrant storage path: `qdrant_storage`

---

## Features

- PDF document ingestion
- Legal text cleaning
- Section-aware chunking
- Cross-page section handling
- Metadata-aware retrieval
- Intent-aware query routing
- Adaptive retrieval depth
- Multi-query retrieval
- Reciprocal Rank Fusion
- Maximum Marginal Relevance
- Scenario-aware legal retrieval
- Grounded answer generation
- Source citations
- CLI-based querying
- ATA-specific evaluation generation and scoring
- Retrieval observability traces
- Versioned retrieval regression evaluation

---

## Project Flow

```
PDF documents
  -> document loading
  -> text cleaning
  -> legal chunking
  -> embedding generation
  -> Qdrant storage
  -> question routing
  -> candidate retrieval
  -> ranking and context selection
  -> prompt construction
  -> Groq answer generation
  -> cited answer output
```

### Ingestion Flow

1. Read PDFs from `data/documents/`.
2. Extract and clean text.
3. Preserve legal structure such as sections, explanations, and enumerations.
4. Split text into retrieval-friendly legal chunks.
5. Generate embeddings.
6. Store chunks and metadata in Qdrant.

Main entry point:

```bash
python ingest.py
```

### Query Flow

1. User asks a question in the CLI.
2. The router classifies the question type.
3. Retrieval queries are expanded.
4. Qdrant returns candidates.
5. Ranker logic selects the best context.
6. The prompt is built from the retrieved legal text.
7. The Groq chat model generates a grounded answer.
8. The answer includes supporting sources when available.

Main entry point:

```bash
python query_cli.py
```

### Retrieval Observability

The answer pipeline can now return an optional retrieval trace for debugging and evaluation.

- `rag/answer_service.py` accepts `include_trace=True` and adds a safe `retrieval_trace` payload.
- The trace records normalized question routing, retrieval queries, candidate summaries, ranked items, selected context summaries, cited source labels, and timing information.
- Retrieval events are recorded for exact, vector, lexical, and neighbor routes without exposing raw prompts or secrets.

This is used by the F-12 retrieval evaluation work and by the local metrics script when trace data is present.

### ATA Evaluation Flow

The Anti-Terrorism Act evaluation flow now has its own dedicated files.

Working files:

- `ata_eval_samples.jsonl`
- `ata_predictions.jsonl`
- `ata_evaluation_matrix.csv`
- `ata_evaluation_summary.json`
- `evaluation/gold/legal_rag_gold_v1.jsonl`

These ATA files are the dedicated evaluation set and outputs for the Anti-Terrorism Act workflow. The versioned gold file is the retrieval regression set used for observability and recall-style evaluation.

ATA scripts:

```bash
python generate_ata_predictions.py
python evaluate_local_metrics.py --samples ata_eval_samples.jsonl --predictions ata_predictions.jsonl
```

The ATA workflow is:

1. Load fixed samples from `ata_eval_samples.jsonl`.
2. Run each question through the existing RAG pipeline.
3. Save generated answers and retrieved contexts.
4. Resume from already completed samples.
5. Compute local metrics without changing the evaluation dataset.

### RAGAS Evaluation Flow

The RAGAS path keeps the same pipeline but adds answer quality scoring.

```bash
python evaluate_ragas.py --input evaluation/ata/ata_eval_samples.jsonl --predictions-output evaluation/ata/ata_predictions.jsonl --output evaluation/ata/ragas/ata_ragas_evaluation_results.json --scores-output evaluation/ata/ragas/ata_ragas_score_matrix.csv
```

Outputs include:

- Generated answers and contexts JSONL
- RAGAS score matrix CSV
- Evaluation summary JSON

### Evaluation Folders

- `evaluation/ata/` stores ATA evaluation batches and ATA evaluation outputs.
- `evaluation_artifacts/` stores copied report files for relevancy, precision, and faithfulness runs.

---

## Repository Layout

```text
MultiDocumentRAG/
|-- data/
|   `-- documents/
|       |-- ANTI-MONEY LAUNDERING ACT, 2010.pdf
|       |-- Constitution_of_pakistan.pdf
|       |-- Pakistan Penal Code.pdf
|       `-- THE ANTI-TERRORISM ACT, 1997.pdf
|-- evaluation/
|   `-- ata/
|       |-- batches/
|       |-- local/
|       `-- ragas/
|-- evaluation_artifacts/
|-- rag/
|   |-- answer_service.py
|   |-- config.py
|   |-- context_builder.py
|   |-- embeddings.py
|   |-- intent_router.py
|   |-- prompt_builder.py
|   |-- ranker.py
|   |-- retriever.py
|   |-- schemas.py
|   |-- vector_store.py
|   `-- ...
|-- ingest.py
|-- query_cli.py
|-- evaluate_local_metrics.py
|-- evaluate_ragas.py
|-- generate_ata_predictions.py
|-- ata_eval_samples.jsonl
|-- ata_predictions.jsonl
|-- ata_evaluation_matrix.csv
|-- ata_evaluation_summary.json
|-- predictions.jsonl
|-- requirements.txt
`-- README.md
```

---

## Supported Question Types

- Section lookup
- Definition
- Punishment
- Comparison
- Scenario-based questions
- Typo-tolerant section lookup
- Invalid-section testing

Examples:

- What does Section 405 state?
- What punishment is provided for theft?
- Compare theft and robbery.
- A person found a lost wallet and kept it. Which provision may apply?
- Which section creates the Proscription Review Committee?

---

## Current Documents and Indexing

The repository currently includes four PDF documents:

- Pakistan Penal Code, 1860
- Constitution of Pakistan
- The Anti-Terrorism Act, 1997
- Anti-Money Laundering Act, 2010

The index is built in Qdrant using legal chunk metadata so retrieval can filter by document, section, provision type, and related fields.

---

## Important Scripts

- `ingest.py` - builds the Qdrant index from the PDFs
- `query_cli.py` - interactive question answering
- `rag/answer_service.py` - main retrieval and generation pipeline
- `generate_ata_predictions.py` - creates ATA predictions with resume support
- `evaluate_local_metrics.py` - computes local retrieval and answer metrics
- `evaluate_ragas.py` - runs RAGAS scoring
- `check_qdrant.py` - inspects the local Qdrant collection
- `deduplicate_qdrant.py` - removes duplicate vectors

---

## Running The Project

### Install

```bash
pip install -r requirements.txt
```

### Ingest Documents

```bash
python ingest.py
```

### Start The Assistant

```bash
python query_cli.py
```

### Generate ATA Predictions

```bash
python generate_ata_predictions.py
```

### Run Local ATA Metrics

```bash
python evaluate_local_metrics.py --samples ata_eval_samples.jsonl --predictions ata_predictions.jsonl --output-csv ata_evaluation_matrix.csv --output-summary ata_evaluation_summary.json
```

The local metrics output now also supports retrieval-focused fields such as recall@k, reciprocal rank, duplicate-context rate, exact-citation completeness, failure rate, and latency summaries when trace data is available.

### Run RAGAS Evaluation

```bash
python evaluate_ragas.py --input evaluation/ata/ata_eval_samples.jsonl --predictions-output evaluation/ata/ata_predictions.jsonl --output evaluation/ata/ragas/ata_ragas_evaluation_results.json --scores-output evaluation/ata/ragas/ata_ragas_score_matrix.csv
```

---

## Notes

- The ATA evaluation files are intentionally separate from the main `eval_samples.jsonl` workflow.
- The retrieval trace is optional and does not change the default answer response shape unless explicitly requested.
- The current pipeline is multi-document, but it still depends on high-quality legal metadata during ingestion.
- If the Qdrant collection is rebuilt, the README commands above still apply as long as the environment variables point to the same collection.

---

## Future Enhancements

- Stronger document-aware retrieval across all supported laws
- Cross-document reasoning
- Hybrid search with BM25 plus dense retrieval
- Better reranking
- Context compression
- Query decomposition
- Citation verification

---

## Backend Auth And Billing Flow

This section summarizes the Django backend that is currently implemented in the repository.

### API Surface

- `POST /api/v1/auth/signup/`
- `POST /api/v1/auth/resend-otp/`
- `POST /api/v1/auth/verify-email/`
- `POST /api/v1/auth/login/`
- `POST /api/v1/auth/refresh/`
- `GET /api/v1/auth/me/`
- `GET /api/v1/billing/plans/`
- `GET /api/v1/billing/subscription/`

### Signup Flow

1. The client submits `username`, `email`, `first_name`, `last_name`, `password`, and `password_confirm`.
2. Unknown fields such as `is_staff`, `is_superuser`, `is_active`, or `role` are rejected.
3. The backend normalizes the username and email to lowercase.
4. The password must match confirmation and pass the password strength checks.
5. A new user is created as inactive with `is_active=False`, `is_staff=False`, and `is_superuser=False`.
6. A 6-digit email verification OTP is generated and stored only as a hash.
7. The OTP is emailed to the user.
8. The API returns a masked email and OTP expiry countdown.

### Email Verification Flow

1. The client submits `email` and `otp`.
2. The backend checks whether the email belongs to an unverified account.
3. The OTP is validated against recent OTP history.
4. Expired, reused, or incorrect OTPs are rejected.
5. Too many failed attempts trigger a lockout for the current OTP.
6. When verification succeeds, the user account is activated.
7. The backend automatically assigns the default Free plan.

### Login Flow

1. The client submits `identifier` and `password`.
2. The identifier may be either a username or an email address.
3. The backend looks up the user case-insensitively.
4. The account must be active before login succeeds.
5. If the password is correct, the backend returns:
   - a JWT access token
   - a JWT refresh token
   - the user profile payload
6. The returned role is derived from the database:
   - `admin` if `is_staff=True` or `is_superuser=True`
   - `user` otherwise

### Refresh Flow

1. The client submits a refresh token.
2. The backend validates the token.
3. If valid, a new access token is returned.

### Profile Flow

1. `GET /api/v1/auth/me/` requires a valid JWT access token.
2. The backend rejects missing, expired, or invalid tokens.
3. The backend rejects inactive users.
4. The response returns the current user profile plus the derived role.

### Billing Flow

#### Plans

1. `GET /api/v1/billing/plans/` is public.
2. Only active plans are returned.
3. The plan payload includes:
   - `id`
   - `name`
   - `code`
   - `price`
   - `billing_period`
   - `document_limit`
   - `query_limit`
   - `is_active`
   - `created_at`
   - `updated_at`

#### Subscription

1. `GET /api/v1/billing/subscription/` requires a valid JWT access token.
2. The account must be active.
3. The backend loads the current user subscription from PostgreSQL.
4. The subscription is rejected if:
   - the plan is inactive
   - the subscription is canceled
   - the subscription is expired
5. The response returns the subscription plus `queries_remaining`.

### Free Plan Logic

1. A default plan with code `FREE` is seeded in the database.
2. The Free plan is monthly, active, and set to:
   - `document_limit = 3`
   - `query_limit = 30`
3. After email verification, the user receives this Free plan automatically.
4. `queries_remaining` is computed from the plan limit minus `queries_used`.

### User Access

- Can register a new account.
- Can request OTP resend.
- Can verify email.
- Can log in.
- Can refresh access tokens.
- Can view their own profile.
- Can view available billing plans.
- Can view their own subscription.

### Admin Access

- The backend currently does not expose dedicated admin API routes in this repository.
- A user is treated as `admin` only when `is_staff=True` or `is_superuser=True`.
- That admin role is used in returned payloads, but it is not yet backed by separate admin endpoints or custom admin permissions here.

### Input Field Access

#### Signup

- Allowed: `username`, `email`, `first_name`, `last_name`, `password`, `password_confirm`
- Rejected: `is_staff`, `is_superuser`, `is_active`, `role`, and any unknown field

#### Login

- Allowed: `identifier`, `password`
- Rejected: everything else

#### Resend OTP

- Allowed: `email`

#### Verify Email

- Allowed: `email`, `otp`

#### Refresh Token

- Allowed: `refresh`

### Implementation Note

- The billing layer currently stores and exposes subscription data.
- The repository does not yet show request-time enforcement of document or query limits inside the RAG pipeline.

### Added Backend Modules

- `billing` now includes staff-only admin plan and subscription APIs under `/api/v1/admin/billing/` and `/api/v1/admin/subscriptions/`.
- `documents` now provides JWT-protected category create/list APIs under `/api/v1/document-categories/`.
- Each new API follows the existing layered pattern: controller for HTTP and permissions, service for queryset/business logic, serializer for validation and response shaping.
- Validation is strict: unknown fields are rejected, codes are normalized to uppercase, and duplicate plan/category data returns conflict errors instead of 500s.
- The new document category model is migration-backed and ready for future `LegalDocument.category` usage without adding that relation yet.

### Legal Document Admin APIs

- `POST /api/v1/documents/` creates a PDF-only legal document with staff JWT auth.
- `GET /api/v1/documents/` returns a paginated admin list with category, status, and search filters.
- `GET /api/v1/documents/<id>/` returns full document metadata, including checksum and ingestion error details.
- `PATCH /api/v1/documents/<id>/` allows only `title` and `category_id`; READY documents are re-queued to `PENDING` after a change.
- `DELETE /api/v1/documents/<id>/` archives the document, moves its file under `data/documents/archived/`, and removes matching Qdrant points by `document_id` when available.
- Upload validation is strict: PDF signature, MIME type, extension, size limit, and checksum duplication are all checked before save.
