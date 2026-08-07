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
- Main answer model: `llama-3.3-70b-versatile` via Groq
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

### ATA Evaluation Flow

The Anti-Terrorism Act evaluation flow now has its own dedicated files.

Working files:

- `ata_eval_samples.jsonl`
- `ata_predictions.jsonl`
- `ata_evaluation_matrix.csv`
- `ata_evaluation_summary.json`

These ATA files are the dedicated evaluation set and outputs for the Anti-Terrorism Act workflow.

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

### Run RAGAS Evaluation

```bash
python evaluate_ragas.py --input evaluation/ata/ata_eval_samples.jsonl --predictions-output evaluation/ata/ata_predictions.jsonl --output evaluation/ata/ragas/ata_ragas_evaluation_results.json --scores-output evaluation/ata/ragas/ata_ragas_score_matrix.csv
```

---

## Notes

- The ATA evaluation files are intentionally separate from the main `eval_samples.jsonl` workflow.
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
