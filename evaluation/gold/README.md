# Legal RAG Gold v1

This folder contains the versioned gold evaluation set used for F-12 retrieval observability.

Schema:
- `dataset_version`: stable version tag for the dataset
- `sample_key` / `id`: stable sample identifiers
- `question`: user-facing test question
- `reference`: ground-truth answer text
- `expected_document_id`: supporting legal document
- `expected_provision_type`: supporting provision type
- `expected_provision_numbers`: expected provision numbers
- `expected_supporting_ids`: canonical provision-level support IDs

The rows in `legal_rag_gold_v1.jsonl` were copied from the verified ATA local evaluation set and annotated with version metadata for retrieval regression testing.
