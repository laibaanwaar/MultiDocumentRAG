from __future__ import annotations

import json
from pathlib import Path

import evaluate_local_metrics as metrics_module
import evaluate_ragas as ragas_module
from evaluate_local_metrics import LocalSample, build_metrics, _load_samples


def _make_sample() -> LocalSample:
    return LocalSample(
        sample_id="s1",
        sample_key="s1",
        raw_id="s1",
        dataset_version="legal_rag_gold_v1",
        category="section_lookup",
        question="What does Section 7 cover?",
        reference="Section 7 covers punishment for acts of terrorism.",
        expected_document_id="ata_1997",
        expected_provision_type="section",
        expected_provision_numbers=["7"],
        expected_supporting_ids=["ata_1997::section::7"],
    )


def _make_prediction() -> dict[str, object]:
    return {
        "answer": "Section 7 covers punishment. [Source 1]",
        "response": "Section 7 covers punishment. [Source 1]",
        "retrieved_contexts": [
            "ATA | Section 7: Punishment for acts of terrorism.",
            "ATA | Section 7: Punishment for acts of terrorism.",
        ],
        "sources": [
            {
                "label": "Source 1",
                "document_id": "ata_1997",
                "provision_type": "section",
                "provision_number": "7",
                "chunk_number": 12,
            }
        ],
        "retrieval_trace": {
            "retrieval_status": "success",
            "timings_ms": {
                "retrieval": 5.0,
                "ranking": 3.0,
                "selection": 2.0,
                "generation": 10.0,
                "total": 20.0,
            },
            "retrieval": {
                "selected_contexts": [
                    {
                        "chunk_id": "chunk-a",
                        "provision_identity": "ata_1997::section::7",
                        "document_id": "ata_1997",
                        "provision_type": "section",
                        "provision_number": "7",
                        "estimated_token_count": 12,
                    },
                    {
                        "chunk_id": "chunk-a",
                        "provision_identity": "ata_1997::section::7",
                        "document_id": "ata_1997",
                        "provision_type": "section",
                        "provision_number": "7",
                        "estimated_token_count": 12,
                    },
                ]
            },
        },
    }


def _make_trace_prediction(
    *,
    supported_identity: str,
    chunk_ids: list[str],
    retrieval_status: str = "success",
    retrieved_contexts: list[str] | None = None,
    source_supporting_id: str | None = None,
) -> dict[str, object]:
    selected_contexts = [
        {
            "chunk_id": chunk_id,
            "provision_identity": supported_identity,
            "document_id": supported_identity.split("::", 1)[0],
            "provision_type": supported_identity.split("::")[1],
            "provision_number": supported_identity.split("::")[2],
            "estimated_token_count": 12,
        }
        for chunk_id in chunk_ids
    ]

    return {
        "answer": "Grounded answer. [Source 1]",
        "response": "Grounded answer. [Source 1]",
        "retrieved_contexts": (
            retrieved_contexts
            if retrieved_contexts is not None
            else ["ATA | Section 7: Punishment."]
        ),
        "sources": [
            {
                "label": "Source 1",
                "document_id": (
                    source_supporting_id.split("::", 1)[0]
                    if source_supporting_id
                    else supported_identity.split("::", 1)[0]
                ),
                "provision_type": (
                    source_supporting_id.split("::")[1]
                    if source_supporting_id
                    else supported_identity.split("::")[1]
                ),
                "provision_number": (
                    source_supporting_id.split("::")[2]
                    if source_supporting_id
                    else supported_identity.split("::")[2]
                ),
                "chunk_id": chunk_ids[0] if chunk_ids else "chunk-a",
            }
        ],
        "retrieval_trace": {
            "retrieval_status": retrieval_status,
            "timings_ms": {
                "retrieval": 5.0,
                "ranking": 3.0,
                "selection": 2.0,
                "generation": 10.0,
                "total": 20.0,
            },
            "retrieval": {
                "selected_contexts": selected_contexts,
            },
        },
    }


def test_versioned_gold_dataset_loads_version_and_supporting_ids() -> None:
    samples = _load_samples(
        Path("evaluation/gold/legal_rag_gold_v1.jsonl")
    )

    assert samples[0].dataset_version == "legal_rag_gold_v1"
    assert samples[0].expected_supporting_ids == [
        "ata_1997::section::1"
    ]
    assert len(samples) >= 10


def test_build_metrics_uses_retrieval_trace_for_recall_and_latency(
    monkeypatch,
) -> None:
    sample = _make_sample()
    prediction = _make_prediction()

    monkeypatch.setattr(
        metrics_module,
        "_cosine_similarity_matrix",
        lambda model, questions, responses: [0.5],
    )

    matrix_rows, summary = build_metrics(
        samples=[sample],
        predictions={"s1": prediction},
        model=object(),
        hit_k=5,
    )

    assert summary["dataset_version"] == "legal_rag_gold_v1"
    assert summary["metrics"]["retrieval_recall_at_5"] == 1.0
    assert summary["metrics"]["reciprocal_rank"] == 1.0
    assert summary["metrics"]["duplicate_context_rate"] == 0.5
    assert summary["metrics"]["context_precision_local"] == 1.0
    assert summary["metrics"]["context_recall_local"] == 1.0
    assert summary["metrics"]["retrieval_failed_rate"] == 0.0
    assert summary["metrics"]["no_match_rate"] == 0.0
    assert summary["metrics"]["exact_citation_completeness"] == 1.0
    assert summary["metrics"]["total_latency_ms"] == 20.0
    assert summary["metrics"]["total_latency_ms_p50"] == 20.0
    assert summary["metrics"]["total_latency_ms_p95"] == 20.0

    row = matrix_rows[0]
    assert row["dataset_version"] == "legal_rag_gold_v1"
    assert row["retrieval_recall_at_5"] == 1.0
    assert row["reciprocal_rank"] == 1.0
    assert row["duplicate_context_rate"] == 0.5
    assert row["matched_context_count"] == 1
    assert row["expected_provision_count"] == 1
    assert row["context_precision_local"] == 1.0
    assert row["context_recall_local"] == 1.0
    assert row["retrieval_failed"] == 0.0
    assert row["no_match"] == 0.0
    assert row["exact_citation_completeness"] == 1.0
    assert row["retrieval_latency_ms"] == 5.0
    assert row["ranking_latency_ms"] == 3.0
    assert row["selection_latency_ms"] == 2.0
    assert row["generation_latency_ms"] == 10.0
    assert row["total_latency_ms"] == 20.0


def test_build_metrics_requires_exact_support_identity_and_ignores_prefixes(
    monkeypatch,
) -> None:
    sample = LocalSample(
        sample_id="s2",
        sample_key="s2",
        raw_id="s2",
        dataset_version="legal_rag_gold_v1",
        category="section_lookup",
        question="What is Section 11EE?",
        reference="Section 11EE applies.",
        expected_document_id="ata_1997",
        expected_provision_type="section",
        expected_provision_numbers=["11EE"],
        expected_supporting_ids=["ata_1997::section::11EE"],
    )
    prediction = _make_trace_prediction(
        supported_identity="ata_1997::section::11E",
        chunk_ids=["chunk-prefix"],
        source_supporting_id="ata_1997::section::11E",
        retrieved_contexts=["ATA | Section 11E: Nearby text."],
    )

    monkeypatch.setattr(
        metrics_module,
        "_cosine_similarity_matrix",
        lambda model, questions, responses: [0.5],
    )

    matrix_rows, summary = build_metrics(
        samples=[sample],
        predictions={"s2": prediction},
        model=object(),
        hit_k=5,
    )

    assert summary["metrics"]["context_precision_local"] == 0.0
    assert summary["metrics"]["context_recall_local"] == 0.0
    assert matrix_rows[0]["matched_context_count"] == 0
    assert matrix_rows[0]["context_precision_local"] == 0.0
    assert matrix_rows[0]["context_recall_local"] == 0.0


def test_build_metrics_legacy_predictions_without_trace_still_work(
    monkeypatch,
) -> None:
    sample = _make_sample()
    prediction = {
        "answer": "Section 7 covers punishment.",
        "response": "Section 7 covers punishment.",
        "retrieved_contexts": [
            "ATA | Section 7: Punishment for acts of terrorism.",
            "ATA | Section 7: Punishment for acts of terrorism.",
        ],
        "sources": [
            {
                "label": "Source 1",
                "document_id": "ata_1997",
                "provision_type": "section",
                "provision_number": "7",
            }
        ],
    }

    monkeypatch.setattr(
        metrics_module,
        "_cosine_similarity_matrix",
        lambda model, questions, responses: [0.5],
    )

    matrix_rows, summary = build_metrics(
        samples=[sample],
        predictions={"s1": prediction},
        model=object(),
        hit_k=5,
    )

    assert summary["metrics"]["context_precision_local"] == 1.0
    assert summary["metrics"]["context_recall_local"] == 1.0
    assert matrix_rows[0]["matched_context_count"] == 1
    assert matrix_rows[0]["retrieved_context_count"] == 2


def test_ragas_prediction_row_requests_and_persists_retrieval_trace(
    monkeypatch,
) -> None:
    calls: list[dict[str, object]] = []

    def fake_answer_question(**kwargs):
        calls.append(kwargs)
        return {
            "answer": "Grounded answer.",
            "retrieved_contexts": ["context-a"],
            "question_type": "section_lookup",
            "retrieval_status": "success",
            "retrieval_trace": {
                "retrieval_status": "success",
                "events": [{"channel": "vector"}],
            },
            "sources": [{"label": "Source 1", "document_id": "ata_1997"}],
        }

    monkeypatch.setattr(
        ragas_module,
        "answer_question",
        fake_answer_question,
    )

    row = ragas_module._build_single_evaluation_row(
        sample_id=7,
        row={
            "question": "What does Section 7 cover?",
            "reference": "Section 7 covers punishment.",
        },
        retriever=object(),
        chat_model=object(),
    )

    assert calls
    assert calls[0]["include_trace"] is True
    assert row["response"] == "Grounded answer."
    assert row["retrieved_contexts"] == ["context-a"]
    assert row["retrieval_trace"] == {
        "retrieval_status": "success",
        "events": [{"channel": "vector"}],
    }
    assert row["retrieval_status"] == "success"
    assert row["sources"] == [{"label": "Source 1", "document_id": "ata_1997"}]
    assert row["sample_id"] == 7
    assert row["user_input"] == "What does Section 7 cover?"
    assert row["reference"] == "Section 7 covers punishment."
