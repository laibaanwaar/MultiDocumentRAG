from __future__ import annotations

import argparse
import csv
import json
import math
import re
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from sentence_transformers import SentenceTransformer


DOCUMENT_ID_ALIASES = {
    "PPC": "ppc_1860",
    "Constitution": "constitution_1973",
    "ATA": "ata_1997",
    "AMLA": "amla_2010",
    "AMLA|": "amla_2010",
}

REJECTION_PHRASES = (
    "not found",
    "does not exist",
    "doesn't exist",
    "no provision",
    "not contain",
    "not retrieved",
    "could not find",
    "unable to find",
    "was not retrieved",
    "no such section",
    "no such article",
)


@dataclass
class LocalSample:
    sample_id: str
    sample_key: str
    raw_id: str
    dataset_version: str | None
    category: str
    question: str
    reference: str
    expected_document_id: str | None
    expected_provision_type: str | None
    expected_provision_numbers: list[str]
    expected_supporting_ids: list[str] = field(default_factory=list)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _normalize_number(value: Any) -> str:
    return str(value).strip().upper()


def _parse_expected_provision_numbers(row: dict[str, Any]) -> list[str]:
    values = row.get("expected_provision_numbers")
    if values is None:
        values = row.get("expected_provision_number")

    if values is None:
        return []

    if isinstance(values, list):
        items = values
    elif isinstance(values, str) and "," in values:
        items = [part.strip() for part in values.split(",")]
    else:
        items = [values]

    cleaned: list[str] = []
    seen: set[str] = set()
    for item in items:
        number = _normalize_number(item)
        if number and number not in seen:
            seen.add(number)
            cleaned.append(number)
    return cleaned


def _expected_provision_numbers_from_row(row: dict[str, Any]) -> list[str]:
    numbers = _parse_expected_provision_numbers(row)
    if numbers:
        return numbers

    return []


def _parse_string_list(value: Any) -> list[str]:
    if value is None:
        return []

    if isinstance(value, list):
        items = value
    elif isinstance(value, str) and "|" in value:
        items = [part.strip() for part in value.split("|")]
    elif isinstance(value, str) and "," in value:
        items = [part.strip() for part in value.split(",")]
    else:
        items = [value]

    cleaned: list[str] = []
    seen: set[str] = set()
    for item in items:
        text = str(item).strip()
        if text and text not in seen:
            seen.add(text)
            cleaned.append(text)

    return cleaned


def _parse_expected_supporting_ids(row: dict[str, Any]) -> list[str]:
    values = (
        row.get("expected_supporting_ids")
        or row.get("expected_supporting_provision_ids")
        or row.get("expected_supporting_chunk_ids")
    )
    return _parse_string_list(values)


def _normalize_supporting_id(
    document_id: str | None,
    provision_type: str | None,
    provision_number: str | None,
) -> str | None:
    if not document_id or not provision_type or not provision_number:
        return None

    return "::".join(
        [
            str(document_id).strip().lower(),
            str(provision_type).strip().lower(),
            str(provision_number).strip().upper(),
        ]
    )


def _supporting_ids_from_sample(sample: "LocalSample") -> set[str]:
    if sample.expected_supporting_ids:
        return {
            str(item).strip()
            for item in sample.expected_supporting_ids
            if str(item).strip()
        }

    normalized: set[str] = set()
    for number in sample.expected_provision_numbers:
        support_id = _normalize_supporting_id(
            sample.expected_document_id,
            sample.expected_provision_type,
            number,
        )
        if support_id:
            normalized.add(support_id)
    return normalized


def _first_str_value(row: dict[str, Any], keys: list[str]) -> str:
    for key in keys:
        value = row.get(key)
        if value is None:
            continue

        text = str(value).strip()
        if text:
            return text

    return ""


def _load_samples(path: Path) -> list[LocalSample]:
    rows = _read_jsonl(path)
    samples: list[LocalSample] = []

    for row in rows:
        sample_key = _first_str_value(
            row,
            ["sample_key", "sample_id", "id"],
        )
        sample_id = _first_str_value(
            row,
            ["sample_id", "sample_key", "id"],
        )
        raw_id = _first_str_value(
            row,
            ["id", "sample_id", "sample_key"],
        )

        if not sample_id and not sample_key and not raw_id:
            raise ValueError(
                "Each sample must contain a non-empty sample_key, sample_id, or id."
            )

        if not sample_key:
            sample_key = sample_id or raw_id

        if not sample_id:
            sample_id = sample_key or raw_id

        if not raw_id:
            raw_id = sample_id or sample_key

        samples.append(
            LocalSample(
                sample_id=sample_id,
                sample_key=sample_key,
                raw_id=raw_id,
                dataset_version=(
                    str(row.get("dataset_version")).strip()
                    if row.get("dataset_version") is not None
                    else None
                ),
                category=str(row.get("category") or "").strip(),
                question=str(row.get("question") or "").strip(),
                reference=str(row.get("reference") or "").strip(),
                expected_document_id=(
                    str(row.get("expected_document_id")).strip()
                    if row.get("expected_document_id") is not None
                    else None
                ),
                expected_provision_type=(
                    str(row.get("expected_provision_type")).strip().lower()
                    if row.get("expected_provision_type") is not None
                    else None
                ),
                expected_provision_numbers=_expected_provision_numbers_from_row(row),
                expected_supporting_ids=_parse_expected_supporting_ids(row),
            )
        )

    return samples


def _load_predictions(path: Path) -> dict[str, dict[str, Any]]:
    rows = _read_jsonl(path)
    predictions: dict[str, dict[str, Any]] = {}

    for row in rows:
        prediction_key = _first_str_value(
            row,
            ["sample_key", "sample_id", "id"],
        )

        if not prediction_key:
            continue

        predictions[prediction_key] = row

    return predictions


def _resolve_prediction(
    sample: LocalSample,
    predictions: dict[str, dict[str, Any]],
) -> dict[str, Any] | None:
    for key in (
        sample.sample_key,
        sample.sample_id,
        sample.raw_id,
    ):
        if key and key in predictions:
            return predictions[key]

    return None


def _map_document_label(label: str) -> str | None:
    return DOCUMENT_ID_ALIASES.get(label.strip(), None)


def _parse_context(context: Any) -> list[dict[str, str]]:
    if isinstance(context, dict):
        metadata = context.get("metadata") if isinstance(context.get("metadata"), dict) else context
        document_id = metadata.get("document_id")
        provision_type = metadata.get("provision_type")
        provision_number = metadata.get("provision_number")

        if document_id or provision_type or provision_number:
            return [
                {
                    "document_id": str(document_id).strip().lower()
                    if document_id is not None
                    else "",
                    "provision_type": str(provision_type).strip().lower()
                    if provision_type is not None
                    else "",
                    "provision_number": str(provision_number).strip().upper()
                    if provision_number is not None
                    else "",
                    "text": str(context.get("text") or context.get("page_content") or "").strip(),
                }
            ]

    text = str(context or "").strip()
    if not text:
        return []

    contexts: list[dict[str, str]] = []
    chunks = re.split(r"\n(?=[A-Z][A-Za-z ]+\s+\|\s+(?:Section|Article)\s+)", text)

    for chunk in chunks:
        match = re.match(
            r"^(?P<label>[A-Za-z]+)\s*\|\s*(?P<ptype>Section|Article)\s*(?P<number>[A-Za-z0-9\-]+)",
            chunk.strip(),
            flags=re.IGNORECASE,
        )
        if not match:
            continue

        label = match.group("label").strip()
        document_id = _map_document_label(label)
        contexts.append(
            {
                "document_id": document_id or label.lower(),
                "provision_type": match.group("ptype").lower(),
                "provision_number": match.group("number").upper(),
                "text": chunk.strip(),
            }
        )

    return contexts


def _flatten_contexts(retrieved_contexts: Any) -> list[dict[str, str]]:
    if not isinstance(retrieved_contexts, list):
        retrieved_contexts = [retrieved_contexts]

    parsed: list[dict[str, str]] = []
    for context in retrieved_contexts:
        parsed.extend(_parse_context(context))
    return parsed


def _count_retrieved_contexts(retrieved_contexts: Any) -> int:
    if not isinstance(retrieved_contexts, list):
        return 1 if str(retrieved_contexts).strip() else 0

    return sum(1 for context in retrieved_contexts if str(context).strip())


def _has_citation(response: str) -> bool:
    return bool(re.search(r"\[Source\s+\d+\]", response))


def _get_generated_response(prediction: dict[str, Any]) -> str:
    return str(
        prediction.get("response")
        or prediction.get("answer")
        or prediction.get("generated_answer")
        or ""
    ).strip()


def _extract_retrieval_trace(
    prediction: dict[str, Any],
) -> dict[str, Any] | None:
    trace = prediction.get("retrieval_trace")

    if isinstance(trace, str):
        trace = trace.strip()
        if not trace:
            return None
        try:
            loaded = json.loads(trace)
        except json.JSONDecodeError:
            return None
        if isinstance(loaded, dict):
            return loaded
        return None

    if isinstance(trace, dict):
        return trace

    return None


def _trace_context_provision_identity(context: dict[str, Any]) -> str:
    provision_identity = str(
        context.get("provision_identity") or ""
    ).strip()
    if provision_identity:
        return provision_identity

    provision_identity = _normalize_supporting_id(
        str(context.get("document_id") or "").strip(),
        str(context.get("provision_type") or "").strip(),
        str(context.get("provision_number") or "").strip(),
    )
    if provision_identity:
        return provision_identity

    return str(context.get("text") or "").strip()


def _trace_context_chunk_identity(context: dict[str, Any]) -> str:
    chunk_id = str(context.get("chunk_id") or "").strip()
    if chunk_id:
        return chunk_id

    return _trace_context_provision_identity(context)


def _trace_context_support_unit(context: dict[str, Any]) -> str:
    provision_identity = str(
        context.get("provision_identity") or ""
    ).strip()
    if provision_identity:
        return provision_identity

    chunk_id = str(context.get("chunk_id") or "").strip()
    if chunk_id:
        return chunk_id

    return ""


def _parsed_context_identity(context: dict[str, str]) -> str:
    provision_identity = _normalize_supporting_id(
        context.get("document_id"),
        context.get("provision_type"),
        context.get("provision_number"),
    )
    if provision_identity:
        return provision_identity

    return str(context.get("text") or "").strip()


def _trace_contexts(
    prediction: dict[str, Any],
    trace: dict[str, Any] | None,
) -> list[dict[str, Any]]:
    if trace is not None:
        retrieval = trace.get("retrieval")
        if isinstance(retrieval, dict):
            selected_contexts = retrieval.get("selected_contexts")
            if isinstance(selected_contexts, list):
                contexts: list[dict[str, Any]] = []
                for context in selected_contexts:
                    if isinstance(context, dict):
                        contexts.append(context)
                if contexts:
                    return contexts

    parsed = _flatten_contexts(prediction.get("retrieved_contexts", []))
    return [
        {
            **context,
            "provision_identity": _parsed_context_identity(context),
            "chunk_id": _parsed_context_identity(context),
        }
        for context in parsed
    ]


def _source_identity(source: dict[str, Any]) -> str:
    chunk_id = str(source.get("chunk_id") or "").strip()
    if chunk_id:
        return chunk_id

    return _normalize_supporting_id(
        str(source.get("document_id") or "").strip(),
        str(source.get("provision_type") or "").strip(),
        str(source.get("provision_number") or "").strip(),
    ) or ""


def _unique_support_units(contexts: list[dict[str, Any]]) -> list[str]:
    seen: set[str] = set()
    support_units: list[str] = []

    for context in contexts:
        support_unit = _trace_context_support_unit(context)
        if support_unit and support_unit not in seen:
            seen.add(support_unit)
            support_units.append(support_unit)

    return support_units


def _safe_percentile(values: list[float], percentile: float) -> float | None:
    cleaned = [value for value in values if not math.isnan(value)]
    if not cleaned:
        return None
    return float(np.percentile(cleaned, percentile))


def _is_invalid_rejection(response: str) -> bool:
    normalized = response.lower()
    return any(phrase in normalized for phrase in REJECTION_PHRASES)


INVALID_PROVISION_CATEGORIES = {
    "invalid_section",
    "nonexistent_provision",
}


def _is_invalid_provision_category(category: str) -> bool:
    return category in INVALID_PROVISION_CATEGORIES


def _cosine_similarity_matrix(
    model: SentenceTransformer,
    questions: list[str],
    responses: list[str],
) -> list[float]:
    question_embeddings = model.encode(
        questions,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=False,
    )
    response_embeddings = model.encode(
        responses,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=False,
    )

    return [
        float(np.dot(question_vec, response_vec))
        for question_vec, response_vec in zip(
            question_embeddings,
            response_embeddings,
            strict=True,
        )
    ]


def _contains_expected_document(
    parsed_contexts: list[dict[str, str]],
    expected_document_id: str | None,
) -> bool:
    if not expected_document_id:
        return False

    return any(
        context.get("document_id") == expected_document_id
        for context in parsed_contexts
    )


def _contains_expected_provisions(
    parsed_contexts: list[dict[str, str]],
    expected_document_id: str | None,
    expected_provision_type: str | None,
    expected_provision_numbers: list[str],
) -> bool:
    if not expected_provision_numbers:
        return False

    normalized_expected = {number.upper() for number in expected_provision_numbers}

    found_numbers: set[str] = set()
    for context in parsed_contexts:
        if expected_document_id and context.get("document_id") != expected_document_id:
            continue

        if expected_provision_type and context.get("provision_type") != expected_provision_type:
            continue

        number = context.get("provision_number", "").upper()
        if number in normalized_expected:
            found_numbers.add(number)

    return found_numbers == normalized_expected


def _count_matched_contexts(
    parsed_contexts: list[dict[str, str]],
    expected_document_id: str | None,
    expected_provision_type: str | None,
    expected_provision_numbers: list[str],
) -> int:
    if not expected_provision_numbers:
        return 0

    normalized_expected = {number.upper() for number in expected_provision_numbers}
    matched = 0

    for context in parsed_contexts:
        if expected_document_id and context.get("document_id") != expected_document_id:
            continue

        if expected_provision_type and context.get("provision_type") != expected_provision_type:
            continue

        if context.get("provision_number", "").upper() in normalized_expected:
            matched += 1

    return matched


def _expected_provision_count(expected_provision_numbers: list[str]) -> int:
    return len({number.upper() for number in expected_provision_numbers})


def _normalized_expected_provision_set(
    expected_provision_numbers: list[str],
) -> set[str]:
    return {
        number.upper()
        for number in expected_provision_numbers
        if str(number).strip()
    }


def _normalized_retrieved_provision_set(
    parsed_contexts: list[dict[str, str]],
    expected_document_id: str | None,
    expected_provision_type: str | None,
) -> set[str]:
    retrieved_numbers: set[str] = set()

    for context in parsed_contexts:
        if expected_document_id and context.get("document_id") != expected_document_id:
            continue

        if expected_provision_type and context.get("provision_type") != expected_provision_type:
            continue

        number = context.get("provision_number", "").strip().upper()
        if number:
            retrieved_numbers.add(number)

    return retrieved_numbers


def _hit_at_k(
    parsed_contexts: list[dict[str, str]],
    expected_document_id: str | None,
    expected_provision_type: str | None,
    expected_provision_numbers: list[str],
    k: int,
) -> bool:
    top_contexts = parsed_contexts[:k]
    if not top_contexts:
        return False

    if expected_document_id and not any(
        context.get("document_id") == expected_document_id
        for context in top_contexts
    ):
        return False

    if not expected_provision_numbers:
        return bool(expected_document_id)

    normalized_expected = {number.upper() for number in expected_provision_numbers}
    found_numbers: set[str] = set()

    for context in top_contexts:
        if expected_document_id and context.get("document_id") != expected_document_id:
            continue

        if expected_provision_type and context.get("provision_type") != expected_provision_type:
            continue

        number = context.get("provision_number", "").upper()
        if number in normalized_expected:
            found_numbers.add(number)

    return found_numbers == normalized_expected


def _safe_mean(values: list[float]) -> float | None:
    cleaned = [value for value in values if not math.isnan(value)]
    if not cleaned:
        return None
    return float(sum(cleaned) / len(cleaned))


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return

    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)


def build_metrics(
    samples: list[LocalSample],
    predictions: dict[str, dict[str, Any]],
    model: SentenceTransformer,
    hit_k: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    matrix_rows: list[dict[str, Any]] = []

    questions: list[str] = []
    responses: list[str] = []
    usable_samples: list[tuple[LocalSample, dict[str, Any]]] = []

    for sample in samples:
        prediction = _resolve_prediction(sample, predictions) or {}
        response = _get_generated_response(prediction)
        questions.append(sample.question)
        responses.append(response)
        usable_samples.append((sample, prediction))

    cosine_scores = _cosine_similarity_matrix(model, questions, responses)

    document_scores: list[float] = []
    provision_scores: list[float] = []
    hit_scores: list[float] = []
    citation_scores: list[float] = []
    invalid_rejection_scores: list[float] = []
    matched_context_counts: list[int] = []
    expected_provision_counts: list[int] = []
    context_precision_scores: list[float] = []
    context_recall_scores: list[float] = []
    retrieval_recall_scores: list[float] = []
    reciprocal_rank_scores: list[float] = []
    duplicate_rate_scores: list[float] = []
    retrieval_failed_scores: list[float] = []
    no_match_scores: list[float] = []
    exact_citation_scores: list[float] = []
    retrieval_latencies: list[float] = []
    ranking_latencies: list[float] = []
    selection_latencies: list[float] = []
    generation_latencies: list[float] = []
    total_latencies: list[float] = []

    for index, (sample, prediction) in enumerate(usable_samples):
        response = _get_generated_response(prediction)
        trace = _extract_retrieval_trace(prediction)
        trace_contexts = _trace_contexts(prediction, trace)
        parsed_contexts = _flatten_contexts(
            prediction.get("retrieved_contexts", [])
        )
        total_retrieved_contexts = len(trace_contexts)
        expected_supporting_set = _supporting_ids_from_sample(sample)
        retrieved_supporting_ids = _unique_support_units(trace_contexts)
        retrieved_chunk_ids = [
            _trace_context_chunk_identity(context)
            for context in trace_contexts
        ]
        source_supporting_ids = [
            supporting_id
            for source in (
                prediction.get("sources")
                if isinstance(prediction.get("sources"), list)
                else []
            )
            if isinstance(source, dict)
            for supporting_id in [_source_identity(source)]
            if supporting_id
        ]
        matched_supporting_set = (
            expected_supporting_set
            & set(
                retrieved_supporting_ids
            )
        )
        expected_provision_count = len(expected_supporting_set)
        matched_context_count = len(matched_supporting_set)
        duplicate_context_rate = (
            1.0
            - (
                len(set(retrieved_chunk_ids))
                / len(retrieved_chunk_ids)
            )
            if retrieved_chunk_ids
            else 0.0
        )
        top_k_ids = retrieved_supporting_ids[:hit_k]
        top_k_hit = bool(
            expected_supporting_set
            and expected_supporting_set.intersection(top_k_ids)
        )
        first_match_rank = next(
            (
                position
                for position, identity in enumerate(
                    retrieved_supporting_ids,
                    start=1,
                )
                if identity in expected_supporting_set
            ),
            None,
        )
        retrieval_recall_at_k = (
            len(
                expected_supporting_set.intersection(top_k_ids)
            )
            / expected_provision_count
            if expected_provision_count
            else 0.0
        )
        reciprocal_rank = (
            1.0 / first_match_rank
            if first_match_rank
            else 0.0
        )
        exact_citation_completeness = (
            len(
                expected_supporting_set.intersection(
                    set(
                        item
                        for item in source_supporting_ids
                        if item
                    )
                )
            )
            / expected_provision_count
            if expected_provision_count
            else 0.0
        )
        trace_timings = (
            trace.get("timings_ms")
            if isinstance(trace, dict)
            else {}
        )
        retrieval_latency = (
            float(trace_timings.get("retrieval"))
            if isinstance(trace_timings, dict)
            and trace_timings.get("retrieval") is not None
            else None
        )
        ranking_latency = (
            float(trace_timings.get("ranking"))
            if isinstance(trace_timings, dict)
            and trace_timings.get("ranking") is not None
            else None
        )
        selection_latency = (
            float(trace_timings.get("selection"))
            if isinstance(trace_timings, dict)
            and trace_timings.get("selection") is not None
            else None
        )
        generation_latency = (
            float(trace_timings.get("generation"))
            if isinstance(trace_timings, dict)
            and trace_timings.get("generation") is not None
            else None
        )
        total_latency = (
            float(trace_timings.get("total"))
            if isinstance(trace_timings, dict)
            and trace_timings.get("total") is not None
            else None
        )
        retrieval_status = str(
            trace.get("retrieval_status") if isinstance(trace, dict) else ""
        ).strip().lower()
        retrieval_failed = 1.0 if retrieval_status == "error" else 0.0
        no_match = 1.0 if retrieval_status == "no_match" else 0.0
        document_hit = _contains_expected_document(
            parsed_contexts,
            sample.expected_document_id,
        )
        provision_hit = _contains_expected_provisions(
            parsed_contexts,
            sample.expected_document_id,
            sample.expected_provision_type,
            sample.expected_provision_numbers,
        )
        hit_at_k = _hit_at_k(
            parsed_contexts,
            sample.expected_document_id,
            sample.expected_provision_type,
            sample.expected_provision_numbers,
            hit_k,
        )
        citation_present = _has_citation(response)
        cosine_similarity = cosine_scores[index]
        context_precision_local = (
            matched_context_count / len(retrieved_supporting_ids)
            if retrieved_supporting_ids
            else 0.0
        )
        context_recall_local = (
            matched_context_count / expected_provision_count
            if expected_provision_count
            else 0.0
        )

        assert 0.0 <= context_precision_local <= 1.0
        assert 0.0 <= context_recall_local <= 1.0

        if _is_invalid_provision_category(sample.category):
            invalid_rejection = _is_invalid_rejection(response)
            invalid_rejection_scores.append(1.0 if invalid_rejection else 0.0)
            invalid_rejection_value = 1.0 if invalid_rejection else 0.0
        else:
            invalid_rejection_value = math.nan

        document_scores.append(1.0 if document_hit else 0.0)
        provision_scores.append(1.0 if provision_hit else 0.0)
        hit_scores.append(1.0 if hit_at_k else 0.0)
        citation_scores.append(1.0 if citation_present else 0.0)
        matched_context_counts.append(matched_context_count)
        expected_provision_counts.append(expected_provision_count)
        context_precision_scores.append(context_precision_local)
        context_recall_scores.append(context_recall_local)
        retrieval_recall_scores.append(retrieval_recall_at_k)
        reciprocal_rank_scores.append(reciprocal_rank)
        duplicate_rate_scores.append(duplicate_context_rate)
        retrieval_failed_scores.append(retrieval_failed)
        no_match_scores.append(no_match)
        exact_citation_scores.append(exact_citation_completeness)
        if retrieval_latency is not None:
            retrieval_latencies.append(retrieval_latency)
        if ranking_latency is not None:
            ranking_latencies.append(ranking_latency)
        if selection_latency is not None:
            selection_latencies.append(selection_latency)
        if generation_latency is not None:
            generation_latencies.append(generation_latency)
        if total_latency is not None:
            total_latencies.append(total_latency)

        matrix_rows.append(
            {
                "sample_id": sample.sample_id,
                "sample_key": sample.sample_key,
                "dataset_version": sample.dataset_version or "",
                "category": sample.category,
                "expected_document_id": sample.expected_document_id or "",
                "expected_provision_type": sample.expected_provision_type or "",
                "expected_provision_numbers": "|".join(sample.expected_provision_numbers),
                "document_accuracy": 1.0 if document_hit else 0.0,
                f"retrieval_hit_at_{hit_k}": 1.0 if hit_at_k else 0.0,
                "provision_accuracy": 1.0 if provision_hit else 0.0,
                "citation_presence": 1.0 if citation_present else 0.0,
                "answer_reference_cosine_similarity": cosine_similarity,
                "invalid_provision_rejection_accuracy": invalid_rejection_value,
                "prediction_present": 1.0 if prediction else 0.0,
                "retrieved_context_count": total_retrieved_contexts,
                "matched_context_count": matched_context_count,
                "expected_provision_count": expected_provision_count,
                "context_precision_local": context_precision_local,
                "context_recall_local": context_recall_local,
                f"retrieval_recall_at_{hit_k}": retrieval_recall_at_k,
                "reciprocal_rank": reciprocal_rank,
                "duplicate_context_rate": duplicate_context_rate,
                "retrieval_failed": retrieval_failed,
                "no_match": no_match,
                "exact_citation_completeness": exact_citation_completeness,
                "retrieval_latency_ms": retrieval_latency
                if retrieval_latency is not None
                else "",
                "ranking_latency_ms": ranking_latency
                if ranking_latency is not None
                else "",
                "selection_latency_ms": selection_latency
                if selection_latency is not None
                else "",
                "generation_latency_ms": generation_latency
                if generation_latency is not None
                else "",
                "total_latency_ms": total_latency
                if total_latency is not None
                else "",
            }
        )

    dataset_versions = sorted(
        {
            sample.dataset_version
            for sample in samples
            if sample.dataset_version
        }
    )
    if len(dataset_versions) == 1:
        dataset_version_value: str | list[str] | None = dataset_versions[0]
    elif dataset_versions:
        dataset_version_value = dataset_versions
    else:
        dataset_version_value = None

    summary = {
        "sample_count": len(samples),
        "dataset_version": dataset_version_value,
        "metrics": {
            "document_accuracy": _safe_mean(document_scores),
            f"retrieval_hit_at_{hit_k}": _safe_mean(hit_scores),
            "provision_accuracy": _safe_mean(provision_scores),
            "citation_presence": _safe_mean(citation_scores),
            "answer_reference_cosine_similarity": _safe_mean(cosine_scores),
            "invalid_provision_rejection_accuracy": _safe_mean(
                invalid_rejection_scores
            ),
            "context_precision_local": _safe_mean(context_precision_scores),
            "context_recall_local": _safe_mean(context_recall_scores),
            f"retrieval_recall_at_{hit_k}": _safe_mean(
                retrieval_recall_scores
            ),
            "reciprocal_rank": _safe_mean(reciprocal_rank_scores),
            "duplicate_context_rate": _safe_mean(duplicate_rate_scores),
            "retrieval_failed_rate": _safe_mean(
                retrieval_failed_scores
            ),
            "no_match_rate": _safe_mean(no_match_scores),
            "exact_citation_completeness": _safe_mean(
                exact_citation_scores
            ),
            "retrieval_latency_ms": _safe_mean(retrieval_latencies),
            "ranking_latency_ms": _safe_mean(ranking_latencies),
            "selection_latency_ms": _safe_mean(selection_latencies),
            "generation_latency_ms": _safe_mean(generation_latencies),
            "total_latency_ms": _safe_mean(total_latencies),
            "total_latency_ms_p50": _safe_percentile(
                total_latencies,
                50.0,
            ),
            "total_latency_ms_p95": _safe_percentile(
                total_latencies,
                95.0,
            ),
        },
        "counts": {
            "samples_with_predictions": sum(
                1 for sample in samples if _resolve_prediction(sample, predictions)
            ),
            "invalid_provision_samples": sum(
                1
                for sample in samples
                if _is_invalid_provision_category(sample.category)
            ),
            "nonexistent_provision_samples": sum(
                1 for sample in samples if sample.category == "nonexistent_provision"
            ),
            "retrieval_failed_samples": int(
                sum(retrieval_failed_scores)
            ),
            "no_match_samples": int(sum(no_match_scores)),
            "samples_with_traces": sum(
                1
                for sample in samples
                if _extract_retrieval_trace(
                    _resolve_prediction(sample, predictions) or {}
                )
            ),
        },
        "hit_k": hit_k,
    }

    return matrix_rows, summary


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Compute local metrics for existing RAG predictions."
    )
    parser.add_argument(
        "--samples",
        type=Path,
        default=Path("eval_samples.jsonl"),
        help="Input samples JSONL file.",
    )
    parser.add_argument(
        "--predictions",
        type=Path,
        default=Path("predictions.jsonl"),
        help="Existing predictions JSONL file.",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=Path("local_evaluation_matrix.csv"),
        help="Path to write the per-sample metric matrix.",
    )
    parser.add_argument(
        "--output-summary",
        type=Path,
        default=Path("local_evaluation_summary.json"),
        help="Path to write the aggregate summary JSON.",
    )
    parser.add_argument(
        "--hit-k",
        type=int,
        default=5,
        help="Top-K value to use for retrieval hit calculation.",
    )
    args = parser.parse_args()

    if not args.samples.exists():
        raise FileNotFoundError(f"Samples file not found: {args.samples}")

    if not args.predictions.exists():
        raise FileNotFoundError(f"Predictions file not found: {args.predictions}")

    samples = _load_samples(args.samples)
    predictions = _load_predictions(args.predictions)

    matched_predictions = sum(
        1 for sample in samples if _resolve_prediction(sample, predictions)
    )
    missing_predictions = len(samples) - matched_predictions

    print(f"loaded samples: {len(samples)}")
    print(f"loaded predictions: {len(predictions)}")
    print(f"matched predictions: {matched_predictions}")
    print(f"missing predictions: {missing_predictions}")

    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

    matrix_rows, summary = build_metrics(
        samples=samples,
        predictions=predictions,
        model=model,
        hit_k=args.hit_k,
    )

    _write_csv(args.output_csv, matrix_rows)
    _write_json(args.output_summary, summary)

    print(f"Saved matrix to {args.output_csv}")
    print(f"Saved summary to {args.output_summary}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
