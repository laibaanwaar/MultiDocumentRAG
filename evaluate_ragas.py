"""Evaluate the current RAG pipeline with RAGAS.

Input file format:
- CSV or JSONL
- Required fields:
  - question
  - ground_truth or reference

The script runs the existing RAG pipeline, captures the retrieved contexts,
and evaluates the generated answers with RAGAS metrics.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import time
import sys
import types
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from langchain_groq import ChatGroq

from rag.answer_service import answer_question, create_rag_components
from rag.embeddings import get_embedding_model


def _install_ragas_compat_shim() -> None:
    """Work around a version mismatch in this environment.

    The installed RAGAS package expects
    `langchain_community.chat_models.vertexai`, which is missing here.
    We provide a tiny placeholder module so the evaluator can import.
    """

    module_name = "langchain_community.chat_models.vertexai"

    if module_name in sys.modules:
        return

    shim = types.ModuleType(module_name)

    class ChatVertexAI:  # pragma: no cover - compatibility shim
        pass

    shim.ChatVertexAI = ChatVertexAI
    sys.modules[module_name] = shim


def _load_ragas_components():
    try:
        from ragas.evaluation import evaluate
        from ragas.metrics import (
            answer_relevancy,
            context_precision,
            context_recall,
            faithfulness,
        )
        return evaluate, [
            answer_relevancy,
            context_precision,
            context_recall,
            faithfulness,
        ]
    except ModuleNotFoundError as error:
        if "langchain_community.chat_models.vertexai" not in str(error):
            raise

        _install_ragas_compat_shim()

        from ragas.evaluation import evaluate
        from ragas.metrics import (
            answer_relevancy,
            context_precision,
            context_recall,
            faithfulness,
        )

        return evaluate, [
            answer_relevancy,
            context_precision,
            context_recall,
            faithfulness,
        ]


def _configure_ragas_metrics(metrics: list[Any]) -> list[Any]:
    for metric in metrics:
        if getattr(metric, "name", "") == "answer_relevancy":
            setattr(metric, "strictness", 1)

    return metrics


def _build_ragas_llm() -> ChatGroq:
    model_name = os.getenv(
        "RAGAS_MODEL",
        "llama-3.1-8b-instant",
    )
    max_tokens = int(os.getenv("RAGAS_MAX_TOKENS", "4096"))
    reasoning_effort = os.getenv("RAGAS_REASONING_EFFORT", "low")

    return ChatGroq(
        model=model_name,
        temperature=0,
        max_tokens=max_tokens,
        reasoning_effort=reasoning_effort,
        timeout=300,
        max_retries=2,
    )


def _build_ragas_run_config():
    from ragas.run_config import RunConfig

    return RunConfig(
        timeout=300,
        max_retries=2,
        max_wait=120,
        max_workers=1,
    )


def _read_rows(path: Path) -> list[dict[str, Any]]:
    suffix = path.suffix.lower()

    if suffix == ".jsonl":
        rows: list[dict[str, Any]] = []
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    rows.append(json.loads(line))
        return rows

    if suffix == ".json":
        with path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
        if isinstance(data, list):
            return data
        if isinstance(data, dict) and isinstance(data.get("rows"), list):
            return data["rows"]
        raise ValueError(
            "JSON input must be a list of rows or an object with a 'rows' key."
        )

    if suffix == ".csv":
        with path.open("r", encoding="utf-8", newline="") as handle:
            return list(csv.DictReader(handle))

    raise ValueError(
        "Unsupported input format. Use .csv, .json, or .jsonl."
    )


def _normalize_sample_id(value: Any) -> int | None:
    try:
        return int(str(value).strip())
    except (TypeError, ValueError):
        return None


def _normalize_row(row: dict[str, Any]) -> dict[str, str]:
    question = str(
        row.get("question")
        or row.get("user_input")
        or ""
    ).strip()

    reference = str(
        row.get("ground_truth")
        or row.get("reference")
        or row.get("answer")
        or ""
    ).strip()

    if not question:
        raise ValueError("Each row must contain a question.")

    if not reference:
        raise ValueError(
            "Each row must contain a ground truth answer in 'ground_truth' or 'reference'."
        )

    return {
        "question": question,
        "reference": reference,
    }


def _load_dataset_rows(path: Path) -> list[dict[str, Any]]:
    raw_rows = _read_rows(path)
    normalized_rows: list[dict[str, Any]] = []

    for index, row in enumerate(raw_rows, start=1):
        normalized_row = _normalize_row(row)
        sample_id = _normalize_sample_id(row.get("sample_id")) or index
        normalized_row["sample_id"] = sample_id
        normalized_rows.append(normalized_row)

    return normalized_rows


def _load_prediction_rows(path: Path) -> dict[int, dict[str, Any]]:
    if not path.exists():
        return {}

    completed_rows: dict[int, dict[str, Any]] = {}

    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue

            row = json.loads(line)
            sample_id = _normalize_sample_id(row.get("sample_id"))

            if sample_id is None:
                continue

            completed_rows[sample_id] = row

    return completed_rows


def _load_score_rows(path: Path) -> dict[int, dict[str, Any]]:
    if not path.exists():
        return {}

    completed_rows: dict[int, dict[str, Any]] = {}

    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue

            row = json.loads(line)
            sample_id = _normalize_sample_id(row.get("sample_id"))

            if sample_id is None:
                continue

            completed_rows[sample_id] = row

    return completed_rows


def _append_prediction(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")
        handle.flush()


def _append_score_row(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")
        handle.flush()


def _truncate_contexts(
    contexts: Any,
    max_contexts: int,
    max_context_chars: int,
) -> list[str]:
    if max_contexts <= 0 or max_context_chars <= 0:
        return []

    if not isinstance(contexts, list):
        contexts = [contexts]

    normalized_contexts: list[str] = []

    for context in contexts[:max_contexts]:
        text = str(context).strip()
        if not text:
            continue

        if len(text) > max_context_chars:
            text = text[:max_context_chars].rstrip()

        normalized_contexts.append(text)

    return normalized_contexts


def _build_scoring_row(
    sample_id: int,
    row: dict[str, Any],
    prediction_row: dict[str, Any],
    max_contexts: int,
    max_context_chars: int,
) -> dict[str, Any]:
    return {
        "user_input": row["question"],
        "reference": row["reference"],
        "response": prediction_row["response"],
        "retrieved_contexts": _truncate_contexts(
            prediction_row.get("retrieved_contexts", []),
            max_contexts=max_contexts,
            max_context_chars=max_context_chars,
        ),
        "question_type": prediction_row.get("question_type"),
        "sample_id": sample_id,
    }


def _score_value_is_valid(value: Any) -> bool:
    if value is None:
        return False

    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return False

    return not math.isnan(numeric)


def _score_row_is_valid(
    row: dict[str, Any],
    metric_names: list[str],
) -> bool:
    return all(
        _score_value_is_valid(row.get(metric_name))
        for metric_name in metric_names
    )


def _average_metric_scores(
    rows: list[dict[str, Any]],
    metric_names: list[str],
) -> list[tuple[str, float]]:
    averages: list[tuple[str, float]] = []

    for metric_name in metric_names:
        values: list[float] = []

        for row in rows:
            value = row.get(metric_name)
            if not _score_value_is_valid(value):
                continue

            values.append(float(value))

        averages.append(
            (
                metric_name,
                sum(values) / len(values) if values else math.nan,
            )
        )

    return averages


def _score_single_sample(
    score_row: dict[str, Any],
    evaluate,
    metrics: list[Any],
    ragas_llm: ChatGroq,
    embedding_model: Any,
    run_config: Any,
    show_progress: bool,
) -> dict[str, Any]:
    from ragas.dataset_schema import EvaluationDataset

    dataset = EvaluationDataset.from_list([score_row])

    result = evaluate(
        dataset=dataset,
        metrics=metrics,
        llm=ragas_llm,
        embeddings=embedding_model,
        run_config=run_config,
        show_progress=show_progress,
        batch_size=1,
        raise_exceptions=True,
    )

    if not getattr(result, "scores", None):
        raise ValueError("RAGAS did not return any scores.")

    scored_row = dict(result.scores[0])
    scored_row["sample_id"] = score_row["sample_id"]
    scored_row["question"] = score_row["user_input"]
    return scored_row


def _score_single_sample_with_retry(
    sample_id: int,
    score_row: dict[str, Any],
    evaluate,
    metrics: list[Any],
    ragas_llm: ChatGroq,
    embedding_model: Any,
    run_config: Any,
    show_progress: bool,
) -> dict[str, Any]:
    while True:
        try:
            return _score_single_sample(
                score_row=score_row,
                evaluate=evaluate,
                metrics=metrics,
                ragas_llm=ragas_llm,
                embedding_model=embedding_model,
                run_config=run_config,
                show_progress=show_progress,
            )
        except Exception as error:
            if not _is_rate_limit_error(error):
                raise

            _log_rate_limit_error(sample_id, error)
            time.sleep(_rate_limit_delay_seconds(error))


def _retry_after_seconds(error: Exception) -> float | None:
    response = getattr(error, "response", None)
    header_sources = [
        getattr(response, "headers", None),
        getattr(error, "headers", None),
    ]

    for headers in header_sources:
        if not headers:
            continue

        for key in ("retry-after", "Retry-After"):
            value = headers.get(key)
            if value is None:
                continue

            if isinstance(value, (int, float)):
                return max(float(value), 0.0)

            text = str(value).strip()

            try:
                return max(float(text), 0.0)
            except ValueError:
                pass

            try:
                retry_time = parsedate_to_datetime(text)
            except (TypeError, ValueError, IndexError):
                continue

            if retry_time.tzinfo is None:
                retry_time = retry_time.replace(tzinfo=timezone.utc)

            delta = (
                retry_time
                - datetime.now(tz=retry_time.tzinfo)
            ).total_seconds()
            if delta > 0:
                return delta

    return None


def _rate_limit_headers(error: Exception) -> dict[str, Any]:
    response = getattr(error, "response", None)
    headers = getattr(response, "headers", None) or getattr(
        error,
        "headers",
        {},
    )

    if not headers:
        return {}

    relevant_keys = (
        "retry-after",
        "Retry-After",
        "x-ratelimit-reset-tokens",
        "X-RateLimit-Reset-Tokens",
    )

    return {
        key: headers.get(key)
        for key in relevant_keys
        if headers.get(key) is not None
    }


def _log_rate_limit_error(
    sample_id: int,
    error: Exception,
) -> None:
    print(f"Sample {sample_id} Groq 429 error: {error}")
    headers = _rate_limit_headers(error)
    if headers:
        print(
            f"Sample {sample_id} rate-limit headers: {headers}"
        )


def _rate_limit_delay_seconds(error: Exception) -> float:
    retry_after = _retry_after_seconds(error)

    if retry_after is None:
        return 70.0

    return retry_after + 5.0


def _is_rate_limit_error(error: Exception) -> bool:
    status_code = getattr(error, "status_code", None)

    if status_code == 429:
        return True

    response = getattr(error, "response", None)
    if getattr(response, "status_code", None) == 429:
        return True

    return "429" in str(error) or "rate limit" in str(error).lower()


def _build_single_evaluation_row(
    sample_id: int,
    row: dict[str, str],
    retriever,
    chat_model,
) -> dict[str, Any]:
    result = answer_question(
        question=row["question"],
        retriever=retriever,
        chat_model=chat_model,
        include_trace=True,
    )

    retrieval_trace = result.get("retrieval_trace") or {}

    return {
        "user_input": row["question"],
        "reference": row["reference"],
        "response": result["answer"],
        "retrieved_contexts": result.get("retrieved_contexts", []),
        "question_type": result.get("question_type"),
        "sample_id": sample_id,
        "retrieval_trace": result.get("retrieval_trace"),
        "retrieval_status": result.get(
            "retrieval_status",
            retrieval_trace.get("retrieval_status"),
        ),
        "sources": result.get("sources", []),
    }


def _evaluate_row_with_retry(
    sample_id: int,
    row: dict[str, str],
    retriever,
    chat_model,
    backoff_schedule: list[int],
) -> dict[str, Any] | None:
    attempt = 0

    while True:
        try:
            return _build_single_evaluation_row(
                sample_id=sample_id,
                row=row,
                retriever=retriever,
                chat_model=chat_model,
            )
        except Exception as error:
            if not _is_rate_limit_error(error):
                print(
                    f"Sample {sample_id} failed with a non-rate-limit error: {error}"
                )
                return None

            if attempt >= len(backoff_schedule):
                print(
                    f"Sample {sample_id} hit Groq rate limits repeatedly and was skipped after retries."
                )
                return None

            retry_after = _retry_after_seconds(error)
            delay_seconds = (
                retry_after
                if retry_after is not None
                else backoff_schedule[attempt]
            )

            print(
                f"Sample {sample_id} hit a Groq rate limit. Retrying in {int(delay_seconds)} seconds..."
            )
            time.sleep(max(float(delay_seconds), 0.0))
            attempt += 1

    return None


def _metric_objects(metric_names: list[str]) -> list[Any]:
    evaluate, available_metrics = _load_ragas_components()
    metric_lookup = {
        metric.name: metric for metric in available_metrics
    }

    if not metric_names:
        return available_metrics

    selected_metrics: list[Any] = []
    for name in metric_names:
        if name not in metric_lookup:
            raise ValueError(
                f"Unknown metric '{name}'. Available metrics: {', '.join(sorted(metric_lookup))}"
            )
        selected_metrics.append(metric_lookup[name])

    return _configure_ragas_metrics(selected_metrics)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False, default=str)


def _format_score(value: Any) -> str:
    if value is None:
        return "-"

    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return str(value)

    if math.isnan(numeric):
        return "nan"

    return f"{numeric:.3f}"


def _print_score_matrix(rows: list[dict[str, Any]]) -> None:
    if not rows:
        print("No per-row scores to display.")
        return

    metric_names = [
        name
        for name in rows[0].keys()
        if name not in {"question", "reference", "response", "retrieved_contexts", "sample_id"}
    ]

    headers = ["sample_id", "question", *metric_names]
    widths = {header: len(header) for header in headers}
    formatted_rows: list[dict[str, str]] = []

    for row in rows:
        formatted = {
            "sample_id": str(row.get("sample_id", "")),
            "question": str(row.get("user_input", "")),
        }

        for metric_name in metric_names:
            formatted[metric_name] = _format_score(row.get(metric_name))

        formatted_rows.append(formatted)

        for header in headers:
            widths[header] = max(widths[header], len(formatted.get(header, "")))

    def render_line(values: dict[str, str]) -> str:
        return " | ".join(
            values.get(header, "").ljust(widths[header])
            for header in headers
        )

    separator = "-+-".join("-" * widths[header] for header in headers)

    print("\nRAGAS Score Matrix")
    print(render_line({header: header for header in headers}))
    print(separator)
    for row in formatted_rows:
        print(render_line(row))


def _write_scores_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return

    metric_names = [
        name
        for name in rows[0].keys()
        if name not in {"question", "reference", "response", "retrieved_contexts", "sample_id"}
    ]
    fieldnames = ["sample_id", "question", *metric_names]

    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "sample_id": row.get("sample_id", ""),
                    "question": row.get("user_input", ""),
                    **{
                        metric_name: _format_score(row.get(metric_name))
                        for metric_name in metric_names
                    },
                }
            )


def main() -> int:
    load_dotenv()

    parser = argparse.ArgumentParser(
        description="Evaluate the RAG pipeline with RAGAS."
    )
    parser.add_argument(
        "--input",
        required=True,
        type=Path,
        help="CSV, JSON, or JSONL file with question and ground truth rows.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("ragas_evaluation_results.json"),
        help="Path to write the evaluation summary JSON.",
    )
    parser.add_argument(
        "--metrics",
        nargs="*",
        default=[],
        help=(
            "Optional subset of metrics to run. "
            "Defaults to answer_relevancy, context_precision, context_recall, faithfulness."
        ),
    )
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable the RAGAS progress bar.",
    )
    parser.add_argument(
        "--predictions-output",
        type=Path,
        default=Path("predictions.jsonl"),
        help="JSONL file for generated answers and retrieved contexts.",
    )
    parser.add_argument(
        "--scores-output",
        type=Path,
        default=Path("ragas_score_matrix.csv"),
        help="CSV file for the per-row RAGAS score matrix.",
    )
    parser.add_argument(
        "--request-delay-seconds",
        type=float,
        default=10.0,
        help="Delay between answer-generation requests.",
    )
    args = parser.parse_args()

    if not args.input.exists():
        raise FileNotFoundError(f"Input file not found: {args.input}")

    evaluate, default_metrics = _load_ragas_components()
    metrics = (
        _metric_objects(args.metrics)
        if args.metrics
        else _configure_ragas_metrics(default_metrics)
    )
    ragas_llm = _build_ragas_llm()
    run_config = _build_ragas_run_config()
    max_contexts = int(os.getenv("RAGAS_MAX_CONTEXTS", "1"))
    max_context_chars = int(os.getenv("RAGAS_MAX_CONTEXT_CHARS", "3500"))
    sample_delay_seconds = float(
        os.getenv("RAGAS_SAMPLE_DELAY_SECONDS", "70")
    )
    score_cache_path = args.scores_output.with_suffix(".jsonl")

    rows = _load_dataset_rows(args.input)
    if not rows:
        raise ValueError("No evaluation rows were found in the input file.")

    requested_sample_ids = [
        row["sample_id"]
        for row in rows
    ]
    requested_sample_id_set = set(requested_sample_ids)
    metric_names = [metric.name for metric in metrics]

    retriever = None
    chat_model = None
    client = None

    try:
        retriever, chat_model, client = create_rag_components()
        embedding_model = get_embedding_model()

        completed_predictions = {
            sample_id: row
            for sample_id, row in _load_prediction_rows(
                args.predictions_output
            ).items()
            if sample_id in requested_sample_id_set
        }
        completed_scores = {
            sample_id: row
            for sample_id, row in _load_score_rows(
                score_cache_path
            ).items()
            if sample_id in requested_sample_id_set
            and _score_row_is_valid(
                row,
                metric_names,
            )
        }

        for row in rows:
            sample_id = row["sample_id"]

            if sample_id in completed_predictions:
                continue

            result_row = _evaluate_row_with_retry(
                sample_id=sample_id,
                row=row,
                retriever=retriever,
                chat_model=chat_model,
                backoff_schedule=[15, 30, 60, 120],
            )

            if result_row is None:
                continue

            completed_predictions[sample_id] = result_row
            _append_prediction(
                args.predictions_output,
                result_row,
            )

            if args.request_delay_seconds > 0:
                time.sleep(args.request_delay_seconds)

        for row in rows:
            sample_id = row["sample_id"]

            if sample_id in completed_scores:
                continue

            prediction_row = completed_predictions.get(sample_id)
            if prediction_row is None:
                print(
                    f"Sample {sample_id} is missing a prediction and cannot be scored."
                )
                continue

            score_input_row = _build_scoring_row(
                sample_id=sample_id,
                row=row,
                prediction_row=prediction_row,
                max_contexts=max_contexts,
                max_context_chars=max_context_chars,
            )

            try:
                scored_row = _score_single_sample_with_retry(
                    sample_id=sample_id,
                    score_row=score_input_row,
                    evaluate=evaluate,
                    metrics=metrics,
                    ragas_llm=ragas_llm,
                    embedding_model=embedding_model,
                    run_config=run_config,
                    show_progress=not args.no_progress,
                )
            except Exception as error:
                print(
                    f"Sample {sample_id} scoring failed: {error}"
                )
                continue

            if not _score_row_is_valid(
                scored_row,
                metric_names,
            ):
                print(
                    f"Sample {sample_id} did not produce a valid score."
                )
                continue

            completed_scores[sample_id] = scored_row
            _append_score_row(
                score_cache_path,
                scored_row,
            )

            if sample_delay_seconds > 0:
                time.sleep(sample_delay_seconds)

        missing_score_ids = [
            sample_id
            for sample_id in requested_sample_ids
            if sample_id not in completed_scores
        ]

        if missing_score_ids:
            print(
                "Skipping final CSV because these requested samples are missing valid scores: "
                + ", ".join(str(sample_id) for sample_id in missing_score_ids)
            )
            return 1

        final_score_rows = [
            completed_scores[sample_id]
            for sample_id in requested_sample_ids
        ]

        summary = {
            "summary": "Per-sample RAGAS evaluation completed.",
            "metrics": _average_metric_scores(
                final_score_rows,
                metric_names,
            ),
            "row_scores": final_score_rows,
            "rows": rows,
        }
        _write_json(args.output, summary)
        _write_scores_csv(args.scores_output, final_score_rows)

        print("Per-sample RAGAS evaluation completed.")
        _print_score_matrix(final_score_rows)
        print(f"Saved evaluation summary to {args.output}")
        print(f"Saved score matrix to {args.scores_output}")
        print(f"Saved generated answers to {args.predictions_output}")

        return 0
    finally:
        if client is not None:
            client.close()


if __name__ == "__main__":
    raise SystemExit(main())
