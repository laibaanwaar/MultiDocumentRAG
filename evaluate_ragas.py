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
import sys
import types
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

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


def _load_dataset_rows(path: Path, limit: int | None) -> list[dict[str, str]]:
    raw_rows = _read_rows(path)
    normalized_rows = [_normalize_row(row) for row in raw_rows]

    if limit is not None:
        return normalized_rows[:limit]

    return normalized_rows


def _build_evaluation_rows(
    rows: list[dict[str, str]],
    retriever,
    chat_model,
) -> list[dict[str, Any]]:
    evaluation_rows: list[dict[str, Any]] = []

    for index, row in enumerate(rows, start=1):
        result = answer_question(
            question=row["question"],
            retriever=retriever,
            chat_model=chat_model,
        )

        evaluation_rows.append(
            {
                "user_input": row["question"],
                "reference": row["reference"],
                "response": result["answer"],
                "retrieved_contexts": result.get("retrieved_contexts", []),
                "question_type": result.get("question_type"),
                "sample_id": index,
            }
        )

    return evaluation_rows


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

    return selected_metrics


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
        "--limit",
        type=int,
        default=None,
        help="Limit the number of rows evaluated.",
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
        default=None,
        help="Optional JSONL file for the generated answers and retrieved contexts.",
    )
    parser.add_argument(
        "--scores-output",
        type=Path,
        default=Path("ragas_score_matrix.csv"),
        help="CSV file for the per-row RAGAS score matrix.",
    )
    args = parser.parse_args()

    if not args.input.exists():
        raise FileNotFoundError(f"Input file not found: {args.input}")

    evaluate, default_metrics = _load_ragas_components()
    metrics = _metric_objects(args.metrics) if args.metrics else default_metrics

    rows = _load_dataset_rows(args.input, args.limit)
    if not rows:
        raise ValueError("No evaluation rows were found in the input file.")

    retriever = None
    chat_model = None
    client = None

    try:
        retriever, chat_model, client = create_rag_components()
        embedding_model = get_embedding_model()

        evaluation_rows = _build_evaluation_rows(
            rows=rows,
            retriever=retriever,
            chat_model=chat_model,
        )

        if args.predictions_output is not None:
            with args.predictions_output.open("w", encoding="utf-8") as handle:
                for row in evaluation_rows:
                    handle.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")

        from ragas.dataset_schema import EvaluationDataset

        dataset = EvaluationDataset.from_list(evaluation_rows)

        result = evaluate(
            dataset=dataset,
            metrics=metrics,
            llm=chat_model,
            embeddings=embedding_model,
            show_progress=not args.no_progress,
        )

        summary = {
            "summary": str(result),
            "metrics": list(result._repr_dict.items()),
            "row_scores": result.scores,
            "rows": evaluation_rows,
        }
        _write_json(args.output, summary)
        _write_scores_csv(args.scores_output, result.scores)

        print(result)
        _print_score_matrix(result.scores)
        print(f"Saved evaluation summary to {args.output}")
        print(f"Saved score matrix to {args.scores_output}")
        if args.predictions_output is not None:
            print(f"Saved generated answers to {args.predictions_output}")

        return 0
    finally:
        if client is not None:
            client.close()


if __name__ == "__main__":
    raise SystemExit(main())
