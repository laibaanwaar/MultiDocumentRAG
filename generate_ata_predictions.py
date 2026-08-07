from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from rag.answer_service import answer_question, create_rag_components


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    if not path.exists():
        return rows

    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))

    return rows


def _normalize_sample_id(value: Any) -> str:
    return str(value).strip()


def _load_completed_ids(path: Path) -> set[str]:
    completed: set[str] = set()

    if not path.exists():
        return completed

    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue

            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue

            sample_id = row.get("id", row.get("sample_id"))
            if sample_id is None:
                continue

            completed.add(_normalize_sample_id(sample_id))

    return completed


def _load_samples(path: Path) -> list[dict[str, Any]]:
    rows = _read_jsonl(path)

    samples: list[dict[str, Any]] = []
    for row in rows:
        sample_id = row.get("id", row.get("sample_id"))
        question = str(row.get("question") or "").strip()

        if not sample_id:
            raise ValueError(
                "Each ATA sample must include an 'id' field."
            )

        if not question:
            raise ValueError(
                f"ATA sample {sample_id} is missing a question."
            )

        samples.append(row)

    return samples


def _write_jsonl_row(path: Path, row: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(
            json.dumps(row, ensure_ascii=False)
        )
        handle.write("\n")


def _build_output_row(
    sample: dict[str, Any],
    result: dict[str, Any],
) -> dict[str, Any]:
    sample_id = _normalize_sample_id(
        sample.get("id", sample.get("sample_id"))
    )

    output: dict[str, Any] = {
        "id": sample_id,
        "sample_id": sample_id,
        "question": sample.get("question"),
        "category": sample.get("category"),
        "reference": sample.get("reference"),
        "expected_document_id": sample.get(
            "expected_document_id"
        ),
        "expected_provision_type": sample.get(
            "expected_provision_type"
        ),
        "expected_provision_numbers": sample.get(
            "expected_provision_numbers", []
        ),
        "answer": result.get("answer", ""),
        "response": result.get("answer", ""),
        "retrieved_contexts": result.get(
            "retrieved_contexts", []
        ),
        "sources": result.get("sources", []),
        "question_type": result.get("question_type"),
        "confidence": result.get("confidence"),
        "retrieved_document_count": result.get(
            "retrieved_document_count"
        ),
    }

    return output


def main() -> None:
    base_dir = Path(__file__).resolve().parent

    parser = argparse.ArgumentParser(
        description=(
            "Generate ATA RAG predictions with incremental JSONL output."
        )
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=base_dir / "ata_eval_samples.jsonl",
        help="Input ATA evaluation samples JSONL file.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=base_dir / "ata_predictions.jsonl",
        help="Output JSONL file for generated predictions.",
    )
    args = parser.parse_args()

    samples = _load_samples(args.input)
    completed_ids = _load_completed_ids(args.output)

    retriever = None
    chat_model = None
    client = None

    completed = 0
    skipped = 0
    failed = 0

    try:
        retriever, chat_model, client = create_rag_components()

        for sample in samples:
            sample_id = _normalize_sample_id(
                sample.get("id", sample.get("sample_id"))
            )

            if sample_id in completed_ids:
                skipped += 1
                continue

            try:
                result = answer_question(
                    question=str(sample.get("question") or "").strip(),
                    retriever=retriever,
                    chat_model=chat_model,
                )
                output_row = _build_output_row(sample, result)
                _write_jsonl_row(args.output, output_row)
                completed_ids.add(sample_id)
                completed += 1
            except Exception as error:  # noqa: BLE001
                failed += 1
                print(
                    f"Failed sample {sample_id}: {error}"
                )

    finally:
        if client is not None:
            client.close()

    remaining = len(samples) - len(completed_ids)

    print(
        "completed="
        f"{completed} skipped={skipped} failed={failed} remaining={remaining}"
    )


if __name__ == "__main__":
    main()
