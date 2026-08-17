from __future__ import annotations

import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Any, Mapping

from rag.legal_reference_normalizer import (
    canonicalize_provision_number,
)

_CHUNK_ID_PATTERN = re.compile(
    r"^(?P<document_id>[^:]+)::"
    r"(?P<provision_type>section|article)::"
    r"(?P<provision_number>[^:]+)::"
    r"part-(?P<part_number>\d+)::"
    r"chunk-(?P<chunk_number>\d+)$"
)


@dataclass(frozen=True, slots=True)
class MetadataIssue:
    """One metadata validation or audit finding."""

    category: str
    field_name: str
    message: str
    point_id: str | None = None
    chunk_id: str | None = None
    document_id: str | None = None
    provision_identity: str | None = None


@dataclass(slots=True)
class ChunkMetadataValidationResult:
    """Validation result for one metadata payload."""

    valid: bool
    issues: list[MetadataIssue] = field(default_factory=list)


@dataclass(slots=True)
class MetadataAuditResult:
    """Structured result for a collection-level metadata audit."""

    total_points: int = 0
    valid_points: int = 0
    invalid_points: int = 0
    duplicate_chunk_ids: list[str] = field(default_factory=list)
    missing_required_fields: dict[str, int] = field(default_factory=dict)
    type_errors: dict[str, int] = field(default_factory=dict)
    consistency_errors: dict[str, int] = field(default_factory=dict)
    sample_errors: list[MetadataIssue] = field(default_factory=list)


def _string_value(value: Any) -> str:
    if value is None:
        return ""

    return str(value).strip()


def _is_non_empty_str(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _is_positive_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value > 0


def _is_bool(value: Any) -> bool:
    return isinstance(value, bool)


def _is_list_of_ints(value: Any) -> bool:
    return (
        isinstance(value, list)
        and bool(value)
        and all(_is_positive_int(item) for item in value)
    )


def _is_list_of_strings(value: Any) -> bool:
    return (
        isinstance(value, list)
        and all(isinstance(item, str) for item in value)
    )


def _issue(
    *,
    category: str,
    field_name: str,
    message: str,
    metadata: Mapping[str, Any],
    point_id: str | None = None,
) -> MetadataIssue:
    return MetadataIssue(
        category=category,
        field_name=field_name,
        message=message,
        point_id=point_id,
        chunk_id=_string_value(metadata.get("chunk_id")) or None,
        document_id=_string_value(metadata.get("document_id")) or None,
        provision_identity=(
            _string_value(metadata.get("provision_identity"))
            or None
        ),
    )


def _chunk_id_fields(metadata: Mapping[str, Any]) -> tuple[str | None, ...]:
    chunk_id = _string_value(metadata.get("chunk_id"))

    if not chunk_id:
        return (None, None, None, None, None)

    match = _CHUNK_ID_PATTERN.match(chunk_id)

    if not match:
        return (None, None, None, None, None)

    return (
        match.group("document_id"),
        match.group("provision_type"),
        match.group("provision_number"),
        match.group("part_number"),
        match.group("chunk_number"),
    )


def _validate_common_metadata(
    metadata: Mapping[str, Any],
    *,
    allow_unsectioned: bool,
    point_id: str | None = None,
) -> list[MetadataIssue]:
    issues: list[MetadataIssue] = []
    is_unsectioned = bool(
        metadata.get("is_unsectioned_chunk")
    )

    if "is_unsectioned_chunk" in metadata and not _is_bool(
        metadata.get("is_unsectioned_chunk")
    ):
        issues.append(
            _issue(
                category="type_error",
                field_name="is_unsectioned_chunk",
                message="is_unsectioned_chunk must be a bool.",
                metadata=metadata,
                point_id=point_id,
            )
        )

    required_string_fields = [
        "document_id",
        "document_name",
        "document_title",
        "document_short_name",
        "document_type",
        "chunk_id",
    ]

    for field_name in required_string_fields:
        value = metadata.get(field_name)

        if not _is_non_empty_str(value):
            issues.append(
                _issue(
                    category="missing_required_field",
                    field_name=field_name,
                    message=(
                        f"{field_name} must be a non-empty string."
                    ),
                    metadata=metadata,
                    point_id=point_id,
                )
            )

    if not _is_non_empty_str(metadata.get("document_id")):
        return issues

    if not _is_non_empty_str(metadata.get("chunk_id")):
        return issues

    if "heading_only_chunk" in metadata and not _is_bool(
        metadata.get("heading_only_chunk")
    ):
        issues.append(
            _issue(
                category="type_error",
                field_name="heading_only_chunk",
                message="heading_only_chunk must be a bool.",
                metadata=metadata,
                point_id=point_id,
            )
        )

    if "subsection_path" in metadata:
        subsection_path = metadata.get("subsection_path")

        if not (
            isinstance(subsection_path, list)
            and all(isinstance(item, str) for item in subsection_path)
        ):
            issues.append(
                _issue(
                    category="type_error",
                    field_name="subsection_path",
                    message="subsection_path must be a list[str].",
                    metadata=metadata,
                    point_id=point_id,
                )
            )
        elif not subsection_path:
            issues.append(
                _issue(
                    category="consistency_error",
                    field_name="subsection_path",
                    message=(
                        "subsection_path cannot be empty when present."
                    ),
                    metadata=metadata,
                    point_id=point_id,
                )
            )

    if "subsection_path_key" in metadata:
        subsection_path_key = metadata.get("subsection_path_key")

        if not _is_non_empty_str(subsection_path_key):
            issues.append(
                _issue(
                    category="type_error",
                    field_name="subsection_path_key",
                    message=(
                        "subsection_path_key must be string-compatible."
                    ),
                    metadata=metadata,
                    point_id=point_id,
                )
            )

    if "component_type" in metadata and metadata.get("component_type") is not None:
        if not _is_non_empty_str(metadata.get("component_type")):
            issues.append(
                _issue(
                    category="type_error",
                    field_name="component_type",
                    message="component_type must be a non-empty string.",
                    metadata=metadata,
                    point_id=point_id,
                )
            )

    if "component_label" in metadata and metadata.get("component_label") is not None:
        if not _is_non_empty_str(metadata.get("component_label")):
            issues.append(
                _issue(
                    category="type_error",
                    field_name="component_label",
                    message="component_label must be a non-empty string.",
                    metadata=metadata,
                    point_id=point_id,
                )
            )

    for field_name in (
        "chunk_number",
        "document_chunk_number",
        "page_start",
        "page_end",
        "provision_part_number",
        "provision_part_count",
    ):
        value = metadata.get(field_name)
        if not _is_positive_int(value):
            issues.append(
                _issue(
                    category="type_error",
                    field_name=field_name,
                    message=(
                        f"{field_name} must be a positive int."
                    ),
                    metadata=metadata,
                    point_id=point_id,
                )
            )

    if not is_unsectioned and not _is_positive_int(
        metadata.get("provision_ordinal")
    ):
        issues.append(
            _issue(
                category="type_error",
                field_name="provision_ordinal",
                message="provision_ordinal must be a positive int.",
                metadata=metadata,
                point_id=point_id,
            )
        )

    source_pages = metadata.get("source_pages")
    if not _is_list_of_ints(source_pages):
        issues.append(
            _issue(
                category="type_error",
                field_name="source_pages",
                message="source_pages must be a non-empty list[int].",
                metadata=metadata,
                point_id=point_id,
            )
        )

    if "page_start" in metadata and "page_end" in metadata:
        page_start = metadata.get("page_start")
        page_end = metadata.get("page_end")
        if (
            _is_positive_int(page_start)
            and _is_positive_int(page_end)
            and int(page_start) > int(page_end)
        ):
            issues.append(
                _issue(
                    category="consistency_error",
                    field_name="page_start",
                    message="page_start must be <= page_end.",
                    metadata=metadata,
                    point_id=point_id,
                )
            )

    if (
        _is_positive_int(metadata.get("provision_part_number"))
        and _is_positive_int(metadata.get("provision_part_count"))
        and int(metadata.get("provision_part_number"))
        > int(metadata.get("provision_part_count"))
    ):
        issues.append(
            _issue(
                category="consistency_error",
                field_name="provision_part_number",
                message=(
                    "provision_part_number must be <= "
                    "provision_part_count."
                ),
                metadata=metadata,
                point_id=point_id,
            )
        )

    provision_type = _string_value(
        metadata.get("provision_type")
    ).lower()

    if provision_type not in {"section", "article"}:
        issues.append(
            _issue(
                category="type_error",
                field_name="provision_type",
                message=(
                    "provision_type must be exactly 'section' or 'article'."
                ),
                metadata=metadata,
                point_id=point_id,
            )
        )
        return issues

    provision_number = _string_value(
        metadata.get("provision_number")
    )
    provision_identity = _string_value(
        metadata.get("provision_identity")
    )
    base_provision_number = _string_value(
        metadata.get("base_provision_number")
    )
    section_number = metadata.get("section_number")
    article_number = metadata.get("article_number")
    section_identity = metadata.get("section_identity")
    article_identity = metadata.get("article_identity")

    if is_unsectioned:
        expected_identity = (
            f"{_string_value(metadata.get('document_id'))}::"
            f"{provision_type}::unsectioned"
        )

        if provision_identity != expected_identity:
            issues.append(
                _issue(
                    category="consistency_error",
                    field_name="provision_identity",
                    message=(
                        "unsectioned chunks must preserve their "
                        "document/type/unsectioned identity."
                    ),
                    metadata=metadata,
                    point_id=point_id,
                )
            )

        if provision_number:
            issues.append(
                _issue(
                    category="consistency_error",
                    field_name="provision_number",
                    message=(
                        "unsectioned chunks should not store a normal "
                        "provision_number."
                    ),
                    metadata=metadata,
                    point_id=point_id,
                )
            )

        if base_provision_number:
            issues.append(
                _issue(
                    category="consistency_error",
                    field_name="base_provision_number",
                    message=(
                        "unsectioned chunks should not store a base "
                        "provision number."
                    ),
                    metadata=metadata,
                    point_id=point_id,
                )
            )

        if metadata.get("provision_ordinal") is not None:
            issues.append(
                _issue(
                    category="consistency_error",
                    field_name="provision_ordinal",
                    message=(
                        "unsectioned chunks should not store a provision ordinal."
                    ),
                    metadata=metadata,
                    point_id=point_id,
                )
            )

        return issues

    if not provision_number:
        issues.append(
            _issue(
                category="missing_required_field",
                field_name="provision_number",
                message="provision_number must be a non-empty string.",
                metadata=metadata,
                point_id=point_id,
            )
        )
        return issues

    canonical_provision_number = canonicalize_provision_number(
        provision_number
    )

    if canonical_provision_number != provision_number:
        issues.append(
            _issue(
                category="consistency_error",
                field_name="provision_number",
                message=(
                    "provision_number must already be stored in canonical form."
                ),
                metadata=metadata,
                point_id=point_id,
            )
        )

    if not provision_identity:
        issues.append(
            _issue(
                category="missing_required_field",
                field_name="provision_identity",
                message="provision_identity must be a non-empty string.",
                metadata=metadata,
                point_id=point_id,
            )
        )

    expected_identity = (
        f"{_string_value(metadata.get('document_id'))}::"
        f"{provision_type}::"
        f"{base_provision_number or provision_number}"
    )

    if provision_identity and provision_identity != expected_identity:
        issues.append(
            _issue(
                category="consistency_error",
                field_name="provision_identity",
                message=(
                    "provision_identity must match the parent provision."
                ),
                metadata=metadata,
                point_id=point_id,
            )
        )

    if not base_provision_number:
        issues.append(
            _issue(
                category="missing_required_field",
                field_name="base_provision_number",
                message=(
                    "base_provision_number must be a non-empty string."
                ),
                metadata=metadata,
                point_id=point_id,
            )
        )
    elif base_provision_number != provision_number:
        issues.append(
            _issue(
                category="consistency_error",
                field_name="base_provision_number",
                message=(
                    "base_provision_number must agree with provision_number."
                ),
                metadata=metadata,
                point_id=point_id,
            )
        )

    if provision_type == "section":
        if section_number != provision_number:
            issues.append(
                _issue(
                    category="consistency_error",
                    field_name="section_number",
                    message=(
                        "section_number must agree with provision_number."
                    ),
                    metadata=metadata,
                    point_id=point_id,
                )
            )

        if article_number is not None:
            issues.append(
                _issue(
                    category="consistency_error",
                    field_name="article_number",
                    message="article_number should be None for Section chunks.",
                    metadata=metadata,
                    point_id=point_id,
                )
            )

        if section_identity is not None and section_identity != provision_identity:
            issues.append(
                _issue(
                    category="consistency_error",
                    field_name="section_identity",
                    message=(
                        "section_identity must agree with provision_identity."
                    ),
                    metadata=metadata,
                    point_id=point_id,
                )
            )

        if article_identity is not None:
            issues.append(
                _issue(
                    category="consistency_error",
                    field_name="article_identity",
                    message="article_identity should be None for Section chunks.",
                    metadata=metadata,
                    point_id=point_id,
                )
            )

    elif provision_type == "article":
        if article_number != provision_number:
            issues.append(
                _issue(
                    category="consistency_error",
                    field_name="article_number",
                    message=(
                        "article_number must agree with provision_number."
                    ),
                    metadata=metadata,
                    point_id=point_id,
                )
            )

        if section_number is not None:
            issues.append(
                _issue(
                    category="consistency_error",
                    field_name="section_number",
                    message="section_number should be None for Article chunks.",
                    metadata=metadata,
                    point_id=point_id,
                )
            )

        if article_identity is not None and article_identity != provision_identity:
            issues.append(
                _issue(
                    category="consistency_error",
                    field_name="article_identity",
                    message=(
                        "article_identity must agree with provision_identity."
                    ),
                    metadata=metadata,
                    point_id=point_id,
                )
            )

        if section_identity is not None:
            issues.append(
                _issue(
                    category="consistency_error",
                    field_name="section_identity",
                    message="section_identity should be None for Article chunks.",
                    metadata=metadata,
                    point_id=point_id,
                )
            )

    if metadata.get("provision_ordinal") is not None and not _is_positive_int(
        metadata.get("provision_ordinal")
    ):
        issues.append(
            _issue(
                category="type_error",
                field_name="provision_ordinal",
                message="provision_ordinal must be a positive int.",
                metadata=metadata,
                point_id=point_id,
            )
        )

    if subsection_path := metadata.get("subsection_path"):
        if isinstance(subsection_path, list) and all(
            isinstance(item, str) for item in subsection_path
        ):
            if not base_provision_number:
                issues.append(
                    _issue(
                        category="consistency_error",
                        field_name="base_provision_number",
                        message=(
                            "child chunks with subsection_path must keep the base "
                            "provision number."
                        ),
                        metadata=metadata,
                        point_id=point_id,
                    )
                )
            if metadata.get("subsection_path_key") != ".".join(
                subsection_path
            ):
                issues.append(
                    _issue(
                        category="consistency_error",
                        field_name="subsection_path_key",
                        message=(
                            "subsection_path_key must join subsection_path with dots."
                        ),
                        metadata=metadata,
                        point_id=point_id,
                    )
                )

            expected_component_type = (
                "subsection"
                if len(subsection_path) == 1
                else "clause"
                if len(subsection_path) == 2
                else "paragraph"
            )

            if metadata.get("component_type") != expected_component_type:
                issues.append(
                    _issue(
                        category="consistency_error",
                        field_name="component_type",
                        message=(
                            "component_type must match subsection_path depth."
                        ),
                        metadata=metadata,
                        point_id=point_id,
                    )
                )

            expected_component_label = f"({subsection_path[-1]})"
            if metadata.get("component_label") != expected_component_label:
                issues.append(
                    _issue(
                        category="consistency_error",
                        field_name="component_label",
                        message=(
                            "component_label must match the final path label."
                        ),
                        metadata=metadata,
                        point_id=point_id,
                    )
                )

    chunk_id = _string_value(metadata.get("chunk_id"))
    chunk_id_fields = _chunk_id_fields(metadata)

    if chunk_id_fields[0] is None:
        issues.append(
            _issue(
                category="consistency_error",
                field_name="chunk_id",
                message=(
                    "chunk_id must follow the expected deterministic format."
                ),
                metadata=metadata,
                point_id=point_id,
            )
        )
    else:
        (
            chunk_document_id,
            chunk_provision_type,
            chunk_provision_number,
            chunk_part_number,
            chunk_document_number,
        ) = chunk_id_fields

        if chunk_document_id != _string_value(metadata.get("document_id")):
            issues.append(
                _issue(
                    category="consistency_error",
                    field_name="chunk_id",
                    message="chunk_id document_id does not match metadata.",
                    metadata=metadata,
                    point_id=point_id,
                )
            )

        if chunk_provision_type != provision_type:
            issues.append(
                _issue(
                    category="consistency_error",
                    field_name="chunk_id",
                    message="chunk_id provision_type does not match metadata.",
                    metadata=metadata,
                    point_id=point_id,
                )
            )

        expected_chunk_provision_number = (
            "unsectioned"
            if is_unsectioned
            else provision_number
        )
        if chunk_provision_number != expected_chunk_provision_number:
            issues.append(
                _issue(
                    category="consistency_error",
                    field_name="chunk_id",
                    message=(
                        "chunk_id provision_number does not match metadata."
                    ),
                    metadata=metadata,
                    point_id=point_id,
                )
            )

        if (
            _is_positive_int(metadata.get("provision_part_number"))
            and int(chunk_part_number)
            != int(metadata.get("provision_part_number"))
        ):
            issues.append(
                _issue(
                    category="consistency_error",
                    field_name="chunk_id",
                    message=(
                        "chunk_id part number does not match provision_part_number."
                    ),
                    metadata=metadata,
                    point_id=point_id,
                )
            )

        if (
            _is_positive_int(metadata.get("document_chunk_number"))
            and int(chunk_document_number)
            != int(metadata.get("document_chunk_number"))
        ):
            issues.append(
                _issue(
                    category="consistency_error",
                    field_name="chunk_id",
                    message=(
                        "chunk_id chunk number does not match document_chunk_number."
                    ),
                    metadata=metadata,
                    point_id=point_id,
                )
            )

    if allow_unsectioned and is_unsectioned:
        return issues

    return issues


def validate_legal_chunk_metadata(
    metadata: Mapping[str, Any],
    *,
    allow_unsectioned: bool = False,
    point_id: str | None = None,
) -> ChunkMetadataValidationResult:
    """Validate one chunk metadata payload without mutating it."""

    issues = _validate_common_metadata(
        metadata,
        allow_unsectioned=allow_unsectioned,
        point_id=point_id,
    )

    return ChunkMetadataValidationResult(
        valid=not issues,
        issues=issues,
    )


def _collect_point_metadata(
    point: Any,
    *,
    metadata_payload_key: str,
) -> dict[str, Any]:
    payload = getattr(point, "payload", {}) or {}

    if not isinstance(payload, Mapping):
        payload = {}

    metadata = payload.get(metadata_payload_key)

    if isinstance(metadata, Mapping):
        return dict(metadata)

    fallback_metadata = payload.get("metadata")
    if isinstance(fallback_metadata, Mapping):
        return dict(fallback_metadata)

    if metadata_payload_key == "metadata":
        return dict(payload)

    return {}


def _record_issue(
    result: MetadataAuditResult,
    issue: MetadataIssue,
    *,
    max_examples: int,
) -> None:
    if len(result.sample_errors) < max_examples:
        result.sample_errors.append(issue)

    if issue.category == "missing_required_field":
        result.missing_required_fields[issue.field_name] = (
            result.missing_required_fields.get(issue.field_name, 0) + 1
        )
    elif issue.category == "type_error":
        result.type_errors[issue.field_name] = (
            result.type_errors.get(issue.field_name, 0) + 1
        )
    else:
        result.consistency_errors[issue.field_name] = (
            result.consistency_errors.get(issue.field_name, 0) + 1
        )


def _append_issue(
    issues_by_point: dict[str, list[MetadataIssue]],
    point_key: str,
    issue: MetadataIssue,
) -> None:
    issues_by_point.setdefault(point_key, []).append(issue)


def audit_collection_metadata(
    client: Any,
    collection_name: str,
    *,
    metadata_payload_key: str = "metadata",
    page_size: int = 256,
    max_examples: int = 10,
) -> MetadataAuditResult:
    """Scroll a collection and audit all stored chunk metadata."""

    result = MetadataAuditResult()
    offset: Any = None
    seen_chunk_ids: set[str] = set()
    duplicate_chunk_ids: set[str] = set()
    issues_by_point: dict[str, list[MetadataIssue]] = defaultdict(list)
    grouped_points: dict[
        tuple[str, str],
        dict[str, list[dict[str, Any]]],
    ] = defaultdict(lambda: defaultdict(list))
    unsectioned_points: set[str] = set()

    point_index = 0

    while True:
        scroll_result = client.scroll(
            collection_name=collection_name,
            scroll_filter=None,
            limit=page_size,
            offset=offset,
            with_payload=True,
            with_vectors=False,
        )

        if isinstance(scroll_result, tuple):
            points, next_offset = scroll_result
        else:
            points = getattr(scroll_result, "points", [])
            next_offset = getattr(
                scroll_result,
                "next_page_offset",
                None,
            )

        if not points:
            break

        for point in points:
            point_index += 1
            result.total_points += 1
            point_id = _string_value(getattr(point, "id", None)) or (
                f"point-{point_index}"
            )
            metadata = _collect_point_metadata(
                point,
                metadata_payload_key=metadata_payload_key,
            )
            chunk_id = _string_value(metadata.get("chunk_id"))

            validation = validate_legal_chunk_metadata(
                metadata,
                allow_unsectioned=True,
                point_id=point_id,
            )
            issues = list(validation.issues)

            if chunk_id:
                if chunk_id in seen_chunk_ids:
                    duplicate_chunk_ids.add(chunk_id)
                    issues.append(
                        MetadataIssue(
                            category="consistency_error",
                            field_name="chunk_id",
                            message="chunk_id must be unique within the collection.",
                            point_id=point_id,
                            chunk_id=chunk_id,
                            document_id=_string_value(
                                metadata.get("document_id")
                            )
                            or None,
                            provision_identity=_string_value(
                                metadata.get("provision_identity")
                            )
                            or None,
                        )
                    )
                else:
                    seen_chunk_ids.add(chunk_id)

            if not issues:
                result.valid_points += 1
            else:
                result.invalid_points += 1
                for issue in issues:
                    _append_issue(issues_by_point, point_id, issue)
                    _record_issue(
                        result,
                        issue,
                        max_examples=max_examples,
                    )

            document_id = _string_value(metadata.get("document_id"))
            provision_type = _string_value(
                metadata.get("provision_type")
            ).lower()
            provision_identity = _string_value(
                metadata.get("provision_identity")
            )

            if (
                document_id
                and provision_type in {"section", "article"}
                and provision_identity
            ):
                if bool(metadata.get("is_unsectioned_chunk")):
                    unsectioned_points.add(point_id)
                    continue

                grouped_points[
                    (document_id, provision_type)
                ][provision_identity].append(
                    {
                        "point_id": point_id,
                        "metadata": metadata,
                    }
                )

        if next_offset is None:
            break

        offset = next_offset

    result.duplicate_chunk_ids = sorted(
        duplicate_chunk_ids
    )

    invalid_point_ids: set[str] = set(
        point_id
        for point_id, issues in issues_by_point.items()
        if issues
    )
    invalid_point_ids.difference_update(unsectioned_points)

    for (
        document_scope,
        provision_groups,
    ) in grouped_points.items():
        unique_ordinals: dict[str, int] = {}
        ordinal_to_identities: dict[int, list[str]] = defaultdict(list)
        ordered_identities: list[tuple[str, int]] = []

        for provision_identity, items in provision_groups.items():
            if all(
                bool(item["metadata"].get("is_unsectioned_chunk"))
                for item in items
            ):
                continue

            ordinals = {
                int(item["metadata"].get("provision_ordinal"))
                for item in items
                if _is_positive_int(item["metadata"].get("provision_ordinal"))
            }

            if len(ordinals) != 1:
                for item in items:
                    issue = MetadataIssue(
                        category="consistency_error",
                        field_name="provision_ordinal",
                        message=(
                            "all chunks of the same provision_identity must "
                            "share one provision_ordinal."
                        ),
                        point_id=item["point_id"],
                        chunk_id=_string_value(
                            item["metadata"].get("chunk_id")
                        )
                        or None,
                        document_id=document_scope[0],
                        provision_identity=provision_identity,
                    )
                    _append_issue(
                        issues_by_point,
                        item["point_id"],
                        issue,
                    )
                    _record_issue(
                        result,
                        issue,
                        max_examples=max_examples,
                    )
                continue

            ordinal = next(iter(ordinals))
            unique_ordinals[provision_identity] = ordinal
            ordinal_to_identities[ordinal].append(provision_identity)
            ordered_identities.append((provision_identity, ordinal))

        for ordinal, identities in ordinal_to_identities.items():
            if len(identities) <= 1:
                continue

            for provision_identity in identities:
                for item in provision_groups[provision_identity]:
                    issue = MetadataIssue(
                        category="consistency_error",
                        field_name="provision_ordinal",
                        message=(
                            "distinct provisions within one document/type "
                            "cannot share a provision_ordinal."
                        ),
                        point_id=item["point_id"],
                        chunk_id=_string_value(
                            item["metadata"].get("chunk_id")
                        )
                        or None,
                        document_id=document_scope[0],
                        provision_identity=provision_identity,
                    )
                    _append_issue(
                        issues_by_point,
                        item["point_id"],
                        issue,
                    )
                    _record_issue(
                        result,
                        issue,
                        max_examples=max_examples,
                    )

        if not ordered_identities:
            continue

        ordered_identities.sort(
            key=lambda item: item[1]
        )

        observed_ordinals = [ordinal for _identity, ordinal in ordered_identities]
        expected_ordinals = list(
            range(1, len(ordered_identities) + 1)
        )

        if observed_ordinals != expected_ordinals:
            for provision_identity, _ordinal in ordered_identities:
                for item in provision_groups[provision_identity]:
                    issue = MetadataIssue(
                        category="consistency_error",
                        field_name="provision_ordinal",
                        message=(
                            "provision_ordinal values must be consecutive "
                            "within a document/type sequence."
                        ),
                        point_id=item["point_id"],
                        chunk_id=_string_value(
                            item["metadata"].get("chunk_id")
                        )
                        or None,
                        document_id=document_scope[0],
                        provision_identity=provision_identity,
                    )
                    _append_issue(
                        issues_by_point,
                        item["point_id"],
                        issue,
                    )
                    _record_issue(
                        result,
                        issue,
                        max_examples=max_examples,
                    )
            continue

        for index, (provision_identity, ordinal) in enumerate(
            ordered_identities,
            start=1,
        ):
            items = provision_groups[provision_identity]
            previous_identity = (
                ordered_identities[index - 2][0]
                if index > 1
                else None
            )
            next_identity = (
                ordered_identities[index][0]
                if index < len(ordered_identities)
                else None
            )
            previous_number = (
                provision_groups[previous_identity][0]["metadata"].get(
                    "provision_number"
                )
                if previous_identity is not None
                else None
            )
            next_number = (
                provision_groups[next_identity][0]["metadata"].get(
                    "provision_number"
                )
                if next_identity is not None
                else None
            )

            for item in items:
                metadata = item["metadata"]
                point_id = item["point_id"]

                if int(metadata.get("provision_ordinal")) != index:
                    issue = MetadataIssue(
                        category="consistency_error",
                        field_name="provision_ordinal",
                        message=(
                            "provision_ordinal must match the sequence order."
                        ),
                        point_id=point_id,
                        chunk_id=_string_value(
                            metadata.get("chunk_id")
                        )
                        or None,
                        document_id=document_scope[0],
                        provision_identity=provision_identity,
                    )
                    _append_issue(issues_by_point, point_id, issue)
                    _record_issue(result, issue, max_examples=max_examples)

                if metadata.get("previous_provision_identity") != previous_identity:
                    issue = MetadataIssue(
                        category="consistency_error",
                        field_name="previous_provision_identity",
                        message=(
                            "previous_provision_identity must match the prior "
                            "provision in source order."
                        ),
                        point_id=point_id,
                        chunk_id=_string_value(
                            metadata.get("chunk_id")
                        )
                        or None,
                        document_id=document_scope[0],
                        provision_identity=provision_identity,
                    )
                    _append_issue(issues_by_point, point_id, issue)
                    _record_issue(result, issue, max_examples=max_examples)

                if metadata.get("next_provision_identity") != next_identity:
                    issue = MetadataIssue(
                        category="consistency_error",
                        field_name="next_provision_identity",
                        message=(
                            "next_provision_identity must match the next "
                            "provision in source order."
                        ),
                        point_id=point_id,
                        chunk_id=_string_value(
                            metadata.get("chunk_id")
                        )
                        or None,
                        document_id=document_scope[0],
                        provision_identity=provision_identity,
                    )
                    _append_issue(issues_by_point, point_id, issue)
                    _record_issue(result, issue, max_examples=max_examples)

                if metadata.get("previous_provision_number") != previous_number:
                    issue = MetadataIssue(
                        category="consistency_error",
                        field_name="previous_provision_number",
                        message=(
                            "previous_provision_number must match the prior "
                            "provision number."
                        ),
                        point_id=point_id,
                        chunk_id=_string_value(
                            metadata.get("chunk_id")
                        )
                        or None,
                        document_id=document_scope[0],
                        provision_identity=provision_identity,
                    )
                    _append_issue(issues_by_point, point_id, issue)
                    _record_issue(result, issue, max_examples=max_examples)

                if metadata.get("next_provision_number") != next_number:
                    issue = MetadataIssue(
                        category="consistency_error",
                        field_name="next_provision_number",
                        message=(
                            "next_provision_number must match the next "
                            "provision number."
                        ),
                        point_id=point_id,
                        chunk_id=_string_value(
                            metadata.get("chunk_id")
                        )
                        or None,
                        document_id=document_scope[0],
                        provision_identity=provision_identity,
                    )
                    _append_issue(issues_by_point, point_id, issue)
                    _record_issue(result, issue, max_examples=max_examples)

                if index == 1 and metadata.get("previous_provision_identity") is not None:
                    issue = MetadataIssue(
                        category="consistency_error",
                        field_name="previous_provision_identity",
                        message=(
                            "the first provision in a sequence must not have a "
                            "previous provision."
                        ),
                        point_id=point_id,
                        chunk_id=_string_value(
                            metadata.get("chunk_id")
                        )
                        or None,
                        document_id=document_scope[0],
                        provision_identity=provision_identity,
                    )
                    _append_issue(issues_by_point, point_id, issue)
                    _record_issue(result, issue, max_examples=max_examples)

                if (
                    index == len(ordered_identities)
                    and metadata.get("next_provision_identity") is not None
                ):
                    issue = MetadataIssue(
                        category="consistency_error",
                        field_name="next_provision_identity",
                        message=(
                            "the final provision in a sequence must not have a "
                            "next provision."
                        ),
                        point_id=point_id,
                        chunk_id=_string_value(
                            metadata.get("chunk_id")
                        )
                        or None,
                        document_id=document_scope[0],
                        provision_identity=provision_identity,
                    )
                    _append_issue(issues_by_point, point_id, issue)
                    _record_issue(result, issue, max_examples=max_examples)

                if ordinal < 1:
                    issue = MetadataIssue(
                        category="consistency_error",
                        field_name="provision_ordinal",
                        message="provision_ordinal must be positive.",
                        point_id=point_id,
                        chunk_id=_string_value(
                            metadata.get("chunk_id")
                        )
                        or None,
                        document_id=document_scope[0],
                        provision_identity=provision_identity,
                    )
                    _append_issue(issues_by_point, point_id, issue)
                    _record_issue(result, issue, max_examples=max_examples)

    invalid_point_ids.update(
        point_id
        for point_id, issues in issues_by_point.items()
        if issues
    )
    result.invalid_points = len(invalid_point_ids)
    result.valid_points = max(
        0,
        result.total_points - result.invalid_points,
    )

    return result


def format_metadata_audit_result(
    result: MetadataAuditResult,
    *,
    collection_name: str | None = None,
    max_examples: int = 5,
) -> str:
    """Format a compact human-readable audit summary."""

    prefix = (
        f"Collection '{collection_name}' metadata audit"
        if collection_name
        else "Collection metadata audit"
    )
    lines = [
        (
            f"{prefix}: {result.invalid_points} invalid of "
            f"{result.total_points} points."
        ),
        (
            f"Duplicate chunk IDs: "
            f"{len(result.duplicate_chunk_ids)}"
        ),
        (
            f"Missing fields: {result.missing_required_fields}"
        ),
        f"Type errors: {result.type_errors}",
        f"Consistency errors: {result.consistency_errors}",
    ]

    if result.sample_errors:
        lines.append("Sample issues:")
        for issue in result.sample_errors[:max_examples]:
            lines.append(
                f"- [{issue.category}] {issue.field_name}: {issue.message}"
            )

    return "\n".join(lines)
