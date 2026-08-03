from __future__ import annotations

from typing import Any

from langchain_core.documents import Document


# -------------------------------------------------------------------
# Metadata helpers
# -------------------------------------------------------------------

def _safe_string(
    value: Any,
    default: str = "",
) -> str:
    """Convert a metadata value to a clean string."""

    if value is None:
        return default

    text = str(value).strip()

    return text or default


def get_document_title(
    metadata: dict[str, Any],
) -> str:
    """Return the best available legal-document title."""

    return _safe_string(
        metadata.get("document_title")
        or metadata.get("document_name"),
        "Unknown document",
    )


def get_document_short_name(
    metadata: dict[str, Any],
) -> str | None:
    """Return a short law name such as PPC, ATA, AMLA, or Constitution."""

    value = (
        metadata.get("document_short_name")
        or metadata.get("short_name")
    )

    normalized = _safe_string(value)

    return normalized or None


def get_provision_type(
    metadata: dict[str, Any],
) -> str | None:
    """Return `section` or `article` from compatible metadata keys."""

    explicit_type = _safe_string(
        metadata.get("provision_type")
    ).lower()

    if explicit_type in {
        "section",
        "article",
    }:
        return explicit_type

    if metadata.get("article_number"):
        return "article"

    if metadata.get("section_number"):
        return "section"

    if (
        metadata.get("document_type")
        == "constitutional_law"
    ):
        return "article"

    return None


def get_provision_number(
    metadata: dict[str, Any],
) -> str | None:
    """Return a Section or Article number."""

    value = (
        metadata.get("provision_number")
        or metadata.get("section_number")
        or metadata.get("article_number")
    )

    normalized = _safe_string(value)

    return normalized or None


def get_provision_title(
    metadata: dict[str, Any],
) -> str | None:
    """Return a Section or Article title."""

    value = (
        metadata.get("provision_title")
        or metadata.get("section_title")
        or metadata.get("article_title")
    )

    normalized = _safe_string(value)

    return normalized or None


def get_provision_label(
    metadata: dict[str, Any],
) -> str:
    """Build a readable Section/Article label."""

    provision_type = get_provision_type(
        metadata
    )
    provision_number = get_provision_number(
        metadata
    )
    provision_title = get_provision_title(
        metadata
    )

    if (
        provision_type
        and provision_number
    ):
        label = (
            f"{provision_type.title()} "
            f"{provision_number}"
        )

        if provision_title:
            label += (
                f" — {provision_title}"
            )

        return label

    if provision_title:
        return provision_title

    return "Unsectioned material"


def format_page_range(
    metadata: dict[str, Any],
) -> str:
    """Format one page number or an inclusive page range."""

    page_start = metadata.get(
        "page_start"
    )
    page_end = metadata.get(
        "page_end"
    )

    if page_start is None:
        page_number = metadata.get(
            "page_number"
        )

        return _safe_string(
            page_number,
            "Unknown",
        )

    if (
        page_end is None
        or page_end == page_start
    ):
        return str(page_start)

    return (
        f"{page_start}-{page_end}"
    )


def get_chunk_numbers(
    metadata: dict[str, Any],
) -> list[Any]:
    """Return all contributing chunk numbers."""

    merged_numbers = metadata.get(
        "merged_chunk_numbers"
    )

    if isinstance(
        merged_numbers,
        list,
    ) and merged_numbers:
        return merged_numbers

    chunk_number = metadata.get(
        "document_chunk_number",
        metadata.get(
            "chunk_number",
            "Unknown",
        ),
    )

    return [
        chunk_number
    ]


def is_merged_provision(
    metadata: dict[str, Any],
) -> bool:
    """Return True when split provision parts were merged."""

    return bool(
        metadata.get(
            "provision_was_merged",
            metadata.get(
                "section_was_merged",
                False,
            ),
        )
    )


def get_quality_status(
    metadata: dict[str, Any],
) -> str:
    """Return a readable source quality label."""

    explicit_status = _safe_string(
        metadata.get(
            "page_quality_status"
        )
    )

    if explicit_status:
        return explicit_status

    if metadata.get(
        "page_quality_suspicious",
        False,
    ):
        return "suspicious"

    return "acceptable"


# -------------------------------------------------------------------
# Context formatting
# -------------------------------------------------------------------

def format_context_document(
    document: Document,
    source_index: int,
) -> str:
    """Format one legal chunk as a citation-ready context block."""

    metadata = document.metadata

    document_title = get_document_title(
        metadata
    )
    document_short_name = (
        get_document_short_name(
            metadata
        )
    )
    provision_label = (
        get_provision_label(
            metadata
        )
    )
    page_range = format_page_range(
        metadata
    )
    chunk_numbers = get_chunk_numbers(
        metadata
    )

    document_line = (
        document_title
    )

    if (
        document_short_name
        and document_short_name.lower()
        not in document_title.lower()
    ):
        document_line += (
            f" ({document_short_name})"
        )

    content = (
        document.page_content.strip()
    )

    return "\n".join(
        (
            f"[Source {source_index}]",
            f"Document: {document_line}",
            f"Document ID: {_safe_string(metadata.get('document_id'), 'Unknown')}",
            f"Document type: {_safe_string(metadata.get('document_type'), 'legal_document')}",
            f"Provision: {provision_label}",
            f"Pages: {page_range}",
            f"Merged provision parts: {is_merged_provision(metadata)}",
            f"Source chunks: {chunk_numbers}",
            f"Quality: {get_quality_status(metadata)}",
            "",
            content,
        )
    ).strip()


def format_context(
    documents: list[Document],
) -> str:
    """
    Convert selected legal documents into LLM-ready context.

    Each block contains its own source label so the answer generator can
    cite evidence using [Source 1], [Source 2], and so on.
    """

    context_parts: list[str] = []

    for index, document in enumerate(
        documents,
        start=1,
    ):
        if not document.page_content.strip():
            continue

        context_parts.append(
            format_context_document(
                document=document,
                source_index=index,
            )
        )

    return (
        "\n\n---\n\n".join(
            context_parts
        )
    )


# -------------------------------------------------------------------
# Structured source records
# -------------------------------------------------------------------

def create_source_record(
    document: Document,
    source_index: int,
) -> dict[str, Any]:
    """Create one structured source record for AnswerResult."""

    metadata = document.metadata

    provision_type = get_provision_type(
        metadata
    )
    provision_number = (
        get_provision_number(
            metadata
        )
    )
    provision_title = (
        get_provision_title(
            metadata
        )
    )
    chunk_numbers = get_chunk_numbers(
        metadata
    )

    return {
        "label": (
            f"Source {source_index}"
        ),
        "document_id": metadata.get(
            "document_id"
        ),
        "document_name": get_document_title(
            metadata
        ),
        "document_title": get_document_title(
            metadata
        ),
        "document_short_name": (
            get_document_short_name(
                metadata
            )
        ),
        "document_type": metadata.get(
            "document_type",
            "legal_document",
        ),
        "provision_type": provision_type,
        "provision_number": provision_number,
        "provision_title": provision_title,
        # Backward-compatible source keys
        "section_number": metadata.get(
            "section_number"
        ),
        "section_title": metadata.get(
            "section_title"
        ),
        # Constitution support
        "article_number": metadata.get(
            "article_number"
        ),
        "article_title": metadata.get(
            "article_title"
        ),
        "page_start": metadata.get(
            "page_start"
        ),
        "page_end": metadata.get(
            "page_end"
        ),
        "page_number": metadata.get(
            "page_number",
            "Unknown",
        ),
        "page_range": format_page_range(
            metadata
        ),
        "chunk_number": (
            chunk_numbers[0]
            if chunk_numbers
            else "Unknown"
        ),
        "chunk_numbers": chunk_numbers,
        "merged_provision_parts": (
            is_merged_provision(
                metadata
            )
        ),
        "source_pages": metadata.get(
            "source_pages",
            [],
        ),
        "quality_status": get_quality_status(
            metadata
        ),
    }


def create_sources(
    documents: list[Document],
) -> list[dict[str, Any]]:
    """Create structured source metadata for all selected documents."""

    sources: list[
        dict[str, Any]
    ] = []

    for index, document in enumerate(
        documents,
        start=1,
    ):
        if not document.page_content.strip():
            continue

        sources.append(
            create_source_record(
                document=document,
                source_index=index,
            )
        )

    return sources


# -------------------------------------------------------------------
# Context diagnostics
# -------------------------------------------------------------------

def estimate_context_characters(
    documents: list[Document],
) -> int:
    """Return the approximate formatted context size in characters."""

    return len(
        format_context(
            documents
        )
    )


def display_context_preview(
    documents: list[Document],
    preview_characters: int = 3000,
) -> None:
    """Print a safe preview of the formatted context."""

    context = format_context(
        documents
    )

    print("\n" + "=" * 70)
    print("FORMATTED CONTEXT PREVIEW")
    print("=" * 70)
    print(
        context[
            :max(
                0,
                preview_characters,
            )
        ]
    )

    if (
        preview_characters >= 0
        and len(context)
        > preview_characters
    ):
        print(
            "\n...context preview truncated..."
        )

    print(
        "\nFormatted context characters: "
        f"{len(context)}"
    )