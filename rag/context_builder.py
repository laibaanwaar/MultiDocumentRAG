from typing import Any

from langchain_core.documents import Document


def format_page_range(metadata: dict[str, Any]) -> str:
    page_start = metadata.get("page_start")
    page_end = metadata.get("page_end")

    if page_start is None:
        return str(metadata.get("page_number", "Unknown"))

    if page_end in {None, page_start}:
        return str(page_start)

    return f"{page_start}-{page_end}"


def format_context(documents: list[Document]) -> str:
    context_parts: list[str] = []

    for index, document in enumerate(documents, start=1):
        metadata = document.metadata
        context_parts.append(
            f"""
[Source {index}]
Document: {metadata.get("document_name", "Unknown document")}
Section: {metadata.get("section_number", "Unsectioned")} - {metadata.get("section_title", "")}
Pages: {format_page_range(metadata)}
Merged section parts: {metadata.get("section_was_merged", False)}
Source chunks: {metadata.get("merged_chunk_numbers", [metadata.get("chunk_number")])}
Quality: {metadata.get("page_quality_status", "unknown")}

{document.page_content}
""".strip()
        )

    return "\n\n---\n\n".join(context_parts)


def create_sources(documents: list[Document]) -> list[dict[str, Any]]:
    sources: list[dict[str, Any]] = []

    for index, document in enumerate(documents, start=1):
        metadata = document.metadata
        sources.append(
            {
                "label": f"Source {index}",
                "document_name": metadata.get("document_name", "Unknown document"),
                "section_number": metadata.get("section_number"),
                "section_title": metadata.get("section_title"),
                "page_start": metadata.get("page_start"),
                "page_end": metadata.get("page_end"),
                "page_number": metadata.get("page_number", "Unknown"),
                "chunk_number": metadata.get(
                    "chunk_number",
                    (
                        metadata.get("merged_chunk_numbers", ["Unknown"])[0]
                        if metadata.get("merged_chunk_numbers")
                        else "Unknown"
                    ),
                ),
                "chunk_numbers": metadata.get(
                    "merged_chunk_numbers",
                    [metadata.get("chunk_number", "Unknown")],
                ),
            }
        )

    return sources
