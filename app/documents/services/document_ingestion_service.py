from __future__ import annotations

import logging
import hashlib
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from uuid import NAMESPACE_URL, uuid5

from langchain_core.documents import Document
from pypdf import PdfReader

CURRENT_FILE = Path(__file__).resolve()
REPO_ROOT = CURRENT_FILE.parents[3]

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from rag.text_cleaner import clean_pdf_pages
from rag.text_splitter import create_chunks
from rag.text_splitter import validate_final_chunks
from rag.vector_store import create_vector_store


logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class UploadedDocumentIngestionInput:
    file_path: Path
    document_id: str
    document_name: str
    document_title: str
    document_short_name: str
    document_type: str = "legal_document"
    provision_type: str = "section"
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class UploadedDocumentIngestionResult:
    document_id: str
    page_count: int
    chunk_count: int
    qdrant_point_ids: list[str] = field(default_factory=list)


def _normalize_text(value: Any) -> str:
    return str(value or "").strip()


def _create_deterministic_chunk_id(chunk: Document) -> str:
    metadata = chunk.metadata

    source_pages = metadata.get("source_pages", [])

    if isinstance(source_pages, (list, tuple, set)):
        source_pages_value = ",".join(
            str(page) for page in source_pages
        )
    else:
        source_pages_value = str(source_pages)

    identity_parts = [
        str(metadata.get("document_id", "")),
        str(metadata.get("document_name", "")),
        str(metadata.get("document_type", "")),
        str(metadata.get("provision_type", "")),
        str(
            metadata.get(
                "provision_number",
                metadata.get("section_number", metadata.get("article_number", "")),
            )
        ),
        str(
            metadata.get(
                "provision_identity",
                metadata.get("section_identity", metadata.get("article_identity", "")),
            )
        ),
        str(
            metadata.get(
                "section_part_number",
                metadata.get("provision_part_number", 1),
            )
        ),
        str(
            metadata.get(
                "section_part_count",
                metadata.get("provision_part_count", 1),
            )
        ),
        str(metadata.get("document_chunk_number", "")),
        str(metadata.get("chunk_id", "")),
        str(metadata.get("page_start", "")),
        str(metadata.get("page_end", "")),
        source_pages_value,
        chunk.page_content.strip(),
    ]

    identity = "|".join(identity_parts)
    content_hash = hashlib.sha256(identity.encode("utf-8")).hexdigest()

    return str(uuid5(NAMESPACE_URL, content_hash))


def load_uploaded_document_pages(
    payload: UploadedDocumentIngestionInput,
) -> list[Document]:
    reader = PdfReader(str(payload.file_path))
    raw_pages = [page.extract_text() or "" for page in reader.pages]

    if not raw_pages:
        raise ValueError(
            f"No pages were found in {payload.file_path.name}."
        )

    cleaned_pages = clean_pdf_pages(raw_pages)
    page_documents: list[Document] = []

    for cleaned_page in cleaned_pages:
        page_text = cleaned_page.text.strip()

        if not page_text:
            continue

        metadata = dict(cleaned_page.metadata)
        page_number = metadata.get("page_number")
        source_path = str(payload.file_path)

        metadata.update(payload.metadata)
        metadata.update(
            {
                "document_id": payload.document_id,
                "document_name": payload.document_name,
                "document_title": payload.document_title,
                "document_short_name": payload.document_short_name,
                "document_type": payload.document_type,
                "provision_type": payload.provision_type,
                "source_path": source_path,
                "source_file_name": payload.file_path.name,
                "page_number": page_number,
                "page": (
                    page_number - 1
                    if isinstance(page_number, int)
                    else None
                ),
            }
        )

        page_documents.append(
            Document(
                page_content=page_text,
                metadata=metadata,
            )
        )

    if not page_documents:
        raise ValueError(
            f"No usable text could be extracted from {payload.file_path.name}."
        )

    return page_documents


def ingest_uploaded_document(
    *,
    file_path: str | Path,
    document_id: str,
    document_name: str,
    document_title: str,
    document_short_name: str,
    document_type: str = "legal_document",
    provision_type: str = "section",
    metadata: dict[str, Any] | None = None,
) -> UploadedDocumentIngestionResult:
    """Parse, chunk, embed, and store one uploaded legal PDF."""

    resolved_file_path = Path(file_path)
    payload = UploadedDocumentIngestionInput(
        file_path=resolved_file_path,
        document_id=_normalize_text(document_id),
        document_name=_normalize_text(document_name),
        document_title=_normalize_text(document_title),
        document_short_name=_normalize_text(document_short_name),
        document_type=_normalize_text(document_type) or "legal_document",
        provision_type=_normalize_text(provision_type) or "section",
        metadata=dict(metadata or {}),
    )

    page_documents = load_uploaded_document_pages(payload)
    chunks = create_chunks(page_documents)

    validate_final_chunks(chunks)
    qdrant_point_ids = [
        _create_deterministic_chunk_id(chunk)
        for chunk in chunks
    ]

    vector_store, client = create_vector_store(reset=False)

    try:
        vector_store.add_documents(
            documents=chunks,
            ids=qdrant_point_ids,
        )
    finally:
        client.close()

    logger.info(
        "Ingested uploaded legal document %s into Qdrant with %s chunks.",
        payload.document_id,
        len(chunks),
    )

    return UploadedDocumentIngestionResult(
        document_id=payload.document_id,
        page_count=len(page_documents),
        chunk_count=len(chunks),
        qdrant_point_ids=qdrant_point_ids,
    )
