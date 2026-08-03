from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from langchain_core.documents import Document
from pypdf import PdfReader

from rag.config import DOCUMENTS_DIR
from rag.document_registry import resolve_document
from rag.text_cleaner import clean_pdf_pages


load_dotenv()


FAIL_ON_DOCUMENT_ERROR = (
    os.getenv(
        "FAIL_ON_DOCUMENT_ERROR",
        "False",
    ).strip().lower()
    in {"1", "true", "yes", "on"}
)


def _deduplicate(values: list[str]) -> list[str]:
    seen: set[str] = set()
    unique_values: list[str] = []

    for value in values:
        cleaned_value = value.strip()

        if cleaned_value and cleaned_value not in seen:
            seen.add(cleaned_value)
            unique_values.append(cleaned_value)

    return unique_values


@dataclass
class QueryPlan:
    original_question: str
    question_type: str
    concepts: list[str]
    section_number: str | None
    retrieval_queries: list[str]

    section_hints: list[str] = field(default_factory=list)
    answer_style: str = "general"

    document_ids: list[str] = field(default_factory=list)
    document_hints: list[str] = field(default_factory=list)

    article_number: str | None = None

    provision_numbers: list[str] = field(default_factory=list)
    provision_type: str | None = None

    def __post_init__(self) -> None:
        self.concepts = _deduplicate(self.concepts)
        self.retrieval_queries = _deduplicate(
            self.retrieval_queries
        )
        self.section_hints = _deduplicate(
            self.section_hints
        )
        self.document_ids = _deduplicate(
            self.document_ids
        )
        self.document_hints = _deduplicate(
            self.document_hints
        )
        self.provision_numbers = _deduplicate(
            self.provision_numbers
        )

        if (
            self.section_number
            and self.section_number
            not in self.provision_numbers
        ):
            self.provision_numbers.append(
                self.section_number
            )

        if (
            self.article_number
            and self.article_number
            not in self.provision_numbers
        ):
            self.provision_numbers.append(
                self.article_number
            )


@dataclass
class CandidateDocument:
    document: Document
    relevance_score: float
    query_index: int
    query_text: str

    retrieval_method: str = "vector"

    @property
    def metadata(self) -> dict[str, Any]:
        return self.document.metadata

    @property
    def document_id(self) -> str | None:
        return self.document.metadata.get(
            "document_id"
        )

    @property
    def provision_number(self) -> str | None:
        metadata = self.document.metadata

        return (
            metadata.get("provision_number")
            or metadata.get("section_number")
            or metadata.get("article_number")
        )


@dataclass
class RankedDocument:
    document: Document
    fusion_score: float

    relevance_score: float | None = None
    matched_queries: int = 1
    keyword_overlap: float = 0.0
    concept_overlap: float = 0.0

    section_boost: float = 0.0

    document_boost: float = 0.0
    provision_boost: float = 0.0

    special_penalty: float = 0.0
    duplicate_penalty: float = 0.0
    final_score: float = 0.0

    @property
    def metadata(self) -> dict[str, Any]:
        return self.document.metadata

    @property
    def document_id(self) -> str | None:
        return self.document.metadata.get(
            "document_id"
        )

    @property
    def provision_number(self) -> str | None:
        metadata = self.document.metadata

        return (
            metadata.get("provision_number")
            or metadata.get("section_number")
            or metadata.get("article_number")
        )


@dataclass
class RetrievalConfidence:
    label: str
    score: float
    top_similarity: float
    average_similarity: float
    section_count: int
    concept_coverage: float

    document_count: int = 0
    document_coverage: float = 0.0
    exact_document_match: bool = False
    exact_provision_match: bool = False


@dataclass
class AnswerResult:
    answer: str
    sources: list[dict[str, Any]]
    question_type: str
    detected_concepts: list[str]
    retrieved_document_count: int
    confidence: RetrievalConfidence

    used_llm: bool = False
    llm_provider: str | None = None
    fallback_used: bool = False


def find_pdf_files() -> list[Path]:
    if not DOCUMENTS_DIR.exists():
        raise FileNotFoundError(
            f"Documents directory not found: {DOCUMENTS_DIR}"
        )

    pdf_files = sorted(
        DOCUMENTS_DIR.rglob("*.pdf"),
        key=lambda path: str(path).lower(),
    )

    return pdf_files


def load_pdf(pdf_path: Path) -> list[Document]:
    document_config = resolve_document(pdf_path.name)

    reader = PdfReader(str(pdf_path))
    raw_pages: list[str] = []

    for page in reader.pages:
        raw_pages.append(page.extract_text() or "")

    if not raw_pages:
        raise ValueError(f"No pages were found in {pdf_path.name}.")

    cleaned_pages = clean_pdf_pages(raw_pages)
    page_documents: list[Document] = []

    for cleaned_page in cleaned_pages:
        page_text = cleaned_page.text.strip()

        if not page_text:
            continue

        metadata = dict(cleaned_page.metadata)
        page_number = metadata.get("page_number")

        metadata.update(
            {
                "document_id": document_config.document_id,
                "document_name": document_config.full_name,
                "document_title": document_config.full_name,
                "document_short_name": document_config.short_name,
                "document_type": document_config.document_type,
                "provision_type": document_config.provision_type,
                "source_path": str(pdf_path),
                "source_file_name": pdf_path.name,
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
            f"No usable text could be extracted from {pdf_path.name}."
        )

    return page_documents


def load_all_documents() -> list[Document]:
    pdf_files = find_pdf_files()

    print(f"PDF files found: {len(pdf_files)}\n")

    all_documents: list[Document] = []
    failed_files: list[tuple[str, str]] = []

    for pdf_path in pdf_files:
        try:
            documents = load_pdf(pdf_path)
            all_documents.extend(documents)

        except Exception as error:
            failed_files.append(
                (pdf_path.name, str(error))
            )

            print(
                f"Failed to load {pdf_path.name}: "
                f"{error}\n"
            )

            if FAIL_ON_DOCUMENT_ERROR:
                raise

    if not all_documents:
        raise ValueError(
            "No usable PDF documents were loaded."
        )

    print("=" * 70)
    print("DOCUMENT LOADING SUMMARY")
    print("=" * 70)
    print(f"PDF files discovered: {len(pdf_files)}")
    print(
        f"PDF files loaded: "
        f"{len(pdf_files) - len(failed_files)}"
    )
    print(f"PDF files failed: {len(failed_files)}")
    print(f"Page documents loaded: {len(all_documents)}")

    return all_documents
