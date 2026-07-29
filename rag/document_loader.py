import hashlib
import os
import re
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document

from rag.text_cleaner import (
    CleanedPage,
    clean_pdf_pages,
    display_cleaning_statistics,
)


load_dotenv()


# -------------------------------------------------------------------
# Environment configuration
# -------------------------------------------------------------------

DOCUMENTS_DIRECTORY = Path(
    os.getenv(
        "DOCUMENTS_DIRECTORY",
        "./data/documents",
    )
)

MIN_PAGE_CHARACTERS = int(
    os.getenv(
        "MIN_PAGE_CHARACTERS",
        "40",
    )
)

FAIL_ON_DOCUMENT_ERROR = (
    os.getenv(
        "FAIL_ON_DOCUMENT_ERROR",
        "False",
    ).lower()
    == "true"
)

DUPLICATE_PAGE_MIN_CHARACTERS = int(
    os.getenv(
        "DUPLICATE_PAGE_MIN_CHARACTERS",
        "300",
    )
)

INHERIT_CHAPTER_METADATA = (
    os.getenv(
        "INHERIT_CHAPTER_METADATA",
        "True",
    ).lower()
    == "true"
)

MAX_CHAPTER_INHERITANCE_PAGES = int(
    os.getenv(
        "MAX_CHAPTER_INHERITANCE_PAGES",
        "15",
    )
)

INCLUDE_AMENDMENT_METADATA = (
    os.getenv(
        "INCLUDE_AMENDMENT_METADATA",
        "True",
    ).lower()
    == "true"
)

SKIP_SUSPICIOUS_PAGES = (
    os.getenv(
        "SKIP_SUSPICIOUS_PAGES",
        "False",
    ).lower()
    == "true"
)

FAIL_ON_SUSPICIOUS_PAGE = (
    os.getenv(
        "FAIL_ON_SUSPICIOUS_PAGE",
        "False",
    ).lower()
    == "true"
)

SKIP_HEADING_ONLY_PAGES = (
    os.getenv(
        "SKIP_HEADING_ONLY_PAGES",
        "False",
    ).lower()
    == "true"
)

MAX_CHARACTER_REMOVAL_RATIO = float(
    os.getenv(
        "MAX_CHARACTER_REMOVAL_RATIO",
        "0.45",
    )
)

MIN_ALPHABETIC_RATIO = float(
    os.getenv(
        "MIN_ALPHABETIC_RATIO",
        "0.55",
    )
)

MAX_ARTIFACTS_PER_1000_CHARS = float(
    os.getenv(
        "MAX_ARTIFACTS_PER_1000_CHARS",
        "25",
    )
)

DEFAULT_DOCUMENT_TYPE = os.getenv(
    "DEFAULT_DOCUMENT_TYPE",
    "legal_document",
)

DOCUMENT_ID_HASH_LENGTH = int(
    os.getenv(
        "DOCUMENT_ID_HASH_LENGTH",
        "12",
    )
)


# -------------------------------------------------------------------
# Multi-document identity helpers
# -------------------------------------------------------------------

def validate_loader_settings() -> None:
    """Validate loader configuration."""

    if MIN_PAGE_CHARACTERS < 0:
        raise ValueError(
            "MIN_PAGE_CHARACTERS cannot be negative."
        )

    if DUPLICATE_PAGE_MIN_CHARACTERS < 0:
        raise ValueError(
            "DUPLICATE_PAGE_MIN_CHARACTERS cannot be negative."
        )

    if MAX_CHAPTER_INHERITANCE_PAGES < 0:
        raise ValueError(
            "MAX_CHAPTER_INHERITANCE_PAGES cannot be negative."
        )

    if not 0.0 <= MAX_CHARACTER_REMOVAL_RATIO <= 1.0:
        raise ValueError(
            "MAX_CHARACTER_REMOVAL_RATIO must be between 0 and 1."
        )

    if not 0.0 <= MIN_ALPHABETIC_RATIO <= 1.0:
        raise ValueError(
            "MIN_ALPHABETIC_RATIO must be between 0 and 1."
        )

    if MAX_ARTIFACTS_PER_1000_CHARS < 0:
        raise ValueError(
            "MAX_ARTIFACTS_PER_1000_CHARS cannot be negative."
        )

    if DOCUMENT_ID_HASH_LENGTH < 8:
        raise ValueError(
            "DOCUMENT_ID_HASH_LENGTH must be at least 8."
        )


def normalize_identifier(value: str) -> str:
    """Convert text into a stable lowercase identifier."""

    value = re.sub(
        r"[^a-z0-9]+",
        "_",
        value.strip().lower(),
    )
    value = re.sub(
        r"_+",
        "_",
        value,
    )

    return value.strip("_") or "document"


def get_relative_source_path(
    pdf_path: Path,
) -> str:
    """Return a stable path relative to the documents folder."""

    resolved_path = pdf_path.resolve()
    resolved_directory = DOCUMENTS_DIRECTORY.resolve()

    try:
        relative_path = resolved_path.relative_to(
            resolved_directory
        )
    except ValueError:
        relative_path = resolved_path

    return relative_path.as_posix()


def create_document_id(
    pdf_path: Path,
) -> str:
    """
    Create a stable unique ID from file name and relative path.

    The path hash avoids collisions when separate folders contain PDFs
    with the same file name.
    """

    relative_path = get_relative_source_path(
        pdf_path
    )
    base_id = normalize_identifier(
        pdf_path.stem
    )
    path_hash = hashlib.sha256(
        relative_path.lower().encode("utf-8")
    ).hexdigest()[:DOCUMENT_ID_HASH_LENGTH]

    return f"{base_id}_{path_hash}"


def create_document_title(
    pdf_path: Path,
) -> str:
    """Create a readable document title from the file name."""

    title = re.sub(
        r"[_\-]+",
        " ",
        pdf_path.stem,
    )
    title = re.sub(
        r"\s+",
        " ",
        title,
    ).strip()

    return title or pdf_path.name


def infer_document_type(
    pdf_path: Path,
) -> str:
    """Infer a broad legal document type from the PDF name."""

    name = normalize_identifier(
        pdf_path.stem
    )

    rules: tuple[
        tuple[tuple[str, ...], str],
        ...
    ] = (
        (
            ("pakistan_penal_code", "penal_code", "ppc"),
            "substantive_criminal_law",
        ),
        (
            (
                "code_of_criminal_procedure",
                "criminal_procedure",
                "crpc",
            ),
            "criminal_procedure",
        ),
        (
            (
                "qanuneshahadat",
                "qanun_e_shahadat",
                "law_of_evidence",
                "qanun_e_shahadat_order",
                "shahadat_order",
                "evidence_order",
            ),
            "law_of_evidence",
        ),
        (
            ("constitution", "constitutional"),
            "constitutional_law",
        ),
        (
            ("peca", "electronic_crimes", "cybercrime"),
            "cybercrime_law",
        ),
        (
            ("anti_terrorism", "terrorism"),
            "anti_terrorism_law",
        ),
    )

    for keywords, document_type in rules:
        if any(
            keyword in name
            for keyword in keywords
        ):
            return document_type

    return DEFAULT_DOCUMENT_TYPE


def build_document_metadata(
    pdf_path: Path,
    total_pages: int,
) -> dict[str, Any]:
    """Build metadata shared by every page of one PDF."""

    relative_path = get_relative_source_path(
        pdf_path
    )

    return {
        "document_id": create_document_id(
            pdf_path
        ),
        "document_name": pdf_path.name,
        "document_title": create_document_title(
            pdf_path
        ),
        "document_type": infer_document_type(
            pdf_path
        ),
        "file_type": "pdf",
        "source_path": str(
            pdf_path.resolve()
        ),
        "relative_source_path": relative_path,
        "source_folder": (
            Path(relative_path).parent.as_posix()
        ),
        "total_pages": total_pages,
    }


# -------------------------------------------------------------------
# File discovery
# -------------------------------------------------------------------

def find_pdf_files() -> list[Path]:
    """
    Find all PDF files inside the configured documents directory.
    """

    validate_loader_settings()

    if not DOCUMENTS_DIRECTORY.exists():
        raise FileNotFoundError(
            "Documents folder was not found:\n"
            f"{DOCUMENTS_DIRECTORY.resolve()}"
        )

    pdf_files = sorted(
        (
            file_path
            for file_path in DOCUMENTS_DIRECTORY.rglob("*")
            if (
                file_path.is_file()
                and file_path.suffix.lower() == ".pdf"
            )
        ),
        key=lambda path: (
            get_relative_source_path(path).lower()
        ),
    )

    if not pdf_files:
        raise FileNotFoundError(
            "No PDF files were found inside:\n"
            f"{DOCUMENTS_DIRECTORY.resolve()}"
        )

    return pdf_files


# -------------------------------------------------------------------
# Stable identifiers and page alignment
# -------------------------------------------------------------------

def create_content_hash(text: str) -> str:
    """
    Create a stable SHA-256 hash for cleaned page content.
    """

    return hashlib.sha256(
        text.encode("utf-8")
    ).hexdigest()


def get_page_number(
    document: Document,
    fallback_page_number: int,
) -> int:
    """
    Resolve a human-readable 1-based page number.

    PyPDFLoader usually stores a zero-based `page` value.
    """

    original_page = document.metadata.get("page")

    if isinstance(original_page, int):
        return original_page + 1

    return fallback_page_number


def validate_page_alignment(
    page_documents: list[Document],
    cleaned_pages: list[CleanedPage],
) -> None:
    """
    Ensure each raw PDF page still matches exactly one cleaned page.
    """

    if len(page_documents) != len(cleaned_pages):
        raise RuntimeError(
            "Cleaned page count does not match loaded page count. "
            f"Loaded pages: {len(page_documents)}, "
            f"cleaned pages: {len(cleaned_pages)}."
        )


# -------------------------------------------------------------------
# Cross-page legal metadata propagation
# -------------------------------------------------------------------

def inherit_chapter_metadata(
    cleaned_pages: list[CleanedPage],
) -> None:
    """
    Propagate chapter and part metadata to nearby continuation pages.

    Chapter inheritance is deliberately limited by
    MAX_CHAPTER_INHERITANCE_PAGES. It is safer to store no chapter
    than to attach an incorrect chapter to distant sections.
    """

    current_chapter_number: str | None = None
    current_chapter_title: str | None = None
    current_part: str | None = None

    chapter_source_page: int | None = None
    part_source_page: int | None = None

    for cleaned_page in cleaned_pages:
        metadata = cleaned_page.metadata
        page_number = metadata.get("page_number")

        detected_chapter_number = metadata.get(
            "chapter_number"
        )
        detected_chapter_title = metadata.get(
            "chapter_title"
        )
        detected_part = metadata.get("part")

        chapter_detected_on_page = bool(
            detected_chapter_number
            or detected_chapter_title
        )

        if detected_chapter_number:
            current_chapter_number = (
                detected_chapter_number
            )
            chapter_source_page = page_number

        if detected_chapter_title:
            current_chapter_title = (
                detected_chapter_title
            )
            chapter_source_page = page_number

        if detected_part:
            current_part = detected_part
            part_source_page = page_number

        chapter_distance: int | None = None

        if (
            isinstance(page_number, int)
            and isinstance(chapter_source_page, int)
        ):
            chapter_distance = (
                page_number - chapter_source_page
            )

        chapter_inheritance_allowed = bool(
            not chapter_detected_on_page
            and chapter_source_page is not None
            and chapter_distance is not None
            and 0 <= chapter_distance
            <= MAX_CHAPTER_INHERITANCE_PAGES
        )

        if chapter_inheritance_allowed:
            if not metadata.get("chapter_number"):
                metadata["chapter_number"] = (
                    current_chapter_number
                )

            if not metadata.get("chapter_title"):
                metadata["chapter_title"] = (
                    current_chapter_title
                )

            metadata["chapter_metadata_inherited"] = True
            metadata["chapter_inheritance_uncertain"] = False

        elif chapter_detected_on_page:
            metadata["chapter_metadata_inherited"] = False
            metadata["chapter_inheritance_uncertain"] = False

        else:
            # Do not keep an old chapter indefinitely.
            metadata["chapter_number"] = None
            metadata["chapter_title"] = None
            metadata["chapter_metadata_inherited"] = False
            metadata["chapter_inheritance_uncertain"] = bool(
                chapter_source_page is not None
            )

        metadata["chapter_source_page"] = (
            chapter_source_page
        )

        metadata["chapter_distance_pages"] = (
            chapter_distance
        )

        # Part inheritance is less risky, but still records its source.
        if not metadata.get("part") and current_part:
            metadata["part"] = current_part
            metadata["part_metadata_inherited"] = True
        else:
            metadata["part_metadata_inherited"] = False

        metadata["part_source_page"] = (
            part_source_page
        )

        if (
            not metadata.get("legal_topic")
            and metadata.get("chapter_title")
        ):
            metadata["legal_topic"] = (
                str(
                    metadata["chapter_title"]
                ).lower()
            )


# -------------------------------------------------------------------
# Cleaning metadata
# -------------------------------------------------------------------

def build_cleaning_metadata(
    cleaned_page: CleanedPage,
) -> dict[str, Any]:
    """
    Convert cleaner statistics into document metadata.
    """

    statistics = cleaned_page.statistics

    metadata: dict[str, Any] = {
        "headers_removed": (
            statistics.headers_removed
        ),
        "footers_removed": (
            statistics.footers_removed
        ),
        "artifacts_removed": (
            statistics.artifacts_removed
        ),
        "footnotes_removed": (
            statistics.footnotes_removed
        ),
        "broken_words_fixed": (
            statistics.broken_words_fixed
        ),
        "hyphenations_fixed": (
            statistics.hyphenations_fixed
        ),
    }

    amendment_notes_removed = int(
        getattr(
            statistics,
            "amendment_notes_removed",
            0,
        )
    )

    amendment_continuations_removed = int(
        getattr(
            statistics,
            "amendment_continuation_lines_removed",
            0,
        )
    )

    if INCLUDE_AMENDMENT_METADATA:
        metadata.update(
            {
                "amendment_notes_removed": (
                    amendment_notes_removed
                ),
                "amendment_continuation_lines_removed": (
                    amendment_continuations_removed
                ),
                "contained_amendment_notes": (
                    amendment_notes_removed > 0
                    or amendment_continuations_removed > 0
                ),
            }
        )

    return metadata


# -------------------------------------------------------------------
# Page-quality validation
# -------------------------------------------------------------------

def calculate_alphabetic_ratio(text: str) -> float:
    """
    Return the proportion of non-whitespace characters that are letters.
    """

    non_whitespace_characters = [
        character
        for character in text
        if not character.isspace()
    ]

    if not non_whitespace_characters:
        return 0.0

    alphabetic_count = sum(
        character.isalpha()
        for character in non_whitespace_characters
    )

    return (
        alphabetic_count
        / len(non_whitespace_characters)
    )


def evaluate_page_quality(
    raw_text: str,
    cleaned_page: CleanedPage,
) -> dict[str, Any]:
    """
    Detect noisy, heading-only, or over-cleaned pages.

    Pages are flagged rather than deleted by default.
    """

    cleaned_text = cleaned_page.text
    statistics = cleaned_page.statistics
    cleaner_metadata = cleaned_page.metadata

    raw_length = len(raw_text)
    cleaned_length = len(cleaned_text)

    removal_ratio = (
        max(0, raw_length - cleaned_length)
        / max(1, raw_length)
    )

    alphabetic_ratio = calculate_alphabetic_ratio(
        cleaned_text
    )

    total_noise_items = (
        statistics.artifacts_removed
        + statistics.footnotes_removed
        + int(
            getattr(
                statistics,
                "amendment_notes_removed",
                0,
            )
        )
        + int(
            getattr(
                statistics,
                "amendment_continuation_lines_removed",
                0,
            )
        )
    )

    artifacts_per_1000_chars = (
        total_noise_items
        / max(1, cleaned_length)
        * 1000
    )

    heading_only_page = bool(
        cleaner_metadata.get(
            "heading_only_page",
            False,
        )
    )

    section_body_present = bool(
        cleaner_metadata.get(
            "section_body_present",
            True,
        )
    )

    reasons: list[str] = []

    if removal_ratio > MAX_CHARACTER_REMOVAL_RATIO:
        reasons.append(
            "high_character_removal_ratio"
        )

    if (
        cleaned_length >= MIN_PAGE_CHARACTERS
        and alphabetic_ratio < MIN_ALPHABETIC_RATIO
    ):
        reasons.append(
            "low_alphabetic_content_ratio"
        )

    if (
        artifacts_per_1000_chars
        > MAX_ARTIFACTS_PER_1000_CHARS
    ):
        reasons.append(
            "high_artifact_density"
        )

    if heading_only_page:
        reasons.append(
            "heading_only_page"
        )

    if (
        cleaner_metadata.get("section_numbers")
        and not section_body_present
        and "heading_only_page" not in reasons
    ):
        reasons.append(
            "section_body_missing"
        )

    if (
        cleaned_length >= 300
        and not cleaner_metadata.get(
            "section_numbers"
        )
        and not cleaner_metadata.get(
            "chapter_number"
        )
        and alphabetic_ratio < 0.65
    ):
        reasons.append(
            "long_unstructured_low_text_quality"
        )

    return {
        "page_quality_status": (
            "suspicious"
            if reasons
            else "acceptable"
        ),
        "page_quality_suspicious": bool(
            reasons
        ),
        "page_quality_reasons": reasons,
        "character_removal_ratio": round(
            removal_ratio,
            4,
        ),
        "alphabetic_ratio": round(
            alphabetic_ratio,
            4,
        ),
        "artifacts_per_1000_chars": round(
            artifacts_per_1000_chars,
            4,
        ),
        "heading_only_page": heading_only_page,
        "section_body_present": (
            section_body_present
        ),
    }


# -------------------------------------------------------------------
# PDF loading
# -------------------------------------------------------------------

def load_pdf(
    pdf_path: Path,
) -> list[Document]:
    """
    Extract, clean, validate, and enrich one PDF.

    The whole PDF is cleaned together so repeated headers and footers
    can be detected across pages.
    """

    print(f"Extracting text from: {pdf_path.name}")

    resolved_pdf_path = pdf_path.resolve()

    loader = PyPDFLoader(
        str(resolved_pdf_path)
    )

    page_documents = loader.load()

    if not page_documents:
        raise ValueError(
            f"No pages were extracted from {pdf_path.name}."
        )

    total_pages = len(page_documents)

    shared_document_metadata = (
        build_document_metadata(
            pdf_path=pdf_path,
            total_pages=total_pages,
        )
    )

    raw_page_texts = [
        document.page_content or ""
        for document in page_documents
    ]

    cleaned_pages = clean_pdf_pages(
        raw_page_texts
    )

    validate_page_alignment(
        page_documents=page_documents,
        cleaned_pages=cleaned_pages,
    )

    if INHERIT_CHAPTER_METADATA:
        inherit_chapter_metadata(
            cleaned_pages
        )

    valid_documents: list[Document] = []
    seen_hashes: set[str] = set()

    empty_pages = 0
    short_pages = 0
    duplicate_pages = 0
    suspicious_pages = 0
    skipped_suspicious_pages = 0
    heading_only_pages = 0
    skipped_heading_only_pages = 0

    suspicious_page_details: list[
        tuple[int, list[str]]
    ] = []

    for index, (
        document,
        raw_text,
        cleaned_page,
    ) in enumerate(
        zip(
            page_documents,
            raw_page_texts,
            cleaned_pages,
        ),
        start=1,
    ):
        cleaned_text = cleaned_page.text.strip()

        if not cleaned_text:
            empty_pages += 1
            continue

        if len(cleaned_text) < MIN_PAGE_CHARACTERS:
            short_pages += 1
            continue

        page_number = get_page_number(
            document=document,
            fallback_page_number=index,
        )

        quality_metadata = evaluate_page_quality(
            raw_text=raw_text,
            cleaned_page=cleaned_page,
        )

        if quality_metadata["heading_only_page"]:
            heading_only_pages += 1

            if SKIP_HEADING_ONLY_PAGES:
                skipped_heading_only_pages += 1
                continue

        if quality_metadata[
            "page_quality_suspicious"
        ]:
            suspicious_pages += 1
            suspicious_page_details.append(
                (
                    page_number,
                    quality_metadata[
                        "page_quality_reasons"
                    ],
                )
            )

            if FAIL_ON_SUSPICIOUS_PAGE:
                raise RuntimeError(
                    "Suspicious PDF page detected. "
                    f"Page: {page_number}; reasons: "
                    f"{quality_metadata['page_quality_reasons']}"
                )

            if SKIP_SUSPICIOUS_PAGES:
                skipped_suspicious_pages += 1
                continue

        content_hash = create_content_hash(
            cleaned_text
        )

        if (
            len(cleaned_text)
            >= DUPLICATE_PAGE_MIN_CHARACTERS
            and content_hash in seen_hashes
        ):
            duplicate_pages += 1
            continue

        if (
            len(cleaned_text)
            >= DUPLICATE_PAGE_MIN_CHARACTERS
        ):
            seen_hashes.add(content_hash)

        document_id = str(
            shared_document_metadata[
                "document_id"
            ]
        )
        page_id = (
            f"{document_id}-page-{page_number}"
        )

        document.page_content = cleaned_text

        # Preserve PyPDFLoader metadata and add legal metadata.
        document.metadata.update(
            cleaned_page.metadata
        )

        document.metadata.update(
            build_cleaning_metadata(
                cleaned_page
            )
        )

        document.metadata.update(
            quality_metadata
        )

        raw_character_count = len(raw_text)
        cleaned_character_count = len(
            cleaned_text
        )

        document.metadata.update(
            shared_document_metadata
        )

        document.metadata.update(
            {
                "page_number": page_number,
                "page_index": page_number - 1,
                "page_id": page_id,
                "text_cleaned": True,
                "raw_character_count": (
                    raw_character_count
                ),
                "cleaned_character_count": (
                    cleaned_character_count
                ),
                "character_count": (
                    cleaned_character_count
                ),
                "characters_removed": max(
                    0,
                    raw_character_count
                    - cleaned_character_count,
                ),
                "content_hash": content_hash,
            }
        )

        valid_documents.append(document)

    total_raw_characters = sum(
        len(text)
        for text in raw_page_texts
    )

    total_cleaned_characters = sum(
        len(document.page_content)
        for document in valid_documents
    )

    print(
        "Document ID: "
        f"{shared_document_metadata['document_id']}"
    )
    print(
        "Document title: "
        f"{shared_document_metadata['document_title']}"
    )
    print(
        "Document type: "
        f"{shared_document_metadata['document_type']}"
    )
    print(f"Pages in PDF: {len(page_documents)}")
    print(
        "Pages containing usable text: "
        f"{len(valid_documents)}"
    )
    print(f"Empty pages skipped: {empty_pages}")
    print(f"Short pages skipped: {short_pages}")
    print(
        "Duplicate pages skipped: "
        f"{duplicate_pages}"
    )
    print(
        "Heading-only pages detected: "
        f"{heading_only_pages}"
    )
    print(
        "Heading-only pages skipped: "
        f"{skipped_heading_only_pages}"
    )
    print(
        "Suspicious pages detected: "
        f"{suspicious_pages}"
    )
    print(
        "Suspicious pages skipped: "
        f"{skipped_suspicious_pages}"
    )
    print(
        "Raw characters extracted: "
        f"{total_raw_characters}"
    )
    print(
        "Cleaned characters extracted: "
        f"{total_cleaned_characters}"
    )
    print(
        "Characters removed: "
        f"{max(0, total_raw_characters - total_cleaned_characters)}"
    )

    if suspicious_page_details:
        print("\nSuspicious page details:")

        for (
            page_number,
            reasons,
        ) in suspicious_page_details[:20]:
            print(
                f"- Page {page_number}: "
                f"{', '.join(reasons)}"
            )

        if len(suspicious_page_details) > 20:
            remaining = (
                len(suspicious_page_details)
                - 20
            )
            print(
                f"- ...and {remaining} more "
                "suspicious pages"
            )

    display_cleaning_statistics(
        cleaned_pages
    )

    if not valid_documents:
        raise ValueError(
            f"No usable page text remained after cleaning "
            f"{pdf_path.name}."
        )

    return valid_documents


# -------------------------------------------------------------------
# Multi-document loading
# -------------------------------------------------------------------

def load_all_documents() -> list[Document]:
    """
    Find, extract, clean, validate, and combine all PDF files.

    Returns:
        A list of cleaned page-level LangChain Document objects.
    """

    pdf_files = find_pdf_files()

    print(f"PDF files found: {len(pdf_files)}\n")

    all_documents: list[Document] = []
    failed_files: list[tuple[str, str]] = []
    loaded_document_ids: set[str] = set()
    document_summaries: list[
        dict[str, Any]
    ] = []

    for pdf_path in pdf_files:
        try:
            documents = load_pdf(
                pdf_path
            )

        except Exception as error:
            failed_files.append(
                (
                    pdf_path.name,
                    str(error),
                )
            )

            print(
                f"Failed to load {pdf_path.name}: "
                f"{error}\n"
            )

            if FAIL_ON_DOCUMENT_ERROR:
                raise

            continue

        document_id = str(
            documents[0].metadata.get(
                "document_id",
                "",
            )
        )

        if not document_id:
            raise RuntimeError(
                f"Document ID was not created for {pdf_path.name}."
            )

        if document_id in loaded_document_ids:
            raise RuntimeError(
                "Duplicate document ID detected: "
                f"{document_id}."
            )

        loaded_document_ids.add(
            document_id
        )

        all_documents.extend(
            documents
        )

        document_summaries.append(
            {
                "document_id": document_id,
                "document_name": documents[0].metadata.get(
                    "document_name",
                    pdf_path.name,
                ),
                "document_title": documents[0].metadata.get(
                    "document_title",
                    pdf_path.stem,
                ),
                "document_type": documents[0].metadata.get(
                    "document_type",
                    DEFAULT_DOCUMENT_TYPE,
                ),
                "pages_loaded": len(documents),
                "total_pages": documents[0].metadata.get(
                    "total_pages",
                    len(documents),
                ),
            }
        )

        print()

    if not all_documents:
        raise ValueError(
            "PDF files were found, but no usable text "
            "was extracted from them."
        )

    print("=" * 70)
    print("DOCUMENT LOADING SUMMARY")
    print("=" * 70)
    print(
        "Total PDF files discovered: "
        f"{len(pdf_files)}"
    )
    print(
        "PDF files loaded successfully: "
        f"{len(pdf_files) - len(failed_files)}"
    )
    print(
        "PDF files failed: "
        f"{len(failed_files)}"
    )
    print(
        "Total page documents loaded: "
        f"{len(all_documents)}"
    )
    print(
        "Unique document IDs: "
        f"{len(loaded_document_ids)}"
    )

    if document_summaries:
        print("\nPer-document summary:")

        for summary in document_summaries:
            print(
                "- "
                f"{summary['document_title']} "
                f"({summary['document_name']})"
            )
            print(
                "  ID: "
                f"{summary['document_id']}"
            )
            print(
                "  Type: "
                f"{summary['document_type']}"
            )
            print(
                "  Pages loaded: "
                f"{summary['pages_loaded']}/"
                f"{summary['total_pages']}"
            )

    if failed_files:
        print("\nFailed files:")

        for file_name, error_message in failed_files:
            print(
                f"- {file_name}: {error_message}"
            )

    return all_documents


# -------------------------------------------------------------------
# Debug sample display
# -------------------------------------------------------------------

def display_document_samples(
    documents: list[Document],
    sample_count: int = 3,
) -> None:
    """
    Print representative cleaned pages and metadata.
    """

    if not documents:
        print(
            "No document samples are available."
        )
        return

    actual_count = min(
        sample_count,
        len(documents),
    )

    if actual_count == 1:
        sample_indexes = [0]
    else:
        sample_indexes = sorted(
            {
                round(
                    index
                    * (len(documents) - 1)
                    / (actual_count - 1)
                )
                for index in range(
                    actual_count
                )
            }
        )

    print("\n" + "=" * 70)
    print("CLEANED TEXT SAMPLES")
    print("=" * 70)

    for (
        sample_number,
        document_index,
    ) in enumerate(
        sample_indexes,
        start=1,
    ):
        document = documents[
            document_index
        ]

        print("\n" + "-" * 70)
        print(f"SAMPLE {sample_number}")
        print("-" * 70)

        print("\nMetadata:")
        print(document.metadata)

        print("\nCleaned text:")
        print(
            document.page_content[:1500]
        )

        print(
            "\nDisplayed characters: "
            f"{min(1500, len(document.page_content))}"
        )


def main() -> None:
    """
    Test PDF loading and text cleaning from the terminal.
    """

    try:
        documents = load_all_documents()

        display_document_samples(
            documents=documents,
            sample_count=3,
        )

    except Exception as error:
        print(
            f"\nDocument loading error: {error}"
        )


if __name__ == "__main__":
    main()
