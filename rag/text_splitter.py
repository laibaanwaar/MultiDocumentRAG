from __future__ import annotations

import os
import re
from collections import Counter
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any

from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


load_dotenv()


# -------------------------------------------------------------------
# Chunking configuration
# -------------------------------------------------------------------

# all-MiniLM-L6-v2 truncates long inputs. A smaller character-based
# chunk is safer than the previous 1600-character default.
CHUNK_SIZE = int(
    os.getenv(
        "CHUNK_SIZE",
        "1000",
    )
)

CHUNK_OVERLAP = int(
    os.getenv(
        "CHUNK_OVERLAP",
        "150",
    )
)

MIN_PROVISION_BODY_CHARACTERS = int(
    os.getenv(
        "MIN_PROVISION_BODY_CHARACTERS",
        os.getenv(
            "MIN_SECTION_BODY_CHARACTERS",
            "80",
        ),
    )
)

# Backward-compatible name used by earlier tests/imports.
MIN_SECTION_BODY_CHARACTERS = MIN_PROVISION_BODY_CHARACTERS

INCLUDE_HEADING_ONLY_CHUNKS = (
    os.getenv(
        "INCLUDE_HEADING_ONLY_CHUNKS",
        "False",
    ).lower()
    == "true"
)

INCLUDE_UNSECTIONED_TEXT = (
    os.getenv(
        "INCLUDE_UNSECTIONED_TEXT",
        "True",
    ).lower()
    == "true"
)

PREPEND_DOCUMENT_CONTEXT = (
    os.getenv(
        "PREPEND_DOCUMENT_CONTEXT",
        "True",
    ).lower()
    == "true"
)

MAX_PROVISION_TITLE_CHARACTERS = int(
    os.getenv(
        "MAX_PROVISION_TITLE_CHARACTERS",
        "220",
    )
)


# -------------------------------------------------------------------
# Legal structure patterns
# -------------------------------------------------------------------

# Supports:
#   379. Punishment for theft
#   298-C. Person ...
#   Section 7.—Punishment for acts of terrorism
#   Article 10A. Right to fair trial
#
# The explicit "Section" / "Article" label is optional because many
# Pakistan statutes print only the number at the start of each heading.
PROVISION_PATTERN = re.compile(
    r"(?mi)"
    r"^[ \t]*"
    r"(?:\d+\[\s*)?"
    r"(?:(?P<label>section|article)\s+)?"
    r"(?P<number>"
    r"\d+"
    r"(?:\s*-\s*[A-Za-z]+|[A-Za-z]+)?"
    r")"
    r"\s*"
    r"(?:\.\s*(?:[-—–]\s*)?|[—–:-]\s*)"
    r"(?P<title>[^\n]+)"
)

# Backward-compatible pattern name.
SECTION_PATTERN = PROVISION_PATTERN

CHAPTER_PATTERN = re.compile(
    r"(?mi)^\s*CHAPTER\s+([IVXLCDM]+|\d+)\s*$"
)

STANDALONE_CHAPTER_PATTERN = re.compile(
    r"(?m)^\s*([IVXLCDM]+)\s*$"
)

UPPERCASE_STRUCTURE_PATTERN = re.compile(
    r"(?m)^\s*(OF\s+[A-Z][A-Z\s,.'()\-]+)\s*$"
)

MIXED_STRUCTURE_PATTERN = re.compile(
    r"(?m)^\s*(Of\s+[A-Z][A-Za-z\s,.'()\-]+)\s*$"
)

PART_PATTERN = re.compile(
    r"(?mi)^\s*PART\s+([IVXLCDM]+|\d+)\s*$"
)

SCHEDULE_PATTERN = re.compile(
    r"(?mi)^\s*(?:THE\s+)?(?:FIRST\s+|SECOND\s+|THIRD\s+|"
    r"FOURTH\s+|FIFTH\s+|SIXTH\s+|SEVENTH\s+|EIGHTH\s+)?"
    r"SCHEDULE\b[^\n]*$"
)

PREAMBLE_PATTERN = re.compile(
    r"(?mi)^\s*PREAMBLE\s*$"
)

EXPLANATION_PATTERN = re.compile(
    r"(?mi)^\s*Explanation(?:\s+\d+)?\s*[:.\-—–]?"
)

ILLUSTRATION_PATTERN = re.compile(
    r"(?mi)^\s*Illustrations?\s*[:.\-—–]?\s*$"
)

SUBSECTION_PATTERN = re.compile(
    r"(?mi)^\s*\((?:\d+|[a-z]|[ivxlcdm]+)\)\s+"
)

BODY_START_PATTERN = re.compile(
    r"^(?:"
    r"\(\d+\)|"
    r"Whoever|Any person|Every person|Nothing|Where|When|If\b|"
    r"A person|Provided|In this section|In this article|"
    r"For the purposes of|This Act|The provisions|"
    r"In this Chapter|In case of|Subject to|Notwithstanding|"
    r"It shall|There shall|No person|A person who"
    r")",
    re.IGNORECASE,
)

INLINE_BODY_PATTERN = re.compile(
    r"(?i)"
    r"(?P<title>.*?)"
    r"(?:\.\s*)?[-—–:]\s*"
    r"(?P<body>"
    r"\(\d+\).*|"
    r"Whoever\b.*|Any person\b.*|Every person\b.*|"
    r"Nothing\b.*|Where\b.*|When\b.*|If\b.*|"
    r"A person\b.*|Provided\b.*|In this (?:section|article)\b.*|"
    r"For the purposes of\b.*|This Act\b.*|"
    r"The provisions\b.*|In this Chapter\b.*|"
    r"In case of\b.*|Subject to\b.*|Notwithstanding\b.*|"
    r"It shall\b.*|There shall\b.*|No person\b.*"
    r")$"
)


# -------------------------------------------------------------------
# Internal data structures
# -------------------------------------------------------------------

@dataclass(slots=True)
class PageSpan:
    page_number: int
    start: int
    end: int
    document: Document


@dataclass(slots=True)
class ProvisionBlock:
    provision_type: str
    provision_number: str | None
    provision_title: str | None
    text: str
    page_start: int | None
    page_end: int | None
    source_pages: list[int] = field(default_factory=list)
    source_metadata: dict[str, Any] = field(default_factory=dict)
    heading_only: bool = False
    provision_body_present: bool = True
    is_unsectioned: bool = False

    @property
    def section_number(self) -> str | None:
        """Compatibility view for earlier section-only code."""

        if self.provision_type == "section":
            return self.provision_number

        return None

    @property
    def section_title(self) -> str | None:
        """Compatibility view for earlier section-only code."""

        if self.provision_type == "section":
            return self.provision_title

        return None

    @property
    def section_body_present(self) -> bool:
        """Compatibility view for earlier section-only code."""

        return self.provision_body_present


# Backward-compatible class name used by the earlier implementation.
SectionBlock = ProvisionBlock


# -------------------------------------------------------------------
# Configuration and metadata validation
# -------------------------------------------------------------------

def validate_chunk_settings() -> None:
    """Validate environment-driven chunking settings."""

    if CHUNK_SIZE <= 0:
        raise ValueError(
            "CHUNK_SIZE must be greater than zero."
        )

    if CHUNK_OVERLAP < 0:
        raise ValueError(
            "CHUNK_OVERLAP cannot be negative."
        )

    if CHUNK_OVERLAP >= CHUNK_SIZE:
        raise ValueError(
            "CHUNK_OVERLAP must be smaller than CHUNK_SIZE."
        )

    if MIN_PROVISION_BODY_CHARACTERS < 0:
        raise ValueError(
            "MIN_PROVISION_BODY_CHARACTERS cannot be negative."
        )

    if MAX_PROVISION_TITLE_CHARACTERS < 20:
        raise ValueError(
            "MAX_PROVISION_TITLE_CHARACTERS must be at least 20."
        )


def normalize_provision_type(value: Any) -> str:
    """
    Normalize provision type to either ``section`` or ``article``.

    The updated document loader assigns:
      - article to the Constitution
      - section to PPC, ATA, and AMLA
    """

    normalized = str(
        value or "section"
    ).strip().lower()

    if normalized not in {
        "section",
        "article",
    }:
        raise ValueError(
            "Unsupported provision_type "
            f"{normalized!r}. Expected 'section' or 'article'."
        )

    return normalized


def normalize_provision_number(value: str) -> str:
    """Normalize values such as ``298 - C`` to ``298-C``."""

    normalized = re.sub(
        r"\s+",
        "",
        value.strip().upper(),
    )

    return normalized


def is_likely_provision_number(
    number: str,
    expected_provision_type: str,
) -> bool:
    normalized_number = normalize_provision_number(number)

    if expected_provision_type == "section":
        return bool(
            re.fullmatch(
                r"\d+(?:-[A-Z]{1,2}|[A-Z]{1,2})?",
                normalized_number,
            )
        )

    return bool(
        re.fullmatch(
            r"\d+(?:-[A-Z]{1,3}|[A-Z]{1,3})?",
            normalized_number,
        )
    )


def get_document_identity(
    document: Document,
) -> tuple[str, str, str, str, str, str]:
    """
    Return stable identity fields required for safe multi-document
    chunking.
    """

    metadata = document.metadata

    document_id = str(
        metadata.get("document_id") or ""
    ).strip()

    document_name = str(
        metadata.get("document_name") or ""
    ).strip()

    document_title = str(
        metadata.get("document_title")
        or document_name
        or document_id
    ).strip()

    document_short_name = str(
        metadata.get("document_short_name")
        or document_title
        or document_id
    ).strip()

    document_type = str(
        metadata.get("document_type")
        or "legal_document"
    ).strip()

    provision_type = normalize_provision_type(
        metadata.get("provision_type")
    )

    if not document_id:
        raise ValueError(
            "Every page document must contain document_id "
            "before multi-document chunking."
        )

    if not document_name:
        raise ValueError(
            f"Document {document_id} is missing document_name."
        )

    return (
        document_id,
        document_name,
        document_title,
        document_short_name,
        document_type,
        provision_type,
    )


def get_page_number(
    document: Document,
    fallback: int,
) -> int:
    """Return a human-readable, one-based PDF page number."""

    page_number = document.metadata.get(
        "page_number"
    )

    if isinstance(page_number, int):
        return page_number

    page = document.metadata.get("page")

    if isinstance(page, int):
        return page + 1

    return fallback


def validate_document_group(
    document_id: str,
    documents: list[Document],
) -> None:
    """Ensure one group contains pages from exactly one legal PDF."""

    if not documents:
        raise ValueError(
            f"Document group {document_id} is empty."
        )

    seen_names: set[str] = set()
    seen_titles: set[str] = set()
    seen_short_names: set[str] = set()
    seen_types: set[str] = set()
    seen_provision_types: set[str] = set()
    seen_page_numbers: set[int] = set()

    for index, document in enumerate(
        documents,
        start=1,
    ):
        (
            current_id,
            document_name,
            document_title,
            document_short_name,
            document_type,
            provision_type,
        ) = get_document_identity(
            document
        )

        if current_id != document_id:
            raise RuntimeError(
                "Cross-document contamination detected. "
                f"Expected document_id={document_id}, "
                f"found {current_id}."
            )

        seen_names.add(document_name)
        seen_titles.add(document_title)
        seen_short_names.add(document_short_name)
        seen_types.add(document_type)
        seen_provision_types.add(provision_type)

        page_number = get_page_number(
            document,
            fallback=index,
        )

        if page_number in seen_page_numbers:
            raise RuntimeError(
                "Duplicate page number detected inside "
                f"{document_name}: {page_number}."
            )

        seen_page_numbers.add(page_number)

    validation_sets = {
        "document names": seen_names,
        "document titles": seen_titles,
        "document short names": seen_short_names,
        "document types": seen_types,
        "provision types": seen_provision_types,
    }

    for label, values in validation_sets.items():
        if len(values) != 1:
            raise RuntimeError(
                f"A document group contains multiple {label}: "
                f"{sorted(values)}"
            )


# -------------------------------------------------------------------
# Text splitter construction
# -------------------------------------------------------------------

SPLIT_SEPARATORS = [
    r"\n(?=Explanation(?:\s+\d+)?\s*[:.\-—–]?)",
    r"\n(?=Illustrations?\s*[:.\-—–]?\s*$)",
    r"\n(?=\(\d+\)\s+)",
    r"\n(?=\([a-z]\)\s+)",
    r"\n(?=\([ivxlcdm]+\)\s+)",
    r"\n\n+",
    r"\n",
    r"(?<=[.!?])\s+",
    r"\s+",
    "",
]


def create_text_splitter() -> RecursiveCharacterTextSplitter:
    """
    Create the secondary splitter used only when one legal provision
    exceeds CHUNK_SIZE.
    """

    validate_chunk_settings()

    return RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len,
        add_start_index=True,
        is_separator_regex=True,
        separators=SPLIT_SEPARATORS,
    )


# -------------------------------------------------------------------
# Page combination and page-range tracking
# -------------------------------------------------------------------

def merge_metadata(
    documents: list[Document],
) -> dict[str, Any]:
    """Merge non-null metadata from source page documents."""

    merged: dict[str, Any] = {}

    for document in documents:
        for key, value in document.metadata.items():
            if key not in merged and value is not None:
                merged[key] = deepcopy(value)

    return merged


def build_combined_text(
    documents: list[Document],
) -> tuple[str, list[PageSpan], dict[int, Document]]:
    """
    Combine pages without inserting artificial page labels.

    Exact character ranges are retained so a provision can be mapped
    back to all PDF pages that it overlaps.
    """

    if not documents:
        return "", [], {}

    document_id = str(
        documents[0].metadata.get(
            "document_id",
            "",
        )
    ).strip()

    validate_document_group(
        document_id=document_id,
        documents=documents,
    )

    text_parts: list[str] = []
    page_spans: list[PageSpan] = []
    page_map: dict[int, Document] = {}
    current_offset = 0

    for index, document in enumerate(
        documents,
        start=1,
    ):
        page_number = get_page_number(
            document,
            fallback=index,
        )

        page_text = (
            document.page_content or ""
        ).strip()

        if text_parts:
            separator = "\n\n"
            text_parts.append(separator)
            current_offset += len(separator)

        start = current_offset
        text_parts.append(page_text)
        current_offset += len(page_text)
        end = current_offset

        page_spans.append(
            PageSpan(
                page_number=page_number,
                start=start,
                end=end,
                document=document,
            )
        )

        page_map[page_number] = document

    return (
        "".join(text_parts),
        page_spans,
        page_map,
    )


def pages_for_offsets(
    start: int,
    end: int,
    page_spans: list[PageSpan],
) -> list[int]:
    """Return exact source pages overlapping ``[start, end)``."""

    pages = [
        span.page_number
        for span in page_spans
        if span.end > start and span.start < end
    ]

    return sorted(set(pages))


def metadata_for_pages(
    page_numbers: list[int],
    page_map: dict[int, Document],
) -> dict[str, Any]:
    """Merge metadata from all pages supporting one provision."""

    documents = [
        page_map[page_number]
        for page_number in page_numbers
        if page_number in page_map
    ]

    return merge_metadata(documents)


# -------------------------------------------------------------------
# Provision detection
# -------------------------------------------------------------------

def is_valid_provision_candidate(
    match: re.Match[str],
    expected_provision_type: str,
) -> bool:
    """Reject years, amendment notes, and mismatched heading labels."""

    number = normalize_provision_number(
        match.group("number")
    )

    title = match.group("title").strip()
    explicit_label = (
        match.group("label") or ""
    ).strip().lower()

    if explicit_label and (
        explicit_label != expected_provision_type
    ):
        return False

    if not is_likely_provision_number(
        number,
        expected_provision_type,
    ):
        return False

    if not title:
        return False

    if re.fullmatch(r"\d{4}", number):
        year = int(number)

        if 1800 <= year <= 2100:
            return False

    if title.startswith(("[", "]")):
        return False

    if re.fullmatch(r"[\W_]+", title):
        return False

    if re.match(
        r"(?i)^"
        r"(?:of\s+\d{4}|schedule\b|article\b|section\b|"
        r"inserted\b|substituted\b|omitted\b|amended\b|"
        r"renumbered\b|repealed\b)",
        title,
    ):
        return False

    if re.match(
        r"(?i)^(?:subs\.?|footnote\b|note\b|page\b|header\b|footer\b)",
        title,
    ):
        return False

    if len(title) < 2:
        return False

    if len(title) > 600:
        return False

    return True


def find_provision_matches(
    combined_text: str,
    expected_provision_type: str,
) -> list[re.Match[str]]:
    """Return valid Section or Article heading matches."""

    return [
        match
        for match in PROVISION_PATTERN.finditer(
            combined_text
        )
        if is_valid_provision_candidate(
            match,
            expected_provision_type,
        )
    ]


def structural_boundaries(
    combined_text: str,
) -> list[tuple[int, int]]:
    """Find chapter, part, subject, preamble, and schedule boundaries."""

    boundaries: list[tuple[int, int]] = []

    patterns = (
        CHAPTER_PATTERN,
        PART_PATTERN,
        SCHEDULE_PATTERN,
        PREAMBLE_PATTERN,
        UPPERCASE_STRUCTURE_PATTERN,
        MIXED_STRUCTURE_PATTERN,
    )

    for pattern in patterns:
        for match in pattern.finditer(
            combined_text
        ):
            boundaries.append(
                (
                    match.start(),
                    match.end(),
                )
            )

    # A standalone Roman numeral counts as a structural boundary only
    # if the next non-empty line looks like a chapter subject.
    for match in STANDALONE_CHAPTER_PATTERN.finditer(
        combined_text
    ):
        following = combined_text[
            match.end():
        ]

        next_line_match = re.search(
            r"\S[^\n]*",
            following,
        )

        if not next_line_match:
            continue

        next_line = (
            next_line_match.group(0).strip()
        )

        if (
            UPPERCASE_STRUCTURE_PATTERN.fullmatch(
                next_line
            )
            or MIXED_STRUCTURE_PATTERN.fullmatch(
                next_line
            )
        ):
            boundaries.append(
                (
                    match.start(),
                    match.end(),
                )
            )

    return sorted(set(boundaries))


def next_structural_boundary(
    start: int,
    proposed_end: int,
    boundaries: list[tuple[int, int]],
) -> int:
    """Stop a provision before a later chapter/part/schedule heading."""

    candidates = [
        boundary_start
        for boundary_start, _ in boundaries
        if start < boundary_start < proposed_end
    ]

    return (
        min(candidates)
        if candidates
        else proposed_end
    )


# -------------------------------------------------------------------
# Heading and body analysis
# -------------------------------------------------------------------

def split_title_and_inline_body(
    heading_line: str,
) -> tuple[str, str]:
    """
    Split headings such as:

        Punishment for theft.—Whoever commits theft...
        Short title.—(1) This Act may be called...
    """

    normalized_line = heading_line.strip()
    match = INLINE_BODY_PATTERN.fullmatch(
        normalized_line
    )

    if match:
        title = (
            match.group("title")
            .strip()
            .rstrip(" :.—–")
        )
        body = match.group("body").strip()

        return title, body

    return normalized_line, ""


def extract_multiline_title(
    combined_text: str,
    match: re.Match[str],
    block_end: int,
) -> tuple[str, str, int]:
    """
    Return provision title, inline body, and body-start offset.

    A short following line is treated as part of a wrapped heading only
    until operative legal text or another structure begins.
    """

    raw_title = match.group(
        "title"
    ).strip()

    title, inline_body = (
        split_title_and_inline_body(
            raw_title
        )
    )

    body_start = match.end()

    if inline_body:
        return (
            title[:MAX_PROVISION_TITLE_CHARACTERS],
            inline_body,
            body_start,
        )

    if raw_title.endswith((".", ":")):
        return (
            raw_title
            .rstrip(" :.")
            [:MAX_PROVISION_TITLE_CHARACTERS],
            "",
            body_start,
        )

    cursor = match.end()
    title_parts = [raw_title]

    while cursor < block_end:
        line_match = re.match(
            r"(?:\n\s*)+([^\n]+)",
            combined_text[
                cursor:block_end
            ],
        )

        if not line_match:
            break

        candidate = (
            line_match.group(1).strip()
        )

        if (
            not candidate
            or BODY_START_PATTERN.match(candidate)
            or PROVISION_PATTERN.match(candidate)
            or CHAPTER_PATTERN.match(candidate)
            or PART_PATTERN.match(candidate)
            or SCHEDULE_PATTERN.match(candidate)
            or UPPERCASE_STRUCTURE_PATTERN.match(
                candidate
            )
            or MIXED_STRUCTURE_PATTERN.match(
                candidate
            )
            or SUBSECTION_PATTERN.match(candidate)
            or EXPLANATION_PATTERN.match(candidate)
            or ILLUSTRATION_PATTERN.match(candidate)
        ):
            break

        if len(candidate) > 160:
            break

        title_parts.append(candidate)
        cursor += line_match.end()

        if candidate.endswith((".", ":")):
            break

    title = " ".join(
        title_parts
    ).rstrip(" :.").strip()

    return (
        title[:MAX_PROVISION_TITLE_CHARACTERS],
        "",
        cursor,
    )


def analyze_provision_body(
    provision_type: str,
    provision_number: str,
    provision_title: str,
    text: str,
    source_metadata: dict[str, Any],
) -> tuple[bool, bool]:
    """Determine whether the block contains usable legal body text."""

    heading_variants = (
        f"{provision_number}. {provision_title}",
        (
            f"{provision_type.title()} "
            f"{provision_number}. "
            f"{provision_title}"
        ),
        (
            f"{provision_type.title()} "
            f"{provision_number}: "
            f"{provision_title}"
        ),
    )

    body = text.strip()

    for heading_prefix in heading_variants:
        if body.lower().startswith(
            heading_prefix.lower()
        ):
            body = body[
                len(heading_prefix):
            ].lstrip(" :.—–\n")
            break

    body_characters = len(
        re.sub(
            r"\s+",
            " ",
            body,
        ).strip()
    )

    inherited_heading_only = bool(
        source_metadata.get(
            "heading_only_page",
            False,
        )
    )

    inherited_body_present = bool(
        source_metadata.get(
            "section_body_present",
            source_metadata.get(
                "provision_body_present",
                True,
            ),
        )
    )

    heading_only = bool(
        body_characters
        < MIN_PROVISION_BODY_CHARACTERS
        or inherited_heading_only
        or not inherited_body_present
    )

    return (
        not heading_only,
        heading_only,
    )


# -------------------------------------------------------------------
# Provision block parsing
# -------------------------------------------------------------------

def create_unsectioned_block(
    provision_type: str,
    text: str,
    start: int,
    end: int,
    page_spans: list[PageSpan],
    page_map: dict[int, Document],
) -> ProvisionBlock:
    """Create a preamble/title/schedule block without a provision ID."""

    pages = pages_for_offsets(
        start,
        end,
        page_spans,
    )

    return ProvisionBlock(
        provision_type=provision_type,
        provision_number=None,
        provision_title=None,
        text=text.strip(),
        page_start=(
            pages[0]
            if pages
            else None
        ),
        page_end=(
            pages[-1]
            if pages
            else None
        ),
        source_pages=pages,
        source_metadata=metadata_for_pages(
            pages,
            page_map,
        ),
        heading_only=False,
        provision_body_present=True,
        is_unsectioned=True,
    )


def parse_provision_blocks(
    documents: list[Document],
) -> tuple[list[ProvisionBlock], dict[str, int]]:
    """
    Parse Section blocks for PPC/ATA/AMLA and Article blocks for the
    Constitution.
    """

    (
        combined_text,
        page_spans,
        page_map,
    ) = build_combined_text(
        documents
    )

    expected_provision_type = (
        normalize_provision_type(
            documents[0].metadata.get(
                "provision_type"
            )
        )
    )

    raw_candidates = list(
        PROVISION_PATTERN.finditer(
            combined_text
        )
    )

    matches = find_provision_matches(
        combined_text,
        expected_provision_type,
    )

    boundaries = structural_boundaries(
        combined_text
    )

    diagnostics = {
        "raw_candidates": len(raw_candidates),
        "valid_candidates": len(matches),
        "rejected_candidates": (
            len(raw_candidates)
            - len(matches)
        ),
        "structural_boundaries": len(
            boundaries
        ),
        "cross_page_provisions": 0,
        "cross_page_sections": 0,
        "missing_page_metadata": 0,
    }

    blocks: list[ProvisionBlock] = []

    if not matches:
        if (
            INCLUDE_UNSECTIONED_TEXT
            and combined_text.strip()
        ):
            blocks.append(
                create_unsectioned_block(
                    provision_type=(
                        expected_provision_type
                    ),
                    text=combined_text,
                    start=0,
                    end=len(combined_text),
                    page_spans=page_spans,
                    page_map=page_map,
                )
            )

        return blocks, diagnostics

    prefix_end = matches[0].start()
    prefix = combined_text[
        :prefix_end
    ].strip()

    if INCLUDE_UNSECTIONED_TEXT and prefix:
        blocks.append(
            create_unsectioned_block(
                provision_type=(
                    expected_provision_type
                ),
                text=prefix,
                start=0,
                end=prefix_end,
                page_spans=page_spans,
                page_map=page_map,
            )
        )

    for index, match in enumerate(matches):
        block_start = match.start()

        next_provision_start = (
            matches[index + 1].start()
            if index + 1 < len(matches)
            else len(combined_text)
        )

        block_end = next_structural_boundary(
            block_start,
            next_provision_start,
            boundaries,
        )

        provision_number = (
            normalize_provision_number(
                match.group("number")
            )
        )

        (
            provision_title,
            inline_body,
            body_start,
        ) = extract_multiline_title(
            combined_text,
            match,
            block_end,
        )

        raw_block = combined_text[
            block_start:block_end
        ].strip()

        if inline_body:
            canonical_heading = (
                f"{provision_number}. "
                f"{provision_title}"
            )

            raw_heading = match.group(0)

            raw_block = re.sub(
                rf"^\s*{re.escape(raw_heading)}",
                (
                    f"{canonical_heading}\n\n"
                    f"{inline_body}"
                ),
                raw_block,
                count=1,
            )

        elif body_start > match.end():
            canonical_heading = (
                f"{provision_number}. "
                f"{provision_title}"
            )

            body = combined_text[
                body_start:block_end
            ].strip()

            raw_block = (
                f"{canonical_heading}\n\n{body}"
                if body
                else canonical_heading
            )

        pages = pages_for_offsets(
            block_start,
            block_end,
            page_spans,
        )

        page_start = (
            pages[0]
            if pages
            else None
        )

        page_end = (
            pages[-1]
            if pages
            else None
        )

        if len(pages) > 1:
            diagnostics[
                "cross_page_provisions"
            ] += 1
            diagnostics[
                "cross_page_sections"
            ] += 1

        if not pages:
            diagnostics[
                "missing_page_metadata"
            ] += 1

        source_metadata = metadata_for_pages(
            pages,
            page_map,
        )

        (
            body_present,
            heading_only,
        ) = analyze_provision_body(
            provision_type=(
                expected_provision_type
            ),
            provision_number=provision_number,
            provision_title=provision_title,
            text=raw_block,
            source_metadata=source_metadata,
        )

        blocks.append(
            ProvisionBlock(
                provision_type=(
                    expected_provision_type
                ),
                provision_number=(
                    provision_number
                ),
                provision_title=(
                    provision_title
                ),
                text=raw_block,
                page_start=page_start,
                page_end=page_end,
                source_pages=pages,
                source_metadata=source_metadata,
                heading_only=heading_only,
                provision_body_present=(
                    body_present
                ),
                is_unsectioned=False,
            )
        )

    return blocks, diagnostics


# Backward-compatible alias for code that still imports the old name.
parse_section_blocks = parse_provision_blocks


# -------------------------------------------------------------------
# Chunk metadata and content construction
# -------------------------------------------------------------------

def build_context_heading(
    metadata: dict[str, Any],
    provision_type: str,
    provision_number: str | None,
    provision_title: str | None,
) -> str:
    """Build a compact retrieval heading for each generated chunk."""

    document_label = str(
        metadata.get("document_short_name")
        or metadata.get("document_title")
        or metadata.get("document_name")
        or metadata.get("document_id")
        or "Legal document"
    ).strip()

    if not provision_number:
        return document_label

    heading = (
        f"{document_label} | "
        f"{provision_type.title()} "
        f"{provision_number}"
    )

    if provision_title:
        heading += f": {provision_title}"

    return heading


def create_provision_metadata(
    block: ProvisionBlock,
) -> dict[str, Any]:
    """Create common section/article metadata for one provision block."""

    metadata = deepcopy(
        block.source_metadata
    )

    document_id = str(
        metadata.get("document_id") or ""
    ).strip()

    if not document_id:
        raise RuntimeError(
            "Provision block is missing document_id."
        )

    provision_number = (
        block.provision_number
    )

    provision_title = (
        block.provision_title
    )

    is_section = (
        block.provision_type == "section"
    )

    is_article = (
        block.provision_type == "article"
    )

    provision_identity = (
        f"{document_id}::"
        f"{block.provision_type}::"
        f"{provision_number or 'unsectioned'}"
    )

    metadata.update(
        {
            "provision_type": block.provision_type,
            "provision_number": provision_number,
            "provision_title": provision_title,
            "provision_identity": provision_identity,
            "provision_numbers": (
                [provision_number]
                if provision_number
                else []
            ),
            "provision_titles": (
                [provision_title]
                if provision_title
                else []
            ),
            "primary_provision": provision_number,
            "primary_provision_title": provision_title,

            # Section-specific compatibility fields.
            "section_number": (
                provision_number
                if is_section
                else None
            ),
            "section_title": (
                provision_title
                if is_section
                else None
            ),
            "section_identity": (
                provision_identity
                if is_section
                else None
            ),
            "section_numbers": (
                [provision_number]
                if is_section
                and provision_number
                else []
            ),
            "section_titles": (
                [provision_title]
                if is_section
                and provision_title
                else []
            ),
            "primary_section": (
                provision_number
                if is_section
                else None
            ),
            "primary_section_title": (
                provision_title
                if is_section
                else None
            ),

            # Constitution-specific fields.
            "article_number": (
                provision_number
                if is_article
                else None
            ),
            "article_title": (
                provision_title
                if is_article
                else None
            ),
            "article_identity": (
                provision_identity
                if is_article
                else None
            ),
            "article_numbers": (
                [provision_number]
                if is_article
                and provision_number
                else []
            ),
            "article_titles": (
                [provision_title]
                if is_article
                and provision_title
                else []
            ),
            "primary_article": (
                provision_number
                if is_article
                else None
            ),
            "primary_article_title": (
                provision_title
                if is_article
                else None
            ),

            "page_start": block.page_start,
            "page_end": block.page_end,
            "page_number": block.page_start,
            "source_pages": list(
                block.source_pages
            ),
            "spans_multiple_pages": (
                len(block.source_pages) > 1
            ),
            "heading_only_chunk": (
                block.heading_only
            ),
            "provision_body_present": (
                block.provision_body_present
            ),

            # Retained because existing retriever filters may still
            # reference this historical field name.
            "section_body_present": (
                block.provision_body_present
            ),
            "is_unsectioned_chunk": (
                block.is_unsectioned
            ),
        }
    )

    if provision_number:
        metadata["legal_topic"] = (
            provision_title.lower()
            if provision_title
            else metadata.get(
                "legal_topic"
            )
        )

    return metadata


# Backward-compatible alias for existing imports.
create_section_metadata = create_provision_metadata


def remove_existing_heading(
    block_text: str,
    provision_number: str,
) -> str:
    """Remove the original numeric heading before adding a canonical one."""

    pattern = re.compile(
        r"(?is)^\s*"
        r"(?:(?:section|article)\s+)?"
        + re.escape(provision_number)
        + r"\s*(?:\.\s*|[—–:-]\s*)"
        + r"[^\n]*"
        + r"(?:\n+|$)"
    )

    return pattern.sub(
        "",
        block_text,
        count=1,
    ).strip()


def split_large_provision(
    block: ProvisionBlock,
    text_splitter: RecursiveCharacterTextSplitter,
) -> list[Document]:
    """
    Split a large Section/Article while repeating its document and
    provision heading in every part.
    """

    metadata = create_provision_metadata(
        block
    )

    context_heading = build_context_heading(
        metadata=metadata,
        provision_type=block.provision_type,
        provision_number=(
            block.provision_number
        ),
        provision_title=(
            block.provision_title
        ),
    )

    body = block.text.strip()

    if block.provision_number:
        body_without_heading = (
            remove_existing_heading(
                block_text=body,
                provision_number=(
                    block.provision_number
                ),
            )
        )

        if body_without_heading:
            body = body_without_heading

    # Account for the repeated heading when deciding whether to split.
    heading_cost = (
        len(context_heading) + 2
        if PREPEND_DOCUMENT_CONTEXT
        else 0
    )

    effective_body_limit = max(
        200,
        CHUNK_SIZE - heading_cost,
    )

    if len(body) <= effective_body_limit:
        content = body

        if PREPEND_DOCUMENT_CONTEXT:
            content = (
                f"{context_heading}\n\n"
                f"{body}"
            ).strip()

        return [
            Document(
                page_content=content,
                metadata=metadata,
            )
        ]

    # A dedicated temporary splitter keeps the final body parts small
    # enough after the canonical heading is added.
    body_splitter = RecursiveCharacterTextSplitter(
        chunk_size=effective_body_limit,
        chunk_overlap=min(
            CHUNK_OVERLAP,
            max(
                0,
                effective_body_limit - 1,
            ),
        ),
        length_function=len,
        add_start_index=True,
        is_separator_regex=True,
        separators=SPLIT_SEPARATORS,
    )

    body_chunks = (
        body_splitter.split_documents(
            [
                Document(
                    page_content=body,
                    metadata=metadata,
                )
            ]
        )
    )

    results: list[Document] = []
    total_parts = len(body_chunks)

    for part_number, chunk in enumerate(
        body_chunks,
        start=1,
    ):
        content = (
            chunk.page_content or ""
        ).strip()

        if PREPEND_DOCUMENT_CONTEXT:
            content = (
                f"{context_heading}\n\n"
                f"{content}"
            ).strip()

        chunk.metadata.update(
            {
                "provision_part_number": (
                    part_number
                ),
                "provision_part_count": (
                    total_parts
                ),
                "provision_was_split": True,

                # Backward-compatible section names.
                "section_part_number": (
                    part_number
                ),
                "section_part_count": (
                    total_parts
                ),
                "section_was_split": True,
            }
        )

        chunk.page_content = content
        results.append(chunk)

    return results


# Backward-compatible alias for existing imports.
split_large_section = split_large_provision


# -------------------------------------------------------------------
# Final chunk validation
# -------------------------------------------------------------------

def validate_final_chunks(
    chunks: list[Document],
) -> None:
    """Fail early if a generated chunk lacks required metadata."""

    errors: list[str] = []

    for index, chunk in enumerate(
        chunks,
        start=1,
    ):
        metadata = chunk.metadata

        if not (
            chunk.page_content or ""
        ).strip():
            errors.append(
                f"Chunk {index}: empty content"
            )

        required_document_fields = (
            "document_id",
            "document_name",
            "document_title",
            "document_short_name",
            "document_type",
            "provision_type",
        )

        for field_name in required_document_fields:
            if not metadata.get(field_name):
                errors.append(
                    f"Chunk {index}: missing "
                    f"{field_name}"
                )

        if metadata.get(
            "is_unsectioned_chunk"
        ):
            continue

        provision_type = metadata.get(
            "provision_type"
        )

        provision_number = metadata.get(
            "provision_number"
        )

        if provision_type not in {
            "section",
            "article",
        }:
            errors.append(
                f"Chunk {index}: invalid "
                "provision_type"
            )

        if not provision_number:
            errors.append(
                f"Chunk {index}: missing "
                "provision_number"
            )

        if not metadata.get(
            "provision_identity"
        ):
            errors.append(
                f"Chunk {index}: missing "
                "provision_identity"
            )

        if (
            provision_type == "section"
            and not metadata.get(
                "section_number"
            )
        ):
            errors.append(
                f"Chunk {index}: section chunk "
                "is missing section_number"
            )

        if (
            provision_type == "article"
            and not metadata.get(
                "article_number"
            )
        ):
            errors.append(
                f"Chunk {index}: article chunk "
                "is missing article_number"
            )

        if metadata.get(
            "page_start"
        ) is None:
            errors.append(
                f"Chunk {index}: missing page_start"
            )

        if metadata.get(
            "page_end"
        ) is None:
            errors.append(
                f"Chunk {index}: missing page_end"
            )

        if not metadata.get(
            "source_pages"
        ):
            errors.append(
                f"Chunk {index}: missing source_pages"
            )

        page_start = metadata.get(
            "page_start"
        )

        page_end = metadata.get(
            "page_end"
        )

        if (
            isinstance(page_start, int)
            and isinstance(page_end, int)
            and page_start > page_end
        ):
            errors.append(
                f"Chunk {index}: invalid page range"
            )

    if errors:
        preview = "\n".join(
            f"- {error}"
            for error in errors[:25]
        )

        raise RuntimeError(
            "Final chunk validation failed:\n"
            f"{preview}"
        )


# -------------------------------------------------------------------
# Public chunking API
# -------------------------------------------------------------------

def create_chunks(
    documents: list[Document],
) -> list[Document]:
    """
    Create section-aware and article-aware chunks from all loaded PDFs.

    Public function retained for compatibility with the existing
    ingestion script.
    """

    if not documents:
        raise ValueError(
            "No documents were provided for chunking."
        )

    validate_chunk_settings()

    document_groups: dict[
        str,
        list[Document],
    ] = {}

    for document in documents:
        (
            document_id,
            _document_name,
            _document_title,
            _document_short_name,
            _document_type,
            _provision_type,
        ) = get_document_identity(
            document
        )

        document_groups.setdefault(
            document_id,
            [],
        ).append(document)

    for (
        document_id,
        grouped_documents,
    ) in document_groups.items():
        grouped_documents.sort(
            key=lambda document: (
                get_page_number(
                    document,
                    fallback=0,
                )
            )
        )

        validate_document_group(
            document_id=document_id,
            documents=grouped_documents,
        )

    text_splitter = create_text_splitter()
    final_chunks: list[Document] = []

    provision_blocks_created = 0
    oversized_provisions_split = 0
    heading_only_blocks_skipped = 0
    unsectioned_blocks_created = 0

    raw_candidates = 0
    valid_candidates = 0
    rejected_candidates = 0
    structural_boundaries_count = 0
    cross_page_provisions = 0
    missing_page_metadata = 0

    per_document_summaries: list[
        dict[str, object]
    ] = []

    for (
        document_id,
        grouped_documents,
    ) in document_groups.items():
        document_metadata = (
            grouped_documents[0].metadata
        )

        provision_type = (
            normalize_provision_type(
                document_metadata.get(
                    "provision_type"
                )
            )
        )

        blocks, diagnostics = (
            parse_provision_blocks(
                grouped_documents
            )
        )

        chunks_before_document = len(
            final_chunks
        )

        document_provision_blocks = 0
        document_unsectioned_blocks = 0
        document_oversized_provisions = 0
        document_heading_only_skipped = 0

        raw_candidates += diagnostics[
            "raw_candidates"
        ]

        valid_candidates += diagnostics[
            "valid_candidates"
        ]

        rejected_candidates += diagnostics[
            "rejected_candidates"
        ]

        structural_boundaries_count += (
            diagnostics[
                "structural_boundaries"
            ]
        )

        cross_page_provisions += diagnostics[
            "cross_page_provisions"
        ]

        missing_page_metadata += diagnostics[
            "missing_page_metadata"
        ]

        for block in blocks:
            if block.is_unsectioned:
                unsectioned_blocks_created += 1
                document_unsectioned_blocks += 1
            else:
                provision_blocks_created += 1
                document_provision_blocks += 1

            if (
                block.heading_only
                and not INCLUDE_HEADING_ONLY_CHUNKS
            ):
                heading_only_blocks_skipped += 1
                document_heading_only_skipped += 1
                continue

            block_chunks = split_large_provision(
                block,
                text_splitter,
            )

            if len(block_chunks) > 1:
                oversized_provisions_split += 1
                document_oversized_provisions += 1

            for chunk in block_chunks:
                chunk_document_id = str(
                    chunk.metadata.get(
                        "document_id",
                        "",
                    )
                )

                if chunk_document_id != document_id:
                    raise RuntimeError(
                        "Chunk inherited incorrect "
                        "document_id. "
                        f"Expected {document_id}, "
                        f"found {chunk_document_id}."
                    )

                chunk_provision_type = (
                    normalize_provision_type(
                        chunk.metadata.get(
                            "provision_type"
                        )
                    )
                )

                if (
                    chunk_provision_type
                    != provision_type
                ):
                    raise RuntimeError(
                        "Chunk inherited incorrect "
                        "provision_type. "
                        f"Expected {provision_type}, "
                        f"found "
                        f"{chunk_provision_type}."
                    )

            final_chunks.extend(
                block_chunks
            )

        per_document_summaries.append(
            {
                "document_id": document_id,
                "document_name": (
                    document_metadata.get(
                        "document_name",
                        "Unknown",
                    )
                ),
                "document_title": (
                    document_metadata.get(
                        "document_title",
                        "Unknown",
                    )
                ),
                "document_short_name": (
                    document_metadata.get(
                        "document_short_name",
                        "Unknown",
                    )
                ),
                "document_type": (
                    document_metadata.get(
                        "document_type",
                        "legal_document",
                    )
                ),
                "provision_type": (
                    provision_type
                ),
                "pages": len(
                    grouped_documents
                ),
                "provision_blocks": (
                    document_provision_blocks
                ),
                "unsectioned_blocks": (
                    document_unsectioned_blocks
                ),
                "oversized_provisions": (
                    document_oversized_provisions
                ),
                "heading_only_skipped": (
                    document_heading_only_skipped
                ),
                "chunks_created": (
                    len(final_chunks)
                    - chunks_before_document
                ),
                "raw_candidates": diagnostics[
                    "raw_candidates"
                ],
                "valid_candidates": diagnostics[
                    "valid_candidates"
                ],
                "cross_page_provisions": (
                    diagnostics[
                        "cross_page_provisions"
                    ]
                ),
            }
        )

    cleaned_chunks: list[Document] = []

    for chunk in final_chunks:
        chunk.page_content = (
            chunk.page_content or ""
        ).strip()

        if not chunk.page_content:
            continue

        chunk.metadata.setdefault(
            "provision_was_split",
            False,
        )

        chunk.metadata.setdefault(
            "provision_part_number",
            1,
        )

        chunk.metadata.setdefault(
            "provision_part_count",
            1,
        )

        # Backward-compatible metadata names.
        chunk.metadata.setdefault(
            "section_was_split",
            chunk.metadata[
                "provision_was_split"
            ],
        )

        chunk.metadata.setdefault(
            "section_part_number",
            chunk.metadata[
                "provision_part_number"
            ],
        )

        chunk.metadata.setdefault(
            "section_part_count",
            chunk.metadata[
                "provision_part_count"
            ],
        )

        cleaned_chunks.append(chunk)

    per_document_chunk_counters: Counter[
        str
    ] = Counter()

    for chunk_number, chunk in enumerate(
        cleaned_chunks,
        start=1,
    ):
        document_id = str(
            chunk.metadata.get(
                "document_id",
                "",
            )
        )

        per_document_chunk_counters[
            document_id
        ] += 1

        document_chunk_number = (
            per_document_chunk_counters[
                document_id
            ]
        )

        provision_type = str(
            chunk.metadata.get(
                "provision_type",
                "section",
            )
        )

        provision_number = str(
            chunk.metadata.get(
                "provision_number",
                "unsectioned",
            )
            or "unsectioned"
        )

        provision_part_number = int(
            chunk.metadata.get(
                "provision_part_number",
                1,
            )
            or 1
        )

        chunk.metadata.update(
            {
                "chunk_number": (
                    chunk_number
                ),
                "document_chunk_number": (
                    document_chunk_number
                ),
                "chunk_id": (
                    f"{document_id}::"
                    f"{provision_type}::"
                    f"{provision_number}::"
                    f"part-"
                    f"{provision_part_number}::"
                    f"chunk-"
                    f"{document_chunk_number}"
                ),
                "chunk_length": len(
                    chunk.page_content
                ),
                "chunk_size_setting": (
                    CHUNK_SIZE
                ),
                "chunk_overlap_setting": (
                    CHUNK_OVERLAP
                ),
                "embedding_model_hint": (
                    "sentence-transformers/"
                    "all-MiniLM-L6-v2"
                ),
            }
        )

    validate_final_chunks(
        cleaned_chunks
    )

    print("=" * 70)
    print("CHUNKING SUMMARY")
    print("=" * 70)
    print(
        "Page documents received: "
        f"{len(documents)}"
    )
    print(
        "Document groups processed: "
        f"{len(document_groups)}"
    )
    print(
        "Raw provision candidates: "
        f"{raw_candidates}"
    )
    print(
        "Valid provision candidates: "
        f"{valid_candidates}"
    )
    print(
        "Invalid candidates rejected: "
        f"{rejected_candidates}"
    )
    print(
        "Structural boundaries detected: "
        f"{structural_boundaries_count}"
    )
    print(
        "Provision blocks parsed: "
        f"{provision_blocks_created}"
    )
    print(
        "Cross-page provisions detected: "
        f"{cross_page_provisions}"
    )
    print(
        "Provisions missing page metadata: "
        f"{missing_page_metadata}"
    )
    print(
        "Unsectioned blocks created: "
        f"{unsectioned_blocks_created}"
    )
    print(
        "Oversized provisions split: "
        f"{oversized_provisions_split}"
    )
    print(
        "Heading-only blocks skipped: "
        f"{heading_only_blocks_skipped}"
    )
    print(f"Chunk size: {CHUNK_SIZE}")
    print(
        f"Chunk overlap: {CHUNK_OVERLAP}"
    )
    print(
        "Total chunks created: "
        f"{len(cleaned_chunks)}"
    )

    if per_document_summaries:
        print("\n" + "=" * 70)
        print(
            "PER-DOCUMENT CHUNKING SUMMARY"
        )
        print("=" * 70)

        for summary in per_document_summaries:
            print(
                "\nDocument: "
                f"{summary['document_title']}"
            )
            print(
                "Short name: "
                f"{summary['document_short_name']}"
            )
            print(
                "Document ID: "
                f"{summary['document_id']}"
            )
            print(
                "Document type: "
                f"{summary['document_type']}"
            )
            print(
                "Provision type: "
                f"{summary['provision_type']}"
            )
            print(
                "Pages processed: "
                f"{summary['pages']}"
            )
            print(
                "Raw candidates: "
                f"{summary['raw_candidates']}"
            )
            print(
                "Valid candidates: "
                f"{summary['valid_candidates']}"
            )
            print(
                "Provision blocks: "
                f"{summary['provision_blocks']}"
            )
            print(
                "Cross-page provisions: "
                f"{summary['cross_page_provisions']}"
            )
            print(
                "Unsectioned blocks: "
                f"{summary['unsectioned_blocks']}"
            )
            print(
                "Oversized provisions split: "
                f"{summary['oversized_provisions']}"
            )
            print(
                "Heading-only blocks skipped: "
                f"{summary['heading_only_skipped']}"
            )
            print(
                "Chunks created: "
                f"{summary['chunks_created']}"
            )

    return cleaned_chunks


# -------------------------------------------------------------------
# Debug display helpers
# -------------------------------------------------------------------

def display_chunks(
    chunks: list[Document],
    count: int = 5,
) -> None:
    """Print representative chunks and their metadata."""

    if not chunks:
        print(
            "No chunks are available."
        )
        return

    print("\n" + "=" * 70)
    print("SAMPLE CHUNKS")
    print("=" * 70)

    for index, chunk in enumerate(
        chunks[
            :min(
                count,
                len(chunks),
            )
        ],
        start=1,
    ):
        print("\n" + "-" * 70)
        print(f"CHUNK {index}")
        print("-" * 70)
        print("\nMetadata:")
        print(chunk.metadata)
        print("\nText:")
        print(chunk.page_content)
        print(
            "\nActual length: "
            f"{len(chunk.page_content)} "
            "characters"
        )


def display_provision_chunks(
    chunks: list[Document],
    provision_number: str,
    document_id: str | None = None,
    provision_type: str | None = None,
) -> None:
    """Display chunks for a Section or Article."""

    normalized_number = (
        normalize_provision_number(
            provision_number
        )
    )

    normalized_type = (
        normalize_provision_type(
            provision_type
        )
        if provision_type
        else None
    )

    matching_chunks = [
        chunk
        for chunk in chunks
        if (
            str(
                chunk.metadata.get(
                    "provision_number",
                    "",
                )
            ).upper()
            == normalized_number
            and (
                document_id is None
                or str(
                    chunk.metadata.get(
                        "document_id",
                        "",
                    )
                )
                == document_id
            )
            and (
                normalized_type is None
                or str(
                    chunk.metadata.get(
                        "provision_type",
                        "",
                    )
                )
                == normalized_type
            )
        )
    ]

    print("\n" + "=" * 70)
    print(
        "CHUNKS FOR "
        f"{normalized_type.title() + ' ' if normalized_type else ''}"
        f"{normalized_number}"
    )
    print("=" * 70)

    if not matching_chunks:
        print(
            "No matching provision chunk was found."
        )
        return

    for chunk in matching_chunks:
        print("\n" + "-" * 70)
        print(
            "Document: "
            f"{chunk.metadata.get('document_title', 'Unknown')} | "
            "Document ID: "
            f"{chunk.metadata.get('document_id', 'Unknown')} | "
            "Provision: "
            f"{chunk.metadata.get('provision_type', 'Unknown').title()} "
            f"{chunk.metadata.get('provision_number', 'Unknown')} | "
            "Pages: "
            f"{chunk.metadata.get('page_start', 'Unknown')}"
            "–"
            f"{chunk.metadata.get('page_end', 'Unknown')} | "
            "Source pages: "
            f"{chunk.metadata.get('source_pages', [])} | "
            "Chunk: "
            f"{chunk.metadata.get('chunk_number', 'Unknown')} | "
            "Part: "
            f"{chunk.metadata.get('provision_part_number', 1)}"
            "/"
            f"{chunk.metadata.get('provision_part_count', 1)}"
        )
        print("-" * 70)
        print(chunk.page_content)


def display_section_chunks(
    chunks: list[Document],
    section_number: str,
    document_id: str | None = None,
) -> None:
    """Backward-compatible Section display helper."""

    display_provision_chunks(
        chunks=chunks,
        provision_number=section_number,
        document_id=document_id,
        provision_type="section",
    )


def display_article_chunks(
    chunks: list[Document],
    article_number: str,
    document_id: str | None = None,
) -> None:
    """Display Constitution Article chunks."""

    display_provision_chunks(
        chunks=chunks,
        provision_number=article_number,
        document_id=document_id,
        provision_type="article",
    )


# -------------------------------------------------------------------
# Command-line test
# -------------------------------------------------------------------

def main() -> None:
    """Test loading and provision-aware chunking from the terminal."""

    from rag.document_loader import (
        load_all_documents,
    )

    try:
        print(
            "Step 1: Loading and cleaning PDFs..."
        )

        documents = load_all_documents()

        print(
            "\nStep 2: Creating "
            "Section/Article-aware chunks..."
        )

        chunks = create_chunks(
            documents
        )

        display_chunks(
            chunks,
            count=5,
        )

        # Representative tests for the selected dataset.
        display_section_chunks(
            chunks,
            section_number="379",
            document_id="ppc_1860",
        )

        display_section_chunks(
            chunks,
            section_number="7",
            document_id="ata_1997",
        )

        display_section_chunks(
            chunks,
            section_number="3",
            document_id="amla_2010",
        )

        display_article_chunks(
            chunks,
            article_number="10A",
            document_id="constitution_1973",
        )

    except Exception as error:
        print(
            f"\nChunking error: {error}"
        )
        raise


if __name__ == "__main__":
    main()
