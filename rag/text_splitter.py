import os
import re
from collections import Counter
from copy import deepcopy
from dataclasses import dataclass, field

from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


load_dotenv()


CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1600"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "250"))
MIN_SECTION_BODY_CHARACTERS = int(
    os.getenv("MIN_SECTION_BODY_CHARACTERS", "80")
)
INCLUDE_HEADING_ONLY_CHUNKS = (
    os.getenv("INCLUDE_HEADING_ONLY_CHUNKS", "False").lower()
    == "true"
)
INCLUDE_UNSECTIONED_TEXT = (
    os.getenv("INCLUDE_UNSECTIONED_TEXT", "True").lower()
    == "true"
)


SECTION_PATTERN = re.compile(
    r"(?m)^(?P<number>\d+(?:-[A-Za-z]+)?[A-Za-z]?)\.\s+"
    r"(?P<title>[^\n]+)"
)

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

EXPLANATION_PATTERN = re.compile(
    r"(?mi)^\s*Explanation(?:\s+\d+)?\s*[:.-]?"
)

ILLUSTRATION_PATTERN = re.compile(
    r"(?mi)^\s*Illustrations?\s*[:.-]?\s*$"
)

SUBSECTION_PATTERN = re.compile(
    r"(?mi)^\s*\((?:\d+|[a-z]|[ivxlcdm]+)\)\s+"
)

BODY_START_PATTERN = re.compile(
    r"^(?:"
    r"Whoever|Any person|Every person|Nothing|Where|When|If\b|"
    r"A person|Provided|In this section|For the purposes of|"
    r"This Act|The provisions|In this Chapter|In case of"
    r")",
    re.IGNORECASE,
)


@dataclass
class PageSpan:
    page_number: int
    start: int
    end: int
    document: Document


@dataclass
class SectionBlock:
    section_number: str | None
    section_title: str | None
    text: str
    page_start: int | None
    page_end: int | None
    source_pages: list[int] = field(default_factory=list)
    source_metadata: dict = field(default_factory=dict)
    heading_only: bool = False
    section_body_present: bool = True
    is_unsectioned: bool = False


def validate_chunk_settings() -> None:
    if CHUNK_SIZE <= 0:
        raise ValueError("CHUNK_SIZE must be greater than zero.")

    if CHUNK_OVERLAP < 0:
        raise ValueError("CHUNK_OVERLAP cannot be negative.")

    if CHUNK_OVERLAP >= CHUNK_SIZE:
        raise ValueError(
            "CHUNK_OVERLAP must be smaller than CHUNK_SIZE."
        )

    if MIN_SECTION_BODY_CHARACTERS < 0:
        raise ValueError(
            "MIN_SECTION_BODY_CHARACTERS cannot be negative."
        )


def get_document_identity(
    document: Document,
) -> tuple[str, str, str, str]:
    """
    Return stable multi-document identity fields.

    document_id is mandatory for safe multi-document chunking.
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
    document_type = str(
        metadata.get("document_type")
        or "legal_document"
    ).strip()

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
        document_type,
    )


def validate_document_group(
    document_id: str,
    documents: list[Document],
) -> None:
    """
    Ensure one group contains pages from exactly one PDF.
    """

    if not documents:
        raise ValueError(
            f"Document group {document_id} is empty."
        )

    seen_names: set[str] = set()
    seen_titles: set[str] = set()
    seen_types: set[str] = set()
    seen_page_numbers: set[int] = set()

    for index, document in enumerate(
        documents,
        start=1,
    ):
        (
            current_id,
            document_name,
            document_title,
            document_type,
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
        seen_types.add(document_type)

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

    if len(seen_names) != 1:
        raise RuntimeError(
            "A document group contains multiple document names: "
            f"{sorted(seen_names)}"
        )

    if len(seen_titles) != 1:
        raise RuntimeError(
            "A document group contains multiple document titles: "
            f"{sorted(seen_titles)}"
        )

    if len(seen_types) != 1:
        raise RuntimeError(
            "A document group contains multiple document types: "
            f"{sorted(seen_types)}"
        )


def create_text_splitter() -> RecursiveCharacterTextSplitter:
    validate_chunk_settings()

    return RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len,
        add_start_index=True,
        is_separator_regex=True,
        separators=[
            r"\n(?=Explanation(?:\s+\d+)?\s*[:.-]?)",
            r"\n(?=Illustrations?\s*[:.-]?\s*$)",
            r"\n(?=\(\d+\)\s+)",
            r"\n(?=\([a-z]\)\s+)",
            r"\n(?=\([ivxlcdm]+\)\s+)",
            r"\n\n+",
            r"\n",
            r"(?<=[.!?])\s+",
            r"\s+",
            "",
        ],
    )


def get_page_number(document: Document, fallback: int) -> int:
    page_number = document.metadata.get("page_number")

    if isinstance(page_number, int):
        return page_number

    page = document.metadata.get("page")

    if isinstance(page, int):
        return page + 1

    return fallback


def merge_metadata(documents: list[Document]) -> dict:
    merged: dict = {}

    for document in documents:
        for key, value in document.metadata.items():
            if key not in merged and value is not None:
                merged[key] = deepcopy(value)

    return merged


def build_combined_text(
    documents: list[Document],
) -> tuple[str, list[PageSpan], dict[int, Document]]:
    """
    Combine pages without embedding page markers in the legal text.

    Each page's exact character range is recorded so sections can be
    mapped to pages even when they start after the page boundary.
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

    for index, document in enumerate(documents, start=1):
        page_number = get_page_number(document, fallback=index)
        page_text = document.page_content.strip()

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

    return "".join(text_parts), page_spans, page_map


def pages_for_offsets(
    start: int,
    end: int,
    page_spans: list[PageSpan],
) -> list[int]:
    """Return exact source pages overlapping [start, end)."""

    pages = [
        span.page_number
        for span in page_spans
        if span.end > start and span.start < end
    ]

    return sorted(set(pages))


def metadata_for_pages(
    page_numbers: list[int],
    page_map: dict[int, Document],
) -> dict:
    documents = [
        page_map[page_number]
        for page_number in page_numbers
        if page_number in page_map
    ]

    return merge_metadata(documents)


def is_valid_section_candidate(
    match: re.Match,
) -> bool:
    number = match.group("number").strip()
    title = match.group("title").strip()

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
        r"(?i)^(?:of\s+\d{4}|schedule\b|article\b|inserted\b|"
        r"substituted\b|omitted\b|amended\b)",
        title,
    ):
        return False

    if len(title) < 2:
        return False

    return True


def find_section_matches(
    combined_text: str,
) -> list[re.Match]:
    return [
        match
        for match in SECTION_PATTERN.finditer(combined_text)
        if is_valid_section_candidate(match)
    ]


def structural_boundaries(
    combined_text: str,
) -> list[tuple[int, int]]:
    boundaries: list[tuple[int, int]] = []

    patterns = (
        CHAPTER_PATTERN,
        PART_PATTERN,
        UPPERCASE_STRUCTURE_PATTERN,
        MIXED_STRUCTURE_PATTERN,
    )

    for pattern in patterns:
        for match in pattern.finditer(combined_text):
            boundaries.append((match.start(), match.end()))

    # A standalone Roman numeral counts only when the next non-empty line
    # is a chapter/subject heading.
    for match in STANDALONE_CHAPTER_PATTERN.finditer(combined_text):
        following = combined_text[match.end():]
        next_line_match = re.search(r"\S[^\n]*", following)

        if not next_line_match:
            continue

        next_line = next_line_match.group(0).strip()

        if (
            UPPERCASE_STRUCTURE_PATTERN.fullmatch(next_line)
            or MIXED_STRUCTURE_PATTERN.fullmatch(next_line)
        ):
            boundaries.append((match.start(), match.end()))

    return sorted(set(boundaries))


def next_structural_boundary(
    start: int,
    proposed_end: int,
    boundaries: list[tuple[int, int]],
) -> int:
    candidates = [
        boundary_start
        for boundary_start, _ in boundaries
        if start < boundary_start < proposed_end
    ]

    return min(candidates) if candidates else proposed_end


def split_title_and_inline_body(
    heading_line: str,
) -> tuple[str, str]:
    starters = (
        "Whoever",
        "Any person",
        "Every person",
        "Nothing",
        "Where",
        "When",
        "If",
        "A person",
        "Provided",
        "In this section",
        "For the purposes of",
        "This Act",
        "The provisions",
        "In this Chapter",
        "In case of",
    )

    for starter in starters:
        marker = f": {starter}"

        if marker.lower() in heading_line.lower():
            split_at = heading_line.lower().index(marker.lower())
            title = heading_line[:split_at].rstrip(" :")
            body = heading_line[
                split_at + 2:
            ].strip()

            return title, body

    return heading_line.strip(), ""


def extract_multiline_title(
    combined_text: str,
    match: re.Match,
    block_end: int,
) -> tuple[str, str, int]:
    """
    Return section title, inline body, and body start offset.

    If the initial title line lacks terminal punctuation, short following
    lines are appended until operative legal body text starts.
    """

    raw_title = match.group("title").strip()
    title, inline_body = split_title_and_inline_body(raw_title)

    body_start = match.end()

    if inline_body:
        return title.rstrip(" :."), inline_body, body_start

    if raw_title.endswith((".", ":")):
        return raw_title.rstrip(" :."), "", body_start

    cursor = match.end()
    title_parts = [raw_title]

    while cursor < block_end:
        line_match = re.match(
            r"(?:\n\s*)+([^\n]+)",
            combined_text[cursor:block_end],
        )

        if not line_match:
            break

        candidate = line_match.group(1).strip()

        if (
            not candidate
            or BODY_START_PATTERN.match(candidate)
            or SECTION_PATTERN.match(candidate)
            or CHAPTER_PATTERN.match(candidate)
            or PART_PATTERN.match(candidate)
            or UPPERCASE_STRUCTURE_PATTERN.match(candidate)
            or MIXED_STRUCTURE_PATTERN.match(candidate)
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

    return (
        " ".join(title_parts).rstrip(" :."),
        "",
        cursor,
    )


def analyze_section_body(
    section_number: str,
    section_title: str,
    text: str,
    source_metadata: dict,
) -> tuple[bool, bool]:
    heading_prefix = f"{section_number}. {section_title}"
    body = text

    if body.startswith(heading_prefix):
        body = body[len(heading_prefix):].lstrip(" :\n")

    body_characters = len(
        re.sub(r"\s+", " ", body).strip()
    )

    inherited_heading_only = bool(
        source_metadata.get("heading_only_page", False)
    )

    inherited_body_present = bool(
        source_metadata.get("section_body_present", True)
    )

    heading_only = bool(
        body_characters < MIN_SECTION_BODY_CHARACTERS
        or inherited_heading_only
        or not inherited_body_present
    )

    return not heading_only, heading_only


def parse_section_blocks(
    documents: list[Document],
) -> tuple[list[SectionBlock], dict[str, int]]:
    combined_text, page_spans, page_map = build_combined_text(
        documents
    )

    raw_candidates = list(
        SECTION_PATTERN.finditer(combined_text)
    )
    matches = find_section_matches(combined_text)
    boundaries = structural_boundaries(combined_text)

    diagnostics = {
        "raw_candidates": len(raw_candidates),
        "valid_candidates": len(matches),
        "rejected_candidates": (
            len(raw_candidates) - len(matches)
        ),
        "structural_boundaries": len(boundaries),
        "cross_page_sections": 0,
        "missing_page_metadata": 0,
    }

    blocks: list[SectionBlock] = []

    if not matches:
        if INCLUDE_UNSECTIONED_TEXT and combined_text.strip():
            pages = pages_for_offsets(
                0,
                len(combined_text),
                page_spans,
            )

            blocks.append(
                SectionBlock(
                    section_number=None,
                    section_title=None,
                    text=combined_text.strip(),
                    page_start=pages[0] if pages else None,
                    page_end=pages[-1] if pages else None,
                    source_pages=pages,
                    source_metadata=metadata_for_pages(
                        pages,
                        page_map,
                    ),
                    is_unsectioned=True,
                )
            )

        return blocks, diagnostics

    prefix_end = matches[0].start()
    prefix = combined_text[:prefix_end].strip()

    if INCLUDE_UNSECTIONED_TEXT and prefix:
        pages = pages_for_offsets(
            0,
            prefix_end,
            page_spans,
        )

        blocks.append(
            SectionBlock(
                section_number=None,
                section_title=None,
                text=prefix,
                page_start=pages[0] if pages else None,
                page_end=pages[-1] if pages else None,
                source_pages=pages,
                source_metadata=metadata_for_pages(
                    pages,
                    page_map,
                ),
                is_unsectioned=True,
            )
        )

    for index, match in enumerate(matches):
        block_start = match.start()

        next_section_start = (
            matches[index + 1].start()
            if index + 1 < len(matches)
            else len(combined_text)
        )

        block_end = next_structural_boundary(
            block_start,
            next_section_start,
            boundaries,
        )

        section_number = match.group("number").strip()

        section_title, inline_body, body_start = (
            extract_multiline_title(
                combined_text,
                match,
                block_end,
            )
        )

        raw_block = combined_text[
            block_start:block_end
        ].strip()

        if inline_body:
            heading = (
                f"{section_number}. {section_title}:"
            )

            raw_block = re.sub(
                rf"^\s*{re.escape(section_number)}"
                rf"\.\s+{re.escape(match.group('title'))}",
                f"{heading}\n\n{inline_body}",
                raw_block,
                count=1,
            )
        elif body_start > match.end():
            heading = (
                f"{section_number}. {section_title}"
            )

            body = combined_text[
                body_start:block_end
            ].strip()

            raw_block = (
                f"{heading}\n\n{body}"
                if body
                else heading
            )

        pages = pages_for_offsets(
            block_start,
            block_end,
            page_spans,
        )

        page_start = pages[0] if pages else None
        page_end = pages[-1] if pages else None

        if len(pages) > 1:
            diagnostics["cross_page_sections"] += 1

        if not pages:
            diagnostics["missing_page_metadata"] += 1

        source_metadata = metadata_for_pages(
            pages,
            page_map,
        )

        body_present, heading_only = analyze_section_body(
            section_number=section_number,
            section_title=section_title,
            text=raw_block,
            source_metadata=source_metadata,
        )

        blocks.append(
            SectionBlock(
                section_number=section_number,
                section_title=section_title,
                text=raw_block,
                page_start=page_start,
                page_end=page_end,
                source_pages=pages,
                source_metadata=source_metadata,
                heading_only=heading_only,
                section_body_present=body_present,
                is_unsectioned=False,
            )
        )

    return blocks, diagnostics


def create_section_metadata(
    block: SectionBlock,
) -> dict:
    metadata = deepcopy(block.source_metadata)

    document_id = str(
        metadata.get("document_id") or ""
    ).strip()

    if not document_id:
        raise RuntimeError(
            "Section block is missing document_id."
        )

    metadata.update(
        {
            "section_number": block.section_number,
            "section_title": block.section_title,
            "section_identity": (
                f"{document_id}::"
                f"{block.section_number or 'unsectioned'}"
            ),
            "section_numbers": (
                [block.section_number]
                if block.section_number
                else []
            ),
            "section_titles": (
                [block.section_title]
                if block.section_title
                else []
            ),
            "primary_section": block.section_number,
            "primary_section_title": block.section_title,
            "page_start": block.page_start,
            "page_end": block.page_end,
            "page_number": block.page_start,
            "source_pages": list(block.source_pages),
            "spans_multiple_pages": (
                len(block.source_pages) > 1
            ),
            "heading_only_chunk": block.heading_only,
            "section_body_present": (
                block.section_body_present
            ),
            "is_unsectioned_chunk": (
                block.is_unsectioned
            ),
        }
    )

    if block.section_number:
        metadata["legal_topic"] = (
            block.section_title.lower()
            if block.section_title
            else None
        )

    return metadata


def split_large_section(
    block: SectionBlock,
    text_splitter: RecursiveCharacterTextSplitter,
) -> list[Document]:
    metadata = create_section_metadata(block)

    if len(block.text) <= CHUNK_SIZE:
        return [
            Document(
                page_content=block.text,
                metadata=metadata,
            )
        ]

    heading = ""

    if block.section_number:
        heading = (
            f"{block.section_number}. "
            f"{block.section_title or ''}"
        ).strip()

    body = block.text

    if heading and body.startswith(heading):
        body = body[len(heading):].lstrip(" :\n")

    body_chunks = text_splitter.split_documents(
        [
            Document(
                page_content=body,
                metadata=metadata,
            )
        ]
    )

    results: list[Document] = []
    total_parts = len(body_chunks)

    for part_number, chunk in enumerate(
        body_chunks,
        start=1,
    ):
        content = chunk.page_content.strip()

        if heading:
            content = (
                f"{heading}\n\n{content}"
            ).strip()

        chunk.metadata.update(
            {
                "section_part_number": part_number,
                "section_part_count": total_parts,
                "section_was_split": True,
            }
        )

        chunk.page_content = content
        results.append(chunk)

    return results


def validate_final_chunks(
    chunks: list[Document],
) -> None:
    errors: list[str] = []

    for index, chunk in enumerate(chunks, start=1):
        metadata = chunk.metadata

        if not chunk.page_content.strip():
            errors.append(
                f"Chunk {index}: empty content"
            )

        required_document_fields = (
            "document_id",
            "document_name",
            "document_title",
            "document_type",
        )

        for field_name in required_document_fields:
            if not metadata.get(field_name):
                errors.append(
                    f"Chunk {index}: missing {field_name}"
                )

        if metadata.get("is_unsectioned_chunk"):
            continue

        if not metadata.get("section_number"):
            errors.append(
                f"Chunk {index}: missing section_number"
            )

        if not metadata.get("section_identity"):
            errors.append(
                f"Chunk {index}: missing section_identity"
            )

        if metadata.get("page_start") is None:
            errors.append(
                f"Chunk {index}: missing page_start"
            )

        if metadata.get("page_end") is None:
            errors.append(
                f"Chunk {index}: missing page_end"
            )

        if not metadata.get("source_pages"):
            errors.append(
                f"Chunk {index}: missing source_pages"
            )

        page_start = metadata.get("page_start")
        page_end = metadata.get("page_end")

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
            for error in errors[:20]
        )

        raise RuntimeError(
            "Final chunk validation failed:\n"
            f"{preview}"
        )


def create_chunks(
    documents: list[Document],
) -> list[Document]:
    if not documents:
        raise ValueError(
            "No documents were provided for chunking."
        )

    validate_chunk_settings()

    document_groups: dict[str, list[Document]] = {}

    for document in documents:
        (
            document_id,
            _document_name,
            _document_title,
            _document_type,
        ) = get_document_identity(
            document
        )

        document_groups.setdefault(
            document_id,
            [],
        ).append(document)

    for document_id, grouped_documents in (
        document_groups.items()
    ):
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

    section_blocks_created = 0
    oversized_sections_split = 0
    heading_only_blocks_skipped = 0
    unsectioned_blocks_created = 0

    raw_candidates = 0
    valid_candidates = 0
    rejected_candidates = 0
    structural_boundaries_count = 0
    cross_page_sections = 0
    missing_page_metadata = 0

    per_document_summaries: list[
        dict[str, object]
    ] = []

    for document_id, grouped_documents in (
        document_groups.items()
    ):
        document_metadata = (
            grouped_documents[0].metadata
        )

        blocks, diagnostics = parse_section_blocks(
            grouped_documents
        )

        chunks_before_document = len(
            final_chunks
        )
        document_section_blocks = 0
        document_unsectioned_blocks = 0
        document_oversized_sections = 0
        document_heading_only_skipped = 0

        raw_candidates += diagnostics["raw_candidates"]
        valid_candidates += diagnostics["valid_candidates"]
        rejected_candidates += diagnostics[
            "rejected_candidates"
        ]
        structural_boundaries_count += diagnostics[
            "structural_boundaries"
        ]
        cross_page_sections += diagnostics[
            "cross_page_sections"
        ]
        missing_page_metadata += diagnostics[
            "missing_page_metadata"
        ]

        for block in blocks:
            if block.is_unsectioned:
                unsectioned_blocks_created += 1
                document_unsectioned_blocks += 1
            else:
                section_blocks_created += 1
                document_section_blocks += 1

            if (
                block.heading_only
                and not INCLUDE_HEADING_ONLY_CHUNKS
            ):
                heading_only_blocks_skipped += 1
                document_heading_only_skipped += 1
                continue

            block_chunks = split_large_section(
                block,
                text_splitter,
            )

            if len(block_chunks) > 1:
                oversized_sections_split += 1
                document_oversized_sections += 1

            for chunk in block_chunks:
                chunk_document_id = str(
                    chunk.metadata.get(
                        "document_id",
                        "",
                    )
                )

                if chunk_document_id != document_id:
                    raise RuntimeError(
                        "Chunk inherited incorrect document_id. "
                        f"Expected {document_id}, found "
                        f"{chunk_document_id}."
                    )

            final_chunks.extend(block_chunks)

        per_document_summaries.append(
            {
                "document_id": document_id,
                "document_name": document_metadata.get(
                    "document_name",
                    "Unknown",
                ),
                "document_title": document_metadata.get(
                    "document_title",
                    "Unknown",
                ),
                "document_type": document_metadata.get(
                    "document_type",
                    "legal_document",
                ),
                "pages": len(grouped_documents),
                "section_blocks": document_section_blocks,
                "unsectioned_blocks": (
                    document_unsectioned_blocks
                ),
                "oversized_sections": (
                    document_oversized_sections
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
                "cross_page_sections": diagnostics[
                    "cross_page_sections"
                ],
            }
        )

    cleaned_chunks: list[Document] = []

    for chunk in final_chunks:
        chunk.page_content = (
            chunk.page_content.strip()
        )

        if not chunk.page_content:
            continue

        chunk.metadata.setdefault(
            "section_was_split",
            False,
        )
        chunk.metadata.setdefault(
            "section_part_number",
            1,
        )
        chunk.metadata.setdefault(
            "section_part_count",
            1,
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

        section_part_number = int(
            chunk.metadata.get(
                "section_part_number",
                1,
            )
            or 1
        )

        section_number = str(
            chunk.metadata.get(
                "section_number",
                "unsectioned",
            )
            or "unsectioned"
        )

        chunk.metadata.update(
            {
                "chunk_number": chunk_number,
                "document_chunk_number": (
                    document_chunk_number
                ),
                "chunk_id": (
                    f"{document_id}::"
                    f"{section_number}::"
                    f"part-{section_part_number}::"
                    f"chunk-{document_chunk_number}"
                ),
                "chunk_length": len(
                    chunk.page_content
                ),
                "chunk_size_setting": CHUNK_SIZE,
                "chunk_overlap_setting": CHUNK_OVERLAP,
            }
        )

    validate_final_chunks(cleaned_chunks)

    print("=" * 70)
    print("CHUNKING SUMMARY")
    print("=" * 70)
    print(
        f"Page documents received: {len(documents)}"
    )
    print(
        f"Document groups processed: {len(document_groups)}"
    )
    print(
        f"Raw section candidates: {raw_candidates}"
    )
    print(
        f"Valid section candidates: {valid_candidates}"
    )
    print(
        f"Invalid candidates rejected: {rejected_candidates}"
    )
    print(
        f"Structural boundaries detected: "
        f"{structural_boundaries_count}"
    )
    print(
        f"Section blocks parsed: {section_blocks_created}"
    )
    print(
        f"Cross-page sections detected: "
        f"{cross_page_sections}"
    )
    print(
        f"Sections missing page metadata: "
        f"{missing_page_metadata}"
    )
    print(
        f"Unsectioned blocks created: "
        f"{unsectioned_blocks_created}"
    )
    print(
        f"Oversized sections split: "
        f"{oversized_sections_split}"
    )
    print(
        f"Heading-only blocks skipped: "
        f"{heading_only_blocks_skipped}"
    )
    print(f"Chunk size: {CHUNK_SIZE}")
    print(f"Chunk overlap: {CHUNK_OVERLAP}")
    print(
        f"Total chunks created: {len(cleaned_chunks)}"
    )

    if per_document_summaries:
        print("\n" + "=" * 70)
        print("PER-DOCUMENT CHUNKING SUMMARY")
        print("=" * 70)

        for summary in per_document_summaries:
            print(
                f"\nDocument: "
                f"{summary['document_title']}"
            )
            print(
                f"Document ID: "
                f"{summary['document_id']}"
            )
            print(
                f"Document type: "
                f"{summary['document_type']}"
            )
            print(
                f"Pages processed: "
                f"{summary['pages']}"
            )
            print(
                f"Raw section candidates: "
                f"{summary['raw_candidates']}"
            )
            print(
                f"Valid section candidates: "
                f"{summary['valid_candidates']}"
            )
            print(
                f"Section blocks: "
                f"{summary['section_blocks']}"
            )
            print(
                f"Cross-page sections: "
                f"{summary['cross_page_sections']}"
            )
            print(
                f"Unsectioned blocks: "
                f"{summary['unsectioned_blocks']}"
            )
            print(
                f"Oversized sections split: "
                f"{summary['oversized_sections']}"
            )
            print(
                f"Heading-only blocks skipped: "
                f"{summary['heading_only_skipped']}"
            )
            print(
                f"Chunks created: "
                f"{summary['chunks_created']}"
            )

    return cleaned_chunks


def display_chunks(
    chunks: list[Document],
    count: int = 5,
) -> None:
    if not chunks:
        print("No chunks are available.")
        return

    print("\n" + "=" * 70)
    print("SAMPLE CHUNKS")
    print("=" * 70)

    for index, chunk in enumerate(
        chunks[:min(count, len(chunks))],
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
            f"{len(chunk.page_content)} characters"
        )


def display_section_chunks(
    chunks: list[Document],
    section_number: str,
    document_id: str | None = None,
) -> None:
    normalized_section = (
        section_number.strip().upper()
    )

    matching_chunks = [
        chunk
        for chunk in chunks
        if (
            str(
                chunk.metadata.get(
                    "section_number",
                    "",
                )
            ).upper()
            == normalized_section
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
        )
    ]

    print("\n" + "=" * 70)
    print(
        f"CHUNKS FOR SECTION {section_number}"
    )
    print("=" * 70)

    if not matching_chunks:
        print(
            f"No chunk for Section "
            f"{section_number} was found."
        )
        return

    for chunk in matching_chunks:
        print("\n" + "-" * 70)
        print(
            f"Document: "
            f"{chunk.metadata.get('document_title', 'Unknown')} | "
            f"Document ID: "
            f"{chunk.metadata.get('document_id', 'Unknown')} | "
            f"Pages: "
            f"{chunk.metadata.get('page_start', 'Unknown')}"
            f"–"
            f"{chunk.metadata.get('page_end', 'Unknown')} | "
            f"Source pages: "
            f"{chunk.metadata.get('source_pages', [])} | "
            f"Chunk: "
            f"{chunk.metadata.get('chunk_number', 'Unknown')} | "
            f"Part: "
            f"{chunk.metadata.get('section_part_number', 1)}"
            f"/"
            f"{chunk.metadata.get('section_part_count', 1)}"
        )
        print("-" * 70)
        print(chunk.page_content)


def main() -> None:
    from rag.document_loader import load_all_documents

    try:
        print(
            "Step 1: Loading and cleaning PDFs..."
        )

        documents = load_all_documents()

        print(
            "\nStep 2: Creating section-aware chunks..."
        )

        chunks = create_chunks(documents)

        display_chunks(chunks, count=5)

        for section_number in [
            "1",
            "3",
            "298-C",
            "299",
            "379",
            "405",
            "408",
            "409",
        ]:
            display_section_chunks(
                chunks,
                section_number=section_number,
            )

    except Exception as error:
        print(f"\nChunking error: {error}")


if __name__ == "__main__":
    main()