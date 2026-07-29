import re
import unicodedata
from collections import Counter
from dataclasses import dataclass, field
from typing import Iterable


# -------------------------------------------------------------------
# Legal structure patterns
# -------------------------------------------------------------------

SECTION_HEADING_PATTERN = re.compile(
    r"^\s*(\d+(?:-[A-Za-z]+)?[A-Za-z]?)\.\s+",
    re.IGNORECASE,
)

SECTION_LINE_PATTERN = re.compile(
    r"^\s*(\d+(?:-[A-Za-z]+)?[A-Za-z]?)\.\s*(.*?)\s*$",
    re.IGNORECASE,
)

CHAPTER_NUMBER_PATTERN = re.compile(
    r"^\s*CHAPTER\s+([IVXLCDM]+|\d+)\s*$",
    re.IGNORECASE,
)

STANDALONE_ROMAN_NUMERAL_PATTERN = re.compile(
    r"^\s*([IVXLCDM]{1,8})\s*$",
    re.IGNORECASE,
)

PART_PATTERN = re.compile(
    r"^\s*PART\s+([IVXLCDM]+|\d+)\s*$",
    re.IGNORECASE,
)

EXPLANATION_PATTERN = re.compile(
    r"^\s*Explanation(?:\s+(\d+))?\s*[:.-]?\s*(.*)$",
    re.IGNORECASE,
)

ILLUSTRATION_PATTERN = re.compile(
    r"^\s*Illustrations?\s*[:.-]?\s*$",
    re.IGNORECASE,
)

ENUMERATION_PATTERN = re.compile(
    r"^\s*\((?:[a-zA-Z]|\d+|[ivxlcdm]+)\)\s*",
    re.IGNORECASE,
)

SECTION_BODY_STARTER_PATTERN = re.compile(
    r"^(?:"
    r"Whoever|"
    r"Any person|"
    r"Every person|"
    r"Nothing|"
    r"Where|"
    r"When|"
    r"If|"
    r"A person|"
    r"Provided|"
    r"In this section|"
    r"For the purposes of"
    r")\b",
    re.IGNORECASE,
)

LEGAL_REFERENCE_PATTERN = re.compile(
    r"\b(?:section|sections|sec\.?)\s+"
    r"(\d+(?:-[A-Za-z]+)?[A-Za-z]?)\b",
    re.IGNORECASE,
)

PAGE_NUMBER_PATTERN = re.compile(
    r"^\s*(?:page\s*)?\d+\s*$",
    re.IGNORECASE,
)

COMMON_HEADER_PATTERNS = [
    re.compile(
        r"^\s*Pakistan\s+Penal\s+Code(?:\s*\([^)]*\))?\s*$",
        re.IGNORECASE,
    ),
    re.compile(
        r"^\s*Pakistan\s+Penal\s+Code\s*"
        r"\(Act\s+XLV\s+of\s+1860\)\s*$",
        re.IGNORECASE,
    ),
]

# Standalone amendment-history lines. These are editorial history, not
# the operative legal text that should normally drive retrieval.
AMENDMENT_NOTE_PATTERN = re.compile(
    r"^\s*\d+\s+"
    r"(?:"
    r"Inserted\b|"
    r"Substituted\b|"
    r"Omitted\b|"
    r"Added\b|"
    r"Amended\b|"
    r"Repealed\b|"
    r"Proviso\s+omitted\b|"
    r"The\s+following\s+was\s+omitted\b|"
    r"The\s+words?.*?(?:inserted|substituted|omitted)\b|"
    r"Sections?\s+\d"
    r")",
    re.IGNORECASE,
)

# Common continuation fragments that belong to an amendment note started
# on the previous line. These are intentionally conservative.
AMENDMENT_CONTINUATION_PATTERN = re.compile(
    r"^\s*(?:"
    r"for\s*:\s*[\"']|"
    r"for\s+the\s+words?\b|"
    r"the\s+words?\b|"
    r"the\s+following\b|"
    r"with\s+effect\s+from\b|"
    r"vide\b|"
    r"see\b|"
    r"read\s+as\b|"
    r"by\s+Act\b|"
    r"by\s+Ordinance\b|"
    r"by\s+the\b|"
    r"\([IVXLCDM]+\s+of\s+\d{4}\)"
    r"|S\.\s*\d+\b"
    r"|and\s+Schedule\b"
    r")",
    re.IGNORECASE,
)

# Lines that are clearly legal content and should terminate amendment
# skipping even if the previous line was an amendment note.
LEGAL_CONTENT_START_PATTERN = re.compile(
    r"^(?:"
    r"\d+(?:-[A-Za-z]+)?[A-Za-z]?\.\s+|"
    r"CHAPTER\b|"
    r"PART\b|"
    r"SECTION\b|"
    r"Explanation(?:\s+\d+)?\b|"
    r"Illustrations?\b|"
    r"\((?:[a-zA-Z]|\d+|[ivxlcdm]+)\)\s+|"
    r"Whoever\b|"
    r"Nothing\b|"
    r"Provided\b|"
    r"Where\b|"
    r"When\b|"
    r"If\b|"
    r"Any\s+person\b|"
    r"Every\s+person\b|"
    r"A\s+person\b"
    r")",
    re.IGNORECASE,
)

# Conservative, document-specific OCR corrections. Explicit mappings are
# safer than general spell correction for legal text.
KNOWN_OCR_CORRECTIONS = {
    "forevery": "for every",
    "afemale": "a female",
    "aJudge": "a Judge",
    "bean received": "been received",
    "wilt of a deceased": "will of a deceased",
    "Mere, if": "Here, if",
    "Codeand": "Code and",
    "dealtwith": "dealt with",
    "andthe": "and the",
    "whichservant": "which servant",
    "ortaking": "or taking",
    "written,or": "written, or",
    "ofcall": "of call",
    "threeyears": "three years",
    "thesubject": "the subject",
    "byany": "by any",
    "himselfas": "himself as",
    "formof": "form of",
    "shallalso": "shall also",
    "shallnot": "shall not",
    "shallbe": "shall be",
    "mayextend": "may extend",
    "of1986": "of 1986",
    "toextra": "to extra",
    "provisionof": "provision of",
    "andnot": "and not",
    "inthe": "in the",
    "ofthe": "of the",
    "tothe": "to the",
    "withthe": "with the",
    "foran": "for an",
    "inthecase": "in the case",
    "ofthecase": "of the case",
    "isliable": "is liable",
}


# -------------------------------------------------------------------
# Structured output models
# -------------------------------------------------------------------

@dataclass
class CleaningStatistics:
    original_characters: int = 0
    cleaned_characters: int = 0
    characters_removed: int = 0
    headers_removed: int = 0
    footers_removed: int = 0
    artifacts_removed: int = 0
    footnotes_removed: int = 0
    amendment_notes_removed: int = 0
    amendment_continuation_lines_removed: int = 0
    broken_words_fixed: int = 0
    hyphenations_fixed: int = 0
    sections_detected: int = 0
    chapters_detected: int = 0
    standalone_chapters_detected: int = 0
    parts_detected: int = 0
    explanations_detected: int = 0
    illustrations_detected: int = 0
    enumerations_detected: int = 0
    inline_section_bodies_split: int = 0
    heading_only_pages_detected: int = 0

    def as_dict(self) -> dict:
        return {
            "original_characters": self.original_characters,
            "cleaned_characters": self.cleaned_characters,
            "characters_removed": self.characters_removed,
            "headers_removed": self.headers_removed,
            "footers_removed": self.footers_removed,
            "artifacts_removed": self.artifacts_removed,
            "footnotes_removed": self.footnotes_removed,
            "amendment_notes_removed": self.amendment_notes_removed,
            "amendment_continuation_lines_removed": (
                self.amendment_continuation_lines_removed
            ),
            "broken_words_fixed": self.broken_words_fixed,
            "hyphenations_fixed": self.hyphenations_fixed,
            "sections_detected": self.sections_detected,
            "chapters_detected": self.chapters_detected,
            "standalone_chapters_detected": (
                self.standalone_chapters_detected
            ),
            "parts_detected": self.parts_detected,
            "explanations_detected": self.explanations_detected,
            "illustrations_detected": self.illustrations_detected,
            "enumerations_detected": self.enumerations_detected,
            "inline_section_bodies_split": (
                self.inline_section_bodies_split
            ),
            "heading_only_pages_detected": (
                self.heading_only_pages_detected
            ),
        }


@dataclass
class CleanedPage:
    text: str
    metadata: dict = field(default_factory=dict)
    statistics: CleaningStatistics = field(
        default_factory=CleaningStatistics
    )


# -------------------------------------------------------------------
# Unicode, whitespace, and OCR repair
# -------------------------------------------------------------------

def normalize_unicode_and_whitespace(text: str) -> str:
    """Normalize Unicode and remove invisible whitespace characters."""

    normalized = unicodedata.normalize("NFKC", text)

    return (
        normalized
        .replace("\u00ad", "")
        .replace("\u200b", "")
        .replace("\u200c", "")
        .replace("\u200d", "")
        .replace("\u2060", "")
        .replace("\ufeff", "")
        .replace("\xa0", " ")
        .replace("\r\n", "\n")
        .replace("\r", "\n")
    )


def repair_hyphenation(
    text: str,
    statistics: CleaningStatistics,
) -> str:
    """
    Repair line-break hyphenation while preserving normal hyphenated
    legal terms.

    Examples:
        imprison-\nment -> imprisonment
        381-\nA.       -> 381-A.
    """

    word_pattern = re.compile(
        r"(?<=[A-Za-z])-\s*\n\s*(?=[a-z])"
    )

    text, count = word_pattern.subn("", text)
    statistics.hyphenations_fixed += count

    section_pattern = re.compile(
        r"\b(\d+)-\s*\n\s*([A-Z])\."
    )

    text, section_count = section_pattern.subn(
        r"\1-\2.",
        text,
    )

    statistics.hyphenations_fixed += section_count

    return text


def repair_known_ocr_errors(
    text: str,
    statistics: CleaningStatistics,
) -> str:
    """
    Apply conservative OCR repairs without general spell correction.

    This function is intentionally safe for legal language and may be
    called more than once. Statistics only count actual replacements.
    """

    for incorrect, corrected in KNOWN_OCR_CORRECTIONS.items():
        pattern = re.compile(
            re.escape(incorrect),
            re.IGNORECASE,
        )

        text, count = pattern.subn(
            corrected,
            text,
        )

        statistics.broken_words_fixed += count

    # Repair lower-case legal function words joined to a capitalized word.
    joined_word_pattern = re.compile(
        r"\b(a|an|the|of|to|for|in|on|by|or|and)"
        r"([A-Z][a-z]{2,})\b"
    )

    text, count = joined_word_pattern.subn(
        r"\1 \2",
        text,
    )
    statistics.broken_words_fixed += count

    # Limited function-word repair for frequent legal phrases only.
    function_word_pattern = re.compile(
        r"\b(and|or|the|of|to|for|with|in|on|by)"
        r"(?=(?:the|a|an|any|every|his|her|their|this|that|"
        r"which|whom|whose|person|property|law|code|court|"
        r"section|offence|imprisonment|fine|government|case|"
        r"accused|convict|servant|agent|public)\b)",
        re.IGNORECASE,
    )

    text, count = function_word_pattern.subn(
        r"\1 ",
        text,
    )
    statistics.broken_words_fixed += count

    # Repair merged years and numeric references such as Act1986.
    letter_number_pattern = re.compile(
        r"(?<=[A-Za-z])(?=\d{2,4}\b)"
    )

    text, count = letter_number_pattern.subn(
        " ",
        text,
    )
    statistics.broken_words_fixed += count

    return text


# -------------------------------------------------------------------
# Repeated headers, footers, and artifacts
# -------------------------------------------------------------------

def normalize_line_for_frequency(line: str) -> str:
    """Normalize a line before repeated header/footer comparison."""

    line = unicodedata.normalize("NFKC", line)
    line = re.sub(r"\s+", " ", line).strip().lower()
    line = re.sub(r"\bpage\s+\d+\b", "page <n>", line)
    line = re.sub(r"^\d+$", "<page-number>", line)

    return line


def detect_repeated_page_lines(
    pages: Iterable[str],
    frequency_threshold: float = 0.80,
    edge_line_limit: int = 4,
) -> set[str]:
    """
    Detect repeated header/footer lines across pages.

    Only short lines near page edges are considered to avoid removing
    repeated legal wording from the body.
    """

    page_list = list(pages)

    non_empty_pages = [
        page
        for page in page_list
        if page and page.strip()
    ]

    if len(non_empty_pages) < 3:
        return set()

    page_occurrences: Counter[str] = Counter()

    for page in non_empty_pages:
        lines = [
            re.sub(r"\s+", " ", line).strip()
            for line in normalize_unicode_and_whitespace(
                page
            ).splitlines()
            if line.strip()
        ]

        if not lines:
            continue

        edge_lines = (
            lines[:edge_line_limit]
            + lines[-edge_line_limit:]
        )

        normalized_unique = {
            normalize_line_for_frequency(line)
            for line in edge_lines
            if len(line) <= 180
        }

        page_occurrences.update(normalized_unique)

    minimum_pages = max(
        2,
        int(
            len(non_empty_pages)
            * frequency_threshold
        ),
    )

    return {
        line
        for line, count in page_occurrences.items()
        if count >= minimum_pages
        and line not in {"", "<page-number>"}
    }


def is_known_header(line: str) -> bool:
    """Return True for known Pakistan Penal Code headers."""

    return any(
        pattern.fullmatch(line)
        for pattern in COMMON_HEADER_PATTERNS
    )


def is_page_footer(line: str) -> bool:
    """Detect isolated page-number footer lines."""

    return bool(
        PAGE_NUMBER_PATTERN.fullmatch(line)
    )


def is_pdf_artifact(line: str) -> bool:
    """Detect isolated amendment, citation, and OCR marker lines."""

    artifact_patterns = [
        r"^\d+\[\]\s*\d+(?:\s+\d+\[\]\s*\d+)*$",
        r"^\d+\[$",
        r"^\]\s*\d+$",
        r"^\[\s*$",
        r"^\]\s*$",
        r"^\d+\[\s*$",
        r"^\d+\[\]\s*$",
        r"^\d+\]\s*$",
        r"^\[\d+\]\s*$",
        r"^\d+\[\(\d+\)\s*$",
        r"^\d+\[\s*\(\d+\)\s*$",
        r"^\d+\s*\[\s*\]\s*$",
        r"^\*+\s*$",
        r"^_+\s*$",
        r"^\d+\s*\]\s*$",
        r"^\[\s*\d+\s*$",
        r"^\d+\s*\[\s*\d*\s*\]\s*$",
    ]

    return any(
        re.fullmatch(pattern, line)
        for pattern in artifact_patterns
    )


def is_amendment_note(line: str) -> bool:
    """Detect the first line of a standalone amendment-history note."""

    return bool(
        AMENDMENT_NOTE_PATTERN.match(line)
    )


def is_amendment_continuation(line: str) -> bool:
    """
    Detect a likely continuation of an amendment-history note.

    This is deliberately conservative because deleting legal body text
    is worse than retaining a small amount of editorial history.
    """

    stripped = line.strip()

    if not stripped:
        return True

    if LEGAL_CONTENT_START_PATTERN.match(stripped):
        return False

    if AMENDMENT_CONTINUATION_PATTERN.match(stripped):
        return True

    # Short quotation fragments frequently follow "Substituted ... for:".
    if (
        len(stripped) <= 140
        and (
            stripped.startswith(('"', "'"))
            or stripped.endswith(('"', "'.", '";', "'."))
            or stripped.endswith('".')
        )
    ):
        return True

    # Editorial citation fragments such as:
    # (VII of 1979), S. 19::
    if re.fullmatch(
        r"\s*\([IVXLCDM]+\s+of\s+\d{4}\)"
        r".{0,80}S\.\s*\d+.*",
        stripped,
        re.IGNORECASE,
    ):
        return True

    return False


def remove_inline_citation_artifacts(line: str) -> str:
    """
    Remove compact amendment markers while preserving inserted legal text.

    Examples:
        personal law115[but shall not include ...] 115;
        -> personal law but shall not include ...;

        1[(1) text
        -> (1) text
    """

    # Preserve subsection markers after a footnote prefix.
    line = re.sub(
        r"\b\d+\[\s*(\(\d+\))",
        r"\1",
        line,
    )

    # Remove numeric marker before an opening amendment bracket.
    line = re.sub(
        r"(?<=\w)\d+\[(?=[A-Za-z(\"'])",
        " ",
        line,
    )

    # Remove repeated closing amendment marker.
    line = re.sub(
        r"\]\s*\d+(?=[;,.:\s]|$)",
        "",
        line,
    )

    # Remove isolated inline markers.
    line = re.sub(
        r"(?<!\w)\d+\[\](?!\w)",
        "",
        line,
    )
    line = re.sub(
        r"(?<!\w)\d+\[(?!\w)",
        "",
        line,
    )
    line = re.sub(
        r"(?<!\w)\[\d+\](?!\w)",
        "",
        line,
    )

    # Remove a lone closing amendment bracket near punctuation.
    line = re.sub(
        r"(?<!\w)\](?=[;,.:\s]|$)",
        "",
        line,
    )

    # Remove a remaining footnote digit attached to a word only when it
    # is followed by clear amendment wording.
    line = re.sub(
        r"(?<=\w)\d+(?=\s+(?:but|provided|except|and)\b)",
        "",
        line,
        flags=re.IGNORECASE,
    )

    return re.sub(
        r"\s{2,}",
        " ",
        line,
    ).strip()


# -------------------------------------------------------------------
# Legal structure helpers
# -------------------------------------------------------------------

def parse_section_heading(line: str) -> tuple[str, str, str] | None:
    """
    Parse a section heading into number, title, and inline body.

    The helper keeps the title conservative. It only splits the body
    when a colon clearly introduces operative legal text.
    """

    match = SECTION_LINE_PATTERN.match(line)

    if not match:
        return None

    number = match.group(1).strip()
    remainder = match.group(2).strip()

    if not remainder:
        return None

    title = remainder
    inline_body = ""

    for colon_match in re.finditer(":", remainder):
        colon_index = colon_match.start()
        before = remainder[:colon_index].strip()
        after = remainder[colon_index + 1:].lstrip()

        if not before or not after:
            continue

        cleaned_after = re.sub(
            r'^[\s"\(\[\']+',
            "",
            after,
        ).strip()

        if SECTION_BODY_STARTER_PATTERN.match(cleaned_after):
            title = before.rstrip(" ,;.-")
            inline_body = after
            break

    return number, title, inline_body


def split_section_heading_and_body(
    line: str,
) -> tuple[str, str, str] | None:
    """
    Split a section heading that contains inline body text.

    Returns:
        (section_number, section_title, inline_body)

    If the line is not a section heading, returns None. If the line is a
    section heading without inline body, the body element is an empty
    string.
    """

    return parse_section_heading(line)


def is_probable_chapter_title(line: str) -> bool:
    """Detect a short chapter subject line."""

    stripped = line.strip()

    if not stripped or len(stripped) > 140:
        return False

    if stripped.endswith((".", ";", ",")):
        return False

    if SECTION_BODY_STARTER_PATTERN.match(stripped):
        return False

    alpha_count = sum(
        character.isalpha()
        for character in stripped
    )

    if not alpha_count:
        return False

    uppercase_ratio = (
        sum(
            character.isupper()
            for character in stripped
        )
        / max(1, alpha_count)
    )

    return (
        stripped.upper().startswith("OF ")
        or uppercase_ratio >= 0.80
    )


def is_standalone_chapter_number_line(line: str) -> bool:
    """Return True for a line that is only a Roman numeral chapter mark."""

    return bool(
        STANDALONE_ROMAN_NUMERAL_PATTERN.fullmatch(line.strip())
    )


def get_chapter_heading_at_index(
    lines: list[str],
    index: int,
) -> tuple[str, str] | None:
    """
    Detect a chapter heading at a specific line index.

    Supports both:
    - CHAPTER XVI
    - XVI followed by a probable chapter title
    """

    line = lines[index].strip()

    if not line:
        return None

    chapter_match = CHAPTER_NUMBER_PATTERN.match(line)
    if chapter_match:
        chapter_number = chapter_match.group(1).upper()
        next_line = get_next_non_empty_line(
            lines,
            index + 1,
        )
        chapter_title = (
            next_line.strip()
            if next_line and is_probable_chapter_title(next_line)
            else None
        )
        return chapter_number, chapter_title

    roman_match = STANDALONE_ROMAN_NUMERAL_PATTERN.match(line)
    if not roman_match:
        return None

    next_line = get_next_non_empty_line(
        lines,
        index + 1,
    )

    if not next_line or not is_probable_chapter_title(next_line):
        return None

    return roman_match.group(1).upper(), next_line.strip()


def get_next_non_empty_line(
    lines: list[str],
    start_index: int,
) -> str | None:
    """Return the next non-empty stripped line from a list of lines."""

    for line in lines[start_index:]:
        stripped = line.strip()
        if stripped:
            return stripped
    return None


def is_legal_heading(line: str) -> bool:
    """Return True when a line begins a legal structural block."""

    return bool(
        parse_section_heading(line)
        or CHAPTER_NUMBER_PATTERN.match(line)
        or PART_PATTERN.match(line)
        or EXPLANATION_PATTERN.match(line)
        or ILLUSTRATION_PATTERN.match(line)
        or ENUMERATION_PATTERN.match(line)
    )


def extract_section_details(
    text: str,
) -> tuple[list[str], list[str]]:
    """Extract section numbers and titles from cleaned page text."""

    numbers: list[str] = []
    titles: list[str] = []

    for line in text.splitlines():
        parsed = parse_section_heading(line.strip())

        if not parsed:
            continue

        number, title, _body = parsed
        title = title.strip()

        if title.endswith(":"):
            title = title[:-1].rstrip()

        if number not in numbers:
            numbers.append(number)

        if title and title not in titles:
            titles.append(title)

    return numbers, titles


def extract_chapter_details(
    text: str,
) -> tuple[str | None, str | None]:
    """Extract the first chapter number and chapter title on a page."""

    lines = [
        line.strip()
        for line in text.splitlines()
        if line.strip()
    ]

    for index, line in enumerate(lines):
        chapter_heading = get_chapter_heading_at_index(
            lines,
            index,
        )

        if chapter_heading is None:
            continue

        chapter_number, chapter_title = chapter_heading
        return chapter_number, chapter_title

    return None, None


def extract_part(text: str) -> str | None:
    """Extract the first PART identifier from the page."""

    for line in text.splitlines():
        match = PART_PATTERN.match(
            line.strip()
        )

        if match:
            return match.group(1).upper()

    return None


def extract_legal_references(
    text: str,
) -> list[str]:
    """Extract section references mentioned inside the page."""

    references: list[str] = []

    for reference in LEGAL_REFERENCE_PATTERN.findall(text):
        normalized = reference.upper()

        if normalized not in references:
            references.append(normalized)

    return references


def extract_keywords(
    text: str,
    limit: int = 15,
) -> list[str]:
    """Extract deterministic legal keywords without external NLP."""

    legal_terms = [
        "abetment",
        "agent",
        "breach of trust",
        "cheating",
        "criminal force",
        "dishonestly",
        "embezzlement",
        "entrusted",
        "fine",
        "fraud",
        "fraudulently",
        "homicide",
        "imprisonment",
        "intention",
        "mischief",
        "misappropriation",
        "movable property",
        "murder",
        "offence",
        "property",
        "punishment",
        "rioting",
        "servant",
        "theft",
        "unlawful assembly",
    ]

    normalized_text = text.lower()
    detected: list[str] = []

    for term in legal_terms:
        if re.search(
            rf"\b{re.escape(term)}\b",
            normalized_text,
        ):
            detected.append(term)

        if len(detected) >= limit:
            break

    return detected


def analyze_section_body_presence(
    text: str,
) -> dict:
    """
    Determine whether a page contains substantive text beyond headings.

    This helps later chunking avoid treating heading-only amendment pages
    as complete answerable provisions.
    """

    lines = [
        line.strip()
        for line in text.splitlines()
        if line.strip()
    ]

    section_heading_indices: set[int] = set()
    chapter_heading_indices: set[int] = set()

    index = 0
    while index < len(lines):
        line = lines[index]

        if parse_section_heading(line):
            section_heading_indices.add(index)
            index += 1
            continue

        chapter_heading = get_chapter_heading_at_index(
            lines,
            index,
        )

        if chapter_heading is not None:
            chapter_heading_indices.add(index)
            if (
                index + 1 < len(lines)
                and is_probable_chapter_title(lines[index + 1])
            ):
                chapter_heading_indices.add(index + 1)
        index += 1

    non_heading_lines = [
        line
        for line_index, line in enumerate(lines)
        if line_index not in section_heading_indices
        and line_index not in chapter_heading_indices
        and not PART_PATTERN.match(line)
        and not EXPLANATION_PATTERN.match(line)
        and not ILLUSTRATION_PATTERN.match(line)
    ]

    body_text = " ".join(non_heading_lines).strip()
    body_character_count = len(body_text)
    section_count = len(section_heading_indices)

    # A page with several section headings but almost no body text is
    # likely an amendment/index-style page rather than full legal text.
    heading_only_page = bool(
        section_count > 0
        and body_character_count < max(
            120,
            section_count * 45,
        )
    )

    section_body_present = bool(
        section_count == 0
        or not heading_only_page
    )

    return {
        "section_body_present": section_body_present,
        "heading_only_page": heading_only_page,
        "section_heading_count": section_count,
        "non_heading_character_count": body_character_count,
    }


def build_metadata(
    text: str,
    page_number: int | None = None,
) -> dict:
    """Build page-level legal metadata from cleaned text."""

    section_numbers, section_titles = (
        extract_section_details(text)
    )

    chapter_number, chapter_title = (
        extract_chapter_details(text)
    )

    contains_explanation = bool(
        re.search(
            r"(?mi)^\s*Explanation"
            r"(?:\s+\d+)?\b",
            text,
        )
    )

    contains_illustration = bool(
        re.search(
            r"(?mi)^\s*Illustrations?\b",
            text,
        )
    )

    metadata = {
        "page_number": page_number,
        "chapter_number": chapter_number,
        "chapter_title": chapter_title,
        "part": extract_part(text),
        "section_numbers": section_numbers,
        "section_titles": section_titles,
        "primary_section": (
            section_numbers[0]
            if section_numbers
            else None
        ),
        "primary_section_title": (
            section_titles[0]
            if section_titles
            else None
        ),
        "contains_explanation": (
            contains_explanation
        ),
        "contains_illustration": (
            contains_illustration
        ),
        "keywords": extract_keywords(text),
        "legal_references": (
            extract_legal_references(text)
        ),
    }

    metadata.update(
        analyze_section_body_presence(text)
    )

    if section_titles:
        metadata["legal_topic"] = (
            section_titles[0].lower()
        )
    elif chapter_title:
        metadata["legal_topic"] = (
            chapter_title.lower()
        )
    else:
        metadata["legal_topic"] = None

    return metadata


# -------------------------------------------------------------------
# Main page cleaning
# -------------------------------------------------------------------

def clean_pdf_page(
    text: str,
    page_number: int | None = None,
    repeated_edge_lines: set[str] | None = None,
) -> CleanedPage:
    """
    Clean one PDF page and return text, metadata, and statistics.

    The cleaner preserves the previous implementation while adding:
    - second-pass OCR repair after paragraph reconstruction;
    - stateful amendment-continuation removal;
    - heading-only page detection;
    - safer handling of inline amendment markers.
    """

    statistics = CleaningStatistics(
        original_characters=len(text or "")
    )

    if not text:
        return CleanedPage(
            text="",
            metadata=build_metadata(
                "",
                page_number=page_number,
            ),
            statistics=statistics,
        )

    text = normalize_unicode_and_whitespace(text)
    text = repair_hyphenation(
        text,
        statistics,
    )
    text = repair_known_ocr_errors(
        text,
        statistics,
    )

    repeated_edge_lines = repeated_edge_lines or set()
    filtered_lines: list[str] = []

    inside_amendment_note = False

    for raw_line in text.splitlines():
        line = re.sub(
            r"[ \t]+",
            " ",
            raw_line,
        ).strip()

        if not line:
            # Keep paragraph boundaries. A blank line also ends a weak
            # amendment continuation unless the next line is clearly
            # another editorial fragment.
            filtered_lines.append("")
            continue

        normalized_frequency_line = (
            normalize_line_for_frequency(line)
        )

        if is_known_header(line):
            statistics.headers_removed += 1
            continue

        if (
            normalized_frequency_line
            in repeated_edge_lines
        ):
            statistics.headers_removed += 1
            continue

        if is_page_footer(line):
            statistics.footers_removed += 1
            continue

        if is_pdf_artifact(line):
            statistics.artifacts_removed += 1
            statistics.footnotes_removed += 1
            continue

        if is_amendment_note(line):
            inside_amendment_note = True
            statistics.amendment_notes_removed += 1
            statistics.footnotes_removed += 1
            continue

        if inside_amendment_note:
            if LEGAL_CONTENT_START_PATTERN.match(line):
                inside_amendment_note = False
            elif is_amendment_continuation(line):
                statistics.amendment_continuation_lines_removed += 1
                statistics.footnotes_removed += 1
                continue
            else:
                # Unknown content after an amendment note is preserved.
                # This avoids accidentally deleting operative legal text.
                inside_amendment_note = False

        cleaned_inline = remove_inline_citation_artifacts(
            line
        )

        if cleaned_inline != line:
            statistics.footnotes_removed += 1

        if not cleaned_inline:
            continue

        parsed_section = split_section_heading_and_body(
            cleaned_inline
        )

        if parsed_section is not None:
            section_number, section_title, inline_body = parsed_section

            if inline_body:
                heading_line = f"{section_number}. {section_title}:"
                filtered_lines.append(heading_line)
                filtered_lines.append("")
                filtered_lines.append(inline_body)
                statistics.sections_detected += 1
                statistics.inline_section_bodies_split += 1
                continue

            # Keep the original line for section headings without inline
            # body text so we preserve existing formatting behavior.
            filtered_lines.append(cleaned_inline)
            statistics.sections_detected += 1
            continue

        if CHAPTER_NUMBER_PATTERN.match(cleaned_inline):
            statistics.chapters_detected += 1

        if PART_PATTERN.match(cleaned_inline):
            statistics.parts_detected += 1

        if EXPLANATION_PATTERN.match(cleaned_inline):
            statistics.explanations_detected += 1

        if ILLUSTRATION_PATTERN.match(cleaned_inline):
            statistics.illustrations_detected += 1

        if ENUMERATION_PATTERN.match(cleaned_inline):
            statistics.enumerations_detected += 1

        filtered_lines.append(cleaned_inline)

    paragraphs: list[str] = []
    current_paragraph: list[str] = []

    def save_paragraph() -> None:
        if current_paragraph:
            paragraphs.append(
                " ".join(
                    current_paragraph
                ).strip()
            )
            current_paragraph.clear()

    for index, line in enumerate(filtered_lines):
        if not line:
            save_paragraph()
            continue

        previous_paragraph = (
            paragraphs[-1]
            if paragraphs
            else ""
        )

        follows_chapter_heading = bool(
            CHAPTER_NUMBER_PATTERN.match(
                previous_paragraph
            )
            or is_standalone_chapter_number_line(
                previous_paragraph
            )
        )

        chapter_heading = get_chapter_heading_at_index(
            filtered_lines,
            index,
        )

        if chapter_heading and is_standalone_chapter_number_line(line):
            save_paragraph()
            paragraphs.append(line)
            statistics.chapters_detected += 1
            statistics.standalone_chapters_detected += 1
            continue

        if (
            is_legal_heading(line)
            or (
                follows_chapter_heading
                and is_probable_chapter_title(line)
            )
        ):
            save_paragraph()
            paragraphs.append(line)
            continue

        current_paragraph.append(line)

    save_paragraph()

    cleaned_text = "\n\n".join(
        paragraph
        for paragraph in paragraphs
        if paragraph
    )

    # Second OCR pass: some joined words only become visible after lines
    # are reconstructed into paragraphs.
    cleaned_text = repair_known_ocr_errors(
        cleaned_text,
        statistics,
    )

    # Punctuation and spacing normalization.
    cleaned_text = re.sub(
        r"\s+([,.;:?!])",
        r"\1",
        cleaned_text,
    )

    cleaned_text = re.sub(
        r"([,.;:?!])(?=[A-Za-z])",
        r"\1 ",
        cleaned_text,
    )

    cleaned_text = re.sub(
        r"\(\s+",
        "(",
        cleaned_text,
    )

    cleaned_text = re.sub(
        r"\s+\)",
        ")",
        cleaned_text,
    )

    cleaned_text = re.sub(
        r"(?<=[A-Za-z])(?=\d{2,4}\b)",
        " ",
        cleaned_text,
    )

    cleaned_text = re.sub(
        r"[ \t]{2,}",
        " ",
        cleaned_text,
    )

    cleaned_text = re.sub(
        r"\n{3,}",
        "\n\n",
        cleaned_text,
    ).strip()

    statistics.cleaned_characters = len(
        cleaned_text
    )

    statistics.characters_removed = max(
        0,
        statistics.original_characters
        - statistics.cleaned_characters,
    )

    metadata = build_metadata(
        cleaned_text,
        page_number=page_number,
    )

    if metadata.get("heading_only_page"):
        statistics.heading_only_pages_detected += 1

    return CleanedPage(
        text=cleaned_text,
        metadata=metadata,
        statistics=statistics,
    )


# -------------------------------------------------------------------
# Backward-compatible APIs
# -------------------------------------------------------------------

def clean_pdf_text(text: str) -> str:
    """
    Backward-compatible API returning only cleaned text.
    """

    return clean_pdf_page(text).text


def clean_pdf_pages(
    pages: list[str],
    frequency_threshold: float = 0.80,
) -> list[CleanedPage]:
    """
    Clean all PDF pages with repeated header/footer detection.
    """

    repeated_lines = detect_repeated_page_lines(
        pages,
        frequency_threshold=frequency_threshold,
    )

    cleaned_pages: list[CleanedPage] = []

    for page_number, page_text in enumerate(
        pages,
        start=1,
    ):
        cleaned_pages.append(
            clean_pdf_page(
                text=page_text,
                page_number=page_number,
                repeated_edge_lines=repeated_lines,
            )
        )

    return cleaned_pages


# -------------------------------------------------------------------
# Statistics
# -------------------------------------------------------------------

def combine_statistics(
    pages: list[CleanedPage],
) -> CleaningStatistics:
    """Combine page-level statistics into one summary."""

    combined = CleaningStatistics()

    for page in pages:
        stats = page.statistics

        for field_name in combined.__dataclass_fields__:
            setattr(
                combined,
                field_name,
                getattr(combined, field_name)
                + getattr(stats, field_name),
            )

    return combined


def display_cleaning_statistics(
    pages: list[CleanedPage],
) -> None:
    """Display combined cleaning statistics."""

    summary = combine_statistics(pages)

    print("\n" + "=" * 70)
    print("TEXT CLEANING SUMMARY")
    print("=" * 70)

    for key, value in summary.as_dict().items():
        label = key.replace("_", " ").title()
        print(f"{label}: {value}")


def _self_test() -> None:
    """Small sanity checks for the requested PDF edge cases."""

    chapter_number, chapter_title = extract_chapter_details(
        "XVI\nOF OFFENCES AFFECTING THE HUMAN BODY"
    )
    assert chapter_number == "XVI"
    assert chapter_title == "OF OFFENCES AFFECTING THE HUMAN BODY"

    parsed_section = split_section_heading_and_body(
        "298-C. Person of Quadiani group, etc., calling himself a Muslim or preaching or propagating his faith: Any person..."
    )
    assert parsed_section is not None
    assert parsed_section[0] == "298-C"
    assert parsed_section[1] == (
        "Person of Quadiani group, etc., calling himself a Muslim or preaching or propagating his faith"
    )
    assert parsed_section[2].startswith("Any person")

    cleaned_case_3 = clean_pdf_text(
        "Every person shall be liable to punishment under this Codeand not otherwise..."
    )
    assert "Code and" in cleaned_case_3

    cleaned_case_4 = clean_pdf_text(
        "written,or by visible representation"
    )
    assert "written, or by visible representation" in cleaned_case_4

    assert extract_chapter_details("(i) a subsection") == (None, None)


if __name__ == "__main__":
    _self_test()
    print("text_cleaner self-test passed.")
