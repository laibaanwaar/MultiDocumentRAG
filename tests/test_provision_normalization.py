import pytest

from rag.intent_router import (
    extract_article_numbers,
    extract_legal_references,
    extract_section_numbers,
)
from rag.legal_reference_normalizer import canonicalize_provision_number
from rag.retriever import normalize_provision_number as retriever_normalize_provision_number
from rag.text_splitter import normalize_provision_number as splitter_normalize_provision_number


@pytest.mark.parametrize(
    ("raw_value", "expected"),
    [
        ("9-A", "9A"),
        ("9A", "9A"),
        (" 9 - A ", "9A"),
        ("Section 9-A", "9A"),
        ("sec. 9A", "9A"),
        ("s 9-A", "9A"),
        ("153-A", "153A"),
        ("298-B", "298B"),
        ("11EE", "11EE"),
        ("Article 10", "10"),
        ("Section 11EE(2)(b)", "11EE"),
        ("not a citation", None),
    ],
)
def test_canonicalize_provision_number_table(raw_value, expected) -> None:
    assert canonicalize_provision_number(raw_value) == expected


def test_query_parsing_uses_the_shared_canonical_base() -> None:
    assert extract_section_numbers("What does s 9-A provide?") == ["9A"]
    assert extract_section_numbers("Compare Section 9A and sec. 9-A") == [
        "9A"
    ]
    for citation in ("art 10(2)", "art. 10(2)", "article 10(2)", "Article 10(2)"):
        references = extract_legal_references(citation)

        assert len(references) == 1
        assert references[0].provision_type == "article"
        assert references[0].base_number == "10"
        assert references[0].subsection_path == ["2"]
        assert references[0].component_type == "subsection"
        assert references[0].original_citation == citation
    assert extract_article_numbers("What does art. 10(2) provide?") == ["10"]

    references = extract_legal_references(
        "What does Section 11EE(2)(b) provide?"
    )

    assert len(references) == 1
    assert references[0].base_number == "11EE"
    assert references[0].subsection_path == ["2", "b"]


def test_shared_canonicalizer_is_consistent_across_layers() -> None:
    raw_value = "Section 9 - A"

    assert (
        canonicalize_provision_number(raw_value)
        == splitter_normalize_provision_number(raw_value)
        == retriever_normalize_provision_number(raw_value)
        == "9A"
    )
