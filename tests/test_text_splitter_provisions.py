from langchain_core.documents import Document

from rag.text_splitter import create_chunks, parse_provision_blocks


def _make_ata_pages() -> list[Document]:
    base_metadata = {
        "document_id": "ata_1997",
        "document_name": "Anti-Terrorism Act, 1997",
        "document_title": "Anti-Terrorism Act, 1997",
        "document_short_name": "ATA",
        "document_type": "special_criminal_law",
        "provision_type": "section",
    }

    return [
        Document(
            page_content=(
                "Anti-Terrorism Act, 1997\n\n"
                "1Subs. This editorial marker must not be treated as a section heading.\n\n"
                "6. Punishment for unlawful assembly\n"
                "Whoever participates in an unlawful assembly under this Act commits an offence. "
                "The text here is long enough to remain a usable section block and should stop "
                "before the next valid section begins."
            ),
            metadata={**base_metadata, "page_number": 1, "page": 0},
        ),
        Document(
            page_content=(
                "7. Punishment for acts of terrorism\n"
                "Whoever does any act of terrorism shall be punished under this section. "
                "This section has its own standalone body text and must not absorb the next section."
            ),
            metadata={**base_metadata, "page_number": 2, "page": 1},
        ),
        Document(
            page_content=(
                "8. Another unrelated ATA section\n"
                "Whoever performs another act covered by the Act commits a separate offence. "
                "This later section should remain outside Section 7."
            ),
            metadata={**base_metadata, "page_number": 3, "page": 2},
        ),
    ]


def _make_ata_footnote_pages() -> list[Document]:
    base_metadata = {
        "document_id": "ata_1997",
        "document_name": "Anti-Terrorism Act, 1997",
        "document_title": "Anti-Terrorism Act, 1997",
        "document_short_name": "ATA",
        "document_type": "special_criminal_law",
        "provision_type": "section",
    }

    return [
        Document(
            page_content=(
                "Anti-Terrorism Act, 1997\n\n"
                "1Subs. This editorial marker must not be treated as a section heading.\n\n"
                "5. Punishment for assembly\n"
                "Whoever commits the offence mentioned in this section shall be punished. "
                "The text ends before the next valid provision begins."
            ),
            metadata={**base_metadata, "page_number": 1, "page": 0},
        ),
        Document(
            page_content=(
                "6. Another ATA section\n"
                "Whoever does the act mentioned in this section commits a separate offence."
            ),
            metadata={**base_metadata, "page_number": 2, "page": 1},
        ),
        Document(
            page_content=(
                "4[7. Punishment for acts of terrorism\n"
                "(1) Whoever commits an act of terrorism shall be punished under this section. "
                "This body text must remain attached to Section 7 only."
            ),
            metadata={**base_metadata, "page_number": 3, "page": 2},
        ),
        Document(
            page_content=(
                "8. Another unrelated ATA section\n"
                "Whoever performs another act covered by the Act commits a separate offence."
            ),
            metadata={**base_metadata, "page_number": 4, "page": 3},
        ),
    ]


def test_ata_footnote_marker_is_not_treated_as_a_section() -> None:
    blocks, _ = parse_provision_blocks(_make_ata_pages())

    provision_numbers = [
        block.provision_number
        for block in blocks
        if block.provision_number
    ]

    assert "1SUBS" not in provision_numbers
    assert "6" in provision_numbers
    assert "7" in provision_numbers


def test_ata_section_7_chunk_has_correct_metadata_and_page_range() -> None:
    chunks = create_chunks(_make_ata_pages())

    section_7_chunks = [
        chunk
        for chunk in chunks
        if chunk.metadata.get("provision_number") == "7"
        and chunk.metadata.get("provision_type") == "section"
    ]

    assert len(section_7_chunks) == 1

    chunk = section_7_chunks[0]

    assert chunk.metadata["section_number"] == "7"
    assert chunk.metadata["provision_number"] == "7"
    assert chunk.metadata["page_start"] == 2
    assert chunk.metadata["page_end"] == 2
    assert chunk.metadata["source_pages"] == [2]
    assert chunk.metadata["chunk_id"].startswith(
        "ata_1997::section::7::part-1::chunk-"
    )
    assert "Section 8" not in chunk.page_content


def test_ata_footnote_wrapper_normalizes_section_7_and_keeps_sections_separate() -> None:
    blocks, _ = parse_provision_blocks(_make_ata_footnote_pages())

    provision_numbers = [
        block.provision_number
        for block in blocks
        if block.provision_number
    ]

    assert "1SUBS" not in provision_numbers
    assert provision_numbers == ["5", "6", "7", "8"]

    chunks = create_chunks(_make_ata_footnote_pages())

    section_7_chunks = [
        chunk
        for chunk in chunks
        if chunk.metadata.get("provision_number") == "7"
        and chunk.metadata.get("provision_type") == "section"
    ]

    section_5_chunks = [
        chunk
        for chunk in chunks
        if chunk.metadata.get("provision_number") == "5"
        and chunk.metadata.get("provision_type") == "section"
    ]

    assert len(section_7_chunks) == 1
    assert len(section_5_chunks) == 1

    section_7_chunk = section_7_chunks[0]
    section_5_chunk = section_5_chunks[0]

    assert section_7_chunk.metadata["document_id"] == "ata_1997"
    assert section_7_chunk.metadata["provision_type"] == "section"
    assert section_7_chunk.metadata["provision_number"] == "7"
    assert section_7_chunk.metadata["section_number"] == "7"
    assert section_7_chunk.metadata["page_start"] == 3
    assert section_7_chunk.metadata["page_end"] == 3
    assert "Punishment for acts of terrorism" in section_7_chunk.page_content
    assert "7. Punishment for acts of terrorism" not in section_5_chunk.page_content
