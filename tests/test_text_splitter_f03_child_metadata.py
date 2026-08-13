from langchain_core.documents import Document

from rag.text_splitter import create_chunks


def _make_child_structure_pages() -> list[Document]:
    base_metadata = {
        "document_id": "ppc_1860",
        "document_name": "Pakistan Penal Code, 1860",
        "document_title": "Pakistan Penal Code, 1860",
        "document_short_name": "PPC",
        "document_type": "legal_document",
        "provision_type": "section",
        "page_number": 1,
        "page": 0,
    }

    return [
        Document(
            page_content=(
                "11EE. Proscription of person\n"
                "(1) Parent subsection one text.\n"
                "(2) Parent subsection two text.\n"
                "(a) First condition.\n"
                "(b) Second condition.\n"
                "(3) Parent subsection three text."
            ),
            metadata=base_metadata,
        )
    ]


def _make_plain_section_pages() -> list[Document]:
    base_metadata = {
        "document_id": "ppc_1860",
        "document_name": "Pakistan Penal Code, 1860",
        "document_title": "Pakistan Penal Code, 1860",
        "document_short_name": "PPC",
        "document_type": "legal_document",
        "provision_type": "section",
        "page_number": 1,
        "page": 0,
    }

    return [
        Document(
            page_content=(
                "7. Simple provision\n"
                "Whoever does a thing under this section is punished by law, "
                "and the punishment continues to apply across repeated acts "
                "without changing the basic section-level structure."
            ),
            metadata=base_metadata,
        )
    ]


def test_child_chunks_carry_hierarchical_metadata() -> None:
    chunks = create_chunks(_make_child_structure_pages())

    child_chunks = [
        chunk
        for chunk in chunks
        if chunk.metadata.get("subsection_path")
    ]

    assert {
        chunk.metadata["subsection_path_key"]
        for chunk in child_chunks
    } == {"1", "2", "2.a", "2.b", "3"}

    subsection_1 = next(
        chunk
        for chunk in child_chunks
        if chunk.metadata["subsection_path_key"] == "1"
    )
    subsection_2 = next(
        chunk
        for chunk in child_chunks
        if chunk.metadata["subsection_path_key"] == "2"
    )
    clause_a = next(
        chunk
        for chunk in child_chunks
        if chunk.metadata["subsection_path_key"] == "2.a"
    )
    clause_b = next(
        chunk
        for chunk in child_chunks
        if chunk.metadata["subsection_path_key"] == "2.b"
    )
    subsection_3 = next(
        chunk
        for chunk in child_chunks
        if chunk.metadata["subsection_path_key"] == "3"
    )

    assert subsection_1.metadata["subsection_path"] == ["1"]
    assert subsection_2.metadata["subsection_path"] == ["2"]
    assert clause_a.metadata["subsection_path"] == ["2", "a"]
    assert clause_b.metadata["subsection_path"] == ["2", "b"]
    assert subsection_3.metadata["subsection_path"] == ["3"]

    assert subsection_1.metadata["component_type"] == "subsection"
    assert subsection_2.metadata["component_type"] == "subsection"
    assert clause_a.metadata["component_type"] == "clause"
    assert clause_b.metadata["component_type"] == "clause"
    assert subsection_3.metadata["component_type"] == "subsection"

    assert clause_a.metadata["component_label"] == "(a)"
    assert clause_b.metadata["component_label"] == "(b)"
    assert clause_a.metadata["base_provision_number"] == "11EE"
    assert clause_b.metadata["base_provision_number"] == "11EE"

    for chunk in child_chunks:
        assert chunk.metadata["provision_number"] == "11EE"
        assert chunk.metadata["document_id"] == "ppc_1860"
        assert chunk.metadata["provision_type"] == "section"
        assert chunk.metadata["section_number"] == "11EE"
        assert chunk.metadata["provision_identity"].startswith(
            "ppc_1860::section::11EE"
        )


def test_sections_without_children_keep_existing_chunk_shape() -> None:
    chunks = create_chunks(_make_plain_section_pages())

    assert len(chunks) == 1
    assert chunks[0].metadata["provision_number"] == "7"
    assert chunks[0].metadata["document_id"] == "ppc_1860"
    assert chunks[0].metadata["provision_type"] == "section"
    assert not chunks[0].metadata.get("subsection_path")
