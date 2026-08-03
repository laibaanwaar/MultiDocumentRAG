from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class LegalDocumentConfig:
    document_id: str
    short_name: str
    full_name: str
    year: int
    document_type: str
    provision_type: str
    filename_aliases: tuple[str, ...]


DOCUMENT_REGISTRY: tuple[LegalDocumentConfig, ...] = (
    LegalDocumentConfig(
        document_id="ppc_1860",
        short_name="PPC",
        full_name="Pakistan Penal Code, 1860",
        year=1860,
        document_type="criminal_code",
        provision_type="section",
        filename_aliases=(
            "pakistan penal code",
            "ppc",
        ),
    ),
    LegalDocumentConfig(
        document_id="constitution_1973",
        short_name="Constitution",
        full_name="Constitution of the Islamic Republic of Pakistan, 1973",
        year=1973,
        document_type="constitutional_law",
        provision_type="article",
        filename_aliases=(
            "constitution of pakistan",
            "constitution of pakistan 1973",
            "constitution",
        ),
    ),
    LegalDocumentConfig(
        document_id="ata_1997",
        short_name="ATA",
        full_name="Anti-Terrorism Act, 1997",
        year=1997,
        document_type="special_criminal_law",
        provision_type="section",
        filename_aliases=(
            "anti terrorism act",
            "anti-terrorism act",
            "terrorism act",
        ),
    ),
    LegalDocumentConfig(
        document_id="amla_2010",
        short_name="AMLA",
        full_name="Anti-Money Laundering Act, 2010",
        year=2010,
        document_type="financial_criminal_law",
        provision_type="section",
        filename_aliases=(
            "anti money laundering act",
            "anti-money laundering act",
            "money laundering act",
        ),
    ),
)


def normalize_filename(filename: str) -> str:
    """Normalize a file name for reliable matching."""

    stem = Path(filename).stem.lower()
    stem = re.sub(r"[_\-]+", " ", stem)
    stem = re.sub(r"[^a-z0-9 ]+", " ", stem)
    stem = re.sub(r"\s+", " ", stem)

    return stem.strip()


def resolve_document(filename: str) -> LegalDocumentConfig:
    """Return registered metadata for a PDF file."""

    normalized_name = normalize_filename(filename)

    for config in DOCUMENT_REGISTRY:
        for alias in config.filename_aliases:
            normalized_alias = normalize_filename(alias)

            if normalized_alias in normalized_name:
                return config

    raise ValueError(
        "Document is not registered: "
        f"{filename}\n"
        "Add this PDF to rag/document_registry.py."
    )