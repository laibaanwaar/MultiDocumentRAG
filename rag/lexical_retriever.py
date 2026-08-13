from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass
from typing import Any, Iterable

from langchain_core.documents import Document

from rag.intent_router import normalize_text, tokenize


def _normalize_document_id(document: Document) -> str:
    return str(
        document.metadata.get("document_id", "")
    ).strip()


def _is_usable_document(document: Document) -> bool:
    body_present = document.metadata.get(
        "provision_body_present",
        document.metadata.get(
            "section_body_present",
            True,
        ),
    )

    return bool(
        document.page_content.strip()
        and not document.metadata.get(
            "heading_only_chunk",
            False,
        )
        and body_present is not False
    )


def _tokenize_document(document: Document) -> list[str]:
    searchable_text = " ".join(
        (
            str(
                document.metadata.get(
                    "document_title",
                    "",
                )
            ),
            str(
                document.metadata.get(
                    "document_short_name",
                    "",
                )
            ),
            str(
                document.metadata.get(
                    "provision_title",
                    document.metadata.get(
                        "section_title",
                        document.metadata.get(
                            "article_title",
                            "",
                        ),
                    ),
                )
            ),
            document.page_content,
        )
    )

    return tokenize(searchable_text)


@dataclass(slots=True)
class LexicalEntry:
    document: Document
    term_frequencies: Counter[str]
    document_length: int
    normalized_text: str
    document_id: str


@dataclass(slots=True)
class LexicalIndex:
    entries: list[LexicalEntry]
    document_frequencies: Counter[str]
    average_document_length: float

    @classmethod
    def from_documents(
        cls,
        documents: Iterable[Document],
    ) -> "LexicalIndex":
        usable_documents = [
            document
            for document in documents
            if _is_usable_document(document)
        ]

        entries: list[LexicalEntry] = []
        document_frequencies: Counter[str] = Counter()
        total_length = 0

        for document in usable_documents:
            tokens = _tokenize_document(document)

            if not tokens:
                continue

            term_frequencies = Counter(tokens)
            total_length += len(tokens)

            entries.append(
                LexicalEntry(
                    document=document,
                    term_frequencies=term_frequencies,
                    document_length=len(tokens),
                    normalized_text=normalize_text(
                        " ".join(
                            (
                                str(
                                    document.metadata.get(
                                        "document_title",
                                        "",
                                    )
                                ),
                                str(
                                    document.metadata.get(
                                        "document_short_name",
                                        "",
                                    )
                                ),
                                str(
                                    document.metadata.get(
                                        "provision_title",
                                        document.metadata.get(
                                            "section_title",
                                            document.metadata.get(
                                                "article_title",
                                                "",
                                            ),
                                        ),
                                    )
                                ),
                                document.page_content,
                            )
                        )
                    ),
                    document_id=_normalize_document_id(
                        document
                    ),
                )
            )

            document_frequencies.update(
                term_frequencies.keys()
            )

        return cls(
            entries=entries,
            document_frequencies=document_frequencies,
            average_document_length=(
                total_length / len(entries)
                if entries
                else 0.0
            ),
        )

    def search(
        self,
        query: str,
        k: int,
        document_ids: list[str] | None = None,
    ) -> list[tuple[Document, float]]:
        if k <= 0:
            return []

        normalized_query = normalize_text(
            query
        ).strip()
        query_tokens = tokenize(
            query
        )

        if not normalized_query and not query_tokens:
            return []

        normalized_document_ids = {
            str(document_id).strip()
            for document_id in (document_ids or [])
            if str(document_id).strip()
        }

        query_counts = Counter(query_tokens)
        scored_documents: list[tuple[Document, float]] = []
        document_count = max(
            1,
            len(self.entries),
        )

        for entry in self.entries:
            if (
                normalized_document_ids
                and entry.document_id
                not in normalized_document_ids
            ):
                continue

            score = self._score_entry(
                entry=entry,
                query_counts=query_counts,
                normalized_query=normalized_query,
                document_count=document_count,
            )

            if score <= 0.0:
                continue

            scored_documents.append(
                (
                    entry.document,
                    score,
                )
            )

        scored_documents.sort(
            key=lambda item: item[1],
            reverse=True,
        )

        return scored_documents[:k]

    def _score_entry(
        self,
        entry: LexicalEntry,
        query_counts: Counter[str],
        normalized_query: str,
        document_count: int,
    ) -> float:
        if not query_counts:
            return 0.0

        if self.average_document_length <= 0:
            average_length = 1.0
        else:
            average_length = self.average_document_length

        k1 = 1.5
        b = 0.75
        score = 0.0
        length_norm = k1 * (
            1
            - b
            + b
            * (
                entry.document_length
                / average_length
            )
        )

        for term, query_frequency in query_counts.items():
            term_frequency = entry.term_frequencies.get(
                term,
                0,
            )

            if term_frequency <= 0:
                continue

            document_frequency = (
                self.document_frequencies.get(
                    term,
                    0,
                )
            )

            idf = math.log1p(
                (
                    document_count
                    - document_frequency
                    + 0.5
                )
                / (
                    document_frequency
                    + 0.5
                )
            )

            score += (
                idf
                * (
                    term_frequency
                    * (k1 + 1)
                    / (
                        term_frequency
                        + length_norm
                    )
                )
                * query_frequency
            )

        if normalized_query and normalized_query in entry.normalized_text:
            score += min(
                0.5,
                0.05 * max(
                    1,
                    len(query_counts),
                ),
            )

        return score
