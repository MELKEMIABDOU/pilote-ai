"""Contrat d'extraction documentaire.

Le reste de l'application dépend de ce port, jamais directement de Docling.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Protocol

SourceType = Literal["file", "email"]


@dataclass(frozen=True, slots=True)
class ExtractedPage:
    page_number: int | None
    markdown: str


@dataclass(frozen=True, slots=True)
class ExtractedChunk:
    chunk_id: int
    page_number: int | None
    text: str


@dataclass(frozen=True, slots=True)
class ExtractedDocument:
    source: str
    source_type: SourceType
    file_type: str
    pages: tuple[ExtractedPage, ...]
    chunks: tuple[ExtractedChunk, ...] = ()

    @property
    def markdown(self) -> str:
        return "\n\n".join(
            f"<!-- page {page.page_number} -->\n{page.markdown}"
            if page.page_number is not None
            else page.markdown
            for page in self.pages
        )

    @property
    def page_count(self) -> int | None:
        page_count = sum(page.page_number is not None for page in self.pages)
        return page_count or None


class DocumentExtractor(Protocol):
    """Transforme un fichier local en document structuré (Markdown par page + passages)."""

    def extract(self, source: str | Path) -> ExtractedDocument: ...
