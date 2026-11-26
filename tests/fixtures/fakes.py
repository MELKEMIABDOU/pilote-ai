"""Faux Docling pour tester l'extracteur sans installer l'extra ``rag``."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace


class FakeDocument:
    def __init__(self, pages: dict[int, object] | None = None) -> None:
        self.pages = pages or {}

    def export_to_markdown(self, *, page_no: int | None = None) -> str:
        if page_no is None:
            return "document content"
        return f"page {page_no} content"


class FakeConverter:
    def __init__(self, document: FakeDocument | None = None, error: Exception | None = None):
        self.document = document or FakeDocument()
        self.error = error
        self.converted_source: Path | None = None

    def convert(self, source: Path) -> SimpleNamespace:
        self.converted_source = source
        if self.error:
            raise self.error
        return SimpleNamespace(document=self.document)


def fake_chunk(
    text: str,
    page: int | None = None,
    headings: tuple[str, ...] = (),
    captions: tuple[str, ...] = (),
) -> SimpleNamespace:
    """Imite un ``DocChunk`` Docling réel.

    Comme chez Docling : ``text`` ne contient PAS les titres ; ils sont dans ``meta.headings``
    (``None`` s'il n'y en a pas), la page dans ``meta.doc_items[].prov[].page_no``.
    """
    provenance = [SimpleNamespace(page_no=page)] if page is not None else []
    return SimpleNamespace(
        text=text,
        meta=SimpleNamespace(
            headings=list(headings) or None,
            captions=list(captions) or None,
            doc_items=[SimpleNamespace(prov=provenance)],
        ),
    )


class FakeChunker:
    def __init__(self, chunks: list[SimpleNamespace] | None = None) -> None:
        self.chunks = chunks or []

    def chunk(self, document: FakeDocument) -> list[SimpleNamespace]:
        return self.chunks
