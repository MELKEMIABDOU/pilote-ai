from pathlib import Path
from types import SimpleNamespace

import pytest

from app.rag.extraction import DoclingExtractor


class FakeDocument:
    def __init__(self, pages: dict[int, object]) -> None:
        self.pages = pages

    def export_to_markdown(self, *, page_no: int | None = None) -> str:
        if page_no is None:
            return "document content"
        return f"page {page_no} content"


class FakeConverter:
    def __init__(self, document: FakeDocument) -> None:
        self.document = document
        self.converted_source: Path | None = None

    def convert(self, source: Path) -> SimpleNamespace:
        self.converted_source = source
        return SimpleNamespace(document=self.document)


class FakeChunker:
    def chunk(self, document: FakeDocument) -> list[object]:
        return []

    def contextualize(self, chunk: object) -> str:
        return ""


def test_extract_preserves_page_numbers_and_uses_a_safe_source_label(tmp_path: Path) -> None:
    source = tmp_path / "order.pdf"
    source.write_bytes(b"document")
    converter = FakeConverter(FakeDocument({2: object(), 1: object()}))

    result = DoclingExtractor(converter=converter, chunker=FakeChunker()).extract(source)

    assert converter.converted_source == source.resolve()
    assert result.source == "order.pdf"
    assert result.page_count == 2
    assert [page.page_number for page in result.pages] == [1, 2]
    assert "<!-- page 1 -->\npage 1 content" in result.markdown
    assert "<!-- page 2 -->\npage 2 content" in result.markdown


def test_extract_rejects_remote_urls_before_calling_docling() -> None:
    converter = FakeConverter(FakeDocument({}))

    with pytest.raises(ValueError, match="remote URLs are not supported"):
        DoclingExtractor(converter=converter, chunker=FakeChunker()).extract(
            "https://example.test/order.pdf"
        )

    assert converter.converted_source is None


def test_extract_keeps_markdown_when_docling_has_no_page_metadata(tmp_path: Path) -> None:
    source = tmp_path / "email.html"
    source.write_text("<p>Order details</p>", encoding="utf-8")
    converter = FakeConverter(FakeDocument({}))

    result = DoclingExtractor(converter=converter, chunker=FakeChunker()).extract(source)

    assert result.page_count is None
    assert result.markdown == "document content"
