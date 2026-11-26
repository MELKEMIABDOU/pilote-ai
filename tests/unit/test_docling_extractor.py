from pathlib import Path

import pytest

from app.adapters.extraction.docling import DoclingExtractor
from app.domain.errors import ExtractionError, UnsupportedDocumentError
from tests.fixtures.fakes import FakeChunker, FakeConverter, FakeDocument, fake_chunk


def make_extractor(
    converter: FakeConverter, chunker: FakeChunker | None = None
) -> DoclingExtractor:
    return DoclingExtractor(converter=converter, chunker=chunker or FakeChunker())


def test_extract_preserves_page_numbers_and_uses_a_safe_source_label(tmp_path: Path) -> None:
    source = tmp_path / "order.pdf"
    source.write_bytes(b"document")
    converter = FakeConverter(FakeDocument({2: object(), 1: object()}))

    result = make_extractor(converter).extract(source)

    assert converter.converted_source == source.resolve()
    assert result.source == "order.pdf"
    assert result.source_type == "file"
    assert result.page_count == 2
    assert [page.page_number for page in result.pages] == [1, 2]
    assert "<!-- page 1 -->\npage 1 content" in result.markdown
    assert "<!-- page 2 -->\npage 2 content" in result.markdown


def test_extract_keeps_markdown_when_docling_has_no_page_metadata(tmp_path: Path) -> None:
    source = tmp_path / "email.html"
    source.write_text("<p>Order details</p>", encoding="utf-8")

    result = make_extractor(FakeConverter(FakeDocument())).extract(source)

    assert result.page_count is None
    assert result.markdown == "document content"


def test_extract_marks_eml_files_as_emails(tmp_path: Path) -> None:
    source = tmp_path / "commande.eml"
    source.write_text("Subject: commande", encoding="utf-8")

    assert make_extractor(FakeConverter()).extract(source).source_type == "email"


def test_extract_returns_numbered_chunks_with_their_page_and_headings(tmp_path: Path) -> None:
    source = tmp_path / "order.pdf"
    source.write_bytes(b"document")
    chunker = FakeChunker(
        [
            fake_chunk("Premier passage.", page=3, headings=("1. Objet",)),
            fake_chunk("Passage sans titre.", page=4),
        ]
    )

    result = make_extractor(FakeConverter(), chunker).extract(source)

    assert [(c.chunk_id, c.page_number, c.text) for c in result.chunks] == [
        (1, 3, "1. Objet\nPremier passage."),
        (2, 4, "Passage sans titre."),
    ]


def test_chunk_spanning_two_pages_keeps_the_whole_page_range(tmp_path: Path) -> None:
    # Cas réel (section 8 du rapport Horizon) : l'introduction est page 4, le tableau page 5.
    source = tmp_path / "order.pdf"
    source.write_bytes(b"document")
    chunker = FakeChunker([fake_chunk("Intro puis tableau.", page=4, more_pages=(5, 5))])

    [chunk] = make_extractor(FakeConverter(), chunker).extract(source).chunks

    assert (chunk.page_number, chunk.page_end) == (4, 5)


def test_extract_rejects_remote_urls_before_calling_docling() -> None:
    converter = FakeConverter()

    with pytest.raises(UnsupportedDocumentError, match="remote URLs are not supported"):
        make_extractor(converter).extract("https://example.test/order.pdf")

    assert converter.converted_source is None


def test_extract_rejects_unsupported_extensions(tmp_path: Path) -> None:
    source = tmp_path / "script.exe"
    source.write_bytes(b"MZ")

    with pytest.raises(UnsupportedDocumentError, match="unsupported file type '.exe'"):
        make_extractor(FakeConverter()).extract(source)


def test_extract_rejects_missing_files(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        make_extractor(FakeConverter()).extract(tmp_path / "absent.pdf")


def test_extract_wraps_docling_failures(tmp_path: Path) -> None:
    source = tmp_path / "broken.pdf"
    source.write_bytes(b"not a pdf")
    converter = FakeConverter(error=RuntimeError("boom"))

    with pytest.raises(ExtractionError, match="broken.pdf"):
        make_extractor(converter).extract(source)


def test_max_tokens_must_be_positive() -> None:
    with pytest.raises(ValueError, match="max_tokens must be positive"):
        DoclingExtractor(max_tokens=0)
