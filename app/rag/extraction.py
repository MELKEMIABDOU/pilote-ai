"""Minimal Docling-based extraction for local files."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal
from urllib.parse import urlsplit


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
    source_type: Literal["file", "email"]
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


class DoclingExtractor:
    """Extract page-aware Markdown and Qwen-tokenized HybridChunker passages."""

    SUPPORTED_EXTENSIONS = {
        ".pdf",
        ".doc",
        ".docx",
        ".xls",
        ".xlsx",
        ".html",
        ".htm",
        ".eml",
        ".msg",
    }

    def __init__(
        self,
        converter: Any | None = None,
        chunker: Any | None = None,
        tokenizer_model: str = "Qwen/Qwen3-Embedding-0.6B",
        max_tokens: int = 512,
    ) -> None:
        if max_tokens < 1:
            raise ValueError("max_tokens must be positive")
        self._converter = converter
        self._chunker = chunker
        self._tokenizer_model = tokenizer_model
        self._max_tokens = max_tokens

    def extract(self, source: str | Path) -> ExtractedDocument:
        converted_source, source_label, source_type, file_type = self._prepare_source(source)
        result = self._load_converter().convert(converted_source)
        document = result.document
        chunker = self._load_chunker()
        document_pages = getattr(document, "pages", None)
        if document_pages:
            pages = tuple(
                ExtractedPage(
                    page_number=page_number,
                    markdown=document.export_to_markdown(page_no=page_number),
                )
                for page_number in sorted(document_pages)
            )
        else:
            pages = (ExtractedPage(page_number=None, markdown=document.export_to_markdown()),)
        raw_chunks = [
            (chunker.contextualize(chunk).strip(), _chunk_page_number(chunk))
            for chunk in chunker.chunk(document)
        ]
        extracted_chunks = _clean_and_repair_chunks(raw_chunks)
        return ExtractedDocument(
            source=source_label,
            source_type=source_type,
            file_type=file_type,
            pages=pages,
            chunks=tuple(extracted_chunks),
        )

    def _load_converter(self) -> Any:
        if self._converter is None:
            try:
                from docling.document_converter import DocumentConverter
            except ImportError as exc:
                raise RuntimeError(
                    "Docling is optional; install it with `uv sync --extra rag`."
                ) from exc
            self._converter = DocumentConverter()
        return self._converter

    def _load_chunker(self) -> Any:
        if self._chunker is None:
            try:
                from docling.chunking import HybridChunker
                from docling_core.transforms.chunker.tokenizer.huggingface import (
                    HuggingFaceTokenizer,
                )
                from transformers import AutoTokenizer
            except ImportError as exc:
                raise RuntimeError(
                    "HybridChunker needs the optional 'rag' dependencies; "
                    "install them with `uv sync --extra rag`."
                ) from exc

            tokenizer = HuggingFaceTokenizer(
                tokenizer=AutoTokenizer.from_pretrained(self._tokenizer_model),
                max_tokens=self._max_tokens,
            )
            self._chunker = HybridChunker(tokenizer=tokenizer)
        return self._chunker

    @classmethod
    def _prepare_source(
        cls, source: str | Path
    ) -> tuple[Path, str, Literal["file", "email"], str]:
        source_value = str(source)
        parsed = urlsplit(source_value)
        if parsed.scheme.lower() in {"http", "https"}:
            raise ValueError(
                "remote URLs are not supported; save email attachments locally first"
            )

        path = Path(source).expanduser().resolve(strict=True)
        if not path.is_file():
            raise ValueError(f"source is not a regular file: {path}")
        file_type = path.suffix.lower()
        if file_type not in cls.SUPPORTED_EXTENSIONS:
            supported = ", ".join(sorted(cls.SUPPORTED_EXTENSIONS))
            raise ValueError(f"unsupported file type {file_type!r}; supported: {supported}")
        source_type: Literal["file", "email"] = (
            "email" if file_type in {".eml", ".msg"} else "file"
        )
        return path, path.name, source_type, file_type


def _chunk_page_number(chunk: Any) -> int | None:
    metadata = getattr(chunk, "meta", None)
    for item in getattr(metadata, "doc_items", ()) or ():
        for provenance in getattr(item, "prov", ()) or ():
            page_number = getattr(provenance, "page_no", None)
            if page_number is not None:
                return int(page_number)
    return None


_HEADING = re.compile(r"^#{1,6}\s+(.+?)\s*#*\s*$")
_NUMBERED_SECTION = re.compile(r"^#{1,6}\s+(?:\d+\.\s+|annexe\b)", re.IGNORECASE)
_TOC_ENTRY = re.compile(r"^\s*\d+\.\s+\d+\.\s+")
_TABLE_SEPARATOR = re.compile(r"^\s*\|?(?:\s*:?-{1,}:?\s*\|)+\s*:?-{0,}:?\s*\|?\s*$")
_RUNNING_HEADER = re.compile(
    r"magasin horizon\s*\|\s*rapport de gestion des produits", re.IGNORECASE
)
_INCOMPLETE_RECORD = re.compile(
    r"(?:\bP\d{3}\s*,\s*$|\bP\d{3}\s*,\s*[^,=\n]{1,40}\s*=)\s*$",
    re.IGNORECASE,
)
_VALUE_FRAGMENT = re.compile(r"^\s*(?:\d+[,.]\d+\s*€|\d+\s*(?:€|%))", re.IGNORECASE)


def _clean_and_repair_chunks(
    raw_chunks: list[tuple[str, int | None]],
) -> tuple[ExtractedChunk, ...]:
    cleaned: list[tuple[str, int | None]] = []
    active_heading: str | None = None
    in_toc = False

    for raw_text, page_number in raw_chunks:
        lines = _clean_chunk_lines(raw_text)
        kept: list[str] = []
        for line in lines:
            heading_match = _HEADING.match(line)
            if heading_match:
                heading = heading_match.group(1).strip()
                if heading.casefold() == "sommaire":
                    in_toc = True
                    continue
                if _RUNNING_HEADER.search(heading):
                    continue
                if _NUMBERED_SECTION.match(line):
                    in_toc = False
                active_heading = heading

            if in_toc:
                if _NUMBERED_SECTION.match(line):
                    in_toc = False
                else:
                    continue
            if _TOC_ENTRY.match(line):
                continue
            kept.append(line)

        text = "\n".join(kept).strip()
        if not text:
            continue
        if active_heading and not any(_HEADING.match(line) for line in kept):
            text = f"## {active_heading}\n\n{text}"
        cleaned.append((text, page_number))

    repaired: list[tuple[str, int | None]] = []
    for text, page_number in cleaned:
        if repaired and _should_join_chunks(repaired[-1][0], text):
            previous_text, previous_page = repaired[-1]
            repaired[-1] = (f"{previous_text.rstrip()}\n{text.lstrip()}", previous_page or page_number)
        else:
            repaired.append((text, page_number))

    return tuple(
        ExtractedChunk(chunk_id=index, page_number=page, text=text)
        for index, (text, page) in enumerate(repaired, start=1)
    )


def _clean_chunk_lines(text: str) -> list[str]:
    cleaned: list[str] = []
    for line in text.splitlines():
        if _TABLE_SEPARATOR.match(line) or re.fullmatch(r"\s*\|\s*-\s*\|\s*", line):
            continue
        if not line.lstrip().startswith("|") and line.rstrip().endswith("|"):
            line = line.rstrip()[:-1].rstrip()
        cleaned.append(line)
    return cleaned


def _should_join_chunks(left: str, right: str) -> bool:
    left_body = left.rstrip()
    right_body = right.lstrip()
    if _INCOMPLETE_RECORD.search(left_body) or left_body.endswith(("=", ",")):
        return True
    if _VALUE_FRAGMENT.match(right_body) and left_body.endswith(("=", ":")):
        return True
    # Keep a short sentence with a table that resumes on the next page.
    right_starts_table = any(line.lstrip().startswith("|") for line in right_body.splitlines()[:3])
    left_is_short_prose = len(left_body) <= 220 and not any(
        line.lstrip().startswith("|") for line in left_body.splitlines()
    )
    return left_is_short_prose and right_starts_table
