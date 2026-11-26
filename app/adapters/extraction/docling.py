"""Adaptateur d'extraction basé sur Docling + HybridChunker (tokenizer Qwen3)."""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

from app.domain.errors import ExtractionError, MissingDependencyError, UnsupportedDocumentError
from app.ports.extraction import (
    ExtractedChunk,
    ExtractedDocument,
    ExtractedPage,
    SourceType,
)

logger = logging.getLogger(__name__)

DEFAULT_TOKENIZER_MODEL = "Qwen/Qwen3-Embedding-0.6B"
DEFAULT_MAX_TOKENS = 512


@dataclass(frozen=True, slots=True)
class ChunkCleaningRules:
    """Règles de nettoyage propres à une famille de documents.

    Par défaut, rien de spécifique à un client : on fournit des règles dédiées
    (ex. l'en-tête répété d'un rapport) seulement pour les documents qui en ont besoin.
    """

    running_headers: tuple[re.Pattern[str], ...] = ()
    toc_titles: frozenset[str] = field(
        default_factory=lambda: frozenset({"sommaire", "table des matières"})
    )


class DoclingExtractor:
    """Extrait un Markdown par page et des passages nettoyés, prêts pour le RAG."""

    SUPPORTED_EXTENSIONS = frozenset(
        {".pdf", ".doc", ".docx", ".xls", ".xlsx", ".html", ".htm", ".eml", ".msg"}
    )
    EMAIL_EXTENSIONS = frozenset({".eml", ".msg"})

    def __init__(
        self,
        converter: Any | None = None,
        chunker: Any | None = None,
        tokenizer_model: str = DEFAULT_TOKENIZER_MODEL,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        cleaning_rules: ChunkCleaningRules | None = None,
    ) -> None:
        if max_tokens < 1:
            raise ValueError("max_tokens must be positive")
        self._converter = converter
        self._chunker = chunker
        self._tokenizer_model = tokenizer_model
        self._max_tokens = max_tokens
        self._rules = cleaning_rules or ChunkCleaningRules()

    def extract(self, source: str | Path) -> ExtractedDocument:
        path, source_type, file_type = self._prepare_source(source)
        converter = self._load_converter()
        chunker = self._load_chunker()

        try:
            document = converter.convert(path).document
            pages = _export_pages(document)
            raw_chunks = [_to_raw_chunk(chunk) for chunk in chunker.chunk(document)]
        except Exception as exc:
            raise ExtractionError(f"extraction failed for {path.name}") from exc

        chunks = _clean_and_repair_chunks(raw_chunks, self._rules)
        logger.info(
            "document extracted",
            extra={"source": path.name, "pages": len(pages), "chunks": len(chunks)},
        )
        return ExtractedDocument(
            source=path.name,
            source_type=source_type,
            file_type=file_type,
            pages=pages,
            chunks=chunks,
        )

    def _load_converter(self) -> Any:
        if self._converter is None:
            try:
                from docling.document_converter import DocumentConverter
            except ImportError as exc:
                raise MissingDependencyError(
                    "Docling is optional; install it with `uv sync --extra rag`."
                ) from exc
            self._converter = DocumentConverter()
        return self._converter

    def _load_chunker(self) -> Any:
        if self._chunker is None:
            try:
                from docling_core.transforms.chunker.hybrid_chunker import HybridChunker
                from docling_core.transforms.chunker.tokenizer.huggingface import (
                    HuggingFaceTokenizer,
                )
                from transformers import AutoTokenizer
            except ImportError as exc:
                raise MissingDependencyError(
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
    def _prepare_source(cls, source: str | Path) -> tuple[Path, SourceType, str]:
        if urlsplit(str(source)).scheme.lower() in {"http", "https"}:
            raise UnsupportedDocumentError(
                "remote URLs are not supported; save email attachments locally first"
            )

        path = Path(source).expanduser().resolve(strict=True)
        if not path.is_file():
            raise UnsupportedDocumentError(f"source is not a regular file: {path.name}")
        file_type = path.suffix.lower()
        if file_type not in cls.SUPPORTED_EXTENSIONS:
            supported = ", ".join(sorted(cls.SUPPORTED_EXTENSIONS))
            raise UnsupportedDocumentError(
                f"unsupported file type {file_type!r}; supported: {supported}"
            )
        source_type: SourceType = "email" if file_type in cls.EMAIL_EXTENSIONS else "file"
        return path, source_type, file_type


def _export_pages(document: Any) -> tuple[ExtractedPage, ...]:
    document_pages = getattr(document, "pages", None)
    if not document_pages:
        return (ExtractedPage(page_number=None, markdown=document.export_to_markdown()),)
    return tuple(
        ExtractedPage(page_number=number, markdown=document.export_to_markdown(page_no=number))
        for number in sorted(document_pages)
    )


def _chunk_page_number(chunk: Any) -> int | None:
    metadata = getattr(chunk, "meta", None)
    for item in getattr(metadata, "doc_items", ()) or ():
        for provenance in getattr(item, "prov", ()) or ():
            page_number = getattr(provenance, "page_no", None)
            if page_number is not None:
                return int(page_number)
    return None


def _to_raw_chunk(chunk: Any) -> _RawChunk:
    """Lit un chunk Docling : titres et légendes sont dans ``meta``, le texte seul dans ``text``."""
    metadata = getattr(chunk, "meta", None)
    return _RawChunk(
        headings=tuple(getattr(metadata, "headings", None) or ()),
        captions=tuple(getattr(metadata, "captions", None) or ()),
        body=str(chunk.text),
        page_number=_chunk_page_number(chunk),
    )


_TOC_ENTRY = re.compile(r"^\s*\d+\.\s+\d+\.\s+")
_TABLE_SEPARATOR = re.compile(r"^\s*\|?(?:\s*:?-{1,}:?\s*\|)+\s*:?-{0,}:?\s*\|?\s*$")
_EMPTY_TABLE_ROW = re.compile(r"\s*\|\s*-\s*\|\s*")
_VALUE_FRAGMENT = re.compile(r"^\s*(?:\d+[,.]\d+\s*€|\d+\s*(?:€|%))", re.IGNORECASE)
_SHORT_PROSE_MAX_CHARS = 220


@dataclass(slots=True)
class _RawChunk:
    """Un passage avant mise en forme : le contexte (titres, légendes) est séparé du texte."""

    headings: tuple[str, ...]
    captions: tuple[str, ...]
    body: str
    page_number: int | None

    def render(self) -> str:
        # Même format que ``HybridChunker.contextualize`` : le contexte puis le texte, une ligne
        # par élément. Les embeddings voient donc le titre de section de chaque passage.
        return "\n".join([*self.headings, *self.captions, self.body])


def _clean_and_repair_chunks(
    raw_chunks: list[_RawChunk],
    rules: ChunkCleaningRules,
) -> tuple[ExtractedChunk, ...]:
    cleaned = [chunk for raw in raw_chunks if (chunk := _clean_chunk(raw, rules)) is not None]

    # Les recollages se décident sur le texte seul, sans les titres devant.
    merged: list[_RawChunk] = []
    for chunk in cleaned:
        if merged and _should_join_chunks(merged[-1].body, chunk.body):
            previous = merged[-1]
            previous.body = f"{previous.body.rstrip()}\n{chunk.body.lstrip()}"
            previous.page_number = previous.page_number or chunk.page_number
        else:
            merged.append(chunk)

    return tuple(
        ExtractedChunk(chunk_id=index, page_number=chunk.page_number, text=chunk.render())
        for index, chunk in enumerate(merged, start=1)
    )


def _clean_chunk(raw: _RawChunk, rules: ChunkCleaningRules) -> _RawChunk | None:
    """Renvoie le passage nettoyé, ou ``None`` s'il ne contient que du bruit (ex. le sommaire)."""
    if any(heading.strip().casefold() in rules.toc_titles for heading in raw.headings):
        return None

    def is_running_header(text: str) -> bool:
        return any(pattern.search(text) for pattern in rules.running_headers)

    lines = [
        line
        for line in _clean_chunk_lines(raw.body)
        if not _TOC_ENTRY.match(line) and not is_running_header(line)
    ]
    body = "\n".join(lines).strip()
    if not body:
        return None
    return _RawChunk(
        headings=tuple(h for h in raw.headings if not is_running_header(h)),
        captions=raw.captions,
        body=body,
        page_number=raw.page_number,
    )


def _clean_chunk_lines(text: str) -> list[str]:
    cleaned: list[str] = []
    for line in text.splitlines():
        if _TABLE_SEPARATOR.match(line) or _EMPTY_TABLE_ROW.fullmatch(line):
            continue
        if not line.lstrip().startswith("|") and line.rstrip().endswith("|"):
            line = line.rstrip()[:-1].rstrip()
        cleaned.append(line)
    return cleaned


def _should_join_chunks(left: str, right: str) -> bool:
    left_body = left.rstrip()
    right_body = right.lstrip()
    # Une ligne coupée en plein milieu ("P001," ou "Prix =").
    if left_body.endswith(("=", ",")):
        return True
    # Un libellé suivi de sa valeur sur le passage suivant ("Prix :" / "12,50 €").
    if left_body.endswith(("=", ":")) and _VALUE_FRAGMENT.match(right_body):
        return True
    # Une courte phrase d'introduction suivie d'un tableau qui reprend page suivante.
    right_starts_table = any(line.lstrip().startswith("|") for line in right_body.splitlines()[:3])
    left_is_short_prose = len(left_body) <= _SHORT_PROSE_MAX_CHARS and not any(
        line.lstrip().startswith("|") for line in left_body.splitlines()
    )
    return left_is_short_prose and right_starts_table
