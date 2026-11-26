"""Faux objets pour les tests : Docling (sans l'extra ``rag``), Embedder et VectorStore."""

from __future__ import annotations

import math
from collections.abc import Sequence
from pathlib import Path
from types import SimpleNamespace

from app.ports.embeddings import Vector
from app.ports.extraction import ExtractedChunk, ExtractedDocument
from app.ports.vector_store import IndexedChunk, SearchResult


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


class FakeExtractor:
    """DocumentExtractor qui renvoie des passages prédéfinis et compte ses appels."""

    def __init__(self, chunks: Sequence[tuple[int | None, str]] = ()) -> None:
        self.chunks = tuple(chunks)
        self.calls: list[Path] = []

    def extract(self, source: str | Path) -> ExtractedDocument:
        path = Path(source)
        self.calls.append(path)
        return ExtractedDocument(
            source=path.name,
            source_type="file",
            file_type=path.suffix,
            pages=(),
            chunks=tuple(
                ExtractedChunk(chunk_id=i, page_number=page, text=text)
                for i, (page, text) in enumerate(self.chunks, start=1)
            ),
        )


class FakeEmbedder:
    """Embedder déterministe : le vecteur compte les mots d'un petit vocabulaire.

    Deux textes qui partagent des mots du vocabulaire ont donc des vecteurs proches.
    """

    def __init__(self, vocabulary: Sequence[str] = ("riz", "prix", "stock", "client")) -> None:
        self.vocabulary = tuple(word.casefold() for word in vocabulary)
        self.model_name = "fake-bag-of-words"
        self.dimension = len(self.vocabulary)

    def embed_documents(self, texts: Sequence[str]) -> list[Vector]:
        return [self.embed_query(text) for text in texts]

    def embed_query(self, text: str) -> Vector:
        words = text.casefold().split()
        # +0.001 : évite le vecteur nul, pour lequel la similarité cosinus n'est pas définie.
        return tuple(words.count(word) + 0.001 for word in self.vocabulary)


class InMemoryVectorStore:
    """VectorStore en mémoire (similarité cosinus) pour les tests."""

    def __init__(self) -> None:
        self._chunks: dict[tuple[str, str, int], IndexedChunk] = {}

    def add(self, chunks: Sequence[IndexedChunk]) -> None:
        for chunk in chunks:
            self._chunks[(chunk.client_id, chunk.document_id, chunk.chunk_id)] = chunk

    def search(self, query: Vector, *, client_id: str, top_k: int = 5) -> list[SearchResult]:
        if top_k < 1:
            raise ValueError("top_k must be >= 1")
        results = []
        for chunk in self._chunks.values():
            if chunk.client_id != client_id:
                continue
            if len(chunk.vector) != len(query):
                raise ValueError("query dimension does not match stored vectors")
            results.append(
                SearchResult(
                    client_id=chunk.client_id,
                    document_id=chunk.document_id,
                    chunk_id=chunk.chunk_id,
                    source=chunk.source,
                    page_number=chunk.page_number,
                    text=chunk.text,
                    score=_cosine(query, chunk.vector),
                )
            )
        return sorted(results, key=lambda result: result.score, reverse=True)[:top_k]

    def has_document(self, document_id: str, *, client_id: str) -> bool:
        return any(key[:2] == (client_id, document_id) for key in self._chunks)

    def delete_document(self, document_id: str, *, client_id: str) -> int:
        keys = [key for key in self._chunks if key[:2] == (client_id, document_id)]
        for key in keys:
            del self._chunks[key]
        return len(keys)


def _cosine(left: Vector, right: Vector) -> float:
    dot = sum(a * b for a, b in zip(left, right, strict=True))
    return dot / (math.hypot(*left) * math.hypot(*right))
