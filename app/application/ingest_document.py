"""Cas d'usage : indexer un document pour un client (extraction → embeddings → stockage).

Ne dépend que des ports : aucune référence à Docling, Qwen ou SQLite ici.
"""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass
from pathlib import Path

from app.domain.errors import ExtractionError
from app.ports.embeddings import Embedder
from app.ports.extraction import DocumentExtractor
from app.ports.vector_store import IndexedChunk, VectorStore

logger = logging.getLogger(__name__)

_HASH_BLOCK_SIZE = 1024 * 1024


@dataclass(frozen=True, slots=True)
class IngestionResult:
    client_id: str
    document_id: str
    source: str
    chunks: int
    already_indexed: bool  # True : le document était déjà là, rien n'a été recalculé


def file_sha256(path: Path) -> str:
    """Identifiant stable d'un fichier : le même contenu donne toujours le même identifiant."""
    digest = hashlib.sha256()
    with path.open("rb") as file:
        while block := file.read(_HASH_BLOCK_SIZE):
            digest.update(block)
    return digest.hexdigest()


class IngestDocument:
    def __init__(
        self, extractor: DocumentExtractor, embedder: Embedder, store: VectorStore
    ) -> None:
        self._extractor = extractor
        self._embedder = embedder
        self._store = store

    def __call__(self, path: Path, *, client_id: str, reindex: bool = False) -> IngestionResult:
        if not client_id.strip():
            raise ValueError("client_id must not be empty")
        document_id = file_sha256(path)

        if not reindex and self._store.has_document(document_id, client_id=client_id):
            logger.info("document already indexed", extra={"document_id": document_id})
            return IngestionResult(client_id, document_id, path.name, 0, already_indexed=True)

        extracted = self._extractor.extract(path)
        if not extracted.chunks:
            # Ex. un PDF scanné sans texte : mieux vaut le dire que d'indexer un document vide.
            raise ExtractionError(f"no text could be extracted from {path.name}")
        vectors = self._embedder.embed_documents([chunk.text for chunk in extracted.chunks])

        # On ne supprime l'ancienne version qu'une fois les nouveaux vecteurs calculés :
        # si l'extraction ou les embeddings échouent, l'ancienne version reste disponible.
        if reindex:
            self._store.delete_document(document_id, client_id=client_id)
        self._store.add(
            [
                IndexedChunk(
                    client_id=client_id,
                    document_id=document_id,
                    chunk_id=chunk.chunk_id,
                    source=extracted.source,
                    page_number=chunk.page_number,
                    text=chunk.text,
                    vector=vector,
                    page_end=chunk.page_end,
                )
                for chunk, vector in zip(extracted.chunks, vectors, strict=True)
            ]
        )
        logger.info(
            "document indexed",
            extra={"document_id": document_id, "source": extracted.source, "chunks": len(vectors)},
        )
        return IngestionResult(
            client_id, document_id, extracted.source, len(vectors), already_indexed=False
        )
