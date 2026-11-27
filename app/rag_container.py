"""Assemblage du RAG : le SEUL endroit qui choisit les implémentations concrètes.

Les cas d'usage (``app/application``) ne connaissent que les ports ; ici on décide que
l'extracteur est Docling, les embeddings Qwen3 et le stockage sqlite-vec. Changer de stockage
(pgvector…) ne demandera de modifier que ce fichier.
"""

from __future__ import annotations

from dataclasses import dataclass
from types import TracebackType

from app.adapters.embeddings.qwen import QwenEmbedder
from app.adapters.extraction.docling import DoclingExtractor
from app.adapters.vector_store.sqlite_vec import SqliteVecStore
from app.application.ingest_document import IngestDocument
from app.application.search_documents import SearchDocuments
from app.config import Settings, get_settings
from app.ports.embeddings import Embedder


@dataclass(frozen=True, slots=True)
class RagServices:
    ingest: IngestDocument
    search: SearchDocuments
    _store: SqliteVecStore
    _embedder: Embedder

    def warm_up(self) -> None:
        """Charge le modèle d'embeddings tout de suite plutôt qu'au premier appel."""
        self._embedder.embed_query("préchargement du modèle")

    def close(self) -> None:
        self._store.close()

    def __enter__(self) -> RagServices:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.close()


def build_rag_services(settings: Settings | None = None) -> RagServices:
    settings = settings or get_settings()
    embedder = QwenEmbedder(settings.embedding_model)
    store = SqliteVecStore(
        settings.vector_store_path, dimension=embedder.dimension, model_name=embedder.model_name
    )
    # Le découpage en passages utilise le tokenizer du modèle d'embeddings : la limite de
    # 512 tokens par passage est ainsi mesurée avec les mêmes tokens que ceux du modèle.
    extractor = DoclingExtractor(tokenizer_model=settings.embedding_model)
    return RagServices(
        ingest=IngestDocument(extractor, embedder, store),
        search=SearchDocuments(embedder, store),
        _store=store,
        _embedder=embedder,
    )
