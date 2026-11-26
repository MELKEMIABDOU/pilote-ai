"""Contrat de stockage et de recherche des passages indexés.

Toutes les opérations sont limitées à un client (``client_id``) : un client ne doit jamais
voir, détecter ou supprimer les documents d'un autre. Le même fichier envoyé par deux clients
a le même ``document_id`` (hash du contenu) mais reste indexé deux fois, une fois par client.
"""

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

from app.ports.embeddings import Vector


def _require_text(value: str, field_name: str) -> None:
    if not value.strip():
        raise ValueError(f"{field_name} must not be empty")


def _check_page_range(first: int | None, last: int | None) -> None:
    if last is not None and (first is None or last < first):
        raise ValueError("page_end must be >= page_number")


@dataclass(frozen=True, slots=True)
class IndexedChunk:
    """Un passage prêt à être stocké : son texte, sa provenance et son vecteur."""

    client_id: str
    document_id: str  # SHA-256 du contenu du fichier source
    chunk_id: int  # numéro du passage dans le document (à partir de 1)
    source: str  # nom du fichier, pour la citation
    page_number: int | None  # première page du passage
    text: str
    vector: Vector
    page_end: int | None = None  # dernière page ; None = même page que page_number

    def __post_init__(self) -> None:
        _require_text(self.client_id, "client_id")
        _require_text(self.document_id, "document_id")
        if self.chunk_id < 1:
            raise ValueError("chunk_id must be >= 1")
        if not self.vector:
            raise ValueError("vector must not be empty")
        _check_page_range(self.page_number, self.page_end)


@dataclass(frozen=True, slots=True)
class SearchResult:
    """Un passage trouvé, avec tout ce qu'il faut pour le citer."""

    client_id: str
    document_id: str
    chunk_id: int
    source: str
    page_number: int | None
    text: str
    score: float  # plus haut = plus pertinent
    page_end: int | None = None

    def __post_init__(self) -> None:
        if not math.isfinite(self.score):
            raise ValueError("score must be a finite number")
        _check_page_range(self.page_number, self.page_end)

    @property
    def pages(self) -> tuple[int, ...]:
        """Toutes les pages couvertes par le passage, ex. ``(4, 5)`` ; vide si inconnues."""
        if self.page_number is None:
            return ()
        return tuple(range(self.page_number, (self.page_end or self.page_number) + 1))


class VectorStore(Protocol):
    """Stocke des passages vectorisés et retrouve les plus proches d'une requête."""

    def add(self, chunks: Sequence[IndexedChunk]) -> None:
        """Ajoute des passages. Un passage déjà présent (même client, document et ``chunk_id``)
        est remplacé : rejouer une ingestion ne crée pas de doublons."""
        ...

    def search(self, query: Vector, *, client_id: str, top_k: int = 5) -> list[SearchResult]:
        """Renvoie au plus ``top_k`` passages du client, triés par score décroissant.

        Lève ``ValueError`` si ``top_k < 1`` ou si la dimension de ``query`` ne correspond pas
        aux vecteurs stockés.
        """
        ...

    def has_document(self, document_id: str, *, client_id: str) -> bool:
        """Indique si ce document est déjà indexé pour ce client."""
        ...

    def delete_document(self, document_id: str, *, client_id: str) -> int:
        """Supprime tous les passages du document pour ce client ; renvoie leur nombre."""
        ...
