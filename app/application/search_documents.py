"""Cas d'usage : chercher dans les documents d'un client, avec la citation de chaque passage.

Pas de seuil « rien trouvé » : l'évaluation a montré que les scores des embeddings ne
séparent pas les questions sans réponse (cf. outputs/eval). Les passages sont renvoyés avec
leur score ; c'est à l'appelant (l'agent) de juger s'ils répondent à la question.
"""

from __future__ import annotations

from app.ports.embeddings import Embedder
from app.ports.vector_store import SearchResult, VectorStore

DEFAULT_TOP_K = 5
# Borne haute : un agent qui demande 1000 passages remplirait son contexte pour rien.
MAX_TOP_K = 20


class SearchDocuments:
    def __init__(self, embedder: Embedder, store: VectorStore) -> None:
        self._embedder = embedder
        self._store = store

    def __call__(
        self, query: str, *, client_id: str, top_k: int = DEFAULT_TOP_K
    ) -> list[SearchResult]:
        if not query.strip():
            raise ValueError("query must not be empty")
        if not client_id.strip():
            raise ValueError("client_id must not be empty")
        if not 1 <= top_k <= MAX_TOP_K:
            raise ValueError(f"top_k must be between 1 and {MAX_TOP_K}")
        return self._store.search(
            self._embedder.embed_query(query), client_id=client_id, top_k=top_k
        )
