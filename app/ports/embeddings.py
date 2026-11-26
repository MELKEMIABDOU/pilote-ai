"""Contrat de calcul des embeddings (texte → vecteur)."""

from collections.abc import Sequence
from typing import Protocol

type Vector = tuple[float, ...]
"""Un embedding. Immuable : un vecteur stocké ne doit pas pouvoir être modifié par erreur."""


class Embedder(Protocol):
    """Transforme des textes en vecteurs comparables entre eux.

    Les requêtes et les documents ont deux méthodes séparées : certains modèles (dont Qwen3)
    ajoutent une instruction devant la requête mais pas devant les documents.
    """

    @property
    def model_name(self) -> str:
        """Nom du modèle. Des vecteurs de modèles différents ne sont pas comparables."""
        ...

    @property
    def dimension(self) -> int:
        """Taille de chaque vecteur produit."""
        ...

    def embed_documents(self, texts: Sequence[str]) -> list[Vector]:
        """Un vecteur par texte, dans le même ordre que ``texts``."""
        ...

    def embed_query(self, text: str) -> Vector:
        """Vecteur d'une question de recherche."""
        ...
