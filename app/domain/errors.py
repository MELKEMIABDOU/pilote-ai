"""Erreurs métier du copilote ADV.

Toutes héritent de ``CopiloteError`` pour que l'API puisse les intercepter en un seul endroit.
"""


class CopiloteError(Exception):
    """Erreur de base de l'application."""


class UnsupportedDocumentError(CopiloteError, ValueError):
    """Le document fourni ne peut pas être traité (format, URL distante, pas un fichier...)."""


class ExtractionError(CopiloteError):
    """L'outil d'extraction a échoué sur un document pourtant accepté."""


class EmbeddingError(CopiloteError):
    """Le modèle d'embeddings a renvoyé un résultat inattendu."""


class MissingDependencyError(CopiloteError, RuntimeError):
    """Une dépendance optionnelle (ex. l'extra ``rag``) n'est pas installée."""
