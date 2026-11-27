"""Contrat d'un modèle de langage qui renvoie du JSON conforme à un schéma.

Le reste de l'application ne connaît que ce port : passer d'Ollama (local) à une API (Claude…)
ne demandera qu'un nouvel adaptateur.
"""

from typing import Any, Protocol


class StructuredLLM(Protocol):
    @property
    def model_name(self) -> str: ...

    def generate_json(self, *, system: str, prompt: str, schema: dict[str, Any]) -> dict[str, Any]:
        """Renvoie un objet JSON qui respecte ``schema`` (JSON Schema).

        Lève ``LLMError`` si le modèle est injoignable ou si la réponse n'est pas un objet JSON.
        """
        ...
