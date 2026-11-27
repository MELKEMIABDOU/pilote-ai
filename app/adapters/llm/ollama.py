"""Adaptateur ``StructuredLLM`` pour Ollama (modèle local : rien ne sort du PC).

API utilisée : ``POST /api/chat`` avec ``format`` = un JSON Schema (sortie structurée : le
modèle est contraint de produire un JSON valide pour ce schéma).
"""

from __future__ import annotations

import json
import logging
from typing import Any

import httpx

from app.domain.errors import LLMError

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "qwen3:8b"
DEFAULT_URL = "http://127.0.0.1:11434"
# Contexte (prompt + réponse) en tokens. 8192 tient avec qwen3:8b dans 8 Go de VRAM ; au-delà,
# une partie du modèle passe sur le CPU et tout ralentit. Ollama TRONQUE en silence un prompt
# trop long : le cas d'usage limite donc la taille du document envoyé.
DEFAULT_CONTEXT_TOKENS = 8192


class OllamaLLM:
    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        *,
        base_url: str = DEFAULT_URL,
        timeout_seconds: float = 300.0,
        context_tokens: int = DEFAULT_CONTEXT_TOKENS,
        gpu_layers: int | None = None,
        transport: httpx.BaseTransport | None = None,  # injectable pour les tests
    ) -> None:
        """``gpu_layers`` : nombre de couches du modèle sur la carte graphique (``None`` =
        Ollama choisit, en général toutes). Moins de couches = le GPU tire moins de puissance,
        le reste tourne sur le CPU, plus lentement. 0 = tout sur le CPU."""
        if gpu_layers is not None and gpu_layers < 0:
            raise ValueError("gpu_layers doit être >= 0")
        self._model = model
        self._base_url = base_url
        self._context_tokens = context_tokens
        self._gpu_layers = gpu_layers
        self._client = httpx.Client(base_url=base_url, timeout=timeout_seconds, transport=transport)

    @property
    def model_name(self) -> str:
        return self._model

    def generate_json(self, *, system: str, prompt: str, schema: dict[str, Any]) -> dict[str, Any]:
        options: dict[str, Any] = {"temperature": 0, "num_ctx": self._context_tokens}
        if self._gpu_layers is not None:
            options["num_gpu"] = self._gpu_layers
        body = {
            "model": self._model,
            "stream": False,
            "format": schema,
            # Extraction = recopier, pas raisonner : le mode « réflexion » de qwen3 ne ferait
            # que ralentir (des centaines de tokens de raisonnement avant la réponse).
            "think": False,
            "options": options,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": prompt},
            ],
        }
        try:
            response = self._client.post("/api/chat", json=body)
        except httpx.ConnectError as exc:
            raise LLMError(
                f"Ollama ne répond pas sur {self._base_url} : lancer l'application Ollama."
            ) from exc
        except httpx.TimeoutException as exc:
            raise LLMError(f"Ollama n'a pas répondu à temps (modèle {self._model}).") from exc

        if response.status_code == httpx.codes.NOT_FOUND:
            raise LLMError(
                f"Modèle {self._model} absent d'Ollama : le télécharger avec "
                f"`ollama pull {self._model}`."
            )
        if response.is_error:
            raise LLMError(f"Ollama a renvoyé l'erreur {response.status_code} : {response.text}")

        payload = response.json()
        logger.info(
            "llm call",
            extra={
                "model": self._model,
                "prompt_tokens": payload.get("prompt_eval_count"),
                "output_tokens": payload.get("eval_count"),
                "seconds": round(payload.get("total_duration", 0) / 1e9, 1),
            },
        )
        try:
            data = json.loads(payload["message"]["content"])
        except (KeyError, TypeError, json.JSONDecodeError) as exc:
            raise LLMError("la réponse d'Ollama n'est pas un JSON valide") from exc
        if not isinstance(data, dict):
            raise LLMError("la réponse d'Ollama n'est pas un objet JSON")
        return data

    def close(self) -> None:
        self._client.close()
