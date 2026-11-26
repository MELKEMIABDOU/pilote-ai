"""Adaptateur d'embeddings Qwen3-Embedding, via ``transformers`` (CPU ou GPU).

Recette officielle du modèle : padding à gauche, vecteur du dernier token, normalisation L2,
et une instruction devant les requêtes seulement (pas devant les documents).
"""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Callable, Sequence
from typing import Any

from app.domain.errors import EmbeddingError, MissingDependencyError
from app.ports.embeddings import Vector

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "Qwen/Qwen3-Embedding-0.6B"
# Le modèle recommande une instruction en anglais, même pour des textes en français.
DEFAULT_QUERY_INSTRUCTION = (
    "Given a question about a business document, retrieve the passages that answer it"
)
# Les passages font au plus 512 tokens (HybridChunker) + leurs titres : 1024 laisse de la marge.
DEFAULT_MAX_LENGTH = 1024
DEFAULT_BATCH_SIZE = 8

type BatchEncoder = Callable[[Sequence[str]], list[Vector]]
"""Encode un lot de textes déjà formatés. Injectable pour tester sans télécharger le modèle."""


class QwenEmbedder:
    """Implémente le port ``Embedder`` avec un modèle Qwen3-Embedding.

    Le modèle est chargé au premier appel (pas à la création) et une seule fois, même si
    plusieurs requêtes de l'API arrivent en même temps.
    """

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        *,
        dimension: int | None = None,
        query_instruction: str = DEFAULT_QUERY_INSTRUCTION,
        batch_size: int = DEFAULT_BATCH_SIZE,
        max_length: int = DEFAULT_MAX_LENGTH,
        device: str | None = None,
        encoder: BatchEncoder | None = None,
    ) -> None:
        if batch_size < 1:
            raise ValueError("batch_size must be >= 1")
        if max_length < 1:
            raise ValueError("max_length must be >= 1")
        if encoder is not None and dimension is None:
            raise ValueError("dimension is required when an encoder is injected")
        self._model_name = model_name
        self._dimension = dimension
        self._query_instruction = query_instruction
        self._batch_size = batch_size
        self._max_length = max_length
        self._device = device
        self._encoder = encoder
        self._load_lock = threading.Lock()

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def dimension(self) -> int:
        if self._dimension is None:
            # Lit seulement config.json (quelques Ko), sans charger les poids du modèle.
            self._dimension = int(_load_config(self._model_name).hidden_size)
        return self._dimension

    def embed_documents(self, texts: Sequence[str]) -> list[Vector]:
        return self._embed(list(texts))

    def embed_query(self, text: str) -> Vector:
        if not text.strip():
            raise ValueError("query must not be empty")
        return self._embed([f"Instruct: {self._query_instruction}\nQuery:{text}"])[0]

    def _embed(self, texts: list[str]) -> list[Vector]:
        if not texts:
            return []
        encode = self._get_encoder()
        vectors: list[Vector] = []
        for start in range(0, len(texts), self._batch_size):
            batch = texts[start : start + self._batch_size]
            encoded = encode(batch)
            if len(encoded) != len(batch):
                raise EmbeddingError(f"expected {len(batch)} vectors, got {len(encoded)}")
            vectors.extend(encoded)
        if any(len(vector) != self.dimension for vector in vectors):
            raise EmbeddingError(f"expected vectors of dimension {self.dimension}")
        return vectors

    def _get_encoder(self) -> BatchEncoder:
        with self._load_lock:
            if self._encoder is None:
                self._encoder = self._load_model()
        return self._encoder

    def _load_model(self) -> BatchEncoder:
        try:
            import torch
            from torch.nn import functional
            from transformers import AutoModel, AutoTokenizer
        except ImportError as exc:
            raise MissingDependencyError(
                "Qwen embeddings need the optional 'rag' dependencies; "
                "install them with `uv sync --extra rag`."
            ) from exc

        started = time.perf_counter()
        device = self._device or ("cuda" if torch.cuda.is_available() else "cpu")
        # Padding à gauche : le dernier token de chaque ligne est alors toujours un vrai token.
        tokenizer = AutoTokenizer.from_pretrained(self._model_name, padding_side="left")
        model = AutoModel.from_pretrained(self._model_name).to(device).eval()
        logger.info(
            "embedding model loaded",
            extra={
                "model": self._model_name,
                "device": device,
                "seconds": round(time.perf_counter() - started, 1),
            },
        )

        def encode(batch: Sequence[str]) -> list[Vector]:
            inputs = tokenizer(
                list(batch),
                padding=True,
                truncation=True,
                max_length=self._max_length,
                return_tensors="pt",
            ).to(device)
            with torch.inference_mode():
                hidden = model(**inputs).last_hidden_state
            pooled = functional.normalize(hidden[:, -1], p=2, dim=1)
            return [tuple(row) for row in pooled.float().cpu().tolist()]

        return encode


def _load_config(model_name: str) -> Any:
    try:
        from transformers import AutoConfig
    except ImportError as exc:
        raise MissingDependencyError(
            "Qwen embeddings need the optional 'rag' dependencies; "
            "install them with `uv sync --extra rag`."
        ) from exc
    return AutoConfig.from_pretrained(model_name)
