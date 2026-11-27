"""Assemblage de la lecture des bons de commande : Docling + LLM Ollama + contrôles.

Comme ``rag_container.py`` : le seul endroit qui choisit les implémentations concrètes.
"""

from __future__ import annotations

from pathlib import Path
from types import TracebackType

from app.adapters.extraction.docling import DoclingExtractor
from app.adapters.llm.ollama import OllamaLLM
from app.application.extract_order import ExtractionOutcome, ExtractOrder, review_extracted_order
from app.config import Settings, get_settings
from app.domain.rules import OrderCheck


class OrderReader:
    """Document → commande extraite → contrôles."""

    def __init__(self, settings: Settings | None = None) -> None:
        settings = settings or get_settings()
        self._extractor = DoclingExtractor(tokenizer_model=settings.embedding_model)
        self._llm = OllamaLLM(
            settings.llm_model, base_url=settings.ollama_url, gpu_layers=settings.llm_gpu_layers
        )
        self._extract = ExtractOrder(self._llm)

    def __call__(self, path: Path) -> tuple[ExtractionOutcome, OrderCheck]:
        outcome = self._extract(self._extractor.extract(path))
        return outcome, review_extracted_order(outcome)

    def close(self) -> None:
        self._llm.close()

    def __enter__(self) -> OrderReader:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.close()
