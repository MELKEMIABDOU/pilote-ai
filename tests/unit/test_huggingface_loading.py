import logging
from typing import Any

import pytest

from app.adapters.huggingface import load_pretrained
from app.logging_config import NOISY_LOGGERS, configure_logging


class FakeLoader:
    """Imite ``from_pretrained`` : échoue hors ligne si le modèle n'est pas en cache."""

    def __init__(self, cached: bool) -> None:
        self.cached = cached
        self.calls: list[dict[str, Any]] = []

    def __call__(self, model_name: str, **kwargs: Any) -> str:
        self.calls.append(kwargs)
        if kwargs.get("local_files_only") and not self.cached:
            raise OSError("not in cache")
        return f"model:{model_name}"


def test_cached_model_is_loaded_without_network() -> None:
    loader = FakeLoader(cached=True)

    assert load_pretrained(loader, "qwen", dtype="x") == "model:qwen"
    assert loader.calls == [{"local_files_only": True, "dtype": "x"}]


def test_missing_model_is_downloaded() -> None:
    loader = FakeLoader(cached=False)

    assert load_pretrained(loader, "qwen", dtype="x") == "model:qwen"
    assert loader.calls == [{"local_files_only": True, "dtype": "x"}, {"dtype": "x"}]


def test_other_errors_are_not_hidden() -> None:
    def broken(model_name: str, **kwargs: Any) -> None:
        raise ValueError("bad argument")

    with pytest.raises(ValueError, match="bad argument"):
        load_pretrained(broken, "qwen")


def test_noisy_libraries_only_log_warnings() -> None:
    configure_logging("INFO")

    assert all(logging.getLogger(name).level == logging.WARNING for name in NOISY_LOGGERS)
