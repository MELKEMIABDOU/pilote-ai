"""Chargement des modèles Hugging Face : le cache local d'abord, Internet seulement si besoin.

Par défaut, ``from_pretrained`` interroge le Hub à chaque lancement (6 requêtes HTTP pour
Qwen3), même quand le modèle est déjà téléchargé : démarrage plus lent, échec sans réseau, et
un serveur qui traite des documents clients contacte l'extérieur sans raison.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any

logger = logging.getLogger(__name__)


def load_pretrained(loader: Callable[..., Any], model_name: str, **kwargs: Any) -> Any:
    """Appelle ``loader(model_name, ...)`` hors ligne, puis en ligne si le cache est incomplet."""
    try:
        return loader(model_name, local_files_only=True, **kwargs)
    except OSError:
        # transformers lève OSError quand un fichier manque dans le cache local.
        logger.info("model not in local cache, downloading it", extra={"model": model_name})
        return loader(model_name, **kwargs)
