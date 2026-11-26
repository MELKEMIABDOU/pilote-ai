"""Règles de nettoyage propres au document de démonstration « Magasin Horizon »."""

import re

from app.adapters.extraction.docling import ChunkCleaningRules

HORIZON_RULES = ChunkCleaningRules(
    running_headers=(
        re.compile(r"magasin horizon\s*\|\s*rapport de gestion des produits", re.IGNORECASE),
    ),
)
