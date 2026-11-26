"""Configuration centrale des logs (à appeler une seule fois, au démarrage)."""

import logging

LOG_FORMAT = "%(asctime)s %(levelname)s %(name)s: %(message)s"

# Bibliothèques trop bavardes en INFO (une ligne par requête HTTP) : on ne garde que leurs
# avertissements et erreurs, pour que les logs de l'application restent lisibles.
NOISY_LOGGERS = ("httpx", "httpcore", "huggingface_hub")


def configure_logging(level: str = "INFO") -> None:
    logging.basicConfig(level=level, format=LOG_FORMAT, force=True)
    for name in NOISY_LOGGERS:
        logging.getLogger(name).setLevel(logging.WARNING)
