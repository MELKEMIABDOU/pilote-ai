"""Configuration centrale des logs (à appeler une seule fois, au démarrage)."""

import logging

LOG_FORMAT = "%(asctime)s %(levelname)s %(name)s: %(message)s"


def configure_logging(level: str = "INFO") -> None:
    logging.basicConfig(level=level, format=LOG_FORMAT, force=True)
