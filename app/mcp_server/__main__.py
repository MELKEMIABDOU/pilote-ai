"""Lance le serveur MCP en stdio :  COPILOTE_MCP_CLIENT_ID=demo uv run python -m app.mcp_server

En stdio, la sortie standard (stdout) EST le canal du protocole : rien d'autre ne doit y être
écrit. Les logs partent sur stderr (comportement par défaut de ``logging``).
"""

from __future__ import annotations

import logging
import sys

from app.config import Settings, get_settings
from app.logging_config import configure_logging
from app.mcp_server.server import create_server
from app.rag_container import build_rag_services

logger = logging.getLogger(__name__)


def main(settings: Settings | None = None) -> int:
    settings = settings or get_settings()
    configure_logging(settings.log_level)
    client_id = (settings.mcp_client_id or "").strip()
    if not client_id:
        print(
            "ERREUR : définir COPILOTE_MCP_CLIENT_ID (le client dont l'agent peut lire "
            "les documents).",
            file=sys.stderr,
        )
        return 2

    with build_rag_services(settings) as services:
        # Charger le modèle maintenant : sinon le 1er appel d'outil attendrait le chargement
        # (plusieurs secondes) et pourrait dépasser le délai d'attente du client MCP.
        services.warm_up()
        logger.info("mcp server ready", extra={"client_id": client_id})
        create_server(services.search, client_id=client_id).run("stdio")
    return 0


if __name__ == "__main__":
    sys.exit(main())
