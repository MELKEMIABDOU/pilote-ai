"""Utiliser le RAG en local, en ligne de commande.

uv run python -m app.cli index documents/rapport.pdf --client client-a
uv run python -m app.cli index documents/*.pdf --client client-a --reindex
uv run python -m app.cli search "Quel est le prix du riz ?" --client client-a
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from app.config import get_settings
from app.domain.errors import CopiloteError
from app.logging_config import configure_logging
from app.rag_container import RagServices, build_rag_services


def _index(services: RagServices, args: argparse.Namespace) -> int:
    failures = 0
    for path in args.files:
        try:
            result = services.ingest(path, client_id=args.client, reindex=args.reindex)
        except (CopiloteError, OSError) as exc:
            failures += 1
            print(f"ERREUR  {path.name} : {exc}", file=sys.stderr)
            continue
        status = "déjà indexé" if result.already_indexed else f"{result.chunks} passages"
        print(f"OK      {result.source} : {status} (id {result.document_id[:12]}…)")
    return 1 if failures else 0


def format_pages(pages: tuple[int, ...]) -> str:
    """``()`` → ``page ?`` ; ``(4,)`` → ``p.4`` ; ``(4, 5, 6)`` → ``p.4-6``."""
    if not pages:
        return "page ?"
    if len(pages) == 1:
        return f"p.{pages[0]}"
    return f"p.{pages[0]}-{pages[-1]}"


def _search(services: RagServices, args: argparse.Namespace) -> int:
    results = services.search(args.query, client_id=args.client, top_k=args.top_k)
    if not results:
        print("Aucun passage indexé pour ce client.")
        return 0
    for rank, result in enumerate(results, start=1):
        print(f"\n[{rank}] {result.source} {format_pages(result.pages)}  score {result.score:.3f}")
        # Texte complet : le couper cacherait parfois la ligne qui contient la réponse.
        print("    " + result.text.replace("\n", "\n    "))
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m app.cli", description="RAG Copilote ADV")
    commands = parser.add_subparsers(dest="command", required=True)

    index = commands.add_parser("index", help="indexer un ou plusieurs documents")
    index.add_argument("files", nargs="+", type=Path)
    index.add_argument("--client", required=True, help="identifiant du client")
    index.add_argument("--reindex", action="store_true", help="recalculer même si déjà indexé")

    search = commands.add_parser("search", help="chercher dans les documents d'un client")
    search.add_argument("query")
    search.add_argument("--client", required=True, help="identifiant du client")
    search.add_argument("--top-k", type=int, default=5)

    args = parser.parse_args(argv)
    settings = get_settings()
    configure_logging(settings.log_level)
    try:
        with build_rag_services(settings) as services:
            return _index(services, args) if args.command == "index" else _search(services, args)
    except (CopiloteError, ValueError) as exc:
        print(f"ERREUR : {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
