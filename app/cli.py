"""Utiliser le copilote en local, en ligne de commande.

uv run python -m app.cli index documents/rapport.pdf --client client-a
uv run python -m app.cli index documents/*.pdf --client client-a --reindex
uv run python -m app.cli search "Quel est le prix du riz ?" --client client-a
uv run python -m app.cli controler documents/commande.pdf
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from app.config import Settings, get_settings
from app.domain.errors import CopiloteError
from app.domain.models import Severity
from app.domain.rules import OrderCheck, format_eur, format_rate
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


def format_order_check(check: OrderCheck, *, source: str, model: str, seconds: float) -> str:
    """Le rapport affiché à l'ADV pour une commande lue et contrôlée."""
    order = check.order
    status = "✅ PRÊTE À SAISIR" if check.is_ready else "❌ À CORRIGER"
    out = [
        f"=== {source} : {status} ===",
        f"Client : {order.customer_name or '—'}"
        + (f" (code {order.customer_code})" if order.customer_code else "")
        + f" | n° de commande : {order.document_number or '—'}",
        f"Date : {order.order_date or '—'} | livraison souhaitée : "
        f"{order.requested_delivery_date or '—'}",
        "Lignes :",
    ]
    for line in order.lines:
        mark = "❌" if line.line_number in check.blocked_line_numbers else "✅"
        price = format_eur(line.unit_price) if line.unit_price is not None else "prix ?"
        vat = format_rate(line.vat_rate) if line.vat_rate is not None else "TVA ?"
        out.append(
            f"  {mark} L{line.line_number} {line.product_code or 'réf. ?'} "
            f"{line.description or ''} : {line.quantity if line.quantity is not None else 'qté ?'}"
            f" × {price} ({vat})"
        )
    if order.computed_total_ht is not None:
        out.append(f"Total HT calculé : {format_eur(order.computed_total_ht)}")
    if check.reliable_total_ttc is not None:
        out.append(f"Total TTC calculé : {format_eur(check.reliable_total_ttc)}")
    if check.anomalies:
        out.append("Anomalies :")
        icons = {Severity.BLOCKING: "❌", Severity.WARNING: "⚠️ "}
        out += [f"  {icons[a.severity]} {a.message}" for a in check.anomalies]
    out.append(f"(lu par {model} en {seconds:.1f} s — l'ADV doit valider avant toute saisie)")
    return "\n".join(out)


def _control(args: argparse.Namespace, settings: Settings) -> int:
    # Import ici : la commande n'a pas besoin du modèle d'embeddings du RAG.
    from app.order_container import OrderReader

    failures = 0
    with OrderReader(settings) as read:
        for path in args.files:
            try:
                outcome, check = read(path)
            except (CopiloteError, OSError) as exc:
                failures += 1
                print(f"ERREUR  {path.name} : {exc}", file=sys.stderr)
                continue
            print(
                "\n"
                + format_order_check(
                    check, source=path.name, model=outcome.model_name, seconds=outcome.seconds
                )
            )
    return 1 if failures else 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m app.cli", description="Copilote ADV")
    commands = parser.add_subparsers(dest="command", required=True)

    control = commands.add_parser("controler", help="lire et contrôler des bons de commande")
    control.add_argument("files", nargs="+", type=Path)

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
        if args.command == "controler":
            return _control(args, settings)
        with build_rag_services(settings) as services:
            return _index(services, args) if args.command == "index" else _search(services, args)
    except (CopiloteError, ValueError) as exc:
        print(f"ERREUR : {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
