"""Évalue l'extraction des bons de commande (document → PurchaseOrder) avec un LLM local.

    uv run python -m scripts.eval_extraction
    uv run python -m scripts.eval_extraction --models qwen3:8b qwen3:4b --save

Pour chaque commande fictive de ``tests/fixtures/orders`` : Docling lit le document, le LLM
remplit le brouillon, le code convertit et vérifie, puis les contrôles s'appliquent. On compare
à ``expected.json`` :
- **champs justes** : chaque valeur (client, n° de commande, dates, totaux, et pour chaque
  ligne référence, quantité, prix, TVA, total) comparée en NOMBRE (196.8 == 196.80) ;
- **commande parfaite** : tous les champs justes ;
- **verdict juste** : la chaîne complète trouve exactement les anomalies attendues ;
- **fausses alertes** : « valeur introuvable dans le document » alors que la valeur est juste.
"""

from __future__ import annotations

import argparse
import json
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, date, datetime
from decimal import Decimal
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
ORDERS = ROOT / "tests" / "fixtures" / "orders"
RESULTS_DIR = ROOT / "outputs" / "eval"

HEADER_FIELDS = (
    "customer_name",
    "customer_code",
    "document_number",
    "order_date",
    "requested_delivery_date",
    "total_ht",
    "total_ttc",
)
LINE_FIELDS = ("product_code", "quantity", "unit_price", "vat_rate", "line_total")
GROUNDING_CODES = {"VALUE_NOT_IN_DOCUMENT", "LINE_NOT_IN_DOCUMENT"}


@dataclass
class DocumentScore:
    filename: str
    note: str
    fields_ok: int = 0
    fields_total: int = 0
    errors: list[str] = field(default_factory=list)
    verdict_ok: bool = False
    found_anomalies: list[str] = field(default_factory=list)
    expected_anomalies: list[str] = field(default_factory=list)
    false_alarms: int = 0
    llm_seconds: float = 0.0

    @property
    def perfect(self) -> bool:
        return self.fields_ok == self.fields_total


# --- Comparaison (fonctions pures, testées) -------------------------------------------------


def same_value(expected: str | None, actual: object) -> bool:
    """Compare une valeur attendue (texte de expected.json) à la valeur extraite."""
    if expected == "*":
        return True  # champ non noté
    if expected is None or actual is None:
        return expected is None and actual is None
    if isinstance(actual, Decimal):
        return actual == Decimal(expected)
    if isinstance(actual, date):
        return actual.isoformat() == expected
    return str(actual).strip().casefold() == expected.strip().casefold()


def score_document(filename: str, expected: dict[str, Any], order: Any) -> DocumentScore:
    score = DocumentScore(filename, expected["note"])

    def compare(label: str, wanted: str | None, actual: object) -> None:
        if wanted == "*":
            return
        score.fields_total += 1
        if same_value(wanted, actual):
            score.fields_ok += 1
        else:
            score.errors.append(f"{label} : attendu {wanted!r}, lu {actual!r}")

    for name in HEADER_FIELDS:
        compare(name, expected[name], getattr(order, name))

    wanted_lines, lines = expected["lines"], order.lines
    if len(lines) != len(wanted_lines):
        score.errors.append(f"nombre de lignes : attendu {len(wanted_lines)}, lu {len(lines)}")
    for index, wanted in enumerate(wanted_lines):
        line = lines[index] if index < len(lines) else None
        for name in LINE_FIELDS:
            compare(f"L{index + 1}.{name}", wanted[name], getattr(line, name, None))
    # Une ligne en trop est une erreur (ex. le modèle a pris « Total HT » pour un produit).
    for extra in lines[len(wanted_lines) :]:
        score.fields_total += 1
        score.errors.append(f"ligne en trop : {extra.description or extra.product_code!r}")
    return score


# --- Chaîne réelle --------------------------------------------------------------------------


def run(
    models: list[str],
    on_progress: Callable[[dict[str, list[DocumentScore]]], object] | None = None,
    *,
    gpu_layers: int | None = None,
    pause_seconds: float = 0.0,
    done: dict[str, list[DocumentScore]] | None = None,
) -> dict[str, list[DocumentScore]]:
    """``on_progress`` est appelé après chaque commande (sauvegarde au fil de l'eau).

    ``done`` : résultats d'une évaluation interrompue ; ces commandes ne sont pas refaites.
    ``pause_seconds`` : repos entre deux commandes, pour laisser le PC refroidir.
    """
    from app.adapters.extraction.docling import DoclingExtractor
    from app.adapters.llm.ollama import OllamaLLM
    from app.application.extract_order import ExtractOrder, review_extracted_order

    expected_all: dict[str, dict[str, Any]] = json.loads(
        (ORDERS / "expected.json").read_text(encoding="utf-8")
    )
    results: dict[str, list[DocumentScore]] = {m: list((done or {}).get(m, [])) for m in models}
    todo = {
        name
        for model in models
        for name in expected_all
        if name not in {s.filename for s in results[model]}
    }
    print(f"Lecture des {len(todo)} documents restants avec Docling…")
    extractor = DoclingExtractor()
    documents = {name: extractor.extract(ORDERS / name) for name in expected_all if name in todo}

    for model in models:
        layers = "toutes" if gpu_layers is None else gpu_layers
        print(f"\nExtraction avec {model} (couches sur le GPU : {layers})…")
        llm = OllamaLLM(model, gpu_layers=gpu_layers)
        extract = ExtractOrder(llm)
        scores = results[model]
        already = {s.filename for s in scores}
        first = True
        for name, expected in expected_all.items():
            if name in already:
                print(f"  {name:<28} déjà fait")
                continue
            if pause_seconds and not first:
                time.sleep(pause_seconds)
            first = False
            outcome = extract(documents[name])
            check = review_extracted_order(outcome)
            score = score_document(name, expected, outcome.order)
            score.llm_seconds = outcome.seconds
            score.found_anomalies = sorted(a.code for a in check.anomalies)
            score.expected_anomalies = sorted(
                expected["expected_anomalies"] + expected.get("expected_document_alerts", [])
            )
            score.verdict_ok = score.found_anomalies == score.expected_anomalies
            # Fausse alerte = « introuvable » sur un champ pourtant juste.
            wrong = {error.split(" : ")[0] for error in score.errors}
            score.false_alarms = sum(
                1
                for a in outcome.anomalies
                if a.code in GROUNDING_CODES and _label(a.field) not in wrong
            )
            scores.append(score)
            print(
                f"  {name:<28} {score.fields_ok}/{score.fields_total} champs  "
                f"{outcome.seconds:5.1f} s",
                flush=True,
            )
            if on_progress:
                on_progress(results)
        llm.close()
    return results


def _label(field_path: str) -> str:
    """« lines[2].unit_price » → « L2.unit_price » (même libellé que dans les erreurs)."""
    if field_path.startswith("lines["):
        number, _, rest = field_path[len("lines[") :].partition("]")
        return f"L{number}{rest}"
    return field_path


def summarize(scores: list[DocumentScore]) -> dict[str, Any]:
    fields_ok = sum(s.fields_ok for s in scores)
    fields_total = sum(s.fields_total for s in scores)
    return {
        "fields": f"{fields_ok}/{fields_total}",
        "field_accuracy": round(fields_ok / fields_total, 3) if fields_total else 0.0,
        "perfect_orders": f"{sum(s.perfect for s in scores)}/{len(scores)}",
        "right_verdicts": f"{sum(s.verdict_ok for s in scores)}/{len(scores)}",
        "false_alarms": sum(s.false_alarms for s in scores),
        "seconds_per_order": round(sum(s.llm_seconds for s in scores) / len(scores), 1),
    }


def print_report(results: dict[str, list[DocumentScore]]) -> None:
    for model, scores in results.items():
        summary = summarize(scores)
        print(f"\n=== {model} ===")
        print(
            f"Champs justes : {summary['fields']} ({summary['field_accuracy']:.0%}) | "
            f"commandes parfaites : {summary['perfect_orders']} | "
            f"verdict juste : {summary['right_verdicts']} | "
            f"fausses alertes : {summary['false_alarms']} | "
            f"{summary['seconds_per_order']} s / commande"
        )
        for score in scores:
            if score.perfect and score.verdict_ok:
                continue
            print(f"\n  {score.filename} — {score.note}")
            for error in score.errors:
                print(f"    ✗ {error}")
            if not score.verdict_ok:
                print(
                    f"    ✗ anomalies : attendues {score.expected_anomalies}, "
                    f"trouvées {score.found_anomalies}"
                )


def load(path: Path) -> dict[str, list[DocumentScore]]:
    """Relit un fichier de ``save`` (pour reprendre une évaluation interrompue)."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {
        model: [DocumentScore(**document) for document in data["documents"]]
        for model, data in payload.items()
    }


def save(results: dict[str, list[DocumentScore]], path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        model: {"summary": summarize(scores), "documents": [vars(s) for s in scores]}
        for model, scores in results.items()
        if scores
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--models", nargs="+", default=["qwen3:8b"])
    parser.add_argument("--save", action="store_true", help="enregistre dans outputs/eval/")
    parser.add_argument(
        "--gpu-layers", type=int, help="couches sur la carte graphique (moins = moins de puissance)"
    )
    parser.add_argument(
        "--pause", type=float, default=0.0, help="secondes de repos entre commandes"
    )
    parser.add_argument(
        "--resume",
        type=Path,
        help="reprendre un fichier extraction_*.json interrompu (implique --save)",
    )
    args = parser.parse_args()

    started = time.perf_counter()
    path = args.resume or RESULTS_DIR / f"extraction_{datetime.now(UTC):%Y%m%dT%H%M%SZ}.json"
    save_each = args.save or args.resume is not None
    # On écrit après CHAQUE commande : un arrêt brutal ne perd que la commande en cours.
    results = run(
        args.models,
        on_progress=(lambda r: save(r, path)) if save_each else None,
        gpu_layers=args.gpu_layers,
        pause_seconds=args.pause,
        done=load(args.resume) if args.resume else None,
    )
    print_report(results)
    print(f"\nDurée totale : {time.perf_counter() - started:.0f} s")
    if save_each:
        print(f"Résultat enregistré : {save(results, path)}")


if __name__ == "__main__":
    main()
