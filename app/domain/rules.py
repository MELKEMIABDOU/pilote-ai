"""Contrôles d'un bon de commande : déterministes, en ``Decimal``, sans IA.

Chaque règle est une petite fonction qui renvoie les anomalies qu'elle trouve. Pour ajouter
un contrôle : écrire une fonction ``(order, settings) -> Iterator[Anomaly]`` et l'ajouter à
``RULES``.

Une règle ne compare des totaux que si elle peut les calculer : une quantité manquante est
signalée UNE fois (« quantité manquante »), sans déclencher en cascade « total incohérent ».
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from decimal import Decimal

from pydantic import BaseModel, ConfigDict

from app.domain.models import Anomaly, PurchaseOrder, PurchaseOrderLine, Severity

BLOCKING, WARNING = Severity.BLOCKING, Severity.WARNING

# Taux de TVA en France (en %) : normal, intermédiaire, réduit, particulier, et 0 (exonération,
# ex. livraison intracommunautaire ou export — à vérifier, d'où un avertissement dédié).
FRENCH_VAT_RATES = frozenset(Decimal(rate) for rate in ("20", "10", "5.5", "2.1", "0"))

# Anomalies qui rendent le TTC calculé faux.
_VAT_BLOCKING_CODES = frozenset({"UNKNOWN_VAT_RATE"})


class RuleSettings(BaseModel):
    model_config = ConfigDict(frozen=True)

    # Écart accepté par ligne entre le total écrit et le total calculé (arrondis du client).
    line_tolerance: Decimal = Decimal("0.01")
    allowed_vat_rates: frozenset[Decimal] = FRENCH_VAT_RATES


class OrderCheck(BaseModel):
    """Résultat des contrôles : ce que l'écran ADV affichera."""

    model_config = ConfigDict(frozen=True)

    order: PurchaseOrder
    anomalies: tuple[Anomaly, ...]

    @property
    def is_ready(self) -> bool:
        """Prête à saisir : aucune anomalie bloquante (les avertissements n'empêchent rien)."""
        return not any(anomaly.severity is BLOCKING for anomaly in self.anomalies)

    @property
    def reliable_total_ttc(self) -> Decimal | None:
        """Le TTC calculé, sauf si un taux de TVA est invalide.

        Un TTC calculé avec un taux faux (ex. 19,6 %) aurait l'air juste : un ADV pressé
        pourrait le recopier. On préfère ne rien afficher et dire pourquoi.
        """
        if any(a.code in _VAT_BLOCKING_CODES for a in self.anomalies):
            return None
        return self.order.computed_total_ttc

    @property
    def blocked_line_numbers(self) -> frozenset[int]:
        return frozenset(
            a.line_number
            for a in self.anomalies
            if a.severity is BLOCKING and a.line_number is not None
        )

    @property
    def ready_lines(self) -> tuple[PurchaseOrderLine, ...]:
        """Lignes sans anomalie bloquante (même si la commande entière ne l'est pas)."""
        return tuple(
            line for line in self.order.lines if line.line_number not in self.blocked_line_numbers
        )


def check_order(order: PurchaseOrder, settings: RuleSettings | None = None) -> OrderCheck:
    settings = settings or RuleSettings()
    anomalies = tuple(anomaly for rule in RULES for anomaly in rule(order, settings))
    return OrderCheck(order=order, anomalies=anomalies)


def format_eur(amount: Decimal) -> str:
    """``Decimal("1234.5")`` → ``"1 234,50 €"`` (format français)."""
    return f"{amount:,.2f}".replace(",", " ").replace(".", ",") + " €"


def format_rate(rate: Decimal) -> str:
    """``Decimal("5.5")`` → ``"5,5 %"`` ; ``Decimal("20")`` → ``"20 %"``."""
    return f"{rate.normalize():f}".replace(".", ",") + " %"


# --- Règles sur la commande -----------------------------------------------------------------


def check_customer(order: PurchaseOrder, settings: RuleSettings) -> Iterator[Anomaly]:
    if not order.customer_code and not order.customer_name:
        yield Anomaly(
            code="MISSING_CUSTOMER",
            severity=BLOCKING,
            message="Client non identifié (ni code ni nom).",
            field="customer",
        )


def check_document_number(order: PurchaseOrder, settings: RuleSettings) -> Iterator[Anomaly]:
    if not order.document_number:
        yield Anomaly(
            code="MISSING_DOCUMENT_NUMBER",
            severity=WARNING,
            message="N° de commande du client absent (souvent exigé sur la facture).",
            field="document_number",
        )


def check_has_lines(order: PurchaseOrder, settings: RuleSettings) -> Iterator[Anomaly]:
    if not order.lines:
        yield Anomaly(
            code="NO_LINES",
            severity=BLOCKING,
            message="La commande ne contient aucune ligne.",
            field="lines",
        )


def check_dates(order: PurchaseOrder, settings: RuleSettings) -> Iterator[Anomaly]:
    if (
        order.order_date is not None
        and order.requested_delivery_date is not None
        and order.requested_delivery_date < order.order_date
    ):
        yield Anomaly(
            code="DELIVERY_BEFORE_ORDER",
            severity=WARNING,
            message=(
                f"Livraison demandée le {order.requested_delivery_date:%d/%m/%Y}, avant la date "
                f"de commande ({order.order_date:%d/%m/%Y})."
            ),
            field="requested_delivery_date",
        )


def check_order_totals(order: PurchaseOrder, settings: RuleSettings) -> Iterator[Anomaly]:
    # Chaque ligne peut porter son propre arrondi : la tolérance s'additionne.
    tolerance_ht = settings.line_tolerance * len(order.lines)
    computed_ht = order.computed_total_ht
    if (
        order.total_ht is not None
        and computed_ht is not None
        and abs(order.total_ht - computed_ht) > tolerance_ht
    ):
        yield Anomaly(
            code="ORDER_TOTAL_HT_MISMATCH",
            severity=BLOCKING,
            message=(
                f"Total HT incohérent : {format_eur(order.total_ht)} écrit, "
                f"{format_eur(computed_ht)} calculé."
            ),
            field="total_ht",
        )

    computed_ttc, breakdown = order.computed_total_ttc, order.vat_breakdown
    if order.total_ttc is not None and computed_ttc is not None and breakdown is not None:
        tolerance_ttc = tolerance_ht + settings.line_tolerance * len(breakdown)
        if abs(order.total_ttc - computed_ttc) > tolerance_ttc:
            yield Anomaly(
                code="ORDER_TOTAL_TTC_MISMATCH",
                severity=BLOCKING,
                message=(
                    f"Total TTC incohérent : {format_eur(order.total_ttc)} écrit, "
                    f"{format_eur(computed_ttc)} calculé."
                ),
                field="total_ttc",
            )


def check_duplicate_products(order: PurchaseOrder, settings: RuleSettings) -> Iterator[Anomaly]:
    # Avertissement seulement : deux lignes du même produit peuvent être voulues (ex. deux
    # dates de livraison), mais c'est souvent une erreur de saisie du client.
    first_line: dict[str, int] = {}
    for line in order.lines:
        if not line.product_code:
            continue
        key = line.product_code.casefold()
        if key in first_line:
            yield Anomaly(
                code="DUPLICATE_PRODUCT_CODE",
                severity=WARNING,
                message=(f"Référence {line.product_code} déjà présente ligne {first_line[key]}."),
                field=f"lines[{line.line_number}].product_code",
                line_number=line.line_number,
            )
        else:
            first_line[key] = line.line_number


# --- Règles sur chaque ligne ----------------------------------------------------------------


def check_lines(order: PurchaseOrder, settings: RuleSettings) -> Iterator[Anomaly]:
    for line in order.lines:
        yield from _check_required_fields(line)
        yield from _check_values(line)
        yield from _check_vat(line, settings)
        yield from _check_line_total(line, settings)


def _line_anomaly(
    line: PurchaseOrderLine, code: str, severity: Severity, field: str, message: str
) -> Anomaly:
    label = line.description or line.product_code or "sans désignation"
    return Anomaly(
        code=code,
        severity=severity,
        message=f"Ligne {line.line_number} ({label}) : {message}",
        field=f"lines[{line.line_number}].{field}",
        line_number=line.line_number,
    )


def _check_required_fields(line: PurchaseOrderLine) -> Iterator[Anomaly]:
    if not line.product_code:
        yield _line_anomaly(
            line, "MISSING_PRODUCT_CODE", BLOCKING, "product_code", "référence manquante."
        )
    if line.quantity is None:
        yield _line_anomaly(line, "MISSING_QUANTITY", BLOCKING, "quantity", "quantité manquante.")
    if line.unit_price is None:
        yield _line_anomaly(line, "MISSING_UNIT_PRICE", BLOCKING, "unit_price", "prix manquant.")


def _check_values(line: PurchaseOrderLine) -> Iterator[Anomaly]:
    if line.quantity is not None and line.quantity <= 0:
        yield _line_anomaly(
            line, "INVALID_QUANTITY", BLOCKING, "quantity", f"quantité invalide ({line.quantity})."
        )
    if line.unit_price is not None and line.unit_price < 0:
        yield _line_anomaly(
            line,
            "NEGATIVE_UNIT_PRICE",
            BLOCKING,
            "unit_price",
            f"prix négatif ({format_eur(line.unit_price)}).",
        )


def _check_vat(line: PurchaseOrderLine, settings: RuleSettings) -> Iterator[Anomaly]:
    if line.vat_rate is None:
        yield _line_anomaly(line, "MISSING_VAT_RATE", WARNING, "vat_rate", "taux de TVA absent.")
    elif line.vat_rate not in settings.allowed_vat_rates:
        yield _line_anomaly(
            line,
            "UNKNOWN_VAT_RATE",
            BLOCKING,
            "vat_rate",
            f"taux de TVA inconnu ({format_rate(line.vat_rate)}).",
        )
    elif line.vat_rate == 0:
        yield _line_anomaly(
            line,
            "ZERO_VAT_RATE",
            WARNING,
            "vat_rate",
            "TVA à 0 % : vérifier le motif d'exonération.",
        )


def _check_line_total(line: PurchaseOrderLine, settings: RuleSettings) -> Iterator[Anomaly]:
    computed = line.computed_total
    if line.line_total is None or computed is None:
        return
    if abs(line.line_total - computed) > settings.line_tolerance:
        yield _line_anomaly(
            line,
            "LINE_TOTAL_MISMATCH",
            BLOCKING,
            "line_total",
            f"total incohérent ({format_eur(line.line_total)} écrit, {format_eur(computed)} "
            f"calculé = {line.quantity} × {format_eur(line.unit_price or Decimal('0'))}).",
        )


Rule = Callable[[PurchaseOrder, RuleSettings], Iterator[Anomaly]]

# Ordre d'affichage : la commande d'abord, puis les lignes, puis les totaux.
RULES: tuple[Rule, ...] = (
    check_customer,
    check_document_number,
    check_has_lines,
    check_dates,
    check_lines,
    check_duplicate_products,
    check_order_totals,
)
