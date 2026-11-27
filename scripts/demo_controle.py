"""Démo des contrôles : affiche les anomalies de deux bons de commande d'exemple.

uv run python -m scripts.demo_controle
"""

from __future__ import annotations

from datetime import date
from decimal import Decimal

from app.domain.models import PurchaseOrder, PurchaseOrderLine, Severity
from app.domain.rules import check_order, format_eur, format_rate

ICONS = {Severity.BLOCKING: "❌", Severity.WARNING: "⚠️ "}


def line(
    n: int, code: str, label: str, qty: str, price: str, vat: str, total: str
) -> PurchaseOrderLine:
    return PurchaseOrderLine(
        line_number=n,
        product_code=code,
        description=label,
        quantity=Decimal(qty),
        unit_price=Decimal(price),
        vat_rate=Decimal(vat),
        line_total=Decimal(total),
    )


EXAMPLES = {
    "Commande correcte": PurchaseOrder(
        customer_name="Épicerie Martin",
        document_number="CMD-2026-118",
        order_date=date(2026, 11, 2),
        lines=(
            line(1, "P006", "Riz 1 kg", "100", "1.85", "5.5", "185.00"),
            line(2, "P019", "Lessive 2 L", "10", "5.90", "20", "59.00"),
        ),
        total_ht=Decimal("244.00"),
        total_ttc=Decimal("265.98"),
    ),
    "Commande avec erreurs (exemple du café)": PurchaseOrder(
        customer_name="Épicerie Martin",
        order_date=date(2026, 11, 2),
        lines=(
            line(1, "P006", "Riz 1 kg", "100", "1.85", "5.5", "185.00"),
            line(2, "P004", "Café 250 g", "50", "4.20", "5.5", "215.00"),
            line(3, "P006", "Riz 1 kg", "20", "1.85", "19.6", "37.00"),
        ),
        total_ht=Decimal("437.00"),
    ),
}


def main() -> None:
    for title, order in EXAMPLES.items():
        check = check_order(order)
        status = "✅ PRÊTE" if check.is_ready else "❌ À CORRIGER"
        print(f"\n=== {title} : {status} ===")
        print(f"Client : {order.customer_name} | n° {order.document_number or '—'}")
        for item in order.lines:
            mark = "❌" if item.line_number in check.blocked_line_numbers else "✅"
            print(
                f"  {mark} L{item.line_number} {item.description} : {item.quantity} × "
                f"{format_eur(item.unit_price or Decimal('0'))}"
            )
        if order.computed_total_ht is not None:
            print(f"Total HT calculé : {format_eur(order.computed_total_ht)}")
        for vat in order.vat_breakdown or ():
            print(
                f"  TVA {format_rate(vat.rate)} sur {format_eur(vat.base)} : "
                f"{format_eur(vat.amount)}"
            )
        if order.computed_total_ttc is not None:
            print(f"Total TTC calculé : {format_eur(order.computed_total_ttc)}")
        if check.anomalies:
            print("Anomalies :")
            for anomaly in check.anomalies:
                print(f"  {ICONS[anomaly.severity]} {anomaly.message}")


if __name__ == "__main__":
    main()
