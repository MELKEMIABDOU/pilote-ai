"""Bons de commande d'exemple pour les tests (montants du rapport Horizon)."""

from __future__ import annotations

from datetime import date
from decimal import Decimal

from app.domain.models import PurchaseOrder, PurchaseOrderLine


def d(value: str | None) -> Decimal | None:
    return None if value is None else Decimal(value)


def make_line(
    line_number: int = 1,
    *,
    product_code: str | None = "P006",
    description: str | None = "Riz 1 kg",
    quantity: str | None = "100",
    unit_price: str | None = "1.85",
    vat_rate: str | None = "5.5",
    line_total: str | None = "185.00",
) -> PurchaseOrderLine:
    return PurchaseOrderLine(
        line_number=line_number,
        product_code=product_code,
        description=description,
        quantity=d(quantity),
        unit_price=d(unit_price),
        vat_rate=d(vat_rate),
        line_total=d(line_total),
    )


LESSIVE = {
    "product_code": "P019",
    "description": "Lessive 2 L",
    "quantity": "10",
    "unit_price": "5.90",
    "vat_rate": "20",
    "line_total": "59.00",
}


def make_order(
    *lines: PurchaseOrderLine,
    customer_name: str | None = "Épicerie Martin",
    document_number: str | None = "CMD-2026-118",
    total_ht: str | None = "244.00",
    total_ttc: str | None = "265.98",
    order_date: date | None = date(2026, 11, 2),
    requested_delivery_date: date | None = date(2026, 11, 9),
) -> PurchaseOrder:
    """Par défaut : une commande CORRECTE (riz à 5,5 % + lessive à 20 %).

    Riz 100 × 1,85 = 185,00 ; lessive 10 × 5,90 = 59,00 ; total HT 244,00.
    TVA : 185,00 × 5,5 % = 10,175 → 10,18 ; 59,00 × 20 % = 11,80 ; total TTC 265,98.
    """
    return PurchaseOrder(
        customer_name=customer_name,
        document_number=document_number,
        order_date=order_date,
        requested_delivery_date=requested_delivery_date,
        lines=lines or (make_line(1), make_line(2, **LESSIVE)),
        total_ht=d(total_ht),
        total_ttc=d(total_ttc),
    )
