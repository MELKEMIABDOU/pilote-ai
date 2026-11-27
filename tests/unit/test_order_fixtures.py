"""Le jeu de commandes fictives est cohérent : sans LLM, on vérifie la « bonne réponse ».

Si on construit un PurchaseOrder à partir de la réponse attendue, les contrôles métier doivent
trouver exactement les anomalies attendues. Sinon, c'est le jeu de test qui est faux, et
l'évaluation de l'extraction mesurerait n'importe quoi.
"""

import json
from datetime import date
from decimal import Decimal
from pathlib import Path
from typing import Any

import pytest

from app.domain.models import PurchaseOrder, PurchaseOrderLine
from app.domain.rules import check_order

ORDERS = Path(__file__).parents[1] / "fixtures" / "orders"
EXPECTED: dict[str, dict[str, Any]] = json.loads((ORDERS / "expected.json").read_text("utf-8"))


def _decimal(value: str | None) -> Decimal | None:
    return None if value is None else Decimal(value)


def _date(value: str | None) -> date | None:
    return None if value in (None, "*") else date.fromisoformat(value)


def order_from_expected(expected: dict[str, Any]) -> PurchaseOrder:
    return PurchaseOrder(
        customer_name=expected["customer_name"],
        customer_code=expected["customer_code"],
        document_number=expected["document_number"],
        order_date=_date(expected["order_date"]),
        requested_delivery_date=_date(expected["requested_delivery_date"]),
        total_ht=_decimal(expected["total_ht"]),
        total_ttc=_decimal(expected["total_ttc"]),
        lines=tuple(
            PurchaseOrderLine(
                line_number=number,
                product_code=line["product_code"],
                quantity=_decimal(line["quantity"]),
                unit_price=_decimal(line["unit_price"]),
                vat_rate=_decimal(line["vat_rate"]),
                line_total=_decimal(line["line_total"]),
            )
            for number, line in enumerate(expected["lines"], start=1)
        ),
    )


def test_every_expected_answer_has_its_document() -> None:
    assert len(EXPECTED) == 10
    assert all((ORDERS / filename).is_file() for filename in EXPECTED)


@pytest.mark.parametrize("filename", sorted(EXPECTED))
def test_rules_find_the_expected_anomalies(filename: str) -> None:
    expected = EXPECTED[filename]

    found = sorted(a.code for a in check_order(order_from_expected(expected)).anomalies)

    assert found == expected["expected_anomalies"], expected["note"]
