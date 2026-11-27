from datetime import date
from decimal import Decimal
from pathlib import Path

import pytest

from app.domain.models import PurchaseOrder, PurchaseOrderLine
from scripts.eval_extraction import (
    DocumentScore,
    load,
    same_value,
    save,
    score_document,
    summarize,
)


def test_a_saved_evaluation_can_be_resumed(tmp_path: Path) -> None:
    done = DocumentScore("o01.pdf", "note", fields_ok=3, fields_total=4, errors=["x"])
    path = tmp_path / "extraction.json"

    save({"qwen3:8b": [done], "qwen3:4b": []}, path)

    assert load(path) == {"qwen3:8b": [done]}


@pytest.mark.parametrize(
    ("expected", "actual", "same"),
    [
        ("196.80", Decimal("196.8"), True),  # même nombre, écriture différente
        ("196.80", Decimal("196.9"), False),
        ("2026-11-02", date(2026, 11, 2), True),
        ("Épicerie Martin", "  épicerie martin ", True),
        (None, None, True),
        (None, "CMD-1", False),
        ("CMD-1", None, False),
        ("*", "n'importe quoi", True),  # champ non noté
    ],
)
def test_same_value(expected: str | None, actual: object, same: bool) -> None:
    assert same_value(expected, actual) is same


EXPECTED = {
    "note": "test",
    "customer_name": "Épicerie Martin",
    "customer_code": None,
    "document_number": "CMD-1",
    "order_date": "*",
    "requested_delivery_date": None,
    "total_ht": "185.00",
    "total_ttc": None,
    "lines": [
        {
            "product_code": "P006",
            "quantity": "100",
            "unit_price": "1.85",
            "vat_rate": "5.5",
            "line_total": "185.00",
        }
    ],
    "expected_anomalies": [],
}


def riz(**changes: object) -> PurchaseOrderLine:
    values: dict[str, object] = {
        "line_number": 1,
        "product_code": "P006",
        "quantity": Decimal("100"),
        "unit_price": Decimal("1.85"),
        "vat_rate": Decimal("5.5"),
        "line_total": Decimal("185.00"),
    }
    return PurchaseOrderLine.model_validate(values | changes)


def order(*lines: PurchaseOrderLine) -> PurchaseOrder:
    return PurchaseOrder(
        customer_name="Épicerie Martin",
        document_number="CMD-1",
        total_ht=Decimal("185"),
        lines=lines,
    )


def test_a_perfect_extraction() -> None:
    score = score_document("a.pdf", EXPECTED, order(riz()))

    assert (score.fields_ok, score.fields_total) == (11, 11)  # 6 en-têtes notés + 5 champs
    assert score.perfect


def test_wrong_and_extra_values_are_counted() -> None:
    score = score_document(
        "a.pdf", EXPECTED, order(riz(unit_price=Decimal("0")), riz(line_number=2))
    )

    assert (score.fields_ok, score.fields_total) == (10, 12)
    assert any("L1.unit_price" in error for error in score.errors)
    assert any("ligne en trop" in error for error in score.errors)


def test_summary() -> None:
    perfect = score_document("a.pdf", EXPECTED, order(riz()))
    perfect.verdict_ok = True

    summary = summarize([perfect])

    assert summary["field_accuracy"] == 1.0
    assert summary["perfect_orders"] == "1/1"
