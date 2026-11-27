"""Le modèle de commande : types, calculs en Decimal, et commandes imparfaites représentables."""

from decimal import Decimal

import pytest
from pydantic import ValidationError

from app.domain.models import PurchaseOrder, PurchaseOrderLine, round_money
from tests.fixtures.orders import make_line, make_order


def test_line_total_is_quantity_times_price_rounded_half_up() -> None:
    # 3 × 3,335 = 10,005 : l'arrondi commercial donne 10,01 (l'arrondi « au pair » donnerait 10,00).
    line = make_line(quantity="3", unit_price="3.335")

    assert line.computed_total == Decimal("10.01")
    assert round_money(Decimal("2.345")) == Decimal("2.35")


def test_decimal_avoids_float_errors() -> None:
    line = make_line(quantity="3", unit_price="1.10")

    assert 3 * 1.10 != 3.30  # le piège des float
    assert line.computed_total == Decimal("3.30")


def test_float_amounts_are_refused() -> None:
    with pytest.raises(ValidationError, match="not float"):
        PurchaseOrderLine.model_validate({"line_number": 1, "unit_price": 1.85})
    with pytest.raises(ValidationError, match="not float"):
        PurchaseOrder.model_validate({"total_ht": 244.0})


def test_decimal_strings_are_accepted() -> None:
    line = PurchaseOrderLine.model_validate({"line_number": 1, "unit_price": "1.85"})

    assert line.unit_price == Decimal("1.85")


def test_incomplete_or_wrong_values_are_still_representable() -> None:
    # Le modèle décrit ce qui a été lu ; ce sont les règles qui signalent les problèmes.
    line = make_line(product_code=None, quantity="0", unit_price="-1")
    order = PurchaseOrder(lines=(line,))

    assert order.lines[0].quantity == 0
    assert PurchaseOrder().lines == ()


def test_order_is_immutable() -> None:
    order = make_order()

    with pytest.raises(ValidationError):
        order.customer_name = "Autre"


def test_unknown_fields_are_refused() -> None:
    with pytest.raises(ValidationError, match="extra"):
        PurchaseOrder.model_validate({"client": "Martin"})


def test_line_numbers_must_be_unique() -> None:
    with pytest.raises(ValidationError, match="line_number values must be unique"):
        PurchaseOrder(lines=(make_line(1), make_line(1)))


def test_currency_is_normalized() -> None:
    assert PurchaseOrder(currency="eur").currency == "EUR"


def test_totals_and_vat_breakdown() -> None:
    order = make_order()

    assert order.computed_total_ht == Decimal("244.00")
    assert [(v.rate, v.base, v.amount) for v in order.vat_breakdown or ()] == [
        (Decimal("20"), Decimal("59.00"), Decimal("11.80")),
        (Decimal("5.5"), Decimal("185.00"), Decimal("10.18")),  # 10,175 → 10,18
    ]
    assert order.computed_total_ttc == Decimal("265.98")


def test_totals_are_unknown_when_a_line_is_incomplete() -> None:
    order = make_order(make_line(1), make_line(2, quantity=None))

    assert order.computed_total_ht is None
    assert order.computed_total_ttc is None
