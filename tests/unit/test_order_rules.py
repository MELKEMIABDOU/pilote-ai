"""Les contrôles d'un bon de commande : une commande juste, puis une règle par test."""

from datetime import date
from decimal import Decimal

import pytest

from app.domain.models import Severity
from app.domain.rules import OrderCheck, RuleSettings, check_order, format_eur, format_rate
from tests.fixtures.orders import LESSIVE, make_line, make_order


def codes(check: OrderCheck) -> list[str]:
    return [anomaly.code for anomaly in check.anomalies]


def test_a_correct_order_has_no_anomaly() -> None:
    check = check_order(make_order())

    assert check.anomalies == ()
    assert check.is_ready
    assert len(check.ready_lines) == 2


def test_the_coffee_example_wrong_line_total() -> None:
    # Café : 50 × 4,20 = 210,00 €, mais le client a écrit 215,00 €.
    cafe = make_line(
        2,
        product_code="P004",
        description="Café 250 g",
        quantity="50",
        unit_price="4.20",
        line_total="215.00",
    )
    order = make_order(make_line(1), cafe, total_ht="400.00", total_ttc=None)

    check = check_order(order)

    assert codes(check) == ["LINE_TOTAL_MISMATCH", "ORDER_TOTAL_HT_MISMATCH"]
    [line_issue, total_issue] = check.anomalies
    assert line_issue.line_number == 2
    assert "215,00 € écrit, 210,00 € calculé" in line_issue.message
    assert "400,00 € écrit, 395,00 € calculé" in total_issue.message
    assert not check.is_ready
    assert [line.line_number for line in check.ready_lines] == [1]


def test_rounding_difference_of_one_cent_is_tolerated() -> None:
    # 3 × 3,335 = 10,005 → 10,01 calculé ; le client a arrondi à 10,00 : simple arrondi.
    line = make_line(quantity="3", unit_price="3.335", line_total="10.00")
    order = make_order(line, total_ht=None, total_ttc=None)

    assert check_order(order).anomalies == ()
    strict = RuleSettings(line_tolerance=Decimal("0"))
    assert codes(check_order(order, strict)) == ["LINE_TOTAL_MISMATCH"]


def test_customer_is_required() -> None:
    check = check_order(make_order(customer_name=None))

    assert codes(check) == ["MISSING_CUSTOMER"]
    assert not check.is_ready


def test_a_customer_code_alone_is_enough() -> None:
    order = make_order(customer_name=None).model_copy(update={"customer_code": "C042"})

    assert check_order(order).is_ready


def test_missing_document_number_is_only_a_warning() -> None:
    check = check_order(make_order(document_number=None))

    assert codes(check) == ["MISSING_DOCUMENT_NUMBER"]
    assert check.anomalies[0].severity is Severity.WARNING
    assert check.is_ready


def test_order_without_lines_is_blocked() -> None:
    order = make_order().model_copy(update={"lines": ()})

    assert codes(check_order(order)) == ["NO_LINES"]


@pytest.mark.parametrize(
    ("field", "code"),
    [
        ("product_code", "MISSING_PRODUCT_CODE"),
        ("quantity", "MISSING_QUANTITY"),
        ("unit_price", "MISSING_UNIT_PRICE"),
    ],
)
def test_required_line_fields(field: str, code: str) -> None:
    order = make_order(make_line(1, **{field: None}), total_ht=None, total_ttc=None)

    check = check_order(order)

    # Une seule anomalie : pas de « total incohérent » en cascade quand on ne peut pas calculer.
    assert codes(check) == [code]
    assert check.anomalies[0].field == f"lines[1].{field}"


def test_missing_quantity_does_not_trigger_order_total_checks() -> None:
    order = make_order(make_line(1, quantity=None), total_ht="999.00", total_ttc="999.00")

    assert codes(check_order(order)) == ["MISSING_QUANTITY"]


@pytest.mark.parametrize("quantity", ["0", "-5"])
def test_quantity_must_be_positive(quantity: str) -> None:
    order = make_order(make_line(quantity=quantity, line_total=None), total_ht=None, total_ttc=None)

    assert codes(check_order(order)) == ["INVALID_QUANTITY"]


def test_negative_price_is_blocked() -> None:
    order = make_order(
        make_line(unit_price="-1.85", line_total=None), total_ht=None, total_ttc=None
    )

    assert codes(check_order(order)) == ["NEGATIVE_UNIT_PRICE"]


def test_missing_vat_rate_is_a_warning() -> None:
    order = make_order(make_line(vat_rate=None), total_ht="185.00", total_ttc=None)

    check = check_order(order)

    assert codes(check) == ["MISSING_VAT_RATE"]
    assert check.is_ready


def test_unknown_vat_rate_is_blocked() -> None:
    # 19,6 % : ancien taux normal (avant 2014), souvent une erreur de modèle de bon de commande.
    order = make_order(make_line(vat_rate="19.6"), total_ht="185.00", total_ttc=None)

    assert codes(check_order(order)) == ["UNKNOWN_VAT_RATE"]


def test_zero_vat_rate_asks_for_the_exemption_reason() -> None:
    order = make_order(make_line(vat_rate="0"), total_ht="185.00", total_ttc="185.00")

    check = check_order(order)

    assert codes(check) == ["ZERO_VAT_RATE"]
    assert check.is_ready


def test_vat_rate_written_with_extra_zeros_is_accepted() -> None:
    order = make_order(make_line(vat_rate="5.50"), total_ht="185.00", total_ttc="195.18")

    assert check_order(order).anomalies == ()


def test_wrong_ttc_total_is_blocked() -> None:
    check = check_order(make_order(total_ttc="270.00"))

    assert codes(check) == ["ORDER_TOTAL_TTC_MISMATCH"]
    assert "270,00 € écrit, 265,98 € calculé" in check.anomalies[0].message


def test_duplicate_product_is_a_warning_on_the_second_line() -> None:
    order = make_order(
        make_line(1),
        make_line(2, **LESSIVE),
        make_line(3, product_code="p006"),
        total_ht="429.00",
        total_ttc=None,
    )

    check = check_order(order)

    assert codes(check) == ["DUPLICATE_PRODUCT_CODE"]
    assert check.anomalies[0].line_number == 3
    assert "déjà présente ligne 1" in check.anomalies[0].message


def test_delivery_date_before_order_date_is_a_warning() -> None:
    order = make_order(order_date=date(2026, 11, 9), requested_delivery_date=date(2026, 11, 2))

    check = check_order(order)

    assert codes(check) == ["DELIVERY_BEFORE_ORDER"]
    assert "02/11/2026" in check.anomalies[0].message


def test_amounts_are_formatted_the_french_way() -> None:
    assert format_eur(Decimal("1234.5")) == "1 234,50 €"
    assert format_eur(Decimal("0.05")) == "0,05 €"


@pytest.mark.parametrize(
    ("rate", "label"), [("5.5", "5,5 %"), ("5.50", "5,5 %"), ("20", "20 %"), ("2.1", "2,1 %")]
)
def test_vat_rates_are_formatted_the_french_way(rate: str, label: str) -> None:
    assert format_rate(Decimal(rate)) == label
