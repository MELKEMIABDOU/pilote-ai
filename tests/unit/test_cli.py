import pytest

from app.cli import format_order_check, format_pages
from app.domain.rules import check_order
from tests.fixtures.orders import make_line, make_order


@pytest.mark.parametrize(
    ("pages", "label"),
    [((), "page ?"), ((4,), "p.4"), ((4, 5), "p.4-5"), ((4, 5, 6), "p.4-6")],
)
def test_format_pages(pages: tuple[int, ...], label: str) -> None:
    assert format_pages(pages) == label


def test_report_of_a_correct_order() -> None:
    report = format_order_check(check_order(make_order()), source="c.pdf", model="m", seconds=2)

    assert "c.pdf : ✅ PRÊTE À SAISIR" in report
    assert "✅ L1 P006 Riz 1 kg : 100 × 1,85 € (5,5 %)" in report
    assert "Total TTC calculé : 265,98 €" in report
    assert "l'ADV doit valider" in report


def test_report_of_an_order_with_anomalies() -> None:
    order = make_order(make_line(quantity=None, line_total=None), total_ht=None, total_ttc=None)

    report = format_order_check(check_order(order), source="c.pdf", model="m", seconds=2)

    assert "❌ À CORRIGER" in report
    assert "❌ L1 P006 Riz 1 kg : qté ?" in report
    assert "quantité manquante" in report
