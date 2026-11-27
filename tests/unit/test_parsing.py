from datetime import date
from decimal import Decimal

import pytest

from app.domain.parsing import find_dates, parse_date, parse_decimal


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("1,85", "1.85"),
        ("1,85 €", "1.85"),
        ("4,20 € HT", "4.20"),
        ("1 234,50 €", "1234.50"),
        ("1 234,50", "1234.50"),  # espace insécable
        ("1 234,50", "1234.50"),  # espace fine insécable (format Word)
        ("1.234,50", "1234.50"),
        ("1,234.50", "1234.50"),
        ("1.85", "1.85"),  # cellule Excel lue par Docling
        ("12 345 678,90", "12345678.90"),
        ("1,234,567", "1234567"),
        ("100", "100"),
        ("50 sachets", "50"),
        ("2,5 kg", "2.5"),
        ("5,5 %", "5.5"),
        ("-12,00", "-12.00"),
        ("Total : 244,00 €.", "244.00"),
    ],
)
def test_parse_decimal(text: str, expected: str) -> None:
    assert parse_decimal(text) == Decimal(expected)


@pytest.mark.parametrize("text", [None, "", "gratuit", "selon devis"])
def test_parse_decimal_returns_none_when_there_is_no_number(text: str | None) -> None:
    assert parse_decimal(text) is None


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("02/11/2026", date(2026, 11, 2)),
        ("2/11/2026", date(2026, 11, 2)),
        ("02-11-2026", date(2026, 11, 2)),
        ("02.11.26", date(2026, 11, 2)),
        ("2026-11-02", date(2026, 11, 2)),
        ("2 novembre 2026", date(2026, 11, 2)),
        ("1er décembre 2026", date(2026, 12, 1)),
        ("le 15 aout 2026", date(2026, 8, 15)),
        ("Livraison souhaitée : 09/11/2026", date(2026, 11, 9)),
        ("2026-11-12T08:30:00+00:00", date(2026, 11, 12)),  # en-tête « Date: » d'un email
    ],
)
def test_parse_date(text: str, expected: date) -> None:
    assert parse_date(text) == expected


def test_find_dates_reads_every_format() -> None:
    text = "Commande du 02/11/2026, envoyée le 2026-11-03T09:00, livrer le 1er décembre 2026."

    assert find_dates(text) == {date(2026, 11, 2), date(2026, 11, 3), date(2026, 12, 1)}


def test_find_dates_ignores_impossible_dates() -> None:
    assert find_dates("le 31/02/2026 ou jamais") == set()


@pytest.mark.parametrize("text", [None, "", "dès que possible", "31/02/2026", "2 brumaire 2026"])
def test_parse_date_returns_none_when_unreadable(text: str | None) -> None:
    assert parse_date(text) is None
