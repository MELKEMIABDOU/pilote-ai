"""Conversion des valeurs écrites sur un bon de commande (« 1 234,50 € », « 2 novembre 2026 »).

Le LLM recopie les valeurs telles qu'elles sont écrites ; c'est CE code, déterministe et
testé, qui les convertit. On ne confie jamais une conversion de nombre à un LLM.

Règle pour les séparateurs : la virgule est le séparateur décimal français. Quand une valeur
contient à la fois « . » et « , », le dernier des deux est le séparateur décimal
(« 1.234,50 » et « 1,234.50 » valent tous deux 1234,50). Un point seul est décimal (« 1.85 »,
format des cellules Excel lues par Docling).
"""

from __future__ import annotations

import re
from datetime import date
from decimal import Decimal, InvalidOperation

# Un nombre, éventuellement avec séparateurs de milliers (espaces, espaces insécables, ').
_NUMBER = re.compile(r"[-+]?\d[\d   .,']*")
_THOUSANDS_SPACES = re.compile(r"[   ']")

_MONTHS = {
    "janvier": 1,
    "février": 2,
    "fevrier": 2,
    "mars": 3,
    "avril": 4,
    "mai": 5,
    "juin": 6,
    "juillet": 7,
    "août": 8,
    "aout": 8,
    "septembre": 9,
    "octobre": 10,
    "novembre": 11,
    "décembre": 12,
    "decembre": 12,
}
_NUMERIC_DATE = re.compile(r"\b(\d{1,2})[/.\-](\d{1,2})[/.\-](\d{2}|\d{4})\b")
# (?!\d) et non \b : « 2026-11-12T08:30:00+00:00 » (date d'un email) doit être lu.
_ISO_DATE = re.compile(r"\b(\d{4})-(\d{1,2})-(\d{1,2})(?!\d)")
_TEXT_DATE = re.compile(r"\b(\d{1,2})(?:er)?\s+([a-zéû]+)\s+(\d{4})\b", re.IGNORECASE)


def parse_decimal(text: str | None) -> Decimal | None:
    """Premier nombre du texte : « 1 234,50 € » → 1234.50 ; « 50 sachets » → 50.

    Renvoie None si le texte ne contient pas de nombre lisible.
    """
    if text is None:
        return None
    match = _NUMBER.search(text)
    if match is None:
        return None
    raw = _THOUSANDS_SPACES.sub("", match.group()).rstrip(".,")
    if "," in raw and "." in raw:
        if raw.rfind(",") > raw.rfind("."):
            raw = raw.replace(".", "").replace(",", ".")  # 1.234,50
        else:
            raw = raw.replace(",", "")  # 1,234.50
    elif "," in raw:
        # Plusieurs virgules = séparateurs de milliers (1,234,567) ; une seule = décimale.
        raw = raw.replace(",", "") if raw.count(",") > 1 else raw.replace(",", ".")
    elif raw.count(".") > 1:
        raw = raw.replace(".", "")  # 1.234.567
    try:
        return Decimal(raw)
    except InvalidOperation:
        return None


def parse_date(text: str | None) -> date | None:
    """« 02/11/2026 », « 2026-11-02 », « 2 novembre 2026 », « 1er décembre 2026 » → date.

    Format numérique : jour/mois/année (convention française). Renvoie None si illisible
    ou si la date n'existe pas (ex. 31/02/2026).
    """
    if text is None:
        return None
    try:
        if match := _ISO_DATE.search(text):
            year, month, day = (int(group) for group in match.groups())
            return date(year, month, day)
        if match := _NUMERIC_DATE.search(text):
            day, month, year = (int(group) for group in match.groups())
            return date(year + 2000 if year < 100 else year, month, day)
        if match := _TEXT_DATE.search(text):
            month_number = _MONTHS.get(match.group(2).casefold())
            if month_number is not None:
                return date(int(match.group(3)), month_number, int(match.group(1)))
    except ValueError:
        return None  # date impossible, ex. 31/02/2026
    return None


def find_dates(text: str) -> set[date]:
    """Toutes les dates lisibles écrites dans le texte, quel que soit leur format.

    Sert à vérifier une date recopiée par le LLM : « 2026-11-20 » est bien dans un document
    qui écrit « 20/11/2026 » (même date, autre écriture).
    """
    found = set()
    for pattern in (_ISO_DATE, _NUMERIC_DATE, _TEXT_DATE):
        for match in pattern.finditer(text):
            if (parsed := parse_date(match.group())) is not None:
                found.add(parsed)
    return found
