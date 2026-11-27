"""Modèle métier ADV : le bon de commande tel qu'il a été LU sur le document du client.

Le modèle DÉCRIT, les règles (``rules.py``) JUGENT :
- ici, seulement les contrôles de TYPE (montants en ``Decimal``, pas de ``float``, numéros de
  ligne uniques) ;
- une quantité nulle, une référence manquante ou un total faux restent représentables : ce
  sont des anomalies à montrer à l'ADV, pas des erreurs qui empêchent de lire la commande.

Prix et totaux sont HORS TAXES (pratique B2B en France), avec un taux de TVA par ligne. On garde
les totaux ÉCRITS par le client (``line_total``, ``total_ht``, ``total_ttc``) à côté des totaux
CALCULÉS : les comparer est l'un des contrôles.
"""

from __future__ import annotations

from datetime import date
from decimal import ROUND_HALF_UP, Decimal
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

CENT = Decimal("0.01")
_MODEL_CONFIG = ConfigDict(extra="forbid", str_strip_whitespace=True, frozen=True)


def round_money(amount: Decimal) -> Decimal:
    """Arrondi commercial au centime : 0,005 → 0,01 (et non l'arrondi « au pair » de Python)."""
    return amount.quantize(CENT, rounding=ROUND_HALF_UP)


def _reject_float(value: object) -> object:
    # 3 * 1.10 == 3.3000000000000003 en float : l'argent passe par une chaîne ou un Decimal.
    if isinstance(value, float):
        raise ValueError("use a decimal string or Decimal, not float")
    return value


class Severity(StrEnum):
    BLOCKING = "bloquant"  # la commande ne peut pas être saisie en l'état
    WARNING = "avertissement"  # à vérifier, mais ne bloque pas


class Anomaly(BaseModel):
    """Un écart détecté par un contrôle, avec ce qu'il faut pour l'afficher à l'ADV."""

    model_config = _MODEL_CONFIG

    code: str  # identifiant stable, ex. "LINE_TOTAL_MISMATCH" (tests, écran, statistiques)
    severity: Severity
    message: str  # phrase lisible par l'ADV
    field: str  # champ concerné, ex. "lines[2].line_total"
    line_number: int | None = None  # None = anomalie sur la commande entière


class PurchaseOrderLine(BaseModel):
    model_config = _MODEL_CONFIG

    line_number: int = Field(gt=0)  # position sur le document (1 = première ligne)
    product_code: str | None = None  # référence produit, ex. "P006"
    description: str | None = None  # désignation, ex. "Riz 1 kg"
    quantity: Decimal | None = None  # Decimal : on peut commander 2,5 kg
    unit_price: Decimal | None = None  # prix unitaire HT
    vat_rate: Decimal | None = None  # en pourcentage : Decimal("5.5") pour 5,5 %
    line_total: Decimal | None = None  # total HT de la ligne ÉCRIT sur le document

    @field_validator("quantity", "unit_price", "vat_rate", "line_total", mode="before")
    @classmethod
    def reject_binary_float_money(cls, value: object) -> object:
        return _reject_float(value)

    @property
    def computed_total(self) -> Decimal | None:
        """Quantité × prix unitaire, arrondi au centime ; None si l'un des deux manque."""
        if self.quantity is None or self.unit_price is None:
            return None
        return round_money(self.quantity * self.unit_price)


class VatAmount(BaseModel):
    model_config = _MODEL_CONFIG

    rate: Decimal  # en pourcentage
    base: Decimal  # total HT soumis à ce taux
    amount: Decimal  # montant de TVA


class PurchaseOrder(BaseModel):
    """Données de commande normalisées ; les valeurs extraites restent des propositions."""

    model_config = _MODEL_CONFIG

    document_number: str | None = None  # n° de commande du client
    customer_code: str | None = None
    customer_name: str | None = None
    order_date: date | None = None
    requested_delivery_date: date | None = None
    currency: str = Field(default="EUR", min_length=3, max_length=3)
    lines: tuple[PurchaseOrderLine, ...] = ()
    total_ht: Decimal | None = None  # totaux ÉCRITS sur le document
    total_ttc: Decimal | None = None

    @field_validator("currency")
    @classmethod
    def normalize_currency(cls, value: str) -> str:
        return value.upper()

    @field_validator("total_ht", "total_ttc", mode="before")
    @classmethod
    def reject_float_total(cls, value: object) -> object:
        return _reject_float(value)

    @model_validator(mode="after")
    def validate_structure(self) -> PurchaseOrder:
        # Contrôle de STRUCTURE (sinon « la ligne 2 » serait ambiguë), pas un contrôle métier.
        line_numbers = [line.line_number for line in self.lines]
        if len(line_numbers) != len(set(line_numbers)):
            raise ValueError("line_number values must be unique")
        return self

    @property
    def computed_total_ht(self) -> Decimal | None:
        """Somme des totaux de ligne calculés ; None si une ligne n'est pas calculable."""
        totals = [line.computed_total for line in self.lines]
        if not totals or any(total is None for total in totals):
            return None
        return sum((total for total in totals if total is not None), Decimal("0"))

    @property
    def vat_breakdown(self) -> tuple[VatAmount, ...] | None:
        """TVA par taux (mention obligatoire sur facture) ; None si une ligne est incomplète.

        La TVA est calculée sur le total HT de chaque taux, puis arrondie une seule fois, et non
        ligne par ligne : on n'accumule pas les erreurs d'arrondi.
        """
        bases: dict[Decimal, Decimal] = {}
        for line in self.lines:
            if line.computed_total is None or line.vat_rate is None:
                return None
            bases[line.vat_rate] = bases.get(line.vat_rate, Decimal("0")) + line.computed_total
        if not bases:
            return None
        return tuple(
            VatAmount(rate=rate, base=base, amount=round_money(base * rate / 100))
            for rate, base in sorted(bases.items(), reverse=True)
        )

    @property
    def computed_total_ttc(self) -> Decimal | None:
        breakdown = self.vat_breakdown
        if self.computed_total_ht is None or breakdown is None:
            return None
        return self.computed_total_ht + sum((vat.amount for vat in breakdown), Decimal("0"))
