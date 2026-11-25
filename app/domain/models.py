"""Validated business models for ADV purchase orders."""

from __future__ import annotations

from datetime import date
from decimal import Decimal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class PurchaseOrderLine(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    line_number: int = Field(gt=0)
    product_code: str | None = None
    description: str = Field(min_length=1)
    quantity: Decimal = Field(gt=0)
    unit_price: Decimal = Field(ge=0)
    line_total: Decimal | None = Field(default=None, ge=0)

    @field_validator("quantity", "unit_price", "line_total", mode="before")
    @classmethod
    def reject_binary_float_money(cls, value: object) -> object:
        if isinstance(value, float):
            raise ValueError("use a decimal string or Decimal, not float")
        return value


class PurchaseOrder(BaseModel):
    """Normalized order data; extracted values remain proposals until validated."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    document_number: str | None = None
    customer_code: str | None = None
    customer_name: str | None = None
    order_date: date | None = None
    requested_delivery_date: date | None = None
    currency: str = Field(default="EUR", min_length=3, max_length=3)
    lines: list[PurchaseOrderLine] = Field(min_length=1)
    total_amount: Decimal | None = Field(default=None, ge=0)

    @field_validator("currency")
    @classmethod
    def normalize_currency(cls, value: str) -> str:
        return value.upper()

    @field_validator("total_amount", mode="before")
    @classmethod
    def reject_float_total(cls, value: object) -> object:
        if isinstance(value, float):
            raise ValueError("use a decimal string or Decimal, not float")
        return value

    @model_validator(mode="after")
    def validate_order(self) -> PurchaseOrder:
        line_numbers = [line.line_number for line in self.lines]
        if len(line_numbers) != len(set(line_numbers)):
            raise ValueError("line_number values must be unique")
        if (
            self.order_date is not None
            and self.requested_delivery_date is not None
            and self.requested_delivery_date < self.order_date
        ):
            raise ValueError("requested_delivery_date cannot precede order_date")
        return self
