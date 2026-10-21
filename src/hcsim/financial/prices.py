"""Stock price models.

A price model does two jobs. `step` advances the actual price once per simulated month, before
any domain runs, and `projection` gives the deterministic planning price that award sizing reads
for a forward year. Each trajectory holds its own copy. The engine passes each step's calendar
month, because the Financial Layer doesn't own the calendar.

Every default is invented.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable


class PriceModelError(ValueError):
    """A price model declaration that can't be used."""


@runtime_checkable
class StockPriceModel(Protocol):
    @property
    def price(self) -> float: ...

    def step(self, calendar_month: int) -> None: ...

    def projection(self, forward_years: int) -> float: ...

    def declaration(self) -> dict[str, Any]: ...


def _check_price(price: float) -> None:
    if not price > 0:
        raise PriceModelError(f"price must be positive, got {price!r}")


@dataclass
class StaticPrice:
    """A price that never moves."""

    opening_price: float

    def __post_init__(self) -> None:
        _check_price(self.opening_price)

    @property
    def price(self) -> float:
        return self.opening_price

    def step(self, calendar_month: int) -> None:
        return None

    def projection(self, forward_years: int) -> float:
        return self.opening_price

    def declaration(self) -> dict[str, Any]:
        return {"type": "StaticPrice", "params": {"opening_price": self.opening_price}}


@dataclass
class AnnualGrowthPrice:
    """A price multiplied by `growth_rate` once a year, in `growth_month`.

    Growth never applies on the first step, so a run starting in `growth_month` opens at the
    opening price. The projection for forward year n is `price * growth_rate ** n`.
    """

    opening_price: float
    growth_rate: float = 1.08
    growth_month: int = 1

    def __post_init__(self) -> None:
        problems = []
        if not self.opening_price > 0:
            problems.append(f"opening_price must be positive, got {self.opening_price!r}")
        if not 0.5 <= self.growth_rate <= 2.0:
            problems.append(
                f"growth_rate is a multiplier between 0.5 and 2.0, got {self.growth_rate!r}; "
                "for 8% growth write 1.08, because 0.08 would mean a 92% annual drop"
            )
        if self.growth_month not in range(1, 13):
            problems.append(
                f"growth_month must be a calendar month 1-12, got {self.growth_month!r}"
            )
        if problems:
            raise PriceModelError("; ".join(problems))
        self._price = self.opening_price
        self._steps = 0

    @property
    def price(self) -> float:
        return self._price

    def step(self, calendar_month: int) -> None:
        if self._steps > 0 and calendar_month == self.growth_month:
            self._price *= self.growth_rate
        self._steps += 1

    def projection(self, forward_years: int) -> float:
        if forward_years < 0:
            raise PriceModelError("forward_years must be non-negative")
        return self._price * self.growth_rate**forward_years

    def declaration(self) -> dict[str, Any]:
        return {
            "type": "AnnualGrowthPrice",
            "params": {
                "opening_price": self.opening_price,
                "growth_rate": self.growth_rate,
                "growth_month": self.growth_month,
            },
        }


PRICE_MODELS: dict[str, type[StaticPrice] | type[AnnualGrowthPrice]] = {
    "StaticPrice": StaticPrice,
    "AnnualGrowthPrice": AnnualGrowthPrice,
}


def from_declaration(declaration: dict[str, Any]) -> StaticPrice | AnnualGrowthPrice:
    """Build a model from `{"type": ..., "params": {...}}`, refusing unknown types by name."""
    kind = declaration.get("type")
    if kind not in PRICE_MODELS:
        raise PriceModelError(f"unknown price model {kind!r}; known: {', '.join(PRICE_MODELS)}")
    return PRICE_MODELS[kind](**declaration.get("params", {}))
