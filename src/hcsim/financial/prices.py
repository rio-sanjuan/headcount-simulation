"""Stock price models.

A price model does two jobs. `step` advances the actual price once per simulated month, before
any domain runs, and `projection` gives the deterministic planning price that award sizing reads
for a forward year. Each trajectory holds its own copy. The engine passes each step's calendar
month, because the Financial Layer doesn't own the calendar.

- `StaticPrice`: a price that never moves.
- `AnnualGrowthPrice`: a price multiplied by a growth rate once a year, in one calendar month.
- `CustomPrice`: a given monthly path, such as the realized month-end closes, and after it ends
  `extend_with`, a model that continues from the path's last price.
- `GBMPrice` (code only): geometric Brownian motion, monthly log-normal steps from its own
  seeded stream. It projects its expected price, and has no world calibration behind it.

Every default is invented.
"""

from __future__ import annotations

import copy
import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

import numpy as np


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


@dataclass
class CustomPrice:
    """A given monthly path, `path[k]` the price in the run's step k, and after it ends,
    `extend_with`: a model that continues from the path's last price.

    The extension opens at the path's last price, and takes its opening step in the path's last
    month, so its first move falls in the month after. Before the first step the price is
    `path[0]`. A projection that lands on the path reads it; one beyond its end is the
    extension's projection over the whole years past the end, rounded up.
    """

    path: Sequence[float]
    extend_with: StockPriceModel | None = None
    _step: int = field(default=-1, init=False)

    def __post_init__(self) -> None:
        values = [float(v) for v in self.path]
        problems = []
        if not values:
            problems.append("a custom path needs at least one price")
        elif not all(math.isfinite(v) and v > 0 for v in values):
            problems.append("a custom path's prices must be finite and positive")
        if problems:
            raise PriceModelError("; ".join(problems))
        self.path = tuple(values)
        if self.extend_with is not None:
            if not math.isclose(self.extend_with.price, values[-1], rel_tol=1e-9):
                raise PriceModelError(
                    f"an extension continues from the path's last price, {values[-1]!r}, "
                    f"not {self.extend_with.price!r}"
                )
            self.extend_with = copy.deepcopy(self.extend_with)

    @property
    def price(self) -> float:
        if self._step < len(self.path):
            return self.path[max(self._step, 0)]
        assert self.extend_with is not None
        return self.extend_with.price

    def step(self, calendar_month: int) -> None:
        self._step += 1
        if self._step >= len(self.path) - 1 and self.extend_with is not None:
            self.extend_with.step(calendar_month)
        elif self._step >= len(self.path):
            raise PriceModelError(
                f"the custom path covers {len(self.path)} months and has no extension for the next"
            )

    def projection(self, forward_years: int) -> float:
        if forward_years < 0:
            raise PriceModelError("forward_years must be non-negative")
        target = max(self._step, 0) + 12 * forward_years
        if target < len(self.path):
            return self.path[target]
        if self.extend_with is None:
            raise PriceModelError(
                f"a projection {forward_years} years ahead runs past the custom path's "
                f"{len(self.path)} months, and it has no extension"
            )
        here = max(self._step, len(self.path) - 1)
        return self.extend_with.projection(math.ceil((target - here) / 12))

    def declaration(self) -> dict[str, Any]:
        extension = None if self.extend_with is None else self.extend_with.declaration()
        return {
            "type": "CustomPrice",
            "params": {"path": list(self.path), "extend_with": extension},
        }


@dataclass
class GBMPrice:
    """Geometric Brownian motion: each month the price moves by a log-normal factor with an
    annual expected return `drift` and volatility `volatility`, from its own stream, seeded by
    `seed`. The projection n years ahead is the expected price, `price * exp(drift * n)`.

    Like the annual growth model, it doesn't move on its first step, so a run's first month
    opens at the opening price.

    Code only: it has no world calibration behind it, and every trajectory's copy draws the same
    path unless it's reseeded (`reseeded`).
    """

    opening_price: float
    drift: float = 0.08
    volatility: float = 0.3
    seed: int = 0
    _price: float = field(default=0.0, init=False)
    _steps: int = field(default=0, init=False)
    _rng: np.random.Generator = field(init=False, repr=False)

    def __post_init__(self) -> None:
        problems = []
        if not (math.isfinite(self.opening_price) and self.opening_price > 0):
            problems.append(f"opening_price must be positive, got {self.opening_price!r}")
        if not math.isfinite(self.drift):
            problems.append(f"drift must be finite, got {self.drift!r}")
        if not (math.isfinite(self.volatility) and self.volatility >= 0):
            problems.append(f"volatility must be non-negative, got {self.volatility!r}")
        if problems:
            raise PriceModelError("; ".join(problems))
        self._price = self.opening_price
        self._rng = np.random.default_rng(self.seed)

    def reseeded(self, seed: int) -> GBMPrice:
        """A fresh copy, from the opening price, on another stream."""
        return GBMPrice(self.opening_price, self.drift, self.volatility, seed)

    @property
    def price(self) -> float:
        return self._price

    def step(self, calendar_month: int) -> None:
        if self._steps > 0:
            sigma = self.volatility / math.sqrt(12)
            shock = float(self._rng.standard_normal())
            self._price *= math.exp((self.drift - self.volatility**2 / 2) / 12 + sigma * shock)
        self._steps += 1

    def projection(self, forward_years: int) -> float:
        if forward_years < 0:
            raise PriceModelError("forward_years must be non-negative")
        return self._price * math.exp(self.drift * forward_years)

    def declaration(self) -> dict[str, Any]:
        return {
            "type": "GBMPrice",
            "params": {
                "opening_price": self.opening_price,
                "drift": self.drift,
                "volatility": self.volatility,
                "seed": self.seed,
            },
        }


PriceModel = StaticPrice | AnnualGrowthPrice | CustomPrice | GBMPrice
PRICE_MODELS: dict[str, type[PriceModel]] = {
    "StaticPrice": StaticPrice,
    "AnnualGrowthPrice": AnnualGrowthPrice,
    "CustomPrice": CustomPrice,
    "GBMPrice": GBMPrice,
}


def from_declaration(declaration: dict[str, Any]) -> PriceModel:
    """Build a model from `{"type": ..., "params": {...}}`, refusing unknown types by name. A
    custom path's extension is itself a declaration."""
    kind = declaration.get("type")
    if kind not in PRICE_MODELS:
        raise PriceModelError(f"unknown price model {kind!r}; known: {', '.join(PRICE_MODELS)}")
    params = dict(declaration.get("params", {}))
    if kind == "CustomPrice" and params.get("extend_with") is not None:
        params["extend_with"] = from_declaration(params["extend_with"])
    return PRICE_MODELS[kind](**params)
