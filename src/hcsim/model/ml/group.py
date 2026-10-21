"""Group models: rates and counts per group, for domains with no per-entity row to score.

Both models give a group its own estimate when it has enough history, and the pooled estimate
otherwise. A group outside the fitted range, including UNATTRIBUTED, reads the pooled value.
"""

from __future__ import annotations

from typing import ClassVar, Self

import numpy as np
from numpy.typing import NDArray

from hcsim.model.ml.entity import ModelNotFittedError

Ints = NDArray[np.int64]
Floats = NDArray[np.float64]


class _PooledFallback:
    _values: Floats | None = None
    _pooled: float | None = None

    def _require(self) -> tuple[Floats, float]:
        if self._values is None or self._pooled is None:
            raise ModelNotFittedError("fit the model first")
        return self._values, self._pooled

    def rate_for(self, group: int) -> float:
        values, pooled = self._require()
        return float(values[group]) if 0 <= group < len(values) else pooled

    def rates(self, groups: Ints) -> Floats:
        values, pooled = self._require()
        out = np.full(groups.shape, pooled, dtype=np.float64)
        known = (groups >= 0) & (groups < len(values))
        out[known] = values[groups[known]]
        return out

    @property
    def pooled(self) -> float:
        return self._require()[1]


class BackfillRateModel(_PooledFallback):
    """The share of departures that open a backfill requisition, per cost center."""

    ALGORITHM_NAME: ClassVar[str] = "backfill_rate"

    def __init__(self, n_groups: int, *, min_departures: int = 20) -> None:
        self.n_groups = n_groups
        self.min_departures = min_departures

    def fit(self, departure_groups: Ints, backfilled: NDArray[np.bool_]) -> Self:
        if len(departure_groups) != len(backfilled):
            raise ValueError("one backfill flag per departure")
        if len(departure_groups) == 0:
            raise ValueError("can't fit a backfill rate on zero departures")
        groups = np.asarray(departure_groups, dtype=np.int64)
        flags = np.asarray(backfilled, dtype=np.float64)
        known = (groups >= 0) & (groups < self.n_groups)
        counts = np.bincount(groups[known], minlength=self.n_groups)
        events = np.bincount(groups[known], weights=flags[known], minlength=self.n_groups)
        pooled = float(flags.mean())
        with np.errstate(invalid="ignore", divide="ignore"):
            own = events / counts
        self._values = np.where(counts >= self.min_departures, own, pooled)
        self._pooled = pooled
        return self

    @classmethod
    def constant(cls, rate: float, n_groups: int) -> BackfillRateModel:
        if not 0.0 <= rate <= 1.0:
            raise ValueError("rate must lie in [0, 1]")
        model = cls(n_groups)
        model._values = np.full(n_groups, rate)
        model._pooled = rate
        return model


class PoissonGrowthModel(_PooledFallback):
    """Growth requisitions opened per cost center per month, drawn as Poisson counts."""

    ALGORITHM_NAME: ClassVar[str] = "poisson_growth"

    def __init__(self, n_groups: int, *, min_months: int = 6) -> None:
        self.n_groups = n_groups
        self.min_months = min_months

    def fit(self, monthly_counts: Floats, observed: NDArray[np.bool_] | None = None) -> Self:
        """`monthly_counts` is (months, groups). `observed` marks the months a group existed."""
        counts = np.asarray(monthly_counts, dtype=np.float64)
        if counts.ndim != 2 or counts.shape[1] != self.n_groups:
            raise ValueError(f"monthly_counts must be (months, {self.n_groups})")
        if (counts < 0).any():
            raise ValueError("counts must be non-negative")
        seen = np.ones_like(counts, dtype=bool) if observed is None else np.asarray(observed)
        months = seen.sum(axis=0)
        if months.sum() == 0:
            raise ValueError("can't fit growth on zero observed months")
        pooled = float(counts[seen].mean())
        with np.errstate(invalid="ignore", divide="ignore"):
            own = np.where(seen, counts, 0).sum(axis=0) / months
        self._values = np.where(months >= self.min_months, own, pooled)
        self._pooled = pooled
        return self

    def sample(self, groups: Ints, rng: np.random.Generator) -> Ints:
        return rng.poisson(self.rates(np.asarray(groups, dtype=np.int64))).astype(np.int64)
