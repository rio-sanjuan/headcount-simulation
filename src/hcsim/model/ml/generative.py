"""Attribute generators for new requisitions.

`EmpiricalAttributeGenerator` draws whole attribute rows, uniformly with replacement, from rows
observed in the requisition's cost center, or from every observed row when the cost center has
too few. It never invents a combination, so a generated requisition always carries a profile a
real one had. A requisition with no generator would carry code 0 in every attribute, and every
hire from it would inherit that one improbable profile.

`ConditionalCategorical` draws one attribute at a time, given a context hierarchy from finest to
coarsest, such as cost center then org. A context reads the distribution at its finest level
with at least `min_count` observations, and the overall distribution when no level has enough.
A code of -1 at a level skips that level. With `smoothing`, each level's counts blend toward its
parent's distribution, `(n_k + a * parent_k) / (n + a)`, so a thin cell borrows from above
rather than jumping between levels.
"""

from __future__ import annotations

from typing import ClassVar, Final, Self

import numpy as np
from numpy.typing import NDArray

from hcsim.model.core.layouts import OpeningColumn
from hcsim.model.ml.entity import ModelNotFittedError

Ints = NDArray[np.int64]

#: The attributes a generator supplies, in this column order.
ATTRIBUTES: Final = (
    OpeningColumn.LEVEL,
    OpeningColumn.LOCATION,
    OpeningColumn.JOB_FAMILY,
    OpeningColumn.TECH_INDICATOR,
    OpeningColumn.IS_MANAGER,
    OpeningColumn.COMPANY,
)


class EmpiricalAttributeGenerator:
    ALGORITHM_NAME = "empirical_attributes"

    def __init__(self, *, min_rows: int = 5) -> None:
        self.min_rows = min_rows
        self._by_group: dict[int, Ints] = {}
        self._pooled: Ints | None = None

    def fit(self, cost_centers: Ints, attributes: Ints) -> Self:
        rows = np.asarray(attributes, dtype=np.int64)
        groups = np.asarray(cost_centers, dtype=np.int64)
        if rows.ndim != 2 or rows.shape[1] != len(ATTRIBUTES):
            raise ValueError(f"attributes must be (N, {len(ATTRIBUTES)}), in ATTRIBUTES order")
        if len(groups) != len(rows):
            raise ValueError("one cost center per attribute row")
        if len(rows) == 0:
            raise ValueError("can't fit a generator on zero rows")
        self._by_group = {
            int(g): rows[groups == g]
            for g in np.unique(groups)
            if int((groups == g).sum()) >= self.min_rows
        }
        self._pooled = rows
        return self

    def sample(self, cost_center: int, n: int, rng: np.random.Generator) -> Ints:
        """`n` attribute rows for a requisition in `cost_center`, in ATTRIBUTES order."""
        if self._pooled is None:
            raise ModelNotFittedError("fit the generator first")
        source = self._by_group.get(int(cost_center), self._pooled)
        return source[rng.integers(len(source), size=n)]

    def covers(self, cost_center: int) -> bool:
        """Whether `cost_center` draws from its own rows rather than the pooled ones."""
        return int(cost_center) in self._by_group


class ConditionalCategorical:
    """A categorical outcome given a context hierarchy, with fallback up the hierarchy."""

    ALGORITHM_NAME: ClassVar[str] = "conditional_categorical"

    def __init__(self, n_categories: int, *, min_count: int = 20, smoothing: float = 0.0) -> None:
        if n_categories < 1:
            raise ValueError("a categorical outcome needs at least one category")
        if min_count < 1 or smoothing < 0:
            raise ValueError("min_count must be at least 1 and smoothing non-negative")
        self.n_categories = n_categories
        self.min_count = min_count
        self.smoothing = smoothing
        self._global: NDArray[np.float64] | None = None
        self._levels: list[dict[int, NDArray[np.float64]]] = []

    def fit(self, contexts: Ints, outcomes: Ints) -> Self:
        """`contexts` is (rows, levels) codes, finest level first; `outcomes` one code per
        row in 0..n_categories-1."""
        ctx = np.asarray(contexts, dtype=np.int64)
        y = np.asarray(outcomes, dtype=np.int64)
        if ctx.ndim != 2 or len(ctx) != len(y):
            raise ValueError("contexts must be (rows, levels), with one outcome per row")
        if len(y) == 0:
            raise ValueError("can't fit a generator on zero rows")
        if y.min() < 0 or y.max() >= self.n_categories:
            raise ValueError(f"outcomes must lie in 0..{self.n_categories - 1}")
        self._global = np.bincount(y, minlength=self.n_categories).astype(np.float64)
        self._levels = []
        for level in range(ctx.shape[1]):
            codes = ctx[:, level]
            known = codes >= 0
            cells: dict[int, NDArray[np.float64]] = {}
            for code in np.unique(codes[known]).tolist():
                rows = y[codes == code]
                cells[int(code)] = np.bincount(rows, minlength=self.n_categories).astype(np.float64)
            self._levels.append(cells)
        return self

    def _require(self) -> NDArray[np.float64]:
        if self._global is None:
            raise ModelNotFittedError("fit the generator first")
        return self._global

    def level_used(self, context: Ints) -> int | None:
        """The finest level a context reads, or None for the overall distribution."""
        self._require()
        for level, code in enumerate(np.asarray(context, dtype=np.int64).tolist()):
            counts = self._levels[level].get(int(code)) if level < len(self._levels) else None
            if counts is not None and counts.sum() >= self.min_count:
                return level
        return None

    def distribution(self, context: Ints) -> NDArray[np.float64]:
        """The outcome's probabilities for one context."""
        overall = self._require()
        codes = np.asarray(context, dtype=np.int64).tolist()
        if len(codes) != len(self._levels):
            raise ValueError(f"a context needs {len(self._levels)} levels, not {len(codes)}")
        probabilities = overall / overall.sum()
        start = self.level_used(context)
        if start is None:
            return probabilities
        # Build from the coarsest usable level down to the one the context reads, blending each
        # toward the one above it when smoothing.
        usable = [
            level
            for level in range(len(codes) - 1, start - 1, -1)
            if (c := self._levels[level].get(int(codes[level]))) is not None
            and c.sum() >= self.min_count
        ]
        for level in usable:
            counts = self._levels[level][int(codes[level])]
            if self.smoothing:
                probabilities = (counts + self.smoothing * probabilities) / (
                    counts.sum() + self.smoothing
                )
            else:
                probabilities = counts / counts.sum()
        return probabilities

    def sample(self, contexts: Ints, rng: np.random.Generator) -> Ints:
        """One outcome per context row, drawn from each row's distribution."""
        ctx = np.asarray(contexts, dtype=np.int64)
        if ctx.ndim != 2:
            raise ValueError("contexts must be (rows, levels)")
        uniform = rng.random(len(ctx))
        out = np.zeros(len(ctx), dtype=np.int64)
        unique, inverse = np.unique(ctx, axis=0, return_inverse=True)
        inverse = inverse.reshape(-1)
        for k, context in enumerate(unique):
            rows = inverse == k
            cumulative = np.cumsum(self.distribution(context))
            out[rows] = np.minimum(
                np.searchsorted(cumulative, uniform[rows], side="right"), self.n_categories - 1
            )
        return out
