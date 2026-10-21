"""Attribute generators for new requisitions.

`EmpiricalAttributeGenerator` draws whole attribute rows, uniformly with replacement, from rows
observed in the requisition's cost center, or from every observed row when the cost center has
too few. It never invents a combination, so a generated requisition always carries a profile a
real one had. A requisition with no generator would carry code 0 in every attribute, and every
hire from it would inherit that one improbable profile.
"""

from __future__ import annotations

from typing import Final, Self

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
