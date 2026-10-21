"""Vest templates: monthly vesting fractions per forward year.

A strategy supplies whole shares per forward year, and the template spreads each year's shares
over its months. The result is padded to a common 72-month buffer, so every award type stacks
in one array. Column `c` holds the shares vesting `c + 1` months after the anchor month.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import numpy as np
from numpy.typing import NDArray

CANONICAL_VEST_MONTHS: Final = 72


class VestTemplateError(ValueError):
    """A template that can't lay out a grant."""


@dataclass(frozen=True)
class VestTemplate:
    name: str
    #: Per forward year, the (month within the year, fraction) pairs. Each year sums to 1.
    years: tuple[tuple[tuple[int, float], ...], ...]

    def __post_init__(self) -> None:
        if not 1 <= len(self.years) <= CANONICAL_VEST_MONTHS // 12:
            raise VestTemplateError("a template covers 1 to 6 forward years")
        for y, chunks in enumerate(self.years, start=1):
            if any(not 1 <= month <= 12 for month, _ in chunks):
                raise VestTemplateError(f"year {y}: months within a year run 1-12")
            if abs(sum(f for _, f in chunks) - 1.0) > 1e-9:
                raise VestTemplateError(f"year {y}: fractions must sum to 1")

    @property
    def n_years(self) -> int:
        return len(self.years)

    def expand(self, shares_per_year: NDArray[np.int64]) -> NDArray[np.int64]:
        """(N, years) whole shares to an (N, 72) monthly vest schedule of whole shares.

        Within each year, chunks are floored and the year's last chunk absorbs the remainder,
        so every share vests exactly once.
        """
        shares = np.asarray(shares_per_year, dtype=np.int64)
        if shares.ndim != 2 or shares.shape[1] != self.n_years:
            raise VestTemplateError(f"expected (N, {self.n_years}) shares, got {shares.shape}")
        if (shares < 0).any():
            raise VestTemplateError("shares must be non-negative")
        schedule = np.zeros((len(shares), CANONICAL_VEST_MONTHS), dtype=np.int64)
        for y, chunks in enumerate(self.years):
            placed = np.zeros(len(shares), dtype=np.int64)
            for k, (month, fraction) in enumerate(chunks):
                column = y * 12 + month - 1
                if k == len(chunks) - 1:
                    chunk = shares[:, y] - placed
                else:
                    chunk = np.floor(shares[:, y] * fraction).astype(np.int64)
                schedule[:, column] += chunk
                placed += chunk
        return schedule


#: A one-year cliff, then quarterly. A generic shape, not a calibrated one.
NEW_HIRE_TEMPLATE: Final = VestTemplate(
    "new_hire_cliff",
    (((12, 1.0),),) + (((3, 0.25), (6, 0.25), (9, 0.25), (12, 0.25)),) * 3,
)
