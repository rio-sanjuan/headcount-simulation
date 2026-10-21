"""Award sizing. The calculator is pure: it reads the award configuration, never the simulation's.

Handlers emit grants, and the apply step sizes them here. In the walking skeleton the one
strategy is the new-hire gap fill: each forward year's grant fills that year's gap between
target total compensation and base salary, at that year's projected price, net of any shares
the recipient already vests that year.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from hcsim.financial.prices import StockPriceModel
from hcsim.financial.ranges import RangeField, RangeTable
from hcsim.financial.vesting import NEW_HIRE_TEMPLATE, VestTemplate
from hcsim.model.core.layouts import EMPLOYEE, Buffer
from hcsim.model.core.schema import JOB_FAMILY, JOB_LEVEL, LOCATION


class SizingError(ValueError):
    """A grant that can't be sized."""


@dataclass(frozen=True)
class AwardBatch:
    award_type: str
    base_salary: NDArray[np.float64]
    #: Whole shares per recipient and forward year.
    shares_per_year: NDArray[np.int64]
    #: Whole shares per recipient and month after the anchor, 72 columns.
    vest_schedule: NDArray[np.int64]

    @property
    def total_shares(self) -> NDArray[np.int64]:
        total: NDArray[np.int64] = self.shares_per_year.sum(axis=1)
        return total


@dataclass(frozen=True)
class NewHireGapFill:
    ranges: RangeTable
    template: VestTemplate = NEW_HIRE_TEMPLATE
    #: Where a new hire sits within the range. Synthetic default: the midpoint.
    range_position: float = 0.5
    award_type: str = "new_hire"

    def size(
        self,
        rows: Buffer,
        prices: StockPriceModel,
        existing_vests_per_year: NDArray[np.float64] | None = None,
    ) -> AwardBatch:
        """Size one grant per employee row."""
        if not 0.0 <= self.range_position <= 1.0:
            raise SizingError("range_position must lie in [0, 1]")
        codes = [
            rows[:, EMPLOYEE.column(name)].astype(np.int64)
            for name in (JOB_FAMILY, JOB_LEVEL, LOCATION)
        ]
        r = self.ranges.lookup(*codes)
        target = r[:, RangeField.MIN_TARGET] + self.range_position * (
            r[:, RangeField.MAX_TARGET] - r[:, RangeField.MIN_TARGET]
        )
        base = (r[:, RangeField.MIN_BASE] + r[:, RangeField.MAX_BASE]) / 2
        years = self.template.n_years
        price = np.array([prices.projection(y) for y in range(1, years + 1)])
        if not (price > 0).all():
            raise SizingError("projected prices must be positive")
        existing = (
            np.zeros((len(rows), years))
            if existing_vests_per_year is None
            else np.asarray(existing_vests_per_year, dtype=np.float64)
        )
        gap = np.maximum(0.0, target[:, None] - base[:, None] - existing * price[None, :])
        shares_float = gap / price[None, :]
        if not np.isfinite(shares_float).all():
            raise SizingError(
                f"{int((~np.isfinite(shares_float)).any(axis=1).sum())} recipients "
                "got a non-finite grant"
            )
        shares = np.rint(shares_float).astype(np.int64)
        return AwardBatch(self.award_type, base, shares, self.template.expand(shares))
