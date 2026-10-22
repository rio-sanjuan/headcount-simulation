"""Award sizing strategies. Each reads the award configuration, never the simulation's, and
sizes grants through the award calculator (see `calculator`).

Handlers emit grants, and the apply step sizes them here. The new-hire gap fill sizes each
forward year's grant to fill that year's gap between target total compensation and base pay,
at that year's projected price, net of any shares the recipient already vests that year.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

from hcsim.financial.calculator import RatingTable, Recipients, SizingError, size
from hcsim.financial.prices import StockPriceModel
from hcsim.financial.ranges import RangeTable
from hcsim.financial.vesting import NEW_HIRE_TEMPLATE, VestTemplate
from hcsim.model.core.layouts import Buffer

__all__ = ["AwardBatch", "NewHireGapFill", "SizingError"]


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
    #: Where each recipient sits in the range, by rating. A new hire is unrated, so takes the
    #: default: the midpoint, a synthetic choice.
    ratings: RatingTable = field(default_factory=RatingTable.flat)
    award_type: str = "new_hire"

    def size(
        self,
        rows: Buffer,
        prices: StockPriceModel,
        existing_vests_per_year: NDArray[np.float64] | None = None,
    ) -> AwardBatch:
        """Size one grant per employee row."""
        projected = np.array([prices.projection(y) for y in range(1, self.template.n_years + 1)])
        if not (np.isfinite(projected) & (projected > 0)).all():
            raise SizingError("projected prices must be positive")
        sized = size(
            Recipients.from_rows(rows),
            self.ranges,
            self.ratings,
            projected,
            self.template,
            existing_vests_per_year,
        )
        return AwardBatch(self.award_type, sized.base, sized.shares, sized.schedule)
