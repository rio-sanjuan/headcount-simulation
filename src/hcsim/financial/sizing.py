"""Award sizing strategies. Each reads the award configuration, never the simulation's, and
sizes grants through the award calculator (see `calculator`).

Handlers emit grants, and the apply step sizes them here, with a `SizingContext`: the award
month, its calendar month, and the holder ledger, from which a strategy reads what each
recipient already vests in each forward year after the award.

- **Gap fill** (`GapFill`): each forward year's grant fills that year's gap between target
  total compensation and base pay, at the year's projected price, net of what the recipient
  already vests that year. `NewHireGapFill` is its multi-year new-hire form, sized over the new
  hire's whole template; `AnnualGapFill` its annual form, the forward years an annual award
  covers.
- **In-year plus forward years** (`InYearPlusForward`), as for a promotion: the template's first
  year is the rest of the current award year, the months until the next annual cycle, so its
  gap is pro-rated by them; the later years are full forward years.
- **Retention** (`RetentionGrant`): a share of each forward year's equity target (target less
  base) on top of whatever the recipient already vests, laid out by its template.

Two simple alternatives size without the year-by-year gap, as baselines to compare against:

- **Fixed fraction** (`FixedFraction`): a fixed share of target total compensation, at the
  first forward year's price, split evenly over the template's years;
- **Single year** (`SingleYear`): only the first forward year's gap, net of what the recipient
  already vests that year, on a one-year template.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from typing import Protocol

import numpy as np
from numpy.typing import NDArray

from hcsim.financial.calculator import RatingTable, Recipients, Sized, SizingError, size
from hcsim.financial.prices import StockPriceModel
from hcsim.financial.ranges import RangeTable
from hcsim.financial.tracker import HolderLedger
from hcsim.financial.vesting import NEW_HIRE_TEMPLATE, VestTemplate
from hcsim.model.core.layouts import Buffer, EmployeeColumn

__all__ = [
    "AnnualGapFill",
    "AwardBatch",
    "FixedFraction",
    "GapFill",
    "InYearPlusForward",
    "NewHireGapFill",
    "RetentionGrant",
    "SingleYear",
    "SizingContext",
    "SizingError",
    "SizingStrategy",
]

Floats = NDArray[np.float64]


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
class SizingContext:
    """What a strategy reads beyond its recipients' rows: the award month (a simulation step),
    its calendar month, 1 to 12, and the holder ledger, when the run keeps one."""

    step: int = 0
    calendar_month: int = 1
    ledger: HolderLedger | None = None

    def __post_init__(self) -> None:
        if self.calendar_month not in range(1, 13):
            raise SizingError(f"a calendar month runs 1-12, not {self.calendar_month}")

    def existing(self, rows: Buffer, years: int) -> Floats:
        """(rows, years): the shares each recipient already vests in each forward year."""
        if self.ledger is None:
            return np.zeros((len(rows), years))
        holders = np.asarray(rows[:, EmployeeColumn.VEST_INDEX], dtype=np.int64)
        return self.ledger.vesting_by_year(holders, self.step, years)


class SizingStrategy(Protocol):
    @property
    def award_type(self) -> str: ...

    def size(
        self, rows: Buffer, prices: StockPriceModel, context: SizingContext | None = None
    ) -> AwardBatch: ...


def _projected(prices: StockPriceModel, years: int) -> Floats:
    return np.array([prices.projection(y) for y in range(1, years + 1)], dtype=np.float64)


def _batch(
    award_type: str, template: VestTemplate, sized: Sized, shares: NDArray[np.int64]
) -> AwardBatch:
    return AwardBatch(award_type, sized.base, shares, template.expand(shares))


@dataclass(frozen=True)
class GapFill:
    """Each forward year's grant fills that year's gap, net of what the recipient already
    vests that year."""

    ranges: RangeTable
    template: VestTemplate
    #: Where each recipient sits in the range, by rating; flat at the midpoint by default.
    ratings: RatingTable = field(default_factory=RatingTable.flat)
    award_type: str = "annual"

    def sized(
        self, rows: Buffer, prices: StockPriceModel, context: SizingContext | None = None
    ) -> Sized:
        """Every step of the calculator, for one grant per row."""
        years = self.template.n_years
        existing = (context or SizingContext()).existing(rows, years)
        return size(
            Recipients.from_rows(rows),
            self.ranges,
            self.ratings,
            _projected(prices, years),
            self.template,
            existing,
        )

    def size(
        self, rows: Buffer, prices: StockPriceModel, context: SizingContext | None = None
    ) -> AwardBatch:
        sized = self.sized(rows, prices, context)
        return _batch(self.award_type, self.template, sized, sized.shares)


@dataclass(frozen=True)
class NewHireGapFill(GapFill):
    """Multi-year new-hire sizing: the gap of every year the new-hire template covers."""

    template: VestTemplate = NEW_HIRE_TEMPLATE
    award_type: str = "new_hire"


@dataclass(frozen=True)
class AnnualGapFill(GapFill):
    """Annual gap fill: the gap of each forward year an annual award covers."""

    award_type: str = "annual"


@dataclass(frozen=True)
class InYearPlusForward(GapFill):
    """The rest of the current award year, then full forward years. The template's first year
    is the in-year portion: its gap is pro-rated by the months from the award month to the next
    annual cycle, in `cycle_month`, over twelve."""

    #: The calendar month the annual cycle awards in.
    cycle_month: int = 1
    award_type: str = "promotion"

    def __post_init__(self) -> None:
        if self.cycle_month not in range(1, 13):
            raise SizingError(f"cycle_month is a calendar month 1-12, not {self.cycle_month}")

    def months_left(self, calendar_month: int) -> int:
        """Months from an award in `calendar_month` to the next cycle: 12 in the cycle month."""
        return (self.cycle_month - calendar_month - 1) % 12 + 1

    def size(
        self, rows: Buffer, prices: StockPriceModel, context: SizingContext | None = None
    ) -> AwardBatch:
        context = context or SizingContext()
        sized = self.sized(rows, prices, context)
        share = self.months_left(context.calendar_month) / 12
        shares = sized.shares.copy()
        shares[:, 0] = np.rint(sized.shares_float[:, 0] * share).astype(np.int64)
        return _batch(self.award_type, self.template, sized, shares)


@dataclass(frozen=True)
class RetentionGrant(GapFill):
    """A share of each forward year's equity target, target less base, on top of whatever
    the recipient already vests."""

    fraction: float = 0.5
    award_type: str = "retention"

    def __post_init__(self) -> None:
        if not 0.0 < self.fraction <= 1.0:
            raise SizingError(f"a retention fraction lies in (0, 1], not {self.fraction}")

    def size(
        self, rows: Buffer, prices: StockPriceModel, context: SizingContext | None = None
    ) -> AwardBatch:
        # Retention isn't net of existing vests: it's sized as if the recipient held nothing.
        sized = self.sized(
            rows, prices, dataclasses.replace(context or SizingContext(), ledger=None)
        )
        shares = np.rint(sized.shares_float * self.fraction).astype(np.int64)
        return _batch(self.award_type, self.template, sized, shares)


@dataclass(frozen=True)
class FixedFraction:
    """A fixed share of target total compensation, at the first forward year's price, split
    evenly over the template's years, the last absorbing the remainder."""

    ranges: RangeTable
    template: VestTemplate
    fraction: float = 0.2
    #: Where each recipient sits in the range, by rating; flat at the midpoint by default.
    ratings: RatingTable = field(default_factory=RatingTable.flat)
    award_type: str = "annual"

    def __post_init__(self) -> None:
        if not 0.0 < self.fraction <= 1.0:
            raise SizingError(f"a fixed fraction lies in (0, 1], not {self.fraction}")

    def size(
        self, rows: Buffer, prices: StockPriceModel, context: SizingContext | None = None
    ) -> AwardBatch:
        years = self.template.n_years
        projected = _projected(prices, years)
        sized = size(
            Recipients.from_rows(rows), self.ranges, self.ratings, projected, self.template
        )
        total = np.rint(self.fraction * sized.target / projected[0]).astype(np.int64)
        shares = np.repeat((total // years)[:, None], years, axis=1)
        shares[:, -1] += total - shares.sum(axis=1)
        return _batch(self.award_type, self.template, sized, shares)


@dataclass(frozen=True)
class SingleYear(GapFill):
    """Only the first forward year's gap, net of what the recipient already vests that year,
    on a one-year template."""

    def __post_init__(self) -> None:
        if self.template.n_years != 1:
            raise SizingError(
                f"a single-year grant needs a one-year template, not {self.template.name!r}'s "
                f"{self.template.n_years} years"
            )
