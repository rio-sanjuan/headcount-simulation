"""Vest templates: how a grant's shares per forward year vest, month by month.

A strategy supplies whole shares per forward year, and the template spreads each year's shares
over its vest months. Every template expands to a common 72-month buffer, so every award type
stacks in one array: column `c` holds the shares vesting `c + 1` months after the award month.
Within a year, chunks are floored and the year's last chunk absorbs the remainder, so every
share vests exactly once.

Three shapes:

- **Quarterly** (`quarterly`): a quarter of each year's shares every three months.
- **Cliff** (`cliff`): nothing until the cliff, which vests the cliff's whole year at once;
  quarterly after it, or nothing more.
- **Calendar-keyed** (`CalendarTemplate`), as for off-cycle awards: shares vest on fixed
  calendar months, the company's vest dates, whatever month the award falls in. The first vest
  date comes at least `wait` months after the award, and forward year y takes the vest dates
  falling 12(y - 1) + 1 to 12y months after it, so the layout depends on the award's calendar
  month.

`VestTemplate` lays out months within each forward year and ignores the calendar.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Final

import numpy as np
from numpy.typing import NDArray

CANONICAL_VEST_MONTHS: Final = 72
MAX_YEARS: Final = CANONICAL_VEST_MONTHS // 12

Ints = NDArray[np.int64]


class VestTemplateError(ValueError):
    """A template that can't lay out a grant."""


def _checked(shares_per_year: Ints, years: int) -> Ints:
    shares = np.asarray(shares_per_year, dtype=np.int64)
    if shares.ndim != 2 or shares.shape[1] != years:
        raise VestTemplateError(f"expected (N, {years}) shares, got {shares.shape}")
    if (shares < 0).any():
        raise VestTemplateError("shares must be non-negative")
    return shares


def _lay_out(shares: Ints, years: Sequence[Sequence[tuple[int, float]]]) -> Ints:
    """Each year's shares at its (month after the award, fraction) chunks."""
    schedule = np.zeros((len(shares), CANONICAL_VEST_MONTHS), dtype=np.int64)
    for y, chunks in enumerate(years):
        placed = np.zeros(len(shares), dtype=np.int64)
        for k, (month, fraction) in enumerate(chunks):
            if k == len(chunks) - 1:
                chunk = shares[:, y] - placed
            else:
                chunk = np.floor(shares[:, y] * fraction).astype(np.int64)
            schedule[:, month - 1] += chunk
            placed += chunk
    return schedule


@dataclass(frozen=True)
class VestTemplate:
    name: str
    #: Per forward year, the (month within the year, fraction) pairs. Each year sums to 1.
    years: tuple[tuple[tuple[int, float], ...], ...]

    def __post_init__(self) -> None:
        if not 1 <= len(self.years) <= MAX_YEARS:
            raise VestTemplateError(f"a template covers 1 to {MAX_YEARS} forward years")
        for y, chunks in enumerate(self.years, start=1):
            if not chunks:
                raise VestTemplateError(f"year {y}: a year needs at least one vest month")
            if any(not 1 <= month <= 12 for month, _ in chunks):
                raise VestTemplateError(f"year {y}: months within a year run 1-12")
            if any(not 0.0 <= f <= 1.0 for _, f in chunks):
                raise VestTemplateError(f"year {y}: fractions lie in [0, 1]")
            if abs(sum(f for _, f in chunks) - 1.0) > 1e-9:
                raise VestTemplateError(f"year {y}: fractions must sum to 1")

    @property
    def n_years(self) -> int:
        return len(self.years)

    def expand(self, shares_per_year: Ints, calendar_month: int = 1) -> Ints:
        """(N, years) whole shares to an (N, 72) monthly vest schedule of whole shares. The
        calendar month doesn't move a month-within-year template."""
        shares = _checked(shares_per_year, self.n_years)
        offsets = [
            [(y * 12 + month, fraction) for month, fraction in chunks]
            for y, chunks in enumerate(self.years)
        ]
        return _lay_out(shares, offsets)


QUARTER: Final = ((3, 0.25), (6, 0.25), (9, 0.25), (12, 0.25))


def quarterly(name: str, years: int) -> VestTemplate:
    """A quarter of each year's shares every three months, for `years` forward years."""
    return VestTemplate(name, (QUARTER,) * years)


def cliff(name: str, years: int, then: str = "quarterly") -> VestTemplate:
    """A one-year cliff, its year's shares all at month 12, then `quarterly` vesting for the
    remaining years, or `none`: nothing more, the whole grant in the first year."""
    if then not in ("quarterly", "none"):
        raise VestTemplateError(f"after a cliff comes 'quarterly' or 'none', not {then!r}")
    if then == "none" and years != 1:
        raise VestTemplateError("a cliff with nothing after it covers one forward year")
    return VestTemplate(name, (((12, 1.0),),) + (QUARTER,) * (years - 1))


@dataclass(frozen=True)
class CalendarTemplate:
    """Shares vest on fixed calendar months, `vest_months`, from at least `wait` months after
    the award. Forward year y takes the vest dates 12(y - 1) + 1 to 12y months after the award
    and splits its shares evenly among them, the last absorbing the remainder."""

    name: str
    vest_months: tuple[int, ...]
    years: int
    wait: int = 1

    def __post_init__(self) -> None:
        problems = []
        if not self.vest_months or any(m not in range(1, 13) for m in self.vest_months):
            problems.append(f"vest months are calendar months 1-12, not {self.vest_months}")
        elif len(set(self.vest_months)) != len(self.vest_months):
            problems.append(f"vest months repeat: {self.vest_months}")
        if not 1 <= self.years <= MAX_YEARS:
            problems.append(f"a template covers 1 to {MAX_YEARS} forward years, not {self.years}")
        if not 1 <= self.wait <= 12:
            problems.append(f"the first vest waits 1 to 12 months, not {self.wait}")
        if problems:
            raise VestTemplateError(f"{self.name}: " + "; ".join(problems))
        object.__setattr__(self, "vest_months", tuple(sorted(self.vest_months)))

    @property
    def n_years(self) -> int:
        return self.years

    def dates(self, calendar_month: int) -> list[list[int]]:
        """Per forward year, the months after an award in `calendar_month` its vest dates fall
        in. A year with no vest date in it, as a short wait can leave the first, vests at its
        last month."""
        if calendar_month not in range(1, 13):
            raise VestTemplateError(f"a calendar month runs 1-12, not {calendar_month}")
        after = [
            m
            for m in range(1, 12 * self.years + 1)
            if (calendar_month - 1 + m) % 12 + 1 in self.vest_months
        ]
        per_year = []
        for y in range(self.years):
            mine = [m for m in after if 12 * y < m <= 12 * (y + 1) and m >= self.wait]
            per_year.append(mine or [12 * (y + 1)])
        return per_year

    def expand(self, shares_per_year: Ints, calendar_month: int = 1) -> Ints:
        """(N, years) whole shares to an (N, 72) monthly vest schedule, for an award in
        `calendar_month`."""
        shares = _checked(shares_per_year, self.n_years)
        chunks = [[(m, 1.0 / len(ms)) for m in ms] for ms in self.dates(calendar_month)]
        return _lay_out(shares, chunks)


Template = VestTemplate | CalendarTemplate

#: A one-year cliff, then quarterly. A generic shape, not a calibrated one.
NEW_HIRE_TEMPLATE: Final = cliff("new_hire_cliff", 4)
