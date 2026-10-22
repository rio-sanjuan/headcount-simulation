"""Award timing: when a grant dispatched in one month is awarded.

A domain dispatches a grant in its step, as when someone is hired or promoted, but the award
itself can land later, as a grant committee meets. Per award type, `AwardTiming` gives:

- **a lag distribution** (`LagDistribution`): the chance of each lag, in whole months from the
  dispatch step to the award month, from zero;
- **lags per calendar month**: a distribution of its own for dispatches in a given calendar
  month, as when awards dispatched in December wait out the holidays;
- **suppressed months**: calendar months with no award, as during a blackout. An award due in
  one slides to the next month that isn't suppressed.

The award month anchors the grant's vesting, and its expense accrues from it. Lags are drawn
from the run's own seeded stream, so a run's timing is reproducible.

**Eligibility** (`AnnualCutoff`). An annual award goes only to those hired before the latest
cutoff month: with an April cycle and an October cutoff, someone hired in October or later waits
for the next cycle. The rule reads each recipient's hire step, so it holds for the seeded
workforce, whose hire steps lie before the run, as for hires the run makes.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Final

import numpy as np
from numpy.typing import NDArray

from hcsim.financial.calculator import SizingError
from hcsim.model.core.layouts import Buffer, EmployeeColumn

Ints = NDArray[np.int64]
#: The longest lag a distribution can give, in months.
MAX_LAG: Final = 24


@dataclass(frozen=True)
class LagDistribution:
    """The probability of a lag of 0, 1, 2, ... months. They sum to 1."""

    probabilities: tuple[float, ...] = (1.0,)

    def __post_init__(self) -> None:
        p = np.asarray(self.probabilities, dtype=np.float64)
        problems = []
        if not 1 <= len(p) <= MAX_LAG + 1:
            problems.append(f"a lag distribution covers lags 0 to at most {MAX_LAG} months")
        if not (np.isfinite(p).all() and (p >= 0).all()):
            problems.append(f"lag probabilities are finite and non-negative, not {list(p)}")
        elif abs(float(p.sum()) - 1.0) > 1e-9:
            problems.append(f"lag probabilities sum to 1, not {float(p.sum()):g}")
        if problems:
            raise SizingError("; ".join(problems))
        object.__setattr__(self, "probabilities", tuple(float(x) for x in p))

    @classmethod
    def fixed(cls, months: int) -> LagDistribution:
        """Every award exactly `months` after its dispatch."""
        return cls(tuple([0.0] * months + [1.0]))

    def draw(self, rng: np.random.Generator, n: int) -> Ints:
        """`n` lags."""
        if len(self.probabilities) == 1:
            return np.zeros(n, dtype=np.int64)
        lags: Ints = rng.choice(len(self.probabilities), size=n, p=self.probabilities).astype(
            np.int64
        )
        return lags


@dataclass(frozen=True)
class AwardTiming:
    """When an award type's grants are awarded: a lag after dispatch, from `by_month`'s
    distribution for the dispatch's calendar month when it has one, else from `lag`, then slid
    past any `suppressed` calendar months."""

    lag: LagDistribution = field(default_factory=LagDistribution)
    by_month: Mapping[int, LagDistribution] = field(default_factory=dict)
    suppressed: frozenset[int] = frozenset()

    def __post_init__(self) -> None:
        problems = []
        months = sorted(m for m in (*self.by_month, *self.suppressed) if m not in range(1, 13))
        if months:
            problems.append(f"calendar months run 1-12, not {months}")
        if len(set(self.suppressed) & set(range(1, 13))) == 12:
            problems.append("every calendar month is suppressed, so nothing could be awarded")
        if problems:
            raise SizingError("award timing: " + "; ".join(problems))
        object.__setattr__(self, "by_month", MappingProxyType(dict(self.by_month)))
        object.__setattr__(self, "suppressed", frozenset(self.suppressed))

    def distribution(self, calendar_month: int) -> LagDistribution:
        """The lag distribution for dispatches in `calendar_month`."""
        return self.by_month.get(calendar_month, self.lag)

    @property
    def immediate(self) -> bool:
        """Whether every award lands in its dispatch month: no lag, nothing suppressed."""
        dists = (self.lag, *self.by_month.values())
        return not self.suppressed and all(d.probabilities == (1.0,) for d in dists)

    def award_steps(self, step: int, calendar_month: int, n: int, rng: np.random.Generator) -> Ints:
        """The award month, as a step, of each of `n` grants dispatched at `step`, in
        `calendar_month`."""
        if calendar_month not in range(1, 13):
            raise SizingError(f"a calendar month runs 1-12, not {calendar_month}")
        lags = self.distribution(calendar_month).draw(rng, n)
        steps = step + lags
        if self.suppressed:
            months = (calendar_month - 1 + lags) % 12 + 1
            for _ in range(12):
                held = np.isin(months, list(self.suppressed))
                if not held.any():
                    break
                steps = np.where(held, steps + 1, steps)
                months = np.where(held, months % 12 + 1, months)
        out: Ints = steps.astype(np.int64)
        return out


@dataclass(frozen=True)
class AnnualCutoff:
    """An annual award's eligibility: recipients hired before the latest `cutoff_month`, a
    calendar month, at or before the award month."""

    cutoff_month: int

    def __post_init__(self) -> None:
        if self.cutoff_month not in range(1, 13):
            raise SizingError(f"a cutoff is a calendar month 1-12, not {self.cutoff_month}")

    def cutoff_step(self, step: int, calendar_month: int) -> int:
        """The step of the latest cutoff month at or before `step`, in `calendar_month`."""
        if calendar_month not in range(1, 13):
            raise SizingError(f"a calendar month runs 1-12, not {calendar_month}")
        return step - (calendar_month - self.cutoff_month) % 12

    def eligible(self, rows: Buffer, step: int, calendar_month: int) -> NDArray[np.bool_]:
        """Which rows were hired before the cutoff, by their hire step."""
        hired = np.asarray(rows[:, EmployeeColumn.HIRE_STEP], dtype=np.float64)
        out: NDArray[np.bool_] = hired < self.cutoff_step(step, calendar_month)
        return out
