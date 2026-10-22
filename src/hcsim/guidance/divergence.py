"""Declared divergences: where two producers of the same counts may differ, why, and how far.

Two producers count guidance categories from the warehouse:

- the guidance pipeline's **measures** (`ingest`), over the aggregate fact and the requisition
  intervals: the history a forecast is fitted to;
- the **realized** totals (`realized`), over the employee-grain fact and the month-end
  snapshots: the volumes a `guided_actuals` run pins.

For the same window and grouping they must agree, category by category, on the window's
total, unless a divergence is declared for the category: a reason, and a tolerance. A
tolerance bounds how far the realized total may fall below the measures' (`below`, a share of
it) or rise above it (`above`), with an absolute `slack` either way for small counts. A
category with no declaration must match exactly.

Two divergences are declared, both for requisitions, which the snapshots see only at
month-ends:

- `requisitions_opened` may fall short: a requisition opened and closed between two month-ends
  leaves no trace unless an accepted offer waited at one, and an opening filled before the
  first sighting counts only when its offer did. Its slack is one requisition at the openings
  cap, as a bulk placeholder opened and closed unfilled in a day can be;
- `requisitions_closed_unfilled` may fall short, and rise a little: a requisition that opened
  and closed unfilled between two month-ends leaves no trace, and a starter credited to a
  closing requisition may have filled one the snapshots never saw, while an offer accepted and
  abandoned between month-ends leaves a closure that looks unfilled.

The head flows and the headcount stock declare none: the employee-grain fact and the aggregate
fact must agree exactly. The tolerances hold for windows without missing snapshots; a window
whose month-ends are missing moves closures past them, beyond any tolerance declared here.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Final

import duckdb
import pandas as pd

from hcsim.guidance.ingest import CATEGORY, MONTH, VALUE, ingest
from hcsim.guidance.realized import Realized
from hcsim.model.core.categories import GuidanceCategory
from hcsim.model.core.layouts import MAX_OPENINGS_PER_REQUISITION

_C = GuidanceCategory
#: How close two counts that must match exactly have to be.
EXACT: Final = 1e-9


@dataclass(frozen=True)
class Tolerance:
    """How far a produced total may lie from its reference: below by `below` of it, above by
    `above` of it, with `slack` either way."""

    below: float = 0.0
    above: float = 0.0
    slack: float = 0.0

    def __post_init__(self) -> None:
        if min(self.below, self.above, self.slack) < 0:
            raise ValueError("a tolerance's shares and slack can't be negative")

    def bounds(self, reference: float) -> tuple[float, float]:
        """The lowest and highest totals within the tolerance of `reference`."""
        size = abs(reference)
        return (
            reference - self.below * size - self.slack - EXACT,
            reference + self.above * size + self.slack + EXACT,
        )


@dataclass(frozen=True)
class Divergence:
    """A declared divergence: the category, why its producers differ, and by how much."""

    category: GuidanceCategory
    reason: str
    tolerance: Tolerance


DIVERGENCES: Final[Mapping[GuidanceCategory, Divergence]] = MappingProxyType(
    {
        d.category: d
        for d in (
            Divergence(
                _C.REQUISITIONS_OPENED,
                "the snapshots miss requisitions opened and closed between month-ends, and "
                "openings filled before a requisition's first sighting with no offer waiting",
                # One unseen requisition can carry as many openings as the cap counts.
                Tolerance(below=0.25, slack=float(MAX_OPENINGS_PER_REQUISITION)),
            ),
            Divergence(
                _C.REQUISITIONS_CLOSED_UNFILLED,
                "a requisition opened and closed unfilled between month-ends is missed, and a "
                "starter credited to a closing requisition may have filled an unseen one, while "
                "an offer accepted and abandoned between month-ends looks like an unfilled closure",
                Tolerance(below=0.3, above=0.15, slack=4.0),
            ),
        )
    }
)
#: No declaration: the producers must match.
UNDECLARED: Final = Tolerance()


@dataclass(frozen=True)
class Check:
    """One category's totals from both producers, and whether they lie within its tolerance."""

    category: GuidanceCategory
    produced: float
    reference: float
    low: float
    high: float
    declared: Divergence | None

    @property
    def gap(self) -> float:
        return self.produced - self.reference

    @property
    def within(self) -> bool:
        return self.low <= self.produced <= self.high

    def describe(self) -> str:
        allowed = "exactly" if self.declared is None else f"within {self.low:g}..{self.high:g}"
        return (
            f"{self.category.value}: {self.produced:g} against {self.reference:g}, which it "
            f"must match {allowed}"
        )


@dataclass(frozen=True)
class Comparison:
    """Every shared category's check, for one window and grouping."""

    checks: tuple[Check, ...]
    producer: str = "realized"
    reference: str = "measures"

    @property
    def within(self) -> bool:
        return all(c.within for c in self.checks)

    def problems(self) -> list[str]:
        """Each category outside its tolerance, described."""
        return [c.describe() for c in self.checks if not c.within]

    def report(self) -> dict[str, Any]:
        return {
            "producer": self.producer,
            "reference": self.reference,
            "within": self.within,
            "categories": {
                c.category.value: {
                    self.producer: c.produced,
                    self.reference: c.reference,
                    "gap": c.gap,
                    "bounds": [c.low, c.high],
                    "declared": None if c.declared is None else c.declared.reason,
                    "within": c.within,
                }
                for c in self.checks
            },
        }


def _totals(frame: pd.DataFrame) -> dict[GuidanceCategory, float]:
    summed = frame.groupby(CATEGORY)[VALUE].sum()
    return {GuidanceCategory(str(c)): float(v) for c, v in summed.items()}


def compare(
    produced: pd.DataFrame,
    reference: pd.DataFrame,
    divergences: Mapping[GuidanceCategory, Divergence] = DIVERGENCES,
) -> Comparison:
    """Each category both long-form frames carry, its totals checked against the reference's
    within its declared tolerance, or exactly when none is declared."""
    ours, theirs = _totals(produced), _totals(reference)
    checks = []
    for category in sorted(set(ours) & set(theirs), key=lambda c: c.value):
        declared = divergences.get(category)
        tolerance = UNDECLARED if declared is None else declared.tolerance
        low, high = tolerance.bounds(theirs[category])
        checks.append(Check(category, ours[category], theirs[category], low, high, declared))
    return Comparison(tuple(checks))


def against_measures(
    con: duckdb.DuckDBPyConnection,
    result: Realized,
    divergences: Mapping[GuidanceCategory, Divergence] = DIVERGENCES,
) -> Comparison:
    """The realized totals against the guidance measures, over the same window and grouping."""
    frame = result.frame
    months = sorted(frame[MONTH].unique())
    categories: Sequence[GuidanceCategory] = sorted(
        {GuidanceCategory(str(c)) for c in frame[CATEGORY].unique()}, key=lambda c: c.value
    )
    series = ingest(con, result.artifact.grouping_column, categories).frame
    window = series[(series[MONTH] >= months[0]) & (series[MONTH] <= months[-1])]
    return compare(frame, window, divergences)
