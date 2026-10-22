"""The award calculator: a pure, seven-step sizing pipeline.

Given each recipient's role and rating, the compensation ranges, a rating table, the projected
price of each forward year, and the shares each recipient already vests, the calculator sizes
one grant per recipient. It reads nothing else: no simulation state, no warehouse, no clock.

1. **Range.** Each recipient's cell, by job family, level, and location.
2. **Penetration.** How far into the range the recipient sits, read from their rating through
   a configurable rating table. An unrated recipient, such as a new hire, takes the table's
   default.
3. **Target.** Target total compensation at that penetration:
   `min_target + p (max_target - min_target)`.
4. **Base.** Base pay imputed at the same penetration and adjusted:
   `base_adj (min_base + p (max_base - min_base))`. The warehouse holds no salaries, so the base
   is always imputed.
5. **Gap.** Each forward year's equity gap: the target, less the base, less the value of what
   the recipient already vests that year at the year's projected price, never below zero.
6. **Shares.** Each year's gap over its projected price.
7. **Whole shares.** Rounded to whole shares, then laid out month by month by the vest template.

`size` returns every step's result, so a grant can be traced from its range to its schedule.

**Nothing is sized to a silent zero.** A projected price that isn't finite and positive, or an
existing vest that isn't finite and non-negative, refuses the batch before any step runs. A
recipient whose grant still comes out non-finite, or too large to count in whole shares, refuses
it too: the error names every such recipient by ID, with the step and forward year where their
sizing went wrong, rather than granting them nothing.
"""

from __future__ import annotations

import itertools
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Final

import numpy as np
from numpy.typing import NDArray

from hcsim.financial.ranges import RangeField, RangeTable
from hcsim.financial.vesting import VestTemplate
from hcsim.model.core.layouts import Buffer, EmployeeColumn

Floats = NDArray[np.float64]
Ints = NDArray[np.int64]
#: The rating rank of an employee who hasn't been rated.
UNRATED: Final = 0
#: The most shares one grant's year can hold: beyond it, whole shares can't be counted exactly.
MAX_SHARES: Final = float(2**53)
#: How many refused recipients an error names before it counts the rest.
NAMED: Final = 20


class SizingError(ValueError):
    """A grant that can't be sized."""


@dataclass(frozen=True)
class RatingTable:
    """The penetration each rating rank sits at, and the default for the unrated.

    Ranks run from 1, the lowest. Every penetration lies in [0, 1], and a better rating never
    sits lower in the range than a worse one.
    """

    positions: Mapping[int, float] = field(default_factory=dict)
    unrated: float = 0.5

    def __post_init__(self) -> None:
        problems = []
        ranks = sorted(self.positions)
        if any(r < 1 for r in ranks):
            problems.append(f"rating ranks run from 1, not {[r for r in ranks if r < 1]}")
        values = [*(self.positions[r] for r in ranks), self.unrated]
        if not all(np.isfinite(v) and 0.0 <= v <= 1.0 for v in values):
            problems.append(f"penetrations lie in [0, 1], not {values}")
        falling = [b for a, b in itertools.pairwise(ranks) if self.positions[b] < self.positions[a]]
        if falling:
            problems.append(f"a better rating can't sit lower in the range: rank {falling[0]}")
        if problems:
            raise SizingError("rating table: " + "; ".join(problems))
        object.__setattr__(self, "positions", MappingProxyType(dict(self.positions)))

    @classmethod
    def flat(cls, position: float = 0.5) -> RatingTable:
        """Every recipient at one penetration, whatever their rating."""
        return cls({}, position)

    def penetration(self, ratings: Ints) -> Floats:
        """Each rating's penetration. A rank the table doesn't list is an error, unless the
        table lists none, when every rank takes the default."""
        ratings = np.asarray(ratings, dtype=np.int64)
        out = np.full(len(ratings), self.unrated, dtype=np.float64)
        if not self.positions:
            return out
        rated = ratings != UNRATED
        unknown = sorted({int(r) for r in ratings[rated]} - set(self.positions))
        if unknown:
            raise SizingError(f"the rating table has no penetration for ranks {unknown}")
        lookup = np.full(max(self.positions) + 1, np.nan)
        for rank, position in self.positions.items():
            lookup[rank] = position
        out[rated] = lookup[ratings[rated]]
        return out


@dataclass(frozen=True)
class Recipients:
    """Who a batch sizes: role codes in the schema's order, and rating ranks."""

    families: Ints
    levels: Ints
    locations: Ints
    ratings: Ints
    #: Their IDs, to name them in errors.
    ids: Ints

    def __len__(self) -> int:
        return len(self.ids)

    @classmethod
    def from_rows(cls, rows: Buffer) -> Recipients:
        def column(c: EmployeeColumn) -> Ints:
            return np.asarray(rows[:, c], dtype=np.int64)

        return cls(
            column(EmployeeColumn.JOB_FAMILY),
            column(EmployeeColumn.LEVEL),
            column(EmployeeColumn.LOCATION),
            column(EmployeeColumn.PERF_RATING),
            column(EmployeeColumn.EMPLOYEE_ID),
        )


@dataclass(frozen=True)
class Sized:
    """Every step's result, one row per recipient; per forward year where it has columns."""

    ranges: Floats
    penetration: Floats
    target: Floats
    base: Floats
    gap: Floats
    shares_float: Floats
    #: Whole shares per recipient and forward year, and per month after the anchor.
    shares: Ints
    schedule: Ints


def size(
    recipients: Recipients,
    ranges: RangeTable,
    ratings: RatingTable,
    prices: Sequence[float] | Floats,
    template: VestTemplate,
    existing: Floats | None = None,
) -> Sized:
    """Size one grant per recipient over the template's forward years, at `prices`, the
    projected price of each forward year, net of `existing`, the shares each already vests per
    forward year."""
    years = template.n_years
    price = np.asarray(prices, dtype=np.float64)
    if price.shape != (years,):
        raise SizingError(f"one projected price per forward year: {years}, not {price.shape}")
    bad_years = [y + 1 for y in np.flatnonzero(~(np.isfinite(price) & (price > 0)))]
    if bad_years:
        raise SizingError(
            "projected prices must be finite and positive: forward year "
            + ", ".join(f"{y} is {float(price[y - 1])!r}" for y in bad_years)
        )
    n = len(recipients)
    held = np.zeros((n, years)) if existing is None else np.asarray(existing, dtype=np.float64)
    if held.shape != (n, years):
        raise SizingError(
            f"existing vests must be (recipients, years), {(n, years)}, not {held.shape}"
        )
    bad_held = ~(np.isfinite(held) & (held >= 0)).all(axis=1)
    if bad_held.any():
        raise SizingError(
            "existing vests must be finite and non-negative, for "
            + _named(recipients.ids[bad_held])
        )
    # 1. Range.
    cells = ranges.lookup(recipients.families, recipients.levels, recipients.locations)
    # 2. Penetration.
    p = ratings.penetration(recipients.ratings)
    # 3. Target.
    low, high = cells[:, RangeField.MIN_TARGET], cells[:, RangeField.MAX_TARGET]
    target = low + p * (high - low)
    # 4. Base.
    low, high = cells[:, RangeField.MIN_BASE], cells[:, RangeField.MAX_BASE]
    base = cells[:, RangeField.BASE_ADJ] * (low + p * (high - low))
    # 5. Gap.
    gap = np.maximum(0.0, target[:, None] - base[:, None] - held * price[None, :])
    # 6. Shares.
    with np.errstate(divide="ignore", invalid="ignore"):
        shares_float = gap / price[None, :]
    # 7. Whole shares.
    _refuse(recipients.ids, {"target": target, "base": base, "gap": gap, "shares": shares_float})
    shares = np.rint(shares_float).astype(np.int64)
    return Sized(cells, p, target, base, gap, shares_float, shares, template.expand(shares))


def _named(ids: Ints) -> str:
    listed = ", ".join(f"employee {int(i)}" for i in ids[:NAMED])
    more = len(ids) - NAMED
    return listed + (f", and {more} more" if more > 0 else "")


def _refuse(ids: Ints, steps: Mapping[str, Floats]) -> None:
    """Refuse every recipient a step left non-finite, or with more shares than can be counted,
    naming each with the first step and forward year it went wrong in."""
    first: dict[int, str] = {}
    for step, values in steps.items():
        grid = values.reshape(len(ids), -1)
        wrong = ~np.isfinite(grid)
        if step == "shares":
            wrong |= np.isfinite(grid) & (grid > MAX_SHARES)
        for row, col in zip(*np.nonzero(wrong), strict=True):
            where = step if grid.shape[1] == 1 else f"{step}, forward year {int(col) + 1}"
            first.setdefault(int(row), where)
    if first:
        rows = sorted(first)
        named = [f"employee {int(ids[r])} (at the {first[r]})" for r in rows[:NAMED]]
        more = len(rows) - NAMED
        raise SizingError(
            f"{len(rows)} recipients got a grant that isn't a finite, countable number of shares, "
            "and none is granted zero instead: "
            + ", ".join(named)
            + (f", and {more} more" if more > 0 else "")
        )
