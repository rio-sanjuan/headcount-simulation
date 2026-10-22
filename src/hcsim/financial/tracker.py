"""The RSU tracker: the holder ledger of every granted tranche, and each holder's vest buffer.

**The holder ledger** keeps one row per holder, award month, and vest month, never summed. Each
trajectory holds its own copy. It supports three changes:

- **forfeit**: on a departure, it stamps the departure month on the holder's live rows that
  vest after it, so a leaver's unvested equity forfeits in the month they leave;
- **move**: when a holder moves cost center, it starts a holder interval at the new cost center
  from that month, so expense after the move can follow them while each tranche keeps the cost
  center it was granted in;
- **timed grants**: a grant recorded before its award month, as award timing lags it, waits in
  the ledger. If its holder leaves before the award month, it's cancelled, never awarded: it
  forfeits before it accrues anything.

**The vest buffer** (`VestBuffer`) keeps each holder's live shares by vest month, so what a
recipient already vests in a forward year is a slice, not a scan of the ledger. It spans every
month of the run and a 120-month tail after it, enough for any grant the run makes: a 72-month
template after an award lagged up to two years. Its rows are allocated lazily: a holder takes a
row the first time they're granted, so a workforce that's never granted costs nothing. A vest
before the run's first month has already happened, so the buffer leaves it out.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Final

import numpy as np
from numpy.typing import NDArray

Ints = NDArray[np.int64]

NOT_FORFEITED = -1
#: The months the vest buffer spans after the run's last.
TAIL: Final = 120


class TrackerError(ValueError):
    """A tranche the tracker can't hold."""


@dataclass
class VestBuffer:
    """Each holder's live shares by vest month, over the run's `months` and a `TAIL` after."""

    months: int = 0
    _row_of: dict[int, int] = field(default_factory=dict)
    _shares: Ints = field(default_factory=lambda: np.zeros((0, TAIL), dtype=np.int64))

    def __post_init__(self) -> None:
        self._shares = np.zeros((0, self.width), dtype=np.int64)

    @property
    def width(self) -> int:
        return self.months + TAIL

    @property
    def allocated(self) -> int:
        """How many holders hold a row."""
        return len(self._row_of)

    def cover(self, months: int) -> None:
        """Widen the buffer to a run of `months` months, and its tail."""
        if months > self.months:
            extra = months - self.months
            self._shares = np.pad(self._shares, ((0, 0), (0, extra)))
            self.months = months

    def _rows(self, holders: Ints, allocate: bool) -> Ints:
        """Each holder's row, allocating new ones when asked, else -1 for those without."""
        out = np.full(len(holders), -1, dtype=np.int64)
        for k, holder in enumerate(holders.tolist()):
            row = self._row_of.get(holder)
            if row is None and allocate:
                row = self._row_of[holder] = len(self._row_of)
            if row is not None:
                out[k] = row
        if allocate and len(self._row_of) > len(self._shares):
            grown = max(len(self._row_of), 2 * len(self._shares), 64)
            self._shares = np.pad(self._shares, ((0, grown - len(self._shares)), (0, 0)))
        return out

    def add(self, holders: Ints, vest_months: Ints, shares: Ints) -> None:
        """Add each tranche's shares at its holder's vest month."""
        holders, months = np.asarray(holders, np.int64), np.asarray(vest_months, np.int64)
        beyond = months >= self.width
        if beyond.any():
            raise TrackerError(
                f"{int(beyond.sum())} tranches vest past the vest buffer's {TAIL}-month tail, "
                f"in month {int(months.max())} of a {self.months}-month run"
            )
        ahead = months >= 0
        rows = self._rows(holders[ahead], allocate=True)
        np.add.at(self._shares, (rows, months[ahead]), np.asarray(shares, np.int64)[ahead])

    def forfeit(self, holders: Ints, month: int) -> None:
        """Clear each holder's vests after `month`."""
        rows = self._rows(np.asarray(holders, np.int64), allocate=False)
        rows = rows[rows >= 0]
        self._shares[rows, max(month + 1, 0) :] = 0

    def by_year(self, holders: Ints, month: int, years: int) -> NDArray[np.float64]:
        """(holders, years): each holder's shares vesting in each forward year after `month`,
        year y covering months `month + 12(y - 1) + 1` through `month + 12y`."""
        rows = self._rows(np.asarray(holders, np.int64), allocate=False)
        out = np.zeros((len(rows), years))
        held = rows >= 0
        for y in range(years):
            first, last = month + 12 * y + 1, min(month + 12 * (y + 1) + 1, self.width)
            if first < last:
                out[held, y] = self._shares[rows[held], max(first, 0) : last].sum(axis=1)
        return out

    def held(self, holders: Ints) -> Ints:
        """(holders, width): each holder's live shares by vest month."""
        rows = self._rows(np.asarray(holders, np.int64), allocate=False)
        out = np.zeros((len(rows), self.width), dtype=np.int64)
        out[rows >= 0] = self._shares[rows[rows >= 0]]
        return out


@dataclass(frozen=True)
class Tranches:
    """Every tranche as parallel arrays. `forfeit_month` is -1 for a live tranche."""

    holder: Ints
    award_month: Ints
    vest_month: Ints
    shares: NDArray[np.float64]
    forfeit_month: Ints
    award_type: Ints
    level: Ints
    cost_center: Ints

    def __len__(self) -> int:
        return len(self.holder)

    @property
    def cancelled(self) -> NDArray[np.bool_]:
        """Timed grants whose holder left before their award month: never awarded."""
        out: NDArray[np.bool_] = (self.forfeit_month >= 0) & (self.forfeit_month < self.award_month)
        return out

    def pending(self, month: int) -> NDArray[np.bool_]:
        """Timed grants still to be awarded after `month`, and not cancelled."""
        out: NDArray[np.bool_] = (self.award_month > month) & ~self.cancelled
        return out


@dataclass(frozen=True)
class Intervals:
    """Holder intervals: from `valid_from`, the holder sits in `cost_center`, until their next
    interval. Sorted by holder, then month."""

    holder: Ints
    valid_from: Ints
    cost_center: Ints

    def __len__(self) -> int:
        return len(self.holder)


@dataclass
class HolderLedger:
    _chunks: list[Ints] = field(default_factory=list)
    _rows: Ints = field(default_factory=lambda: np.zeros((0, 8), dtype=np.int64))
    buffer: VestBuffer = field(default_factory=VestBuffer)
    _moves: list[Ints] = field(default_factory=list)

    # Column order of the consolidated rows.
    HOLDER, AWARD, VEST, SHARES, FORFEIT, TYPE, LEVEL, CC = range(8)

    def _consolidate(self) -> Ints:
        if self._chunks:
            self._rows = np.concatenate([self._rows, *self._chunks])
            self._chunks.clear()
        return self._rows

    def record(
        self,
        holders: Ints,
        award_month: int,
        schedule: Ints,
        award_type: int,
        levels: Ints,
        cost_centers: Ints,
        anchor_month: int | None = None,
    ) -> int:
        """Record one grant per holder. Vests are laid out from `anchor_month`, which defaults to
        the award month. Returns the number of tranches recorded."""
        anchor = award_month if anchor_month is None else anchor_month
        holder_idx, column = np.nonzero(schedule)
        if len(holder_idx) == 0:
            return 0
        rows = np.empty((len(holder_idx), 8), dtype=np.int64)
        rows[:, self.HOLDER] = np.asarray(holders)[holder_idx]
        rows[:, self.AWARD] = award_month
        rows[:, self.VEST] = anchor + column + 1
        rows[:, self.SHARES] = schedule[holder_idx, column]
        rows[:, self.FORFEIT] = NOT_FORFEITED
        rows[:, self.TYPE] = award_type
        rows[:, self.LEVEL] = np.asarray(levels)[holder_idx]
        rows[:, self.CC] = np.asarray(cost_centers)[holder_idx]
        self.buffer.add(rows[:, self.HOLDER], rows[:, self.VEST], rows[:, self.SHARES])
        self._chunks.append(rows)
        return len(rows)

    def forfeit(self, holders: Ints, month: int) -> int:
        """Forfeit each holder's live tranches that vest after `month`. Returns rows forfeited."""
        rows = self._consolidate()
        hit = (
            np.isin(rows[:, self.HOLDER], holders)
            & (rows[:, self.FORFEIT] == NOT_FORFEITED)
            & (rows[:, self.VEST] > month)
        )
        rows[hit, self.FORFEIT] = month
        self.buffer.forfeit(np.asarray(holders, np.int64), month)
        return int(hit.sum())

    def vesting_by_year(self, holders: Ints, month: int, years: int) -> NDArray[np.float64]:
        """(holders, years): the shares each holder's live tranches vest in each forward year
        after `month`, from the vest buffer. Forward year y (from 1) covers vest months
        `month + 12(y - 1) + 1` through `month + 12y`."""
        return self.buffer.by_year(np.asarray(holders, dtype=np.int64), month, years)

    def move(self, holders: Ints, month: int, cost_centers: Ints) -> int:
        """Start each holder's interval at `cost_centers` from `month`. Returns the moves."""
        holders = np.asarray(holders, dtype=np.int64)
        if len(holders):
            rows = np.empty((len(holders), 3), dtype=np.int64)
            rows[:, 0], rows[:, 1], rows[:, 2] = holders, month, np.asarray(cost_centers)
            self._moves.append(rows)
        return len(holders)

    def intervals(self) -> Intervals:
        """Every move, as holder intervals. A holder moved twice in one month keeps the last."""
        if not self._moves:
            empty = np.zeros(0, dtype=np.int64)
            return Intervals(empty, empty.copy(), empty.copy())
        rows = np.concatenate(self._moves)
        order = np.lexsort((np.arange(len(rows)), rows[:, 1], rows[:, 0]))
        rows = rows[order]
        last = np.ones(len(rows), dtype=bool)
        last[:-1] = (rows[1:, 0] != rows[:-1, 0]) | (rows[1:, 1] != rows[:-1, 1])
        rows = rows[last]
        return Intervals(rows[:, 0].copy(), rows[:, 1].copy(), rows[:, 2].copy())

    def cost_center_at(self, holders: Ints, months: Ints, granted: Ints) -> Ints:
        """Each (holder, month)'s cost center: their latest interval's at or before the month,
        else `granted`, the cost center they were granted in."""
        holders, months = np.asarray(holders, np.int64), np.asarray(months, np.int64)
        out = np.asarray(granted, dtype=np.int64).copy()
        moves = self.intervals()
        if not len(moves):
            return out
        # Sort the intervals by (holder, month); find each query's last interval at or before it.
        key = moves.holder * (1 << 32) + moves.valid_from
        query = holders * (1 << 32) + months
        at = np.searchsorted(key, query, side="right") - 1
        found = (at >= 0) & (moves.holder[np.maximum(at, 0)] == holders)
        out[found] = moves.cost_center[at[found]]
        return out

    def tranches(self) -> Tranches:
        rows = self._consolidate()
        return Tranches(
            holder=rows[:, self.HOLDER].copy(),
            award_month=rows[:, self.AWARD].copy(),
            vest_month=rows[:, self.VEST].copy(),
            shares=rows[:, self.SHARES].astype(np.float64),
            forfeit_month=rows[:, self.FORFEIT].copy(),
            award_type=rows[:, self.TYPE].copy(),
            level=rows[:, self.LEVEL].copy(),
            cost_center=rows[:, self.CC].copy(),
        )

    def __len__(self) -> int:
        return len(self._consolidate())
