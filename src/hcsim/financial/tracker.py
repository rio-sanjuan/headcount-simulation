"""The RSU tracker's holder ledger: every granted tranche, by recipient.

One row per holder, award month, and vest month, never summed. On a departure, the ledger
stamps the departure month on the holder's live rows that vest after it, so a leaver's
unvested equity forfeits in the month they leave. Each trajectory holds its own copy.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

Ints = NDArray[np.int64]

NOT_FORFEITED = -1


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


@dataclass
class HolderLedger:
    _chunks: list[Ints] = field(default_factory=list)
    _rows: Ints = field(default_factory=lambda: np.zeros((0, 8), dtype=np.int64))

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
        return int(hit.sum())

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
