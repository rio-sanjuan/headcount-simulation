"""The world's requisition book: every requisition opened, and every state interval it passed
through.

A requisition carries one or more openings. Each fill takes one opening and ends the current
state interval; the next interval starts the same day with one opening fewer. A requisition
closes when its last opening fills, or when it exits unfilled, which ends its interval with
every remaining opening. Requisition ids start high, so they never collide with employee ids.
"""

from __future__ import annotations

from typing import Final

import numpy as np
from numpy.typing import DTypeLike, NDArray

from hcsim.synth.columns import EventLog, Index, Table
from hcsim.synth.population import ROLE_DTYPES, Roles

REQ_TYPES: Final = ("backfill", "campus", "growth")
CLOSE_REASONS: Final = (
    "filled_external",
    "filled_transfer",
    "unfilled",
    "filled_conversion",
    "filled_campus",
)
OPEN: Final = -1
UNFILLED: Final = CLOSE_REASONS.index("unfilled")
INTERVAL_DTYPES: Final[dict[str, DTypeLike]] = {
    "req": np.int64,
    "interval_start": "datetime64[D]",
    "interval_end": "datetime64[D]",
    "openings": np.int16,
    "end_reason": np.int8,
    "filled_by": np.int64,
}


class RequisitionBook:
    def __init__(self, first_id: int = 500_001) -> None:
        self.reqs = Table(
            {
                "req_id": np.int64,
                "req_type": np.int8,
                **ROLE_DTYPES,
                "opened_date": "datetime64[D]",
                "opened_month": np.int32,
                "openings": np.int16,  # at opening
                "remaining": np.int16,  # still to fill
                "interval_start": "datetime64[D]",  # of the current state interval
                "closed_date": "datetime64[D]",
                "close_reason": np.int8,
                "backfill_of": np.int64,  # the person whose role a backfill replaces
            }
        )
        self.intervals = EventLog(list(INTERVAL_DTYPES))
        self.next_id = first_id

    def __len__(self) -> int:
        return len(self.reqs)

    def column(self, name: str) -> np.ndarray:
        return self.reqs.column(name)

    def roles(self, index: Index) -> Roles:
        return {name: self.reqs.column(name)[index] for name in ROLE_DTYPES}

    def open(
        self,
        req_type: str,
        roles: Roles,
        dates: NDArray[np.datetime64],
        month: int,
        backfill_of: NDArray[np.int64] | None = None,
        openings: NDArray[np.integer] | None = None,
    ) -> Index:
        n = len(dates)
        ids = self.next_id + np.arange(n, dtype=np.int64)
        self.next_id += n
        count = np.ones(n, dtype=np.int16) if openings is None else openings
        return self.reqs.append(
            n,
            req_id=ids,
            req_type=REQ_TYPES.index(req_type),
            opened_date=dates,
            opened_month=month,
            openings=count,
            remaining=count,
            interval_start=dates,
            close_reason=OPEN,
            backfill_of=-1 if backfill_of is None else backfill_of,
            **roles,
        )

    def open_requisitions(self) -> Index:
        return np.flatnonzero(self.reqs.column("close_reason") == OPEN)

    def _end_intervals(
        self, index: Index, dates: NDArray[np.datetime64], reason: int, fillers: NDArray[np.int64]
    ) -> None:
        self.intervals.add(
            len(index),
            req=index,
            interval_start=self.reqs.column("interval_start")[index],
            interval_end=dates,
            openings=self.reqs.column("remaining")[index],
            end_reason=reason,
            filled_by=fillers,
        )

    def fill(
        self,
        index: Index,
        dates: NDArray[np.datetime64],
        reasons: NDArray[np.integer],
        fillers: NDArray[np.int64],
    ) -> None:
        """Fill one opening per entry, in date order within each requisition."""
        order = np.lexsort((dates, index))
        index, dates = index[order], dates[order]
        reasons, fillers = np.asarray(reasons)[order], fillers[order]
        rank = np.zeros(len(index), dtype=np.int64)
        if len(index):
            starts = np.r_[True, index[1:] != index[:-1]]
            rank = np.arange(len(index)) - np.maximum.accumulate(
                np.where(starts, np.arange(len(index)), 0)
            )
        for r in range(int(rank.max(initial=-1)) + 1):  # the r-th fill of each requisition
            pick = rank == r
            j, day = index[pick], dates[pick]
            for reason in np.unique(reasons[pick]).tolist():
                same = reasons[pick] == reason
                self._end_intervals(j[same], day[same], reason, fillers[pick][same])
            left = self.reqs.column("remaining")[j] - 1
            self.reqs.column("remaining")[j] = left
            self.reqs.column("interval_start")[j] = day
            done = j[left == 0]
            self.reqs.column("closed_date")[done] = day[left == 0]
            self.reqs.column("close_reason")[done] = reasons[pick][left == 0]

    def exit(self, index: Index, dates: NDArray[np.datetime64]) -> None:
        """Close requisitions unfilled, with every opening they still carry."""
        self._end_intervals(index, dates, UNFILLED, np.full(len(index), -1, dtype=np.int64))
        self.reqs.column("closed_date")[index] = dates
        self.reqs.column("close_reason")[index] = UNFILLED
