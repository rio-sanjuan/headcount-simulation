"""The world's requisition book: every requisition opened, open or closed.

Every requisition carries one opening in this world so far. Requisitions are rows of
integer-coded columns. A closed requisition records its date, its reason, and the employee who
filled it. Requisition ids start high, so they never collide with employee ids.
"""

from __future__ import annotations

from typing import Final

import numpy as np
from numpy.typing import NDArray

from hcsim.synth.columns import Index, Table
from hcsim.synth.population import ROLE_DTYPES, Roles

REQ_TYPES: Final = ("backfill", "campus", "growth")
CLOSE_REASONS: Final = ("filled_external", "filled_transfer", "unfilled", "filled_conversion")
OPEN: Final = -1


class RequisitionBook:
    def __init__(self, first_id: int = 500_001) -> None:
        self.reqs = Table(
            {
                "req_id": np.int64,
                "req_type": np.int8,
                **ROLE_DTYPES,
                "opened_date": "datetime64[D]",
                "opened_month": np.int32,
                "closed_date": "datetime64[D]",
                "close_reason": np.int8,
                "filled_by": np.int64,
            }
        )
        self.next_id = first_id

    def __len__(self) -> int:
        return len(self.reqs)

    def column(self, name: str) -> np.ndarray:
        return self.reqs.column(name)

    def roles(self, index: Index) -> Roles:
        return {name: self.reqs.column(name)[index] for name in ROLE_DTYPES}

    def open(self, req_type: str, roles: Roles, dates: NDArray[np.datetime64], month: int) -> Index:
        n = len(dates)
        ids = self.next_id + np.arange(n, dtype=np.int64)
        self.next_id += n
        return self.reqs.append(
            n,
            req_id=ids,
            req_type=REQ_TYPES.index(req_type),
            opened_date=dates,
            opened_month=month,
            close_reason=OPEN,
            filled_by=-1,
            **roles,
        )

    def open_requisitions(self) -> Index:
        return np.flatnonzero(self.reqs.column("close_reason") == OPEN)

    def close(
        self,
        index: Index,
        dates: NDArray[np.datetime64],
        reason: str,
        filled_by: NDArray[np.int64] | None = None,
    ) -> None:
        self.reqs.column("closed_date")[index] = dates
        self.reqs.column("close_reason")[index] = CLOSE_REASONS.index(reason)
        if filled_by is not None:
            self.reqs.column("filled_by")[index] = filled_by
