"""The world's workforce register, and the transactions every change records.

`Workforce` is the world's own register of who is employed: one row per person ever employed,
as integer-coded columns, with an active flag. Every change goes through it, and each change
appends transaction rows that the data contract carries: `HIRE` (+1), `TERM` (-1), the two legs
of a move between roles, `XFER_OUT` (-1) and `XFER_IN` (+1), and one `ACTIVE` stock row per
employee at every month-end. A row's index never changes, so indices stay valid all run.
"""

from __future__ import annotations

from typing import Final

import numpy as np
from numpy.typing import DTypeLike, NDArray

from hcsim.synth.columns import EventLog, Index, Table
from hcsim.synth.population import ROLE_DTYPES, Roles

TRANS_TYPES: Final = ("ACTIVE", "HIRE", "TERM", "XFER_IN", "XFER_OUT")
_CODE = {name: code for code, name in enumerate(TRANS_TYPES)}

#: Transaction columns, coded. `*_previous` hold the role before a move.
TXN_DTYPES: Final[dict[str, DTypeLike]] = {
    "employee_id": np.int64,
    "event_date": "datetime64[D]",
    "trans_type": np.int8,
    "amount": np.int8,
    **ROLE_DTYPES,
    "cost_center_previous": np.int32,
    "family_previous": np.int16,
    "level_previous": np.int16,
    "hire_date": "datetime64[D]",
}


class Workforce:
    def __init__(
        self, ids: NDArray[np.int64], roles: Roles, hire_dates: NDArray[np.datetime64]
    ) -> None:
        self.people = Table(
            {"employee_id": np.int64, **ROLE_DTYPES, "hire_date": "datetime64[D]", "active": bool}
        )
        self.people.append(len(ids), employee_id=ids, hire_date=hire_dates, active=True, **roles)
        self.next_id = int(ids.max()) + 1 if len(ids) else 1
        self.log = EventLog(list(TXN_DTYPES))

    def __len__(self) -> int:
        return int(self.people.column("active").sum())

    def active(self) -> Index:
        return np.flatnonzero(self.people.column("active"))

    def roles(self, index: Index) -> Roles:
        return {name: self.people.column(name)[index] for name in ROLE_DTYPES}

    def column(self, name: str) -> np.ndarray:
        return self.people.column(name)

    def _record(
        self,
        index: Index,
        trans_type: str,
        amount: int,
        dates: NDArray[np.datetime64] | np.datetime64,
        previous: Roles | None = None,
    ) -> None:
        now = self.roles(index)
        before = previous if previous is not None else now
        self.log.add(
            len(index),
            employee_id=self.people.column("employee_id")[index],
            event_date=dates,
            trans_type=_CODE[trans_type],
            amount=amount,
            cost_center_previous=before["cost_center"],
            family_previous=before["family"],
            level_previous=before["level"],
            hire_date=self.people.column("hire_date")[index],
            **now,
        )

    def hire(self, roles: Roles, dates: NDArray[np.datetime64]) -> NDArray[np.int64]:
        """Hire one person per role, on each date. Returns the new employee ids."""
        n = len(dates)
        ids = self.next_id + np.arange(n, dtype=np.int64)
        self.next_id += n
        index = self.people.append(n, employee_id=ids, hire_date=dates, active=True, **roles)
        self._record(index, "HIRE", +1, dates)
        return ids

    def terminate(self, index: Index, dates: NDArray[np.datetime64]) -> None:
        self._record(index, "TERM", -1, dates)
        self.people.column("active")[index] = False

    def reassign(self, index: Index, roles: Roles, dates: NDArray[np.datetime64]) -> Roles:
        """Move employees into new roles: XFER_OUT in the old role, then XFER_IN in the new one.
        Returns the roles they left."""
        before = self.roles(index)
        self._record(index, "XFER_OUT", -1, dates)
        for name, values in roles.items():
            self.people.column(name)[index] = values
        self._record(index, "XFER_IN", +1, dates, previous=before)
        return before

    def snapshot(self, month_end: np.datetime64) -> None:
        self._record(self.active(), "ACTIVE", 1, month_end)


def tenure_months(
    hire_dates: NDArray[np.datetime64], month_end: np.datetime64
) -> NDArray[np.float64]:
    days = (month_end - hire_dates).astype("timedelta64[D]").astype(np.float64)
    return days / 30.4
