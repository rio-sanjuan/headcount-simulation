"""The world's workforce register, and the transactions every change records.

`Workforce` is the world's own register of who works there: one row per person ever present, as
integer-coded columns, with an active flag and a contingent flag. Every change goes through it,
and each change appends transaction rows that the data contract carries:

- `HIRE` (+1) and `TERM` (-1) for regular employees, and `RIF` (-1) for a termination program.
  Notice of a future departure is `TERM_PENDING` (0): reported on the day of notice, and dated
  on the day the person will leave.
- Two legs for a move between roles, `XFER_OUT` (-1) then `XFER_IN` (+1), and for a
  reorganization, `REORG_OUT` (-1) then `REORG_IN` (+1).
- `PROMO` (0) and `RECLASS` (0), which change the job in place.
- `CTG_START` (+1) and `CTG_END` (-1) for contingent workers, who book to a contingent account.
  A conversion out is `CONV_OUT` (-1) then `CTG_START` (+1), and a conversion in is `CTG_END`
  (-1) then `CONV_IN` (+1), so each account's stock stays in step with its flows.
- One `ACTIVE` stock row per person, regular or contingent, at every month-end.

A row's index never changes, so indices stay valid all run.
"""

from __future__ import annotations

from typing import Final

import numpy as np
from numpy.typing import DTypeLike, NDArray

from hcsim.synth.columns import EventLog, Index, Table
from hcsim.synth.population import ROLE_DTYPES, Roles

TRANS_TYPES: Final = (
    "ACTIVE",
    "HIRE",
    "TERM",
    "XFER_IN",
    "XFER_OUT",
    "PROMO",
    "REORG_IN",
    "REORG_OUT",
    "RECLASS",
    "CONV_IN",
    "CONV_OUT",
    "CTG_START",
    "CTG_END",
    "RIF",
    "TERM_PENDING",
    "ADMIN",
    # Emitted only by the data quirks (see `quirks`), never by the world's processes.
    "HIRE_REV",
    "TERM_REV",
    "HIRE_PENDING",
)
_CODE = {name: code for code, name in enumerate(TRANS_TYPES)}

#: Transaction columns, coded. `*_previous` hold the role before a move.
TXN_DTYPES: Final[dict[str, DTypeLike]] = {
    "employee_id": np.int64,
    "reporting_date": "datetime64[D]",
    "event_date": "datetime64[D]",
    "trans_type": np.int8,
    "amount": np.int8,
    **ROLE_DTYPES,
    "cost_center_previous": np.int32,
    "family_previous": np.int16,
    "level_previous": np.int16,
    "location_previous": np.int16,
    "company_previous": np.int16,
    "account_previous": np.int16,
    "is_manager_previous": np.bool_,
    "hire_date": "datetime64[D]",
}


class Workforce:
    def __init__(
        self, ids: NDArray[np.int64], roles: Roles, hire_dates: NDArray[np.datetime64]
    ) -> None:
        self.people = Table(
            {
                "employee_id": np.int64,
                **ROLE_DTYPES,
                "hire_date": "datetime64[D]",
                "active": bool,
                "rating": np.int8,  # a code into the rating scale; -1 until first rated
                "readiness": np.float32,  # latent promotion readiness
                "contingent": np.bool_,  # in the contingent pool, outside regular headcount
                "exit_month": np.int32,  # a departure given notice for this month; -1 for none
                "exit_date": "datetime64[D]",  # and its day
            }
        )
        self.people.append(
            len(ids),
            employee_id=ids,
            hire_date=hire_dates,
            active=True,
            rating=-1,
            exit_month=-1,
            **roles,
        )
        self.next_id = int(ids.max()) + 1 if len(ids) else 1
        self.log = EventLog(list(TXN_DTYPES))

    def __len__(self) -> int:
        return int(self.people.column("active").sum())

    def active(self) -> Index:
        """Regular employees in post."""
        return np.flatnonzero(self.people.column("active") & ~self.people.column("contingent"))

    def available(self) -> Index:
        """Regular employees in post who haven't given notice: the ones a move can take."""
        regular = self.people.column("active") & ~self.people.column("contingent")
        return np.flatnonzero(regular & (self.people.column("exit_month") < 0))

    def contingent_pool(self) -> Index:
        return np.flatnonzero(self.people.column("active") & self.people.column("contingent"))

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
        reported: NDArray[np.datetime64] | None = None,
    ) -> None:
        now = self.roles(index)
        before = previous if previous is not None else now
        self.log.add(
            len(index),
            employee_id=self.people.column("employee_id")[index],
            reporting_date=dates if reported is None else reported,
            event_date=dates,
            trans_type=_CODE[trans_type],
            amount=amount,
            cost_center_previous=before["cost_center"],
            family_previous=before["family"],
            level_previous=before["level"],
            location_previous=before["location"],
            company_previous=before["company"],
            account_previous=before["account"],
            is_manager_previous=before["is_manager"],
            hire_date=self.people.column("hire_date")[index],
            **now,
        )

    def reserve_ids(self, n: int) -> NDArray[np.int64]:
        """Ids for people who will join later, such as accepted offers."""
        ids = self.next_id + np.arange(n, dtype=np.int64)
        self.next_id += n
        return ids

    def hire(
        self,
        roles: Roles,
        dates: NDArray[np.datetime64],
        ids: NDArray[np.int64] | None = None,
    ) -> NDArray[np.int64]:
        """Hire one person per role, on each date, with reserved ids or new ones. Returns the
        employee ids."""
        n = len(dates)
        if ids is None:
            ids = self.reserve_ids(n)
        index = self.people.append(
            n, employee_id=ids, hire_date=dates, active=True, rating=-1, exit_month=-1, **roles
        )
        self._record(index, "HIRE", +1, dates)
        return ids

    def terminate(
        self, index: Index, dates: NDArray[np.datetime64], trans_type: str = "TERM"
    ) -> None:
        self._record(index, trans_type, -1, dates)
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

    def promote(self, index: Index, roles: Roles, dates: NDArray[np.datetime64]) -> None:
        """Change jobs in place, recorded as one `PROMO` row with the job before."""
        before = self.roles(index)
        for name, values in roles.items():
            self.people.column(name)[index] = values
        self._record(index, "PROMO", 0, dates, previous=before)

    def give_notice(
        self,
        index: Index,
        notice_dates: NDArray[np.datetime64],
        exit_dates: NDArray[np.datetime64],
        exit_months: NDArray[np.integer],
    ) -> None:
        """Schedule departures: a `TERM_PENDING` row reported today and dated on the exit."""
        self._record(index, "TERM_PENDING", 0, exit_dates, reported=notice_dates)
        self.people.column("exit_month")[index] = exit_months
        self.people.column("exit_date")[index] = exit_dates

    def reorganize(self, index: Index, cost_center: int, dates: NDArray[np.datetime64]) -> None:
        """Move people to another cost center in a batch: `REORG_OUT`, then `REORG_IN`."""
        before = self.roles(index)
        self._record(index, "REORG_OUT", -1, dates)
        self.people.column("cost_center")[index] = cost_center
        self._record(index, "REORG_IN", +1, dates, previous=before)

    def reclassify(self, index: Index, family: int, dates: NDArray[np.datetime64]) -> None:
        """Change people's family in place: one `RECLASS` row with the job before."""
        before = self.roles(index)
        self.people.column("family")[index] = family
        self._record(index, "RECLASS", 0, dates, previous=before)

    def enroll_contingent(self, roles: Roles, start_dates: NDArray[np.datetime64]) -> None:
        """Add contingent workers already in post when the world starts, with no event."""
        n = len(start_dates)
        ids = self.next_id + np.arange(n, dtype=np.int64)
        self.next_id += n
        self.people.append(
            n,
            employee_id=ids,
            hire_date=start_dates,
            active=True,
            rating=-1,
            exit_month=-1,
            contingent=True,
            **roles,
        )

    def start_contingent(self, roles: Roles, dates: NDArray[np.datetime64]) -> NDArray[np.int64]:
        n = len(dates)
        ids = self.next_id + np.arange(n, dtype=np.int64)
        self.next_id += n
        index = self.people.append(
            n,
            employee_id=ids,
            hire_date=dates,
            active=True,
            rating=-1,
            exit_month=-1,
            contingent=True,
            **roles,
        )
        self._record(index, "CTG_START", +1, dates)
        return ids

    def end_contingent(self, index: Index, dates: NDArray[np.datetime64]) -> None:
        self._record(index, "CTG_END", -1, dates)
        self.people.column("active")[index] = False

    def convert_out(self, index: Index, account: int, dates: NDArray[np.datetime64]) -> Roles:
        """Regular employees join the contingent pool. Returns the roles they left."""
        before = self.roles(index)
        self._record(index, "CONV_OUT", -1, dates)
        self.people.column("account")[index] = account
        self.people.column("contingent")[index] = True
        self.people.column("rating")[index] = -1
        self._record(index, "CTG_START", +1, dates, previous=before)
        return before

    def convert_in(self, index: Index, roles: Roles, dates: NDArray[np.datetime64]) -> None:
        """Contingent workers become regular employees in `roles`, with service from today."""
        before = self.roles(index)
        self._record(index, "CTG_END", -1, dates)
        for name, values in roles.items():
            self.people.column(name)[index] = values
        self.people.column("contingent")[index] = False
        self.people.column("hire_date")[index] = dates
        self.people.column("readiness")[index] = 0.0
        self._record(index, "CONV_IN", +1, dates, previous=before)

    def administer(self, index: Index, dates: NDArray[np.datetime64]) -> None:
        """Administrative changes with no effect on headcount: one `ADMIN` row each."""
        self._record(index, "ADMIN", 0, dates)

    def snapshot(self, month_end: np.datetime64) -> None:
        """One `ACTIVE` row per person in post, regular or contingent."""
        everyone = np.flatnonzero(self.people.column("active"))
        self._record(everyone, "ACTIVE", 1, month_end)


def tenure_months(
    hire_dates: NDArray[np.datetime64], month_end: np.datetime64
) -> NDArray[np.float64]:
    days = (month_end - hire_dates).astype("timedelta64[D]").astype(np.float64)
    return days / 30.4
