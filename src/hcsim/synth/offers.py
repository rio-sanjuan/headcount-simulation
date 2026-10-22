"""Offers, pending starts, no-shows, and hires with no requisition (see `OfferTruth`).

An external fill makes an accepted offer: the opening closes on the offer day, naming the
person's future employee id. The person starts on the first start weekday after a lag, and is
a pending start at every month-end in between. A no-show never starts; their opening reopens on
the day they were due.
"""

from __future__ import annotations

from typing import Final

import numpy as np
import pandas as pd
from numpy.typing import DTypeLike, NDArray

from hcsim.synth.columns import EventLog, Table
from hcsim.synth.config import WorldConfig
from hcsim.synth.dates import MonthDays
from hcsim.synth.population import ROLE_DTYPES, Roles, draw_roles
from hcsim.synth.requisitions import RequisitionBook
from hcsim.synth.workforce import Workforce

Dates = NDArray[np.datetime64]
PENDING_DTYPES: Final[dict[str, DTypeLike]] = {
    "snapshot_date": "datetime64[D]",
    "offer": np.int64,
}


def next_weekday(dates: Dates, weekday: int) -> Dates:
    """The first `weekday` (Monday is 0) on or after each date."""
    day = (dates.view("int64") + 3) % 7
    result: Dates = dates + ((weekday - day) % 7).astype("timedelta64[D]")
    return result


class Offers:
    def __init__(self, config: WorldConfig, rng: np.random.Generator) -> None:
        self.config = config
        self.truth = config.offers
        self.rng = rng
        self.book = Table(
            {
                "employee_id": np.int64,
                "req": np.int64,
                **ROLE_DTYPES,
                "offer_date": "datetime64[D]",
                "start_date": "datetime64[D]",
                "no_show": np.bool_,
                "done": np.bool_,
            }
        )
        self.pending = EventLog(list(PENDING_DTYPES))

    def make(
        self,
        workforce: Workforce,
        reqs: NDArray[np.int64],
        roles: Roles,
        dates: Dates,
    ) -> NDArray[np.int64]:
        """Accept one offer per filled opening. Returns the people's future employee ids."""
        n = len(dates)
        ids = workforce.reserve_ids(n)
        low, high = self.truth.start_lag_days
        lag = self.rng.integers(low, high + 1, n).astype("timedelta64[D]")
        start = next_weekday(dates + lag, self.config.dates.start_weekday)
        no_show = self.rng.random(n) < self.truth.no_show_probability
        self.book.append(
            n,
            employee_id=ids,
            req=reqs,
            offer_date=dates,
            start_date=start,
            no_show=no_show,
            **roles,
        )
        return ids

    def start(self, workforce: Workforce, requisitions: RequisitionBook, month: pd.Period) -> None:
        """Start everyone due this month, and reopen the openings of those who don't show."""
        last = np.datetime64(month.end_time.date(), "D")
        due = np.flatnonzero(~self.book.column("done") & (self.book.column("start_date") <= last))
        start = self.book.column("start_date")[due]
        shows = ~self.book.column("no_show")[due]
        joining = due[shows]
        roles = {name: self.book.column(name)[joining] for name in ROLE_DTYPES}
        workforce.hire(roles, start[shows], self.book.column("employee_id")[joining])
        # A reopening never predates the requisition's current interval.
        missed = self.book.column("req")[due[~shows]]
        day = np.maximum(start[~shows], requisitions.column("interval_start")[missed])
        requisitions.reopen(missed, day)
        self.book.column("done")[due] = True

    def snapshot(self, month_end: np.datetime64) -> None:
        """Record every accepted offer that hasn't started by this month-end."""
        waiting = np.flatnonzero(
            (self.book.column("offer_date") <= month_end)
            & (self.book.column("start_date") > month_end)
            & ~self.book.column("done")
        )
        self.pending.add(len(waiting), snapshot_date=month_end, offer=waiting)


def direct_hires(
    config: WorldConfig, workforce: Workforce, days: MonthDays, rng: np.random.Generator
) -> None:
    """Hires with no requisition: a Poisson count against regular headcount, at any level."""
    n = int(rng.poisson(config.offers.no_requisition_hire_rate * len(workforce.active())))
    cost_centers = rng.integers(config.cost_centers, size=n)
    levels = rng.choice(np.asarray(config.levels), size=n, p=list(config.level_mix))
    roles = draw_roles(config, cost_centers, levels, rng)
    workforce.hire(roles, days.sample("start", n, rng))
