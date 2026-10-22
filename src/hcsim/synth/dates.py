"""Event days within a month, drawn from a profile per kind of event.

- `start`: a hire's first day, on the configured weekday.
- `exit`: a departure, on a business day, with the month's last business day weighted up.
- `business`: any business day, for every other event.
- `any`: any calendar day.

Each event may carry an earliest day, such as a requisition's opening date for its fill. Days
before it are excluded, and when the profile leaves no day, the event falls uniformly on any
day from its earliest one to the month's end.
"""

from __future__ import annotations

from typing import Final

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from hcsim.synth.config import DateTruth

Dates = NDArray[np.datetime64]
PROFILES: Final = ("start", "exit", "business", "any")


class MonthDays:
    def __init__(self, month: pd.Period, truth: DateTruth) -> None:
        self.first = np.datetime64(month.start_time.date(), "D")
        self.days = self.first + np.arange(month.days_in_month).astype("timedelta64[D]")
        weekday = (
            self.days.astype("datetime64[D]").view("int64") - 4
        ) % 7  # 1970-01-01 was a Thursday
        business = weekday < 5
        exit_weight = business.astype(np.float64)
        exit_weight[np.flatnonzero(business)[-1]] = truth.exit_month_end_weight
        self.weights = {
            "start": (weekday == truth.start_weekday).astype(np.float64),
            "exit": exit_weight,
            "business": business.astype(np.float64),
            "any": np.ones(len(self.days)),
        }

    def sample(
        self, profile: str, n: int, rng: np.random.Generator, earliest: Dates | None = None
    ) -> Dates:
        weight = np.broadcast_to(self.weights[profile], (n, len(self.days)))
        if earliest is not None:
            late_enough = self.days[None, :] >= np.asarray(earliest, dtype="datetime64[D]")[:, None]
            weight = weight * late_enough
            none_left = weight.sum(axis=1) == 0
            weight = np.where(none_left[:, None], late_enough.astype(np.float64), weight)
        cumulative = np.cumsum(weight, axis=1)
        u = rng.random(n)[:, None] * cumulative[:, -1:]
        index = np.minimum((u >= cumulative).sum(axis=1), len(self.days) - 1)
        result: Dates = self.days[index]
        return result
