"""Performance ratings: calibration cycles, a Markov chain between them, and first ratings.

A calibration takes effect on the first day of each cycle month, before that month's events,
so departures and moves in the cycle month already read the new ratings. Everyone active with
at least `min_tenure_months` of tenure on that day is rated: from the entry distribution if
they've never been rated, and otherwise by one step of the Markov chain from their last rating.
When the world starts, incumbents carry a rating from the last cycle before it, drawn from the
chain's long-run distribution. A rating is a code into the truth's scale; -1 means unrated.
"""

from __future__ import annotations

from typing import Final

import numpy as np
import pandas as pd
from numpy.typing import DTypeLike

from hcsim.synth.columns import EventLog
from hcsim.synth.config import RatingTruth
from hcsim.synth.workforce import Workforce, tenure_months

UNRATED: Final = -1
RATING_DTYPES: Final[dict[str, DTypeLike]] = {
    "employee_id": np.int64,
    "cycle_date": "datetime64[D]",
    "rating": np.int8,
    "is_first": np.bool_,
}


def last_cycle_before(truth: RatingTruth, first_month: pd.Period) -> pd.Period:
    """The latest cycle month strictly before `first_month`."""
    for back in range(1, 13):
        month = first_month - back
        if month.month in truth.cycle_months:
            return month
    raise AssertionError("cycle_months is never empty")  # pragma: no cover - validated


def _draw(probabilities: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """One category per row of `probabilities`."""
    cumulative = np.cumsum(probabilities, axis=-1)
    u = rng.random(len(probabilities))[:, None]
    return np.minimum((u > cumulative).sum(axis=1), probabilities.shape[-1] - 1).astype(np.int8)


class RatingProcess:
    def __init__(self, truth: RatingTruth, workforce: Workforce, rng: np.random.Generator) -> None:
        self.truth = truth
        self.workforce = workforce
        self.rng = rng
        self.log = EventLog(list(RATING_DTYPES))
        self._transitions = np.asarray(truth.transitions, dtype=np.float64)
        self._entry = np.asarray(truth.entry_distribution, dtype=np.float64)

    def _eligible(self, day: np.datetime64) -> np.ndarray:
        active = self.workforce.active()
        tenure = tenure_months(self.workforce.column("hire_date")[active], day)
        return active[tenure >= self.truth.min_tenure_months]

    def _record(
        self, index: np.ndarray, day: np.datetime64, ratings: np.ndarray, first: np.ndarray
    ) -> None:
        self.workforce.column("rating")[index] = ratings
        self.log.add(
            len(index),
            employee_id=self.workforce.column("employee_id")[index],
            cycle_date=day,
            rating=ratings,
            is_first=first,
        )

    def seed(self, first_month: pd.Period) -> None:
        """Rate the incumbents as of the last cycle before the world starts."""
        day = np.datetime64(last_cycle_before(self.truth, first_month).start_time.date(), "D")
        index = self._eligible(day)
        stationary = np.broadcast_to(self.truth.stationary(), (len(index), len(self._entry)))
        self._record(index, day, _draw(stationary, self.rng), np.zeros(len(index), dtype=bool))

    def is_cycle(self, month: pd.Period) -> bool:
        return month.month in self.truth.cycle_months

    def calibrate(self, month: pd.Period) -> None:
        """Run a calibration effective on the first day of `month`."""
        day = np.datetime64(month.start_time.date(), "D")
        index = self._eligible(day)
        current = self.workforce.column("rating")[index]
        first = current == UNRATED
        rows = np.where(
            first[:, None], self._entry[None, :], self._transitions[np.maximum(current, 0)]
        )
        self._record(index, day, _draw(rows, self.rng), first)
