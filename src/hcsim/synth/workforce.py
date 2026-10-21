"""The world's monthly workforce process, and the transactions it records.

`Workforce` is the world's own register of who is employed. Every change goes through it, and
each change appends the transaction rows the data contract carries: `HIRE` (+1), `TERM` (-1),
the two legs of a move between cost centers, `XFER_OUT` (-1) and `XFER_IN` (+1), and one
`ACTIVE` stock row per employee at every month-end. Event dates fall within the month.

The departure hazard is the world's ground truth, and it differs in form from the engine's
models on purpose: it's nonlinear in tenure, seasonal, and carries a random effect per cost
center. Every parameter is invented, and the truth manifest records them all.
"""

from __future__ import annotations

import datetime as dt
from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

ROLE_FIELDS = (
    "cost_center",
    "job_code",
    "job_family",
    "job_level",
    "location",
    "company",
    "account",
    "is_manager",
)


@dataclass(frozen=True)
class AttritionTruth:
    """The true monthly departure hazard. Invented, round values."""

    base_monthly_hazard: float = 0.008
    level_factor: dict[int, float] = field(
        default_factory=lambda: {1: 1.25, 2: 1.0, 3: 0.9, 4: 0.8, 5: 0.7}
    )
    first_year_factor: float = 1.5
    seasonal_amplitude: float = 0.2
    seasonal_peak_month: int = 3
    cost_center_sigma: float = 0.2

    def hazard(
        self, level: int, tenure_months: float, month_of_year: int, cost_center_effect: float
    ) -> float:
        season = 1 + self.seasonal_amplitude * np.cos(
            2 * np.pi * (month_of_year - self.seasonal_peak_month) / 12
        )
        tenure = self.first_year_factor if tenure_months < 12 else 1.0
        h = self.base_monthly_hazard * self.level_factor[level] * tenure * season
        return float(min(1.0, h * cost_center_effect))


def month_days(month_end: pd.Timestamp) -> int:
    return int(month_end.day)


def random_day(month_end: pd.Timestamp, rng: np.random.Generator) -> dt.date:
    return dt.date(month_end.year, month_end.month, int(rng.integers(1, month_days(month_end) + 1)))


class Workforce:
    """The register of active employees, and the transactions every change records."""

    def __init__(self, initial: pd.DataFrame, next_id: int) -> None:
        self.people: dict[int, dict[str, Any]] = {
            int(row["employee_id"]): {str(k): v for k, v in row.items()}
            for row in initial.to_dict("records")
        }
        self.next_id = next_id
        self.rows: list[dict[str, Any]] = []

    def __len__(self) -> int:
        return len(self.people)

    def __iter__(self) -> Iterator[tuple[int, dict[str, Any]]]:
        return iter(list(self.people.items()))

    def _record(
        self,
        person: dict[str, Any],
        trans_type: str,
        amount: int,
        date: dt.date,
        previous: dict[str, Any] | None = None,
    ) -> None:
        before = previous or person
        self.rows.append(
            {
                "employee_id": person["employee_id"],
                "reporting_date": date,
                "event_date": date,
                "trans_type": trans_type,
                "amount": amount,
                "cost_center": person["cost_center"],
                "cost_center_previous": before["cost_center"],
                "job_code": person["job_code"],
                "job_code_previous": before["job_code"],
                "location": person["location"],
                "company": person["company"],
                "account": person["account"],
                "is_manager": person["is_manager"],
                "hire_date": person["hire_date"],
            }
        )

    def hire(self, role: dict[str, Any], date: dt.date) -> int:
        employee_id = self.next_id
        self.next_id += 1
        person = {
            "employee_id": employee_id,
            **{k: role[k] for k in ROLE_FIELDS},
            "hire_date": date,
        }
        self.people[employee_id] = person
        self._record(person, "HIRE", +1, date)
        return employee_id

    def terminate(self, employee_id: int, date: dt.date) -> dict[str, Any]:
        person = self.people.pop(employee_id)
        self._record(person, "TERM", -1, date)
        return person

    def reassign(self, employee_id: int, role: dict[str, Any], date: dt.date) -> dict[str, Any]:
        """Move an employee into `role`: a transfer, recorded as XFER_OUT then XFER_IN."""
        before = dict(self.people[employee_id])
        self._record(before, "XFER_OUT", -1, date)
        self.people[employee_id].update({k: role[k] for k in ROLE_FIELDS})
        self._record(self.people[employee_id], "XFER_IN", +1, date, previous=before)
        return before

    def snapshot(self, month_end: pd.Timestamp) -> None:
        for person in self.people.values():
            self._record(person, "ACTIVE", 1, month_end.date())


def tenure_months(person: dict[str, Any], month_end: pd.Timestamp) -> float:
    return (month_end.date() - person["hire_date"]).days / 30.4
