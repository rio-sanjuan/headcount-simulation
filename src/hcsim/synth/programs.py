"""Termination programs: planned reductions in one level-1 org over a multi-month window.

At the start of each month, a program may start in an org that isn't already in one. Its
target is a share of the org's regular headcount that day, apportioned evenly over its window.
Each month of the window, before any sampled departure, it terminates that month's part,
chosen without replacement with weights by rating (a Gumbel top-k), on days spread over the
month. Program terminations are `RIF` rows, and they open no backfill. Each program is recorded
for the truth manifest.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from hcsim.synth.config import WorldConfig
from hcsim.synth.dates import MonthDays
from hcsim.synth.structure import level_one_names
from hcsim.synth.workforce import Workforce


@dataclass
class _Program:
    org: int
    start: pd.Period
    plan: list[int]  # terminations per month of the window
    done: list[int] = field(default_factory=list)


class TerminationPrograms:
    def __init__(
        self,
        config: WorldConfig,
        workforce: Workforce,
        org_of_cost_center: NDArray[np.int64],
        rng: np.random.Generator,
    ) -> None:
        self.truth = config.programs
        self._date_truth = config.dates
        self.workforce = workforce
        self.org = org_of_cost_center
        self.rng = rng
        self.orgs = level_one_names(int(org_of_cost_center.max()) + 1)
        self.running: list[_Program] = []
        self.finished: list[_Program] = []
        self._weight = np.asarray(self.truth.rating_weight, dtype=np.float64)

    def _days(self, month: pd.Period) -> MonthDays:
        return MonthDays(month, self._date_truth)

    def _members(self, org: int) -> NDArray[np.int64]:
        active = self.workforce.active()
        members: NDArray[np.int64] = active[
            self.org[self.workforce.column("cost_center")[active]] == org
        ]
        return members

    def _maybe_start(self, month: pd.Period) -> None:
        if self.rng.random() >= self.truth.start_probability:
            return
        busy = {p.org for p in self.running}
        free = [o for o in range(len(self.orgs)) if o not in busy]
        if not free:
            return
        org = int(self.rng.choice(free))
        months = int(
            self.rng.integers(self.truth.window_months[0], self.truth.window_months[1] + 1)
        )
        target = round(self.rng.uniform(*self.truth.share) * len(self._members(org)))
        plan = [target // months + (1 if k < target % months else 0) for k in range(months)]
        self.running.append(_Program(org, month, plan))

    def run(self, month: pd.Period) -> int:
        """Start a program, perhaps, then carry out this month's part of every running one.
        Returns the number of people terminated."""
        self._maybe_start(month)
        total = 0
        for program in list(self.running):
            k = len(program.done)
            members = self._members(program.org)
            n = min(program.plan[k], len(members))
            rating = self.workforce.column("rating")[members]
            weight = np.where(
                rating >= 0, self._weight[np.maximum(rating, 0)], self.truth.unrated_weight
            )
            keys = np.log(weight) + self.rng.gumbel(size=len(members))
            chosen = members[np.argsort(-keys)[:n]]
            days = self._days(month).sample("exit", n, self.rng)
            self.workforce.terminate(chosen, days, "RIF")
            program.done.append(n)
            total += n
            if len(program.done) == len(program.plan):
                self.running.remove(program)
                self.finished.append(program)
        return total

    def manifest(self) -> list[dict[str, Any]]:
        return [
            {
                "org": self.orgs[p.org],
                "start": str(p.start),
                "months": len(p.plan),
                "planned": int(sum(p.plan)),
                "terminated": int(sum(p.done)),
            }
            for p in self.finished + self.running
        ]
