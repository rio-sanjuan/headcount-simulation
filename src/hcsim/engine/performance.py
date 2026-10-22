"""The performance domain: rating cycles.

Ratings change once a year, in the configured cycle month, before anything else is sampled
that month, so the month's attrition and promotions read the new ratings:

- a rated employee moves from their rating by the Markov model's transitions;
- an unrated employee receives a first rating from the model's entry distribution, once their
  tenure at the cycle reaches the first-rating cutoff. Anyone newer stays unrated until the
  next cycle.

`PERF_RATING` holds a rating's Markov state plus one, and 0 for the unrated, so on a single
rating scale it's the rating's rank. Eligibility rules on `performance` keep their rows out of
a cycle. Without a performance model, the domain doesn't register, and ratings stay as seeded.
"""

from __future__ import annotations

from typing import Final

import numpy as np

from hcsim.engine.domains import StepContext
from hcsim.engine.results import DomainResult, RowChange
from hcsim.engine.state import SimulationState
from hcsim.model.core.layouts import EMPLOYEE, EmployeeColumn

PERFORMANCE: Final = "performance"


def performance(state: SimulationState, ctx: StepContext) -> DomainResult:
    model = ctx.config.models.performance
    if model is None or ctx.month != ctx.config.performance_cycle_month:
        return DomainResult(PERFORMANCE)
    rows = state.live_employees
    rating = rows[:, EmployeeColumn.PERF_RATING].astype(np.int64)
    tenure = ctx.step - rows[:, EmployeeColumn.HIRE_STEP]
    first = (rating == 0) & (tenure >= ctx.config.performance_first_rating_min_tenure_months)
    rated = (rating > 0) | first
    rated &= ctx.config.resolved_eligibility.eligible(PERFORMANCE, rows)
    index = np.flatnonzero(rated)
    previous = np.where(rating[index] > 0, rating[index] - 1, -1)
    drawn = model.sample(previous[:, None], ctx.rng(PERFORMANCE))
    written = rows[index].copy()
    written[:, EmployeeColumn.PERF_RATING] = drawn + 1
    return DomainResult(
        PERFORMANCE, employees=RowChange(EMPLOYEE, write_index=index, write_rows=written)
    )
