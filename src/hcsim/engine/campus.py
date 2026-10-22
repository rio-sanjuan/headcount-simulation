"""The campus domain: campus requisitions on a schedule, filled in the months campus hires start.

**The schedule lookup.** A `CampusSchedule` names, for calendar months, how many campus
requisitions each cost center opens (by label), and the calendar months campus hires start.
In a month the schedule lists, the domain opens that many one-opening campus requisitions per
cost center.

**The required attribute generator.** A campus requisition's role is drawn from the campus
attribute generator, given its cost center, as growth's are from its own. A run with a campus
schedule must have one.

**Isolation from the ledger.** Campus requisitions follow the schedule, not the guidance: they
open outside the requisitions-opened quota and its ledger, and the regular fills, transfers,
and cancellation leave them alone. They count as `G_camp` when they open.

**The campus fill-model hook.** In a start month, campus requisitions fill: by the campus fill
model's probability for each, where the run has one, and every one of them where it doesn't.
Each fill hires into the requisition's role with a new-hire grant, as `F_camp`.

A run without a campus schedule has no campus domain, and any campus requisitions it seeds are
filled and cancelled like any other.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import TYPE_CHECKING, Final

import numpy as np
from numpy.typing import NDArray

from hcsim.engine.results import DomainResult, Grant, RowChange
from hcsim.engine.state import SimulationState
from hcsim.model.core.layouts import (
    EMPLOYEE,
    OPENING,
    Buffer,
    EmployeeColumn,
    OpeningColumn,
    copy_role,
)
from hcsim.model.core.schema import COST_CENTER, REQ_TYPE, SimulationSchema
from hcsim.model.ml.generative import ATTRIBUTES
from hcsim.model.ml.sampling import bernoulli

if TYPE_CHECKING:
    from hcsim.engine.domains import StepContext

Ints = NDArray[np.int64]
CAMPUS: Final = "campus"


@dataclass(frozen=True)
class CampusSchedule:
    #: Calendar month to campus requisitions per cost-center label.
    openings: Mapping[int, Mapping[str, int]]
    #: The calendar months campus hires start.
    start_months: tuple[int, ...] = (7,)
    _codes: dict[int, Ints] = field(default_factory=dict, repr=False, compare=False)

    def __post_init__(self) -> None:
        frozen = {int(m): MappingProxyType(dict(c)) for m, c in self.openings.items()}
        object.__setattr__(self, "openings", MappingProxyType(frozen))
        object.__setattr__(self, "start_months", tuple(self.start_months))

    def problems(self, schema: SimulationSchema) -> list[str]:
        """What keeps the schedule from fitting the run; it resolves cost centers on the way."""
        found = []
        months = [*self.openings, *self.start_months]
        if any(not 1 <= m <= 12 for m in months):
            found.append(f"campus months must lie in 1..12: {sorted(set(months))}")
        if REQ_TYPE not in schema.vocabularies or CAMPUS not in schema.vocabulary(REQ_TYPE):
            found.append("a campus schedule needs 'campus' among the requisition types")
        vocab = schema.vocabulary(COST_CENTER)
        for month, counts in sorted(self.openings.items()):
            codes = np.zeros(len(vocab), dtype=np.int64)
            for label, count in sorted(counts.items()):
                if label not in vocab:
                    found.append(f"no cost center {label!r} for campus in month {month}")
                elif count < 0:
                    found.append(f"campus openings for {label} in month {month} are negative")
                else:
                    codes[vocab.code(label)] = count
            self._codes[month] = codes
        return found

    def opened(self, month: int, n: int) -> Ints:
        """Campus requisitions to open in a calendar month, per cost-center code."""
        result: Ints = self._codes.get(month, np.zeros(n, dtype=np.int64))
        return result


def is_campus(reqs: Buffer, ctx: StepContext) -> NDArray[np.bool_]:
    """The campus requisitions, when the run has a campus schedule to fill them."""
    if ctx.config.campus is None:
        return np.zeros(len(reqs), dtype=bool)
    code = ctx.config.schema.vocabulary(REQ_TYPE).code(CAMPUS)
    result: NDArray[np.bool_] = reqs[:, OpeningColumn.REQ_TYPE] == code
    return result


def campus(state: SimulationState, ctx: StepContext) -> DomainResult:
    schedule = ctx.config.campus
    generator = ctx.config.models.campus_attributes
    if schedule is None or generator is None:
        return DomainResult(CAMPUS)
    n = state.n_cost_centers
    rng = ctx.rng(CAMPUS)
    counts = schedule.opened(ctx.month, n)
    code = ctx.config.schema.vocabulary(REQ_TYPE).code(CAMPUS)
    opened = OPENING.empty(int(counts.sum()))
    opened[:, OpeningColumn.REQ_ID] = -1
    opened[:, OpeningColumn.CREATION_STEP] = ctx.step
    opened[:, OpeningColumn.REQ_TYPE] = code
    opened[:, OpeningColumn.NUM_OPENINGS] = 1
    start = 0
    for cc in np.flatnonzero(counts):
        k = int(counts[cc])
        opened[start : start + k, OpeningColumn.COST_CENTER] = cc
        opened[start : start + k][:, list(ATTRIBUTES)] = generator.sample(int(cc), k, rng)
        start += k
    reqs = state.live_openings
    filled = np.zeros(len(reqs), dtype=bool)
    if ctx.month in schedule.start_months:
        mine = is_campus(reqs, ctx)
        model = ctx.config.models.campus_fill
        p = np.ones(len(reqs)) if model is None else model.compute_probability(reqs)
        filled = mine & bernoulli(np.where(mine, p, 0.0), rng)
    rows = np.flatnonzero(filled)
    per_row = reqs[rows, OpeningColumn.NUM_OPENINGS].astype(np.int64)
    source = np.repeat(rows, per_row)  # every opening of a filled campus requisition
    hires = EMPLOYEE.empty(len(source))
    copy_role(reqs[source], hires)
    hires[:, EmployeeColumn.EMPLOYEE_ID] = -1
    hires[:, EmployeeColumn.HIRE_STEP] = ctx.step
    return DomainResult(
        CAMPUS,
        employees=RowChange(EMPLOYEE, append=hires),
        openings=RowChange(OPENING, drop=rows, append=opened),
        grants=(Grant("new_hire"),),
        flows={
            "G_camp": counts,
            "F_camp": np.bincount(
                hires[:, EmployeeColumn.COST_CENTER].astype(np.int64), minlength=n
            ).astype(np.int64),
        },
    )
