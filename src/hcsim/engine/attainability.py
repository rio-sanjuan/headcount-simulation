"""The plan's reach: the known-event credit, the expected path, the natural floor, and the
attainability classes.

**The known-event credit** (`known_credit`) is the net headcount change known events make to
each cost center over the lookahead `(t, t+L̂]`, steps `t+1` through `t+L̂`, that committed
strength `S = H + O` doesn't already hold. The supply rule subtracts it from the plan gap, so
growth opens for a known departure before it happens, and not for a known arrival:

- a known termination or conversion out takes one from its cost center;
- a reorganization, other movement, or promotion that moves someone to another cost center takes
  one from the source and adds one to the destination, and one across the run's boundary only
  takes one out, or adds one in;
- a known transfer takes one from its source: its destination fills an opening `S` already
  holds;
- a pending start adds nothing, since its reserved opening is already in `S`.

**Hazards** (`Hazards.of`) are each cost center's monthly rates from the run's own models: `a`,
the mean attrition probability over its employees; `f`, its open openings' mean summed fill
hazard over the channels, weighted by openings; and `u`, the mean chance a requisition exits
unfilled, weighted by openings. They're model rates, not guidance's counts.

**The expected path** (`expected_path`) holds the hazards and steps the expectation forward a
month at a time, as the step order does: each head leaves with `a`, each open opening fills
with `f`, and one that doesn't exits with `u`. New openings join the stock after the month's
fills, as growth's do, and known net inflows join headcount in their month:

    fills(k) = f O(k)
    H(k+1)   = (1 - a) H(k) + fills(k) + inflow(k)
    O(k+1)   = (1 - u) (O(k) - fills(k)) + new(k)

**The natural floor** `H_floor(c, t)` is the expected path with no new supply: the seed
surviving attrition, plus the expected fills of the openings already open, plus the known net
inflows (`natural_floor`).

**Attainability** (`attainability`) classes each cost center's target at the plan's evaluation
month `t*`, on the headcount at that month's end:

- `attainable`: the floor doesn't pass the target by more than its band, the cap leaves room for
  it, and, if the floor falls short, openings have time to fill;
- `below_floor`: the floor alone passes the target by more than its band. Supply only adds, so
  the target needs exits beyond attrition;
- `above_cap`: the target is above the capacity cap, which vetoes reaching it;
- `out_of_time`: the floor falls short, and no opening can fill by `t*`, since growth's openings
  first fill the month after they open, or the cost center's fill hazard is zero;
- `no_target`: the plan has no target for the cost center at `t*`.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from hcsim.engine.campus import is_campus
from hcsim.engine.ceiling import Ceiling
from hcsim.engine.known_events import MOVEMENTS, EventKind
from hcsim.engine.known_resolution import ResolvedPlan
from hcsim.engine.state import SimulationState
from hcsim.model.core.layouts import EmployeeColumn, OpeningColumn

if TYPE_CHECKING:
    from hcsim.engine.domains import StepContext

Floats = NDArray[np.float64]
Ints = NDArray[np.int64]
_OUT = frozenset({EventKind.TERMINATION, EventKind.CONVERSION_OUT})


def known_credit(
    state: SimulationState, known: ResolvedPlan | None, step: int, lookahead: int
) -> Floats:
    """Each cost center's net headcount change from known events in steps `step + 1` through
    `step + lookahead`, that committed strength doesn't already hold."""
    n = state.n_cost_centers
    credit = np.zeros(n)
    if known is None or lookahead < 1:
        return credit
    live = state.live_employees
    where = dict(
        zip(
            live[:, EmployeeColumn.EMPLOYEE_ID].astype(np.int64).tolist(),
            live[:, EmployeeColumn.COST_CENTER].astype(np.int64).tolist(),
            strict=True,
        )
    )
    for s in range(step + 1, step + lookahead + 1):
        for resolved in known.at(s):
            kind = resolved.event.kind
            source = where.get(resolved.event.employee_id)
            destination = resolved.codes.get(EmployeeColumn.COST_CENTER)
            if kind in _OUT or resolved.boundary == "out":
                if source is not None:
                    credit[source] -= 1
            elif resolved.boundary == "in":
                if destination is not None:
                    credit[destination] += 1
            elif kind in MOVEMENTS and destination is not None and destination != source:
                if source is not None:
                    credit[source] -= 1
                if kind is not EventKind.TRANSFER:
                    credit[destination] += 1
    return credit


@dataclass(frozen=True)
class Hazards:
    """Each cost center's monthly hazards: attrition per head, fill per open opening, and exit
    unfilled per open opening."""

    attrition: Floats
    fill: Floats
    unfilled: Floats

    @classmethod
    def of(cls, state: SimulationState, ctx: StepContext) -> Hazards:
        """The run's model hazards in `state`, at `ctx`'s step."""
        from hcsim.engine.domains import ATTRITION, CANCELLATION
        from hcsim.engine.fills import channel_hazards

        n, models = state.n_cost_centers, ctx.config.models
        eligibility = ctx.config.resolved_eligibility
        people = state.live_employees
        p = eligibility.restrict(ATTRITION, people, models.attrition.compute_probability(people))
        attrition = _mean(people[:, EmployeeColumn.COST_CENTER], p, np.ones(len(people)), n)
        reqs = state.live_openings
        weights = reqs[:, OpeningColumn.NUM_OPENINGS].astype(np.float64)
        cost_centers = reqs[:, OpeningColumn.COST_CENTER]
        fill = _mean(cost_centers, channel_hazards(reqs, ctx).sum(axis=1), weights, n)
        q = eligibility.restrict(CANCELLATION, reqs, models.cancellation.compute_probability(reqs))
        q = np.where((ctx.reserved(reqs) > 0) | is_campus(reqs, ctx), 0.0, q)
        return cls(attrition, fill, _mean(cost_centers, q, weights, n))


def _mean(groups: NDArray[np.float64], values: Floats, weights: Floats, n: int) -> Floats:
    codes = groups.astype(np.int64)
    total = np.bincount(codes, weights=weights, minlength=n)
    summed = np.bincount(codes, weights=weights * values, minlength=n)
    out: Floats = np.divide(summed, total, out=np.zeros(n), where=total > 0)
    return out


def expected_path(
    headcount: Floats,
    openings: Floats,
    hazards: Hazards,
    months: int,
    new: Floats | None = None,
    inflow: Floats | None = None,
) -> tuple[Floats, Floats]:
    """(months + 1, cost centers) expected headcount and open openings from `headcount` and
    `openings`, with `new` openings joining after each month's fills and `inflow` heads joining
    in each month, both (months, cost centers) and zero by default."""
    n = len(headcount)
    added = np.zeros((months, n)) if new is None else np.asarray(new, dtype=np.float64)
    joined = np.zeros((months, n)) if inflow is None else np.asarray(inflow, dtype=np.float64)
    h, o = np.zeros((months + 1, n)), np.zeros((months + 1, n))
    h[0], o[0] = headcount, openings
    a, f, u = hazards.attrition, hazards.fill, hazards.unfilled
    for k in range(months):
        fills = f * o[k]
        h[k + 1] = (1 - a) * h[k] + fills + joined[k]
        o[k + 1] = (1 - u) * (o[k] - fills) + added[k]
    return h, o


def known_inflows(state: SimulationState, known: ResolvedPlan | None, months: int) -> Floats:
    """(months, cost centers): each month's net headcount change from known events that
    committed strength doesn't already hold, as the credit counts them."""
    return np.array([known_credit(state, known, k - 1, 1) for k in range(months)]).reshape(
        months, state.n_cost_centers
    )


def natural_floor(
    state: SimulationState, hazards: Hazards, known: ResolvedPlan | None, months: int
) -> Floats:
    """`H_floor`: (months + 1, cost centers), the expected path with no new supply."""
    floor, _ = expected_path(
        state.headcount.astype(np.float64),
        state.open_openings.astype(np.float64),
        hazards,
        months,
        inflow=known_inflows(state, known, months),
    )
    return floor


class Attainability(StrEnum):
    ATTAINABLE = "attainable"
    BELOW_FLOOR = "below_floor"
    ABOVE_CAP = "above_cap"
    OUT_OF_TIME = "out_of_time"
    NO_TARGET = "no_target"


@dataclass(frozen=True)
class Reach:
    """Each cost center's target at `t*`, its floor at that month's end, and its class."""

    step: int
    target: Floats
    floor: Floats
    classes: tuple[Attainability, ...]

    def counts(self) -> dict[str, int]:
        return {c.value: self.classes.count(c) for c in Attainability}


def attainability(state: SimulationState, ctx: StepContext) -> Reach | None:
    """Each cost center's attainability at the plan's evaluation month, from `state` at `ctx`'s
    step, or None for a run with no plan or one that ends before `t*`. Every target counts,
    whichever valve the plan drives for it."""
    config = ctx.config
    if config.plan is None:
        return None
    s = (config.plan.evaluation_month - pd.Period(config.run_date, freq="M")).n - ctx.step
    if not 0 <= s < config.simulation_months - ctx.step:
        return None
    hazards = Hazards.of(state, ctx)
    floor = natural_floor(state, hazards, ctx.known, s + 1)[s + 1]
    whole = Ceiling.build(config.schema, {}, config.plan)
    target = whole.targets(config.run_date, ctx.step + s)
    band = whole.bands(config.run_date, ctx.step + s)
    caps = config.ceiling.caps
    classes = []
    for c in range(len(target)):
        if np.isnan(target[c]):
            classes.append(Attainability.NO_TARGET)
        elif floor[c] > target[c] + band[c]:
            classes.append(Attainability.BELOW_FLOOR)
        elif target[c] > caps[c]:
            classes.append(Attainability.ABOVE_CAP)
        elif floor[c] < target[c] - band[c] and (s < 1 or hazards.fill[c] <= 0):
            classes.append(Attainability.OUT_OF_TIME)
        else:
            classes.append(Attainability.ATTAINABLE)
    return Reach(s, target, floor, tuple(classes))
