"""Rate calibration: logit shifts to the attrition and fill hazards that make the expected path
meet the plan.

Under `rate_calibration` (see `alignment`), the plan drives the domain rates. Each cost center
with a target at the plan's evaluation month `t*` takes two logit shifts, `δ_A` on every
attrition probability in it and `δ_F` on every fill-channel hazard of its openings,
`expit(logit p + δ)`. A shift applies after the model and before eligibility rules and known-event
exclusions, which always win.

**The two equations.** From the seed, the expected path to the end of `t*` (see `attainability`),
with backfill opening `b` requisitions per departure and growth at its model's rate, gives the
departures `D₀` and fills `F₀` the model expects. The gap to the target, `g = H* - H(t*)`, is
shared between them in proportion to their size, so departures fall by `g D₀ / (D₀ + F₀)` and
fills rise by `g F₀ / (D₀ + F₀)`. The shifts solve

    Σ_k a(δ_A) H(k) = D*        Σ_k f(δ_F) O(k) = F*

where `a` and `f` are the cost center's mean shifted hazards, over its employees and its open
openings as the seed holds them, and `H(k)` and `O(k)` the path the shifts themselves make. So
they're a fixed point: each equation is solved for its shift by bisection on the current path,
the path is recomputed, and the two alternate until neither shift moves. When both hold, the
target is met exactly in expectation, since `H(t*) = H(0) - D* + F* + inflows = H*`.

A cost center with no target at `t*` takes no shift. A shift is bounded at `±MAX_SHIFT`, and a
cost center whose shift reaches the bound, or whose iteration doesn't settle, is reported
unconverged, with its residual: the expected headcount at `t*` less the target.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Final, cast

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from hcsim.engine.attainability import known_inflows
from hcsim.engine.campus import is_campus
from hcsim.engine.ceiling import Ceiling
from hcsim.engine.config import BackfillSource
from hcsim.engine.logit import logit_shift
from hcsim.engine.state import SimulationState
from hcsim.model.core.layouts import EmployeeColumn, OpeningColumn
from hcsim.model.interface import EntityModel, GroupRateModel

if TYPE_CHECKING:
    from hcsim.engine.domains import StepContext

Floats = NDArray[np.float64]
Mask = NDArray[np.bool_]
#: The largest logit shift either hazard takes.
MAX_SHIFT: Final = 10.0
#: How far a shift may still move, in logits, when the fixed point counts as settled.
TOLERANCE: Final = 1e-9
MAX_ITERATIONS: Final = 200
_BISECTIONS: Final = 80


@dataclass(frozen=True)
class Calibration:
    """Each cost center's two shifts, whether its fixed point settled within the bounds, and its
    residual at `t*`, from the seed at step 0. Cost centers with no target take no shift."""

    attrition: Floats
    fill: Floats
    converged: Mask
    residual: Floats
    #: The evaluation step `t*`, and the fixed point's iterations.
    step: int
    iterations: int

    def to_dict(self) -> dict[str, object]:
        return {
            "step": self.step,
            "iterations": self.iterations,
            "attrition": self.attrition.tolist(),
            "fill": self.fill.tolist(),
            "converged": self.converged.tolist(),
            "residual": self.residual.tolist(),
        }


@dataclass(frozen=True)
class _Seed:
    """What the equations read from the seed: each row's hazard and cost center."""

    people: Floats  # attrition probability per employee, eligible or zero
    people_cc: NDArray[np.int64]
    hazards: Floats  # (openings, channels), from the models, before any shift
    eligible: Floats  # (openings, channels), 1 where the channel may fill the row
    weights: Floats  # openings per requisition
    reqs_cc: NDArray[np.int64]
    unfilled: Floats  # per cost center
    backfill: Floats  # requisitions per departure, per cost center
    growth: Floats  # requisitions per month, per cost center
    n: int


def calibrate(state: SimulationState, ctx: StepContext) -> Calibration | None:
    """The shifts that make the expected path from `state` meet the plan at `t*`, or None for a
    run whose plan isn't scored inside it."""
    config = ctx.config
    if config.plan is None:
        return None
    s = (config.plan.evaluation_month - pd.Period(config.run_date, freq="M")).n
    if not 0 <= s < config.simulation_months:
        return None
    seed = _seed(state, ctx)
    whole = Ceiling.build(config.schema, {}, config.plan)
    target = whole.targets(config.run_date, s)
    planned = ~np.isnan(target)
    months = s + 1
    inflow = known_inflows(state, ctx.known, months)
    h0 = state.headcount.astype(np.float64)
    o0 = state.open_openings.astype(np.float64)
    zero = np.zeros(seed.n)
    base = _path(seed, h0, o0, zero, zero, inflow, months)
    departures, fills = base[2].sum(axis=0), base[3].sum(axis=0)
    flows = departures + fills
    gap = np.where(planned, np.nan_to_num(target) - base[0][months], 0.0)
    share = np.divide(departures, flows, out=np.full(seed.n, 0.5), where=flows > 0)
    wanted_d = np.maximum(0.0, departures - share * gap)
    wanted_f = np.maximum(0.0, fills + (1 - share) * gap)
    delta_a, delta_f = zero.copy(), zero.copy()
    iterations = 0
    while iterations < MAX_ITERATIONS:
        iterations += 1
        h, held, _, _ = _path(seed, h0, o0, delta_a, delta_f, inflow, months)
        exposure_h, exposure_o = h[:months].sum(axis=0), held.sum(axis=0)
        rate_a = np.divide(wanted_d, exposure_h, out=zero.copy(), where=exposure_h > 0)
        rate_f = np.divide(wanted_f, exposure_o, out=zero.copy(), where=exposure_o > 0)
        new_a = np.where(planned, _solve(lambda d: _attrition(seed, d), rate_a), 0.0)
        new_f = np.where(planned, _solve(lambda d: _fill(seed, d), rate_f), 0.0)
        moved = max(np.abs(new_a - delta_a).max(initial=0), np.abs(new_f - delta_f).max(initial=0))
        delta_a, delta_f = new_a, new_f
        if moved < TOLERANCE:
            break
    h, _, _, _ = _path(seed, h0, o0, delta_a, delta_f, inflow, months)
    residual = np.where(planned, h[months] - np.nan_to_num(target), 0.0)
    bounded = (np.abs(delta_a) >= MAX_SHIFT - 1e-6) | (np.abs(delta_f) >= MAX_SHIFT - 1e-6)
    converged = ~planned | (~bounded & (np.abs(residual) < 1e-6 * np.maximum(1.0, target)))
    return Calibration(delta_a, delta_f, np.asarray(converged), residual, s, iterations)


def _seed(state: SimulationState, ctx: StepContext) -> _Seed:
    from hcsim.engine.domains import ATTRITION, CANCELLATION
    from hcsim.engine.fills import CHANNEL_DOMAINS, model_hazards

    config, models, n = ctx.config, ctx.config.models, state.n_cost_centers
    eligibility = config.resolved_eligibility
    people = state.live_employees
    p = models.attrition.compute_probability(people)
    p = eligibility.restrict(ATTRITION, people, np.asarray(p, dtype=np.float64))
    reqs = state.live_openings
    open_rows = ~is_campus(reqs, ctx)
    eligible = (
        np.stack(
            [eligibility.restrict(d, reqs, np.ones(len(reqs))) for d in CHANNEL_DOMAINS], axis=1
        )
        * open_rows[:, None]
    )
    weights = reqs[:, OpeningColumn.NUM_OPENINGS].astype(np.float64)
    reqs_cc = reqs[:, OpeningColumn.COST_CENTER].astype(np.int64)
    q = eligibility.restrict(CANCELLATION, reqs, models.cancellation.compute_probability(reqs))
    q = np.where((ctx.reserved(reqs) > 0) | ~open_rows, 0.0, q)
    if config.backfill_source is BackfillSource.CLASSIFIER:
        scores = cast(EntityModel, models.backfill).compute_probability(people)
        backfill = _mean(people[:, EmployeeColumn.COST_CENTER], scores, np.ones(len(people)), n)
    else:
        backfill = _rates(cast(GroupRateModel, models.backfill), n)
    return _Seed(
        p,
        people[:, EmployeeColumn.COST_CENTER].astype(np.int64),
        model_hazards(reqs, ctx),
        eligible,
        weights,
        reqs_cc,
        _mean(reqs_cc, q, weights, n),
        backfill,
        _rates(models.growth, n),
        n,
    )


def _rates(model: GroupRateModel, n: int) -> Floats:
    return np.array([model.rate_for(c) for c in range(n)], dtype=np.float64)


def _mean(groups: NDArray[Any], values: Floats, weights: Floats, n: int) -> Floats:
    codes = np.asarray(groups, dtype=np.int64)
    total = np.bincount(codes, weights=weights, minlength=n)
    summed = np.bincount(codes, weights=weights * np.asarray(values, dtype=np.float64), minlength=n)
    out: Floats = np.divide(summed, total, out=np.zeros(n), where=total > 0)
    return out


def _attrition(seed: _Seed, delta: Floats) -> Floats:
    """Each cost center's mean attrition hazard under the shift `delta`."""
    shifted = logit_shift(seed.people, delta[seed.people_cc])
    return _mean(seed.people_cc, shifted, np.ones(len(shifted)), seed.n)


def _fill(seed: _Seed, delta: Floats) -> Floats:
    """Each cost center's mean fill hazard per opening under the shift `delta`, the channels
    rescaled as the fills domain rescales them."""
    from hcsim.engine.fills import rescale

    shifted = logit_shift(seed.hazards, delta[seed.reqs_cc][:, None]) * seed.eligible
    return _mean(seed.reqs_cc, rescale(shifted).sum(axis=1), seed.weights, seed.n)


def _solve(rate: Callable[[Floats], Floats], wanted: Floats) -> Floats:
    """The shift per cost center at which the monotone `rate` reaches `wanted`, by bisection
    within the bounds."""
    lo, hi = np.full(len(wanted), -MAX_SHIFT), np.full(len(wanted), MAX_SHIFT)
    for _ in range(_BISECTIONS):
        mid = (lo + hi) / 2
        above = rate(mid) > wanted
        hi, lo = np.where(above, mid, hi), np.where(above, lo, mid)
    out: Floats = (lo + hi) / 2
    return out


def _path(
    seed: _Seed,
    h0: Floats,
    o0: Floats,
    delta_a: Floats,
    delta_f: Floats,
    inflow: Floats,
    months: int,
) -> tuple[Floats, Floats, Floats, Floats]:
    """The expected path under the shifts, in the step order, departures, then backfill for
    them, then fills, then exits unfilled, then growth: headcount (months + 1 rows), and each
    month's openings held when fills draw, departures, and fills (months rows)."""
    a, f, u = _attrition(seed, delta_a), _fill(seed, delta_f), seed.unfilled
    h, o = np.zeros((months + 1, seed.n)), np.zeros((months + 1, seed.n))
    leaving, held = np.zeros((months, seed.n)), np.zeros((months, seed.n))
    h[0], o[0] = h0, o0
    for k in range(months):
        leaving[k] = a * h[k]
        held[k] = o[k] + seed.backfill * leaving[k]  # backfill opens before the month's fills
        filling = f * held[k]
        h[k + 1] = h[k] - leaving[k] + filling + inflow[k]
        o[k + 1] = (1 - u) * (held[k] - filling) + seed.growth  # growth opens last
    return h, held, leaving, f * held
