"""The alignment known-answer suite: deterministic runs whose every month can be worked by hand.

Every case runs the toy state, 30 heads and 4 open openings in each of three cost centers, on
models with no randomness left in the counts: no one leaves unless a known event says so, every
open opening fills the month it's open, no requisition exits unfilled, no departure is backfilled,
and growth outside the plan opens nothing. Fills come before growth in the step order, so the
seed's 4 openings fill in the first month, and an opening growth adds fills the month after. Each
case says, month by month, why the headcount is what it is.
"""

from __future__ import annotations

import dataclasses
from typing import Any

import numpy as np
import pandas as pd

from hcsim.engine.attainability import attainability
from hcsim.engine.domains import StepContext
from hcsim.engine.known_events import EventKind, KnownEvent, KnownEvents, Source
from hcsim.engine.known_plan import sanitize
from hcsim.engine.known_resolution import resolve_plan
from hcsim.engine.loop import SimulationResult, simulate
from hcsim.engine.seeds import SeedManager
from hcsim.model.core.alignment import AlignmentTarget, Interpolation, marginal
from hcsim.model.ml.group import BackfillRateModel, PoissonGrowthModel
from tests.engine.test_loop import SCHEMA, config, guidance, initial_state, models

#: The toy state through deterministic models: only the plan's supply and known events move it.
STILL = dataclasses.replace(
    models(attrition=0.0, fill=1.0, cancel=0.0),
    backfill=BackfillRateModel.constant(0.0, 3),
    growth=PoissonGrowthModel(3, min_months=1).fit(np.zeros((1, 3))),
)
#: With no supply, each cost center's seed openings fill in the first month, and that's all.
FLAT = [30, 34, 34, 34, 34, 34, 34]


def plan(values: dict[str, list[float]], dates: list[str] | None = None) -> AlignmentTarget:
    """Targets at `dates`, January by default, held between and after, over the run's six
    months."""
    dates = dates or ["2026-01"]
    m = marginal("cost_center", list(values), dates, list(values.values()))
    return AlignmentTarget((m,), pd.Period("2026-01", "M"), 6, 6, Interpolation.HOLD_LAST)


def run(**overrides: Any) -> SimulationResult:
    known = overrides.pop("known", None)
    cfg = config(models=STILL, num_trajectories=2, **overrides)
    return simulate(cfg, initial_state(), known=known, validate=True)


def headcount(result: SimulationResult, cost_center: int) -> list[int]:
    paths = result.headcount[:, :, cost_center]
    assert (paths == paths[0]).all()  # deterministic: every trajectory alike
    return paths[0].tolist()


def test_strength_closes_the_whole_gap_at_a_gain_of_one() -> None:
    # January: the 4 seed openings fill, 34 heads, and growth opens the 6 short of 40. February:
    # those 6 fill, and there's no gap left.
    result = run(plan=plan({"CC-01": [40.0]}))
    assert headcount(result, 0) == [30, 34, 40, 40, 40, 40, 40]
    assert headcount(result, 1) == FLAT  # no target, so no supply


def test_strength_closes_half_the_gap_at_a_gain_of_one_half() -> None:
    # Gaps of 6, 3, and 1 open ceil(3), ceil(1.5), and ceil(0.5): 3, 2, and 1.
    result = run(plan=plan({"CC-01": [40.0]}), requisitions_supply_gain=0.5)
    assert headcount(result, 0) == [30, 34, 37, 39, 40, 40, 40]


def test_the_soft_bridge_spreads_the_gap_to_the_next_date_evenly() -> None:
    # January's target is already met. In February, April's 46 is 12 off with 3 months to go:
    # 4 open. In March it's 8 off with 2 to go, and in April 4 off with none: 4 each time.
    target = plan({"CC-01": [34.0, 46.0]}, ["2026-01", "2026-04"])
    result = run(plan=target, requisitions_supply_rule="soft_bridge")
    assert headcount(result, 0) == [30, 34, 34, 38, 42, 46, 46]


def test_expected_fill_overshoots_when_openings_fill_better_than_it_assumes() -> None:
    # At an assumed yield of a half, 6 short asks for 12 openings, and all 12 fill.
    result = run(
        plan=plan({"CC-01": [40.0]}),
        requisitions_supply_rule="expected_fill",
        requisitions_fill_yield=0.5,
    )
    assert headcount(result, 0) == [30, 34, 46, 46, 46, 46, 46]


def test_the_cap_stops_supply_short_of_the_target() -> None:
    result = run(plan=plan({"CC-01": [40.0]}), capacity_caps={"CC-01": 37})
    assert headcount(result, 0) == [30, 34, 37, 37, 37, 37, 37]


def test_a_target_below_the_floor_opens_nothing() -> None:
    target = plan({"CC-01": [20.0]})
    result = run(plan=target)
    assert headcount(result, 0) == FLAT
    reach = attainability(
        initial_state(), StepContext(config(models=STILL, plan=target), 0, 0, SeedManager(1))
    )
    assert reach is not None
    assert reach.classes[0].value == "below_floor"


def test_the_credit_opens_ahead_of_known_departures() -> None:
    # Two CC-01 people are known to leave in March. Without the credit, supply sees the gap only
    # once they've gone: 40, then 38, then 40 again. With a two-month lookahead, January opens
    # for them too: 8, so February holds 42 and March lands on 40.
    events = KnownEvents(
        tuple(KnownEvent(EventKind.TERMINATION, 2, Source.DECLARED, e) for e in (0, 1))
    )
    known = resolve_plan(sanitize(events, range(90), []), SCHEMA)
    target = plan({"CC-01": [40.0]})
    late = run(plan=target, known=known)
    assert headcount(late, 0) == [30, 34, 40, 38, 40, 40, 40]
    early = run(plan=target, known=known, requisitions_credit_lookahead=2)
    assert headcount(early, 0) == [30, 34, 42, 40, 40, 40, 40]


def test_under_none_the_plan_drives_nothing() -> None:
    result = run(plan=plan({"CC-01": [40.0]}), alignment_method="none")
    assert headcount(result, 0) == FLAT


def aligned(target: AlignmentTarget) -> Any:
    covering = guidance({"requisitions_opened": {"CC-01": 2.0}})
    return dataclasses.replace(covering, aligned_to=target.digest())


def test_hybrid_lets_guidance_drive_the_cost_centers_it_covers() -> None:
    # Guidance opens CC-01's 2 a month, each filling the month after; the plan's supply closes
    # CC-02's gap at once.
    target = plan({"CC-01": [40.0], "CC-02": [40.0]})
    result = run(plan=target, guidance=aligned(target), alignment_method="hybrid")
    assert headcount(result, 0) == [30, 34, 36, 38, 40, 42, 44]
    assert headcount(result, 1) == [30, 34, 40, 40, 40, 40, 40]


def test_demand_leaves_supply_to_guidance_alone() -> None:
    # CC-02's target has no guidance to drive it, so nothing does.
    target = plan({"CC-01": [40.0], "CC-02": [40.0]})
    result = run(plan=target, guidance=aligned(target), alignment_method="demand")
    assert headcount(result, 0) == [30, 34, 36, 38, 40, 42, 44]
    assert headcount(result, 1) == FLAT
    assert result.config.alignment.undriven == ("CC-02",)


def test_rate_calibration_cant_move_a_certain_rate() -> None:
    # No one leaves and every opening fills: a logit shift leaves 0 and 1 where they are, so the
    # target is out of the rates' reach, the shifts run to their bound unconverged, and nothing
    # moves.
    result = run(plan=plan({"CC-01": [40.0]}), alignment_method="rate_calibration")
    assert headcount(result, 0) == FLAT
    assert result.calibration is not None
    assert not result.calibration.converged[0]
    assert result.calibration.converged[1:].all()  # no target, nothing to meet
    assert result.calibration.residual[0] == -6.0
