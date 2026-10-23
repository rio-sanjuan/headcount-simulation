"""Tests for the plan's reach: the known-event credit, the expected path, the natural floor, and
the attainability classes."""

from __future__ import annotations

import dataclasses

import numpy as np
import pandas as pd
import pytest

from hcsim.engine.attainability import (
    Attainability,
    Hazards,
    expected_path,
    known_credit,
    natural_floor,
)
from hcsim.engine.attainability import attainability as reach
from hcsim.engine.config import ConfigError
from hcsim.engine.domains import StepContext, growth
from hcsim.engine.known_events import EventKind, KnownEvent, KnownEvents, Source
from hcsim.engine.known_plan import KnownEventPlan
from hcsim.engine.known_resolution import ResolvedEvent, ResolvedPlan
from hcsim.engine.seeds import SeedManager
from hcsim.model.core.alignment import AlignmentTarget, Interpolation, marginal
from hcsim.model.core.layouts import EmployeeColumn
from tests.engine.test_growth import opened
from tests.engine.test_loop import config, initial_state, models
from tests.engine.test_supply_rules import dated

CC = EmployeeColumn.COST_CENTER
K = EventKind


def event(kind: EventKind, step: int, employee: int | None, **resolved: object) -> ResolvedEvent:
    return ResolvedEvent(KnownEvent(kind, step, Source.DECLARED, employee), **resolved)  # type: ignore[arg-type]


def known(*events: ResolvedEvent) -> ResolvedPlan:
    return ResolvedPlan(KnownEventPlan(KnownEvents(())), events)


#: Employees 0-29 sit in CC-01, 30-59 in CC-02, and 60-89 in CC-03.
EVENTS = known(
    event(K.TERMINATION, 2, 0),
    event(K.REORGANIZATION, 1, 30, codes={CC: 2}),  # CC-02 to CC-03
    event(K.TRANSFER, 2, 61, codes={CC: 0}),  # fills an opening in CC-01 that S already holds
    event(K.PENDING_START, 1, None),  # its reserved opening is in S
    event(K.PROMOTION, 1, 62, codes={CC: 2}),  # stays in CC-03
    event(K.REORGANIZATION, 3, None, codes={CC: 1}, boundary="in"),
    event(K.REORGANIZATION, 3, 63, boundary="out"),
    event(K.CONVERSION_OUT, 5, 1),  # past the lookahead
)


def test_the_credit_is_the_known_net_change_committed_strength_doesnt_hold() -> None:
    state = initial_state()
    assert known_credit(state, EVENTS, 0, 3).tolist() == [-1.0, 0.0, -1.0]
    assert known_credit(state, EVENTS, 2, 1).tolist() == [0.0, 1.0, -1.0]
    assert known_credit(state, EVENTS, 0, 0).tolist() == [0.0, 0.0, 0.0]
    assert known_credit(state, None, 0, 3).tolist() == [0.0, 0.0, 0.0]


def test_the_expected_path_steps_the_expectation_a_month_at_a_time() -> None:
    hazards = Hazards(np.array([0.1]), np.array([0.5]), np.array([0.2]))
    h, o = expected_path(np.array([100.0]), np.array([10.0]), hazards, 2)
    assert h[:, 0].tolist() == pytest.approx([100.0, 95.0, 87.5])  # 90 + 5, then 85.5 + 2
    assert o[:, 0].tolist() == pytest.approx([10.0, 4.0, 1.6])  # 0.8 (10 - 5), then 0.8 (4 - 2)
    h, o = expected_path(
        np.array([100.0]),
        np.array([10.0]),
        hazards,
        2,
        new=np.array([[3.0], [0.0]]),
        inflow=np.array([[0.0], [2.0]]),
    )
    assert h[:, 0].tolist() == pytest.approx([100.0, 95.0, 91.0])  # 85.5 + 3.5 + 2
    assert o[:, 0].tolist() == pytest.approx([10.0, 7.0, 2.8])


def test_hazards_are_the_run_models_rates() -> None:
    state = initial_state()
    ctx = StepContext(
        config(models=models(attrition=0.05, fill=0.3, cancel=0.1)), 0, 0, SeedManager(1)
    )
    hazards = Hazards.of(state, ctx)
    assert hazards.attrition.tolist() == pytest.approx([0.05] * 3)
    assert hazards.fill.tolist() == pytest.approx([0.3] * 3)
    assert hazards.unfilled.tolist() == pytest.approx([0.1] * 3)


def test_the_natural_floor_is_the_path_with_no_new_supply() -> None:
    state = initial_state()
    hazards = Hazards(np.full(3, 0.1), np.full(3, 0.5), np.full(3, 0.0))
    floor = natural_floor(state, hazards, EVENTS, 2)
    # 30 heads and 4 openings: 27 + 2 after a month, then 26.1 + 1 and the reorganization out of
    # CC-02 and into CC-03 in step 1.
    assert floor[1].tolist() == pytest.approx([29.0, 29.0, 29.0])
    assert floor[2].tolist() == pytest.approx([27.1, 26.1, 28.1])


def june(values: dict[str, float]) -> AlignmentTarget:
    """Targets for June, in a plan that runs January to June and is scored at June's end."""
    m = marginal("cost_center", list(values), ["2026-06"], [[v] for v in values.values()])
    return AlignmentTarget((m,), pd.Period("2026-01", "M"), 6, 6, Interpolation.HOLD_LAST)


def classes(**overrides: object) -> list[str]:
    run = config(**overrides)
    found = reach(initial_state(), StepContext(run, 0, 0, SeedManager(1)))
    assert found is not None
    return [c.value for c in found.classes]


def test_each_target_falls_in_one_attainability_class() -> None:
    # The floor at June's end: 30 heads and 4 openings at a 5% attrition and a 30% fill hazard
    # come to about 27.5.
    plan = june({"CC-01": 20.0, "CC-02": 40.0, "CC-03": 60.0})
    assert classes(plan=plan, capacity_caps={"CC-03": 50}) == [
        "below_floor",
        "attainable",
        "above_cap",
    ]
    short = dated({"CC-01": [40.0]}, ["2026-01"], months=1)  # t* is the first month
    assert classes(plan=short) == ["out_of_time", "no_target", "no_target"]
    stuck = models(fill=0.0)
    assert classes(plan=plan, models=stuck)[1] == "out_of_time"
    assert reach(initial_state(), StepContext(config(), 0, 0, SeedManager(1))) is None
    found = reach(initial_state(), StepContext(config(plan=plan), 0, 0, SeedManager(1)))
    assert found is not None
    assert found.step == 5
    assert found.counts() == {c.value: 0 for c in Attainability} | {
        "below_floor": 1,
        "attainable": 2,
    }


def test_the_supply_rule_opens_for_known_departures_in_the_lookahead() -> None:
    plan = dated({"CC-02": [40.0]}, ["2026-01"])
    leaving = known(event(K.TERMINATION, 2, 30), event(K.TERMINATION, 3, 31))
    run = config(plan=plan, requisitions_credit_lookahead=3)
    with_credit = StepContext(run, 0, 0, SeedManager(8), known=leaving)
    assert opened(growth(initial_state(), with_credit))[1] == 8  # 40 less 34, and 2 to leave
    plain = StepContext(
        dataclasses.replace(run, requisitions_credit_lookahead=0),
        0,
        0,
        SeedManager(8),
        known=leaving,
    )
    assert opened(growth(initial_state(), plain))[1] == 6
    with pytest.raises(ConfigError, match="requisitions_credit_lookahead must be non-negative"):
        config(requisitions_credit_lookahead=-1)


def test_a_plan_scored_past_the_runs_end_has_no_reach_and_no_plan_no_band() -> None:
    from hcsim.engine.ceiling import Ceiling
    from tests.engine.test_loop import RUN_DATE, SCHEMA

    late = dated({"CC-01": [40.0]}, ["2026-06"])  # its window runs June to November
    assert reach(initial_state(), StepContext(config(plan=late), 0, 0, SeedManager(1))) is None
    assert Ceiling.build(SCHEMA).bands(RUN_DATE, 0).tolist() == [0.0, 0.0, 0.0]
