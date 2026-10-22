"""Tests for the known-event domains: conversion out, reorganization, and other movement."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from hcsim.engine.config import SimulationConfig
from hcsim.engine.domains import StepContext
from hcsim.engine.gates import StepGate
from hcsim.engine.known_domains import reorganization
from hcsim.engine.known_events import EventKind, KnownEvent, KnownEventError, KnownEvents, Source
from hcsim.engine.known_plan import sanitize
from hcsim.engine.known_resolution import ResolvedEvent, ResolvedPlan, resolve_plan
from hcsim.engine.loop import FLOW_SYMBOLS, HEADCOUNT_IDENTITY, SimulationResult, simulate
from hcsim.engine.seeds import SeedManager
from hcsim.model.core.layouts import EmployeeColumn
from hcsim.model.core.schema import (
    AWARD_TYPE,
    COMPANY,
    COST_CENTER,
    JOB_FAMILY,
    JOB_LEVEL,
    LOCATION,
    REQ_TYPE,
    TECH_INDICATOR,
    SimulationSchema,
)
from tests.engine.test_loop import RUN_DATE, initial_state, models

SCHEMA = SimulationSchema.build(
    {
        COST_CENTER: ["CC-01", "CC-02", "CC-03"],
        JOB_FAMILY: ["F"],
        JOB_LEVEL: [1, 2],
        TECH_INDICATOR: ["T"],
        LOCATION: ["L"],
        COMPANY: ["C"],
        REQ_TYPE: ["backfill", "campus", "growth"],
        AWARD_TYPE: ["new_hire"],
    }
)
ROLE = {
    "cost_center": "CC-01",
    "job_family": "F",
    "job_level": "2",
    "tech_indicator": "T",
    "location": "L",
    "company": "C",
}
EVENTS = (
    KnownEvent(EventKind.CONVERSION_OUT, 0, Source.DECLARED, employee_id=0),
    KnownEvent(
        EventKind.REORGANIZATION, 1, Source.DECLARED, 31, destination={"cost_center": "CC-03"}
    ),
    KnownEvent(EventKind.REORGANIZATION, 1, Source.DECLARED, 32, destination={"job_level": "2"}),
    KnownEvent(EventKind.REORGANIZATION, 2, Source.DECLARED, 60, destination={"boundary": "out"}),
    KnownEvent(EventKind.REORGANIZATION, 2, Source.DECLARED, 999, destination=ROLE),
)


def plan(*events: KnownEvent) -> ResolvedPlan:
    return resolve_plan(sanitize(KnownEvents(events), range(90), []), SCHEMA)


def config(**overrides: Any) -> SimulationConfig:
    args: dict[str, Any] = {
        "run_date": RUN_DATE,
        "schema": SCHEMA,
        "models": models(attrition=0.2),
        "simulation_months": 4,
        "num_trajectories": 3,
        "seed": 11,
        "forecast_scope": "hcf",
    } | overrides
    return SimulationConfig(**args)


def run(events: tuple[KnownEvent, ...] = EVENTS, **overrides: Any) -> SimulationResult:
    return simulate(config(**overrides), initial_state(), known=plan(*events), validate=True)


def test_known_events_fire_and_count_as_their_flows() -> None:
    result = run()
    for t in result.trajectories:
        assert t.known_fired == {"conversion_out": 1, "reorganization": 4}
        assert t.flows["V_conv"][0].tolist() == [1, 0, 0]
        assert t.flows["R_out"][1].tolist() == [0, 1, 0]  # 31 leaves CC-02 ...
        assert t.flows["R_in"][1].tolist() == [
            0,
            0,
            1,
        ]  # ... for CC-03; 32's new level moves no one
        assert t.flows["R_out"][2].tolist() == [0, 0, 1]  # 60 leaves the run from CC-03
        assert t.flows["R_in"][2].tolist() == [1, 0, 0]  # 999 joins it in CC-01
    h = result.headcount
    f = {s: result.flow(s) for s in FLOW_SYMBOLS}
    change = sum(sign * f[s] for s, sign in HEADCOUNT_IDENTITY.items())
    assert np.array_equal(h[:, 1:], h[:, :-1] + change)


def test_people_waiting_on_a_known_event_are_never_sampled_away() -> None:
    # A high hazard would take some of them before their month, failing the firing check.
    result = run(models=models(attrition=0.6))
    assert all(t.known_fired["reorganization"] == 4 for t in result.trajectories)


def test_a_move_writes_the_destination_and_a_boundary_in_keeps_its_id() -> None:
    state = initial_state()
    known = plan(*EVENTS)
    ctx = StepContext(config(), 1, 0, SeedManager(1), known=known)
    result = reorganization(state, ctx)
    assert result.employees.write_index.tolist() == [31, 32]
    written = result.employees.write_rows
    assert written[0, EmployeeColumn.COST_CENTER] == 2
    assert written[1, EmployeeColumn.LEVEL] == SCHEMA.vocabulary(JOB_LEVEL).code(2)
    assert not result.forfeit
    later = StepContext(config(), 2, 0, SeedManager(1), known=known)
    joined = reorganization(state, later)
    (row,) = joined.employees.append
    assert row[EmployeeColumn.EMPLOYEE_ID] == 999
    assert row[EmployeeColumn.HIRE_STEP] == 2
    assert joined.employees.drop.tolist() == [60]


def test_an_event_that_doesn_t_fire_fails_the_run() -> None:
    with pytest.raises(KnownEventError, match="conversion_out of 0 at step 0"):
        run(gates=(StepGate("conversion_out", months=(6,)),))


def test_other_movement_runs_only_in_replay() -> None:
    other = KnownEvent(
        EventKind.OTHER_MOVEMENT,
        0,
        Source.REPLAY,
        employee_id=5,
        destination={"cost_center": "CC-02"},
    )
    replayed = run((other,), run_mode="replay")
    assert all(t.known_fired == {"other_movement": 1} for t in replayed.trajectories)
    assert replayed.flow("W_out")[:, 0].tolist() == [[1, 0, 0]] * 3
    assert replayed.flow("W_in")[:, 0].tolist() == [[0, 1, 0]] * 3
    # Outside replay it never registers, so its events are neither fired nor required.
    unguided = run((other,))
    assert all(t.known_fired == {} for t in unguided.trajectories)
    assert "other_movement" in unguided.unregistered


def test_without_known_events_nothing_changes() -> None:
    plain = simulate(config(), initial_state(), validate=True)
    empty = simulate(config(), initial_state(), known=plan(), validate=True)
    assert np.array_equal(plain.headcount, empty.headcount)
    assert all(t.known_fired == {} for t in plain.trajectories)
    assert all(t.known_fired == {} for t in empty.trajectories)


def test_a_move_for_someone_absent_or_already_present_is_left_unfired() -> None:
    state = initial_state()
    present = ResolvedEvent(
        KnownEvent(EventKind.REORGANIZATION, 0, Source.DECLARED, 5), boundary="in"
    )
    absent = ResolvedEvent(
        KnownEvent(EventKind.REORGANIZATION, 0, Source.DECLARED, 4_000),
        {EmployeeColumn.COST_CENTER: 1},
    )
    known = ResolvedPlan(plan().plan, (present, absent))
    ctx = StepContext(config(), 0, 0, SeedManager(1), known=known)
    result = reorganization(state, ctx)
    assert result.employees.is_empty
    assert ctx.fired.unfired(known) == [present.event, absent.event]
