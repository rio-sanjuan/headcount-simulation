"""Tests for attrition: known terminations first, the quota remainder, executives, and the
anticipated flag."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import duckdb
import numpy as np
import pytest

from hcsim.engine.domains import StepContext, attrition
from hcsim.engine.gates import StepGate
from hcsim.engine.known_events import EventKind, KnownEvent, KnownEventError, KnownEvents, Source
from hcsim.engine.known_plan import sanitize
from hcsim.engine.known_resolution import ResolvedPlan, resolve_plan
from hcsim.engine.loop import simulate
from hcsim.engine.results import DomainResult
from hcsim.engine.seeding import executive_min_level
from hcsim.engine.seeds import SeedManager
from hcsim.model.core.layouts import EmployeeColumn
from hcsim.synth.config import preset
from hcsim.synth.world import generate_world, write_world
from tests.engine.test_loop import SCHEMA, config, guidance, initial_state, models


def known(*people: int, step: int = 0) -> ResolvedPlan:
    events = [
        KnownEvent(EventKind.TERMINATION, step, Source.PROGRAM, employee_id=p) for p in people
    ]
    return resolve_plan(sanitize(KnownEvents(tuple(events)), range(90), []), SCHEMA)


def step(**overrides: Any) -> StepContext:
    plan = overrides.pop("known", None)
    return StepContext(config(**overrides), 0, 0, SeedManager(3), known=plan)


def test_known_terminations_leave_first_and_are_anticipated() -> None:
    ctx = step(models=models(attrition=0.0), known=known(1, 2, 40))
    result = attrition(initial_state(), ctx)
    assert result.employees.drop.tolist() == [1, 2, 40]
    assert result.anticipated.tolist() == [True, True, True]
    assert len(ctx.fired.fired) == 3


def test_sampled_departures_join_them_unflagged() -> None:
    ctx = step(models=models(attrition=0.5), known=known(1, 2))
    result = attrition(initial_state(), ctx)
    leavers = result.employees.drop.tolist()
    flagged = dict(zip(leavers, result.anticipated.tolist(), strict=True))
    assert flagged[1]
    assert flagged[2]
    assert sum(flagged.values()) == 2
    assert len(leavers) > 2


def test_known_terminations_count_against_the_quota() -> None:
    # CC-01 holds employees 0-29. Two of them are known to leave; the quota is three.
    ctx = step(
        models=models(attrition=0.5),
        guidance=guidance({"attrition": {"CC-01": 3.0}}),
        known=known(1, 2),
    )
    result = attrition(initial_state(), ctx)
    in_cc01 = [r for r in result.employees.drop.tolist() if r < 30]
    assert len(in_cc01) == 3
    assert {1, 2} <= set(in_cc01)
    record = result.quotas["attrition"]
    assert record.quota[0] == 3
    assert record.realized[0] == 3


def test_known_terminations_beyond_the_quota_still_leave() -> None:
    ctx = step(
        models=models(attrition=0.5),
        guidance=guidance({"attrition": {"CC-01": 1.0}}),
        known=known(1, 2, 3),
    )
    result = attrition(initial_state(), ctx)
    assert [r for r in result.employees.drop.tolist() if r < 30] == [1, 2, 3]
    assert result.quotas["attrition"].realized[0] == 3


def test_executives_leave_only_by_known_events() -> None:
    state = initial_state()  # levels alternate 1 and 2
    levels = state.live_employees[:, EmployeeColumn.LEVEL]
    ctx = step(models=models(attrition=1.0), attrition_executive_min_level=2, known=known(1))
    result = attrition(state, ctx)
    dropped = result.employees.drop
    executive = levels[dropped] == SCHEMA.vocabulary("job_level").code(2)
    # Every level-1 employee leaves, and the only executive is the one known to.
    assert dropped[executive].tolist() == [1]
    assert int((levels == 0).sum()) == int((~executive).sum())


def test_a_known_termination_that_doesn_t_fire_fails_the_run() -> None:
    with pytest.raises(KnownEventError, match="termination of 5 at step 1"):
        simulate(
            config(gates=(StepGate("attrition", months=(1,)),)),
            initial_state(),
            known=known(5, step=1),
        )
    run = simulate(config(models=models(attrition=0.0)), initial_state(), known=known(5, step=1))
    assert all(t.known_fired == {"termination": 1} for t in run.trajectories)
    assert all(int(t.flows["A"][1].sum()) == 1 for t in run.trajectories)


def test_the_executive_level_comes_from_the_world(tmp_path: Path) -> None:
    path = tmp_path / "w.duckdb"
    write_world(generate_world(preset("tiny", seed=2)), path)
    with duckdb.connect(str(path), read_only=True) as con:
        level = executive_min_level(con)
        (expected,) = con.execute(
            "SELECT min(job_level) FROM dim_job WHERE is_executive"
        ).fetchone() or (None,)
    assert level == (None if expected is None else int(expected))
    with duckdb.connect() as bare:
        bare.execute("CREATE TABLE dim_job (job_level INT, is_executive BOOLEAN)")
        assert executive_min_level(bare) is None


def test_a_known_event_for_someone_absent_stays_unfired() -> None:
    ctx = step(known=known(1))
    taken = ctx.fire_known(initial_state().live_employees[10:], EventKind.TERMINATION)
    assert not taken.any()
    assert ctx.fired.fired == set()


def test_each_departure_needs_its_own_anticipated_flag() -> None:
    rows = initial_state().live_employees[:2]
    with pytest.raises(ValueError, match="one anticipated flag per departure"):
        DomainResult("attrition", departures=rows, anticipated=np.array([True]))
    assert DomainResult("attrition", departures=rows).anticipated.tolist() == [False, False]
