"""Tests for replay: realized history routed to the domains, with boundary fills."""

from __future__ import annotations

import dataclasses
import datetime as dt
from collections import Counter
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import duckdb
import numpy as np
import pytest

from hcsim.engine.apply import apply
from hcsim.engine.config import ForecastScope, RunMode, SimulationConfig
from hcsim.engine.domains import StepContext
from hcsim.engine.known_events import EventKind, Source, collect, from_replay
from hcsim.engine.known_plan import sanitize
from hcsim.engine.known_resolution import jobs_from_world, resolve_plan
from hcsim.engine.loop import FLOW_SYMBOLS, HEADCOUNT_IDENTITY, OPENING_IDENTITY, simulate
from hcsim.engine.replay import replay_requisitions, replayed_requisitions
from hcsim.engine.results import DomainResult, Grant, RowChange
from hcsim.engine.seeding import schema_from_world, seed_state
from hcsim.engine.seeds import SeedManager
from hcsim.financial.prices import StaticPrice
from hcsim.financial.sizing import AwardBatch
from hcsim.financial.tracker import HolderLedger
from hcsim.model.core.layouts import EMPLOYEE, EmployeeColumn, OpeningColumn
from hcsim.model.core.schema import SimulationSchema
from hcsim.studies.inputs import fit_models
from hcsim.synth.config import preset
from hcsim.synth.world import generate_world, write_world
from tests.engine.test_loop import SCHEMA, awards, initial_state, models

RUN = dt.date(2025, 1, 1)
MONTHS = 6


@pytest.fixture(scope="module")
def con(tmp_path_factory: pytest.TempPathFactory) -> Iterator[duckdb.DuckDBPyConnection]:
    path = tmp_path_factory.mktemp("world") / "w.duckdb"
    write_world(generate_world(preset("small", seed=6)), path)
    with duckdb.connect(str(path), read_only=True) as connection:
        yield connection


def replay_run(con: duckdb.DuckDBPyConnection, **overrides: Any) -> Any:
    schema = schema_from_world(con)
    state, _ = seed_state(con, schema, RUN)
    events = collect(RUN, MONTHS, [(Source.REPLAY, from_replay(con, RUN))])
    employees = state.live_employees[:, EmployeeColumn.EMPLOYEE_ID].astype(int).tolist()
    reqs = state.live_openings[:, OpeningColumn.REQ_ID].astype(int).tolist()
    plan = resolve_plan(sanitize(events, employees, reqs), schema, jobs_from_world(con))
    config = SimulationConfig(
        run_date=RUN,
        schema=schema,
        models=fit_models(con, schema, RUN).models,
        simulation_months=MONTHS,
        num_trajectories=2,
        seed=3,
        forecast_scope=ForecastScope.HEADCOUNT,
        run_mode=RunMode.REPLAY,
        **overrides,
    )
    replay = replayed_requisitions(con, schema, RUN, MONTHS)
    return simulate(config, state, known=plan, replay=replay, validate=True), plan, replay


def test_realized_movements_become_replay_events(con: duckdb.DuckDBPyConnection) -> None:
    events = from_replay(con, RUN)
    kinds = Counter(e.kind for e in events)
    assert {EventKind.TERMINATION, EventKind.PROMOTION, EventKind.PENDING_START} <= set(kinds)
    assert all(e.source is Source.REPLAY for e in events)
    hires = [e for e in events if e.kind is EventKind.PENDING_START]
    assert sum(e.req_id is not None for e in hires) > len(hires) / 2
    transfers = [e for e in events if e.kind is EventKind.TRANSFER]
    assert all(e.req_id is not None for e in transfers)
    assert all("cost_center" in e.destination for e in transfers + hires)


def test_realized_requisitions_open_and_close_by_month(con: duckdb.DuckDBPyConnection) -> None:
    replay = replayed_requisitions(con, schema_from_world(con), RUN, MONTHS)
    assert set(replay.opened) <= set(range(MONTHS))
    opened = np.concatenate(list(replay.opened.values()))
    assert (opened[:, OpeningColumn.REQ_ID] > 0).all()
    assert (
        (opened[:, OpeningColumn.NUM_OPENINGS] >= 1) & (opened[:, OpeningColumn.NUM_OPENINGS] <= 10)
    ).all()
    assert sum(len(v) for v in replay.closed.values()) > 0
    assert replay.pruned == {}


def test_a_replay_fires_its_history_and_keeps_the_identities(
    con: duckdb.DuckDBPyConnection,
) -> None:
    result, plan, _ = replay_run(con)
    terminations = sum(1 for e in plan.events if e.event.kind is EventKind.TERMINATION)
    for t in result.trajectories:
        assert int(t.flows["A"].sum()) == terminations  # every departure is history's
        assert t.known_fired["termination"] == terminations
    # Nothing is sampled: both trajectories replay the same history.
    assert np.array_equal(result.headcount[0], result.headcount[1])
    f = {s: result.flow(s) for s in FLOW_SYMBOLS}
    assert int(f["G"].sum() + f["B_att"].sum()) > 0
    assert int(f["F_pend"].sum()) > 0
    h, o = result.headcount, result.open_openings
    heads = sum(v * f[s] for s, v in HEADCOUNT_IDENTITY.items())
    openings = sum(v * f[s] for s, v in OPENING_IDENTITY.items())
    assert np.array_equal(h[:, 1:], h[:, :-1] + heads)
    assert np.array_equal(o[:, 1:], o[:, :-1] + openings)
    assert "replay_requisitions" not in result.unregistered


def test_boundary_fills_follow_their_setting(con: duckdb.DuckDBPyConnection) -> None:
    with_boundary, _, _ = replay_run(con)
    without, _, _ = replay_run(con, replay_boundary_fill=False)
    assert int(with_boundary.flow("F_bdy").sum()) > 0
    assert int(without.flow("F_bdy").sum()) == 0


class Recorded:
    """Replayed awards: 7 shares a month for a year, for new hires only."""

    def batch(self, rows: Any, step: int, award_type: str, prices: Any) -> AwardBatch | None:
        if award_type != "new_hire":
            return None
        schedule = np.zeros((len(rows), 72), dtype=np.int64)
        schedule[:, :12] = 7
        return AwardBatch(
            award_type, np.full(len(rows), 1.0), np.full((len(rows), 6), 14), schedule
        )


def test_replayed_awards_replace_sizing_for_the_grants_they_cover(tmp_path: Path) -> None:
    settings = dataclasses.replace(awards(), replayed=Recorded())
    state = initial_state()
    state.ledger, state.price = HolderLedger(), StaticPrice(100.0)
    hires = EMPLOYEE.empty(2)
    hires[:, EmployeeColumn.EMPLOYEE_ID] = -1
    result = DomainResult(
        "fills", employees=RowChange(EMPLOYEE, append=hires), grants=(Grant("new_hire"),)
    )
    applied = apply(state, result, 0, SCHEMA, settings)
    assert applied.recorded_tranches > 0
    assert (
        state.employees[state.n_employees - 2 : state.n_employees, EmployeeColumn.BASE_SALARY]
        == 1.0
    ).all()
    assert int(state.ledger.tranches().shares.sum()) == 2 * 12 * 7


def test_requisitions_the_schema_can_t_hold_are_pruned(con: duckdb.DuckDBPyConnection) -> None:
    full = schema_from_world(con)
    labels = {n: list(v.labels) for n, v in full.vocabularies.items() if not n.startswith("org_")}
    labels["cost_center"] = labels["cost_center"][1:]
    labels["req_type"] = ["backfill", "growth"]
    narrow = SimulationSchema.build(labels)
    replay = replayed_requisitions(con, narrow, RUN, MONTHS)
    assert set(replay.pruned) <= {"unknown_cost_center", "unknown_req_type"}
    assert replay.pruned["unknown_req_type"] > 0
    state = initial_state()
    ctx = StepContext(
        dataclasses.replace(config_for(narrow), run_mode=RunMode.REPLAY),
        min(replay.opened),
        0,
        SeedManager(1),
        replay=replay,
    )
    result = replay_requisitions(state, ctx)
    assert int(result.flows["G_camp"].sum()) == 0


def config_for(schema: SimulationSchema) -> SimulationConfig:
    return SimulationConfig(
        run_date=RUN, schema=schema, models=models(), forecast_scope=ForecastScope.HEADCOUNT
    )
