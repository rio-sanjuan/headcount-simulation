"""Tests for replay: realized history routed to the domains, with boundary fills, and
requisitions kept on history's track."""

from __future__ import annotations

import dataclasses
import datetime as dt
from collections import Counter
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import duckdb
import numpy as np
import pandas as pd
import pytest

from hcsim.engine.apply import apply
from hcsim.engine.config import ForecastScope, RunMode, SimulationConfig
from hcsim.engine.domains import StepContext
from hcsim.engine.known_events import EventKind, Source, collect, from_replay, step_of
from hcsim.engine.known_plan import sanitize
from hcsim.engine.known_resolution import jobs_from_world, resolve_plan
from hcsim.engine.loop import FLOW_SYMBOLS, HEADCOUNT_IDENTITY, OPENING_IDENTITY, simulate
from hcsim.engine.replay import (
    RESERVED_OPENINGS,
    ReplayedRequisitions,
    recorded_awards,
    replay_requisitions,
    replayed_requisitions,
)
from hcsim.engine.results import DomainResult, Grant, RowChange
from hcsim.engine.seeding import schema_from_world, seed_state
from hcsim.engine.seeds import SeedManager
from hcsim.financial.prices import StaticPrice
from hcsim.financial.routing import RecordedAwards
from hcsim.financial.tracker import HolderLedger
from hcsim.model.core.layouts import EMPLOYEE, OPENING, EmployeeColumn, OpeningColumn
from hcsim.model.core.schema import REQ_TYPE, SimulationSchema
from hcsim.studies.inputs import fit_models
from hcsim.synth.config import preset
from hcsim.synth.world import generate_world, write_world
from tests.engine.test_loop import SCHEMA, awards, initial_state, models
from tests.engine.test_loop import config as loop_config

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


def test_hires_whose_offers_predate_the_run_carry_no_requisition(
    con: duckdb.DuckDBPyConnection,
) -> None:
    waiting = con.execute(
        "SELECT employee_id, start_date FROM pending_start WHERE snapshot_date = ?",
        [RUN - dt.timedelta(days=1)],
    ).df()
    due = {
        (int(e), step_of(RUN, d))
        for e, d in zip(waiting["employee_id"], waiting["start_date"], strict=True)
    }
    starts = [
        e for e in from_replay(con, RUN) if e.reason == "HIRE" and (e.employee_id, e.step) in due
    ]
    assert starts
    assert all(e.req_id is None for e in starts)


def test_realized_requisitions_carry_their_openings_by_month(
    con: duckdb.DuckDBPyConnection,
) -> None:
    replay = replayed_requisitions(con, schema_from_world(con), RUN, MONTHS)
    assert set(replay.held) == set(replay.reserved) == set(range(MONTHS))
    assert (np.diff(replay.ids) > 0).all()
    assert (replay.rows[:, OpeningColumn.NUM_OPENINGS] == 0).all()
    for step, (place, openings) in replay.held.items():
        assert ((openings >= 0) & (openings <= 10)).all()
        cost_centers = replay.rows[place, OpeningColumn.COST_CENTER].astype(int)
        shows = np.bincount(cost_centers, weights=openings, minlength=len(replay.reserved[step]))
        assert (replay.reserved[step] <= shows).all()
    assert sum(int(r.sum()) for r in replay.reserved.values()) > 0
    assert replay.pruned == {}


def test_a_replay_holds_history_s_openings_and_its_reservations(
    con: duckdb.DuckDBPyConnection,
) -> None:
    result, _, _ = replay_run(con)
    schema = schema_from_world(con)
    truth = np.stack(
        [seed_state(con, schema, first_of(RUN, k))[0].open_openings for k in range(MONTHS + 1)]
    )
    for t in result.trajectories:
        reserved = t.diagnostics[RESERVED_OPENINGS]
        assert int(reserved.sum()) > 0
        assert np.array_equal(t.open_openings[0], truth[0])
        assert np.array_equal(t.open_openings[1:] - reserved, truth[1:])


def test_the_domain_moves_each_requisition_to_its_month_end_openings() -> None:
    state = initial_state()  # requisitions 500-505, with 1 or 3 openings
    rows = OPENING.empty(4)
    rows[:, OpeningColumn.REQ_ID] = [500, 501, 502, 900]
    rows[:, OpeningColumn.COST_CENTER] = [0, 0, 1, 2]
    rows[:, OpeningColumn.REQ_TYPE] = SCHEMA.vocabulary(REQ_TYPE).code("growth")
    # 500 is no longer held, 501 grows from 3 to 5, 502 stays at 1, and 900 opens with 2.
    held = {0: (np.array([1, 2, 3]), np.array([5, 1, 2]))}
    replay = ReplayedRequisitions(rows, held, {0: np.array([0, 2, 0])})
    config = dataclasses.replace(config_for(SCHEMA), run_mode=RunMode.REPLAY)
    result = replay_requisitions(state, StepContext(config, 0, 0, SeedManager(1), replay=replay))
    # Openings count by the type of the row they land on: 501 is a backfill, 900 growth.
    assert result.flows["B_att"].tolist() == [2, 0, 0]
    assert result.flows["G"].tolist() == [0, 0, 2]
    assert result.flows["U"].tolist() == [1, 0, 0]
    assert result.diagnostics[RESERVED_OPENINGS].tolist() == [0, 2, 0]
    apply(state, result, 0, SCHEMA)
    live = state.live_openings
    openings = dict(
        zip(
            live[:, OpeningColumn.REQ_ID].astype(int).tolist(),
            live[:, OpeningColumn.NUM_OPENINGS].astype(int).tolist(),
            strict=True,
        )
    )
    assert openings == {501: 5, 502: 1, 503: 3, 504: 1, 505: 3, 900: 2}
    nothing = StepContext(config, 0, 0, SeedManager(1), replay=ReplayedRequisitions())
    assert replay_requisitions(state, nothing).openings.is_empty


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


#: A recorded new-hire award for employee 900: 7 shares a month for a year.
RECORDED = RecordedAwards({("new_hire", 0): {900: np.array([7] * 12 + [0] * 60)}})


def test_replayed_awards_replace_sizing_for_the_grants_they_cover(tmp_path: Path) -> None:
    settings = dataclasses.replace(awards(), replayed=RECORDED)
    state = initial_state()
    state.ledger, state.price = HolderLedger(), StaticPrice(100.0)
    hires = EMPLOYEE.empty(2)
    hires[:, EmployeeColumn.EMPLOYEE_ID] = [900, 901]  # 900 holds a recorded award, 901 doesn't
    result = DomainResult(
        "fills", employees=RowChange(EMPLOYEE, append=hires), grants=(Grant("new_hire"),)
    )
    applied = apply(state, result, 0, SCHEMA, settings)
    assert (applied.replayed_grants, applied.unsized_grants) == (1, 0)
    # Both bases are imputed by the sizer; only 901's grant is sized.
    assert (
        state.employees[state.n_employees - 2 : state.n_employees, EmployeeColumn.BASE_SALARY] > 0
    ).all()
    tranches = state.ledger.tranches()
    holder_900 = (
        tranches.holder == state.employees[state.n_employees - 2, EmployeeColumn.VEST_INDEX]
    )
    assert int(tranches.shares[holder_900].sum()) == 12 * 7
    assert int(tranches.shares[~holder_900].sum()) > 12 * 7
    # After the hybrid window's last step, the recorded award no longer routes.
    later = dataclasses.replace(settings, replay_through=-1)
    state = initial_state()
    state.ledger, state.price = HolderLedger(), StaticPrice(100.0)
    assert apply(state, result, 0, SCHEMA, later).replayed_grants == 0


def test_a_replay_grants_nothing_where_history_s_complete_record_holds_no_award() -> None:
    settings = dataclasses.replace(awards(), replayed=RECORDED)
    replay = loop_config(run_mode=RunMode.REPLAY, awards=settings)
    window = replay.award_window
    assert window is not None
    assert window.complete_through == replay.simulation_months - 1
    # A forecast's grants aren't history's, and a replay with no record of awards has nothing
    # to say a grant was granted none, so each sizes every grant that finds no award.
    for other in (loop_config(awards=settings), loop_config(run_mode=RunMode.REPLAY)):
        assert other.award_window is not None
        assert other.award_window.complete_through is None
    state = initial_state()
    state.ledger, state.price = HolderLedger(), StaticPrice(100.0)
    hires = EMPLOYEE.empty(2)
    hires[:, EmployeeColumn.EMPLOYEE_ID] = [900, 901]  # 900 holds a recorded award, 901 doesn't
    result = DomainResult(
        "fills", employees=RowChange(EMPLOYEE, append=hires), grants=(Grant("new_hire"),)
    )
    applied = apply(state, result, 0, SCHEMA, settings, window=window)
    assert (applied.replayed_grants, applied.ungranted_grants, applied.unsized_grants) == (1, 1, 0)
    # The ledger holds 900's award alone, and both bases are imputed.
    joined = state.employees[state.n_employees - 2 : state.n_employees]
    assert (joined[:, EmployeeColumn.BASE_SALARY] > 0).all()
    holders = state.ledger.tranches().holder
    assert set(holders.tolist()) == {int(joined[0, EmployeeColumn.VEST_INDEX])}


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
        0,
        0,
        SeedManager(1),
        replay=replay,
    )
    result = replay_requisitions(state, ctx)
    assert int(result.flows["G"].sum()) > 0
    assert int(result.flows["G_camp"].sum()) == 0


def first_of(run: dt.date, months: int) -> dt.date:
    """The first of the month `months` after `run`'s."""
    return (pd.Period(run, freq="M") + months).start_time.date()


def config_for(schema: SimulationSchema) -> SimulationConfig:
    return SimulationConfig(
        run_date=RUN, schema=schema, models=models(), forecast_scope=ForecastScope.HEADCOUNT
    )


def test_the_recorded_awards_are_the_windows_primary_plan_awards(
    con: duckdb.DuckDBPyConnection,
) -> None:
    recorded = recorded_awards(con, RUN, MONTHS)
    expected = con.execute(
        "SELECT a.award_type, count(DISTINCT a.employee_id) AS holders, sum(v.shares) AS shares "
        "FROM equity_award a JOIN equity_vest v USING (award_id) "
        "WHERE a.plan = 'primary' AND a.award_date >= ? AND a.award_date < ? GROUP BY ALL",
        [RUN, dt.date(2025, 7, 1)],
    ).df()
    assert len(recorded) > 0
    by_type: dict[str, int] = {}
    for (award_type, step), by_id in recorded.schedules.items():
        assert 0 <= step < MONTHS
        by_type[award_type] = by_type.get(award_type, 0) + sum(int(s.sum()) for s in by_id.values())
    assert by_type == dict(zip(expected["award_type"], expected["shares"].astype(int), strict=True))
