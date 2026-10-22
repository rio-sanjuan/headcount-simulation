"""Tests for pending starts: matching, reservations, boundary fills, and quota deduction."""

from __future__ import annotations

import datetime as dt
from pathlib import Path
from typing import Any

import duckdb
import numpy as np

from hcsim.engine.config import DomainModels, ForecastScope, SimulationConfig
from hcsim.engine.domains import StepContext
from hcsim.engine.fills import fills
from hcsim.engine.known_events import (
    EventKind,
    KnownEvent,
    KnownEvents,
    Source,
    from_pending_starts,
    known_events,
)
from hcsim.engine.known_plan import sanitize
from hcsim.engine.known_resolution import ResolvedPlan, jobs_from_world, resolve_plan
from hcsim.engine.loop import FLOW_SYMBOLS, HEADCOUNT_IDENTITY, OPENING_IDENTITY, simulate
from hcsim.engine.pending import pending_starts
from hcsim.engine.seeding import schema_from_world, seed_state
from hcsim.engine.seeds import SeedManager
from hcsim.model.core.layouts import EmployeeColumn, OpeningColumn
from hcsim.model.core.schema import COST_CENTER
from hcsim.model.ml.entity import ConstantModel
from hcsim.model.ml.generative import EmpiricalAttributeGenerator
from hcsim.model.ml.group import BackfillRateModel, PoissonGrowthModel
from hcsim.synth.config import preset
from hcsim.synth.world import generate_world, write_world
from tests.engine.test_loop import SCHEMA, config, guidance, initial_state, models

ROLE = {"cost_center": "CC-02", "job_family": "F", "job_level": "1", "location": "L"}


def start(person: int, req: int, step: int = 1, role: dict[str, str] | None = None) -> KnownEvent:
    return KnownEvent(
        EventKind.PENDING_START, step, Source.PENDING_START, person, req, role or ROLE
    )


def plan(*events: KnownEvent) -> ResolvedPlan:
    reqs = initial_state().live_openings[:, OpeningColumn.REQ_ID].astype(int).tolist()
    return resolve_plan(sanitize(KnownEvents(events), range(90), reqs), SCHEMA)


def step_context(step: int, known: ResolvedPlan, **overrides: Any) -> StepContext:
    return StepContext(config(**overrides), step, 0, SeedManager(2), known=known)


def test_a_matched_start_fills_its_requisition_under_its_own_id() -> None:
    state = initial_state()  # requisition 501 is in CC-01 at level code 1, with 3 openings
    known = plan(start(900, 501))
    ctx = step_context(1, known)
    result = pending_starts(state, ctx)
    (hire,) = result.employees.append
    assert hire[EmployeeColumn.EMPLOYEE_ID] == 900
    assert hire[EmployeeColumn.COST_CENTER] == 0
    assert hire[EmployeeColumn.LEVEL] == 1
    assert hire[EmployeeColumn.HIRE_STEP] == 1
    assert result.openings.write_index.tolist() == [1]
    assert result.openings.write_rows[0, OpeningColumn.NUM_OPENINGS] == 2
    assert result.flows["F_pend"].tolist() == [1, 0, 0]
    assert result.flows["F_bdy"].tolist() == [0, 0, 0]
    assert ctx.fired.unfired(known) == []
    # A one-opening requisition closes.
    closing = pending_starts(state, step_context(1, plan(start(901, 500))))
    assert closing.openings.drop.tolist() == [0]


def test_an_unmatched_start_joins_in_its_offer_s_role_as_a_boundary_fill() -> None:
    result = pending_starts(initial_state(), step_context(1, plan(start(902, 9_999))))
    (hire,) = result.employees.append
    assert hire[EmployeeColumn.COST_CENTER] == 1
    assert result.openings.is_empty
    assert result.flows["F_bdy"].tolist() == [0, 1, 0]


def test_a_reserved_opening_survives_fills_and_cancellation_until_its_start() -> None:
    known = plan(start(900, 501, step=2))
    result = simulate(
        config(models=models(attrition=0.0, fill=1.0, cancel=1.0), simulation_months=3),
        initial_state(),
        known=known,
        validate=True,
    )
    for t in result.trajectories:
        assert t.flows["F_pend"][2].tolist() == [1, 0, 0]
        assert int(t.flows["F_bdy"].sum()) == 0
        assert t.known_fired == {"pending_start": 1}
    h, o = result.headcount, result.open_openings
    f = {s: result.flow(s) for s in FLOW_SYMBOLS}
    assert np.array_equal(
        h[:, 1:], h[:, :-1] + sum(v * f[s] for s, v in HEADCOUNT_IDENTITY.items())
    )
    assert np.array_equal(o[:, 1:], o[:, :-1] + sum(v * f[s] for s, v in OPENING_IDENTITY.items()))


def test_starts_count_against_the_external_hires_quota() -> None:
    state = initial_state()
    known = plan(start(900, 501), start(901, 500))  # both in CC-01
    cfg: dict[str, Any] = {
        "models": models(fill=0.9),
        "guidance": guidance({"external_hires": {"CC-01": 3.0}}),
    }
    ctx = step_context(1, known, **cfg)
    started = pending_starts(state, ctx)
    assert started.ledger["external_hires"].tolist() == [2]
    state.category_ledger = {k: v.copy() for k, v in started.ledger.items()}
    filled = fills(state, ctx)
    cost_centers = filled.employees.append[:, EmployeeColumn.COST_CENTER]
    assert int((cost_centers == 0).sum()) == 1  # three in all, two of them pending starts
    assert filled.quotas["external_hires"].realized.tolist() == [3]


def test_an_offer_recently_due_is_still_expected(tmp_path: Path) -> None:
    path = tmp_path / "w.duckdb"
    write_world(generate_world(preset("small", seed=4)), path)
    run = dt.date(2025, 7, 1)
    with duckdb.connect(str(path), read_only=True) as con:
        due = from_pending_starts(con, run)
        lenient = from_pending_starts(con, run, staleness_days=45)
        (late,) = con.execute(
            "SELECT count(*) FROM pending_start WHERE snapshot_date = "
            "(SELECT max(snapshot_date) FROM pending_start WHERE snapshot_date < ?) "
            "AND start_date < ? AND start_date >= ?",
            [run, run, run - dt.timedelta(days=45)],
        ).fetchone() or (0,)
    assert len(lenient) == len(due) + late
    assert min(e.step for e in lenient) == 0


def still(n: int) -> DomainModels:
    """Models under which nothing is sampled: no one leaves, and no opening opens, fills, or
    exits."""
    return DomainModels(
        attrition=ConstantModel(0.0),
        external_fill=ConstantModel(0.0),
        cancellation=ConstantModel(0.0),
        backfill=BackfillRateModel.constant(0.0, n),
        growth=PoissonGrowthModel(n, min_months=1).fit(np.zeros((1, n))),
        attributes=EmpiricalAttributeGenerator(min_rows=1).fit(
            np.zeros(1, dtype=np.int64), np.zeros((1, 6), dtype=np.int64)
        ),
    )


def test_starts_on_offers_from_before_the_run_leave_the_seeded_openings_whole(
    tmp_path: Path,
) -> None:
    """History closed each pending offer's opening on the day it was accepted, so the seed's
    openings already exclude it. A start on it, even when the seed holds its requisition open
    with other openings, reserves none of them and takes none: it's a boundary fill, and the
    run's openings stay as the seed has them."""
    path = tmp_path / "w.duckdb"
    write_world(generate_world(preset("small", seed=6)), path)
    run = dt.date(2025, 1, 1)
    with duckdb.connect(str(path), read_only=True) as con:
        schema = schema_from_world(con)
        state, report = seed_state(con, schema, run)
        reqs = state.live_openings[:, OpeningColumn.REQ_ID].astype(np.int64).tolist()
        (shared,) = con.execute(
            "SELECT count(*) FROM pending_start WHERE snapshot_date = $snapshot "
            "AND req_id IN (SELECT unnest($reqs))",
            {"snapshot": report.snapshot_date, "reqs": reqs},
        ).fetchone() or (0,)
        employees = state.live_employees[:, EmployeeColumn.EMPLOYEE_ID].astype(int).tolist()
        events = known_events(con, run, 6, history=False)
        known = resolve_plan(sanitize(events, employees, reqs), schema, jobs_from_world(con))
    assert shared > 0  # offers whose requisition the seed holds open, with other openings
    openings = state.live_openings
    assert dict(known.plan.reservations) == {}
    seeded = openings[:, OpeningColumn.NUM_OPENINGS].astype(np.int64)
    assert known.plan.fillable(openings, 0).tolist() == seeded.tolist()

    config = SimulationConfig(
        run_date=run,
        schema=schema,
        models=still(len(schema.vocabulary(COST_CENTER))),
        simulation_months=6,
        num_trajectories=1,
        seed=1,
        forecast_scope=ForecastScope.HEADCOUNT,
    )
    (t,) = simulate(config, state, known=known, validate=True).trajectories
    starts = sum(e.event.kind is EventKind.PENDING_START for e in known.events)
    assert t.known_fired == {"pending_start": starts}
    assert (t.open_openings == state.open_openings).all()  # every month, the seed's
    assert int(t.flows["F_pend"].sum()) == 0
    assert int(t.flows["F_bdy"].sum()) == starts
    assert int((t.headcount[-1] - t.headcount[0]).sum()) == starts
