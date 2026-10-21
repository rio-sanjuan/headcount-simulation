"""Tests for seeding a run from a synthetic world."""

from __future__ import annotations

import datetime as dt
from collections.abc import Iterator

import duckdb
import numpy as np
import pytest

from hcsim.engine.seeding import (
    SIMULATED_ID_BASE,
    SeedingError,
    opening_price,
    range_table_from_world,
    schema_from_world,
    seed_state,
    snapshot_before,
)
from hcsim.model.core.layouts import EmployeeColumn, OpeningColumn
from hcsim.model.core.schema import COST_CENTER, JOB_LEVEL, REQ_TYPE, SimulationSchema
from hcsim.synth.config import preset
from hcsim.synth.world import generate_world, write_world

RUN = dt.date(2026, 1, 1)


@pytest.fixture(scope="module")
def con(tmp_path_factory: pytest.TempPathFactory) -> Iterator[duckdb.DuckDBPyConnection]:
    path = tmp_path_factory.mktemp("world") / "world.duckdb"
    write_world(generate_world(preset("tiny", seed=9)), path)
    with duckdb.connect(str(path), read_only=True) as connection:
        yield connection


def test_the_schema_comes_from_the_dimensions(con: duckdb.DuckDBPyConnection) -> None:
    schema = schema_from_world(con)
    assert len(schema.vocabulary(COST_CENTER)) == 10
    assert schema.vocabulary(JOB_LEVEL).labels == ("1", "2", "3", "4", "5")
    assert schema.vocabulary(REQ_TYPE).labels == ("backfill", "campus", "growth")
    assert schema.vocabulary("org_level1").labels == ("ORG-A", "ORG-B")


def test_employees_seed_from_the_last_month_end_stock(con: duckdb.DuckDBPyConnection) -> None:
    state, report = seed_state(con, schema_from_world(con), RUN)
    (active,) = con.execute(
        "SELECT count(*) FROM workforce_txn WHERE trans_type='ACTIVE' "
        "AND event_date = DATE '2025-12-31' AND account = 'ACCT-100'"
    ).fetchone() or (0,)
    assert report.snapshot_date == dt.date(2025, 12, 31)
    assert report.employees_seeded == state.n_employees == active
    assert report.employees_pruned == 0
    assert state.counters_agree()
    assert int(state.headcount.sum()) == active


def test_tenure_is_seeded_as_a_negative_hire_step(con: duckdb.DuckDBPyConnection) -> None:
    state, _ = seed_state(con, schema_from_world(con), RUN)
    rows = state.live_employees
    employee_id = int(rows[0, EmployeeColumn.EMPLOYEE_ID])
    (hire_date,) = con.execute(
        "SELECT hire_date FROM workforce_txn WHERE employee_id = ? LIMIT 1", [employee_id]
    ).fetchone() or (None,)
    expected = (hire_date.year - 2026) * 12 + hire_date.month - 1
    assert rows[0, EmployeeColumn.HIRE_STEP] == expected
    assert (rows[:, EmployeeColumn.HIRE_STEP] < 0).all()
    assert rows[:, EmployeeColumn.VEST_INDEX].tolist() == list(range(state.n_employees))


def test_openings_seed_from_the_same_month_end_snapshot(con: duckdb.DuckDBPyConnection) -> None:
    state, report = seed_state(con, schema_from_world(con), RUN)
    (open_reqs,) = con.execute(
        "SELECT count(*) FROM req_snapshot WHERE snapshot_date = DATE '2025-12-31'"
    ).fetchone() or (0,)
    assert report.openings_seeded == state.n_openings == open_reqs > 0
    live = state.live_openings
    assert (live[:, OpeningColumn.CREATION_STEP] < 0).all()
    assert (
        (live[:, OpeningColumn.NUM_OPENINGS] >= 1) & (live[:, OpeningColumn.NUM_OPENINGS] <= 10)
    ).all()
    assert int(state.open_openings.sum()) == int(live[:, OpeningColumn.NUM_OPENINGS].sum())


def test_seeding_can_skip_the_standing_backlog(con: duckdb.DuckDBPyConnection) -> None:
    state, report = seed_state(con, schema_from_world(con), RUN, include_openings=False)
    assert state.n_openings == report.openings_seeded == 0


def test_capacity_leaves_room_and_ids_start_from_a_high_base(
    con: duckdb.DuckDBPyConnection,
) -> None:
    state, report = seed_state(con, schema_from_world(con), RUN, with_ledger=True)
    assert report.employee_capacity >= state.n_employees + 1_000
    assert report.opening_capacity >= state.n_openings + 500
    assert state.next_employee_id == state.next_requisition_id == SIMULATED_ID_BASE
    assert state.ledger is not None


def test_rows_the_schema_cant_encode_are_pruned_and_counted(
    con: duckdb.DuckDBPyConnection,
) -> None:
    full = schema_from_world(con)
    labels = {name: list(v.labels) for name, v in full.vocabularies.items() if name != "org_level1"}
    labels[COST_CENTER] = [cc for cc in labels[COST_CENTER] if cc != "CC-01"]
    narrow = SimulationSchema.build(labels)
    state, report = seed_state(con, narrow, RUN)
    _, baseline = seed_state(con, full, RUN)
    assert report.employees_pruned > 0
    assert report.employees_seeded + report.employees_pruned == baseline.employees_seeded
    assert state.counters_agree()


def test_sbc_inputs_load_from_the_world(con: duckdb.DuckDBPyConnection) -> None:
    schema = schema_from_world(con)
    assert range_table_from_world(con, schema).values.shape[:2] == (4, 5)
    (close,) = con.execute(
        "SELECT close FROM stock_price WHERE month_end = DATE '2025-12-31'"
    ).fetchone() or (0.0,)
    assert opening_price(con, RUN) == pytest.approx(close)


def test_a_run_before_the_world_starts_is_refused(con: duckdb.DuckDBPyConnection) -> None:
    with pytest.raises(SeedingError, match="no active stock"):
        snapshot_before(con, dt.date(2020, 1, 1))
    with pytest.raises(SeedingError, match="no stock price"):
        opening_price(con, dt.date(2020, 1, 1))
    assert np.isfinite(opening_price(con, RUN))
