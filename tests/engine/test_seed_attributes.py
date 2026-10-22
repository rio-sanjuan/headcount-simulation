"""Tests for the seeded rating, manager flag, opening clip, and requisition-type mapping."""

from __future__ import annotations

import datetime as dt
from collections.abc import Iterator

import duckdb
import numpy as np
import pandas as pd
import pytest

from hcsim.engine.seeding import (
    REQ_TYPE_MAP,
    SeedingError,
    _mapped_types,
    ratings_as_of,
    schema_from_world,
    seed_state,
)
from hcsim.model.core.layouts import EmployeeColumn, OpeningColumn
from hcsim.model.data.subset import SubsetSpec
from hcsim.synth.config import preset
from hcsim.synth.world import generate_world, write_world

RUN = dt.date(2026, 1, 1)


@pytest.fixture(scope="module")
def con(tmp_path_factory: pytest.TempPathFactory) -> Iterator[duckdb.DuckDBPyConnection]:
    path = tmp_path_factory.mktemp("world") / "w.duckdb"
    write_world(generate_world(preset("tiny", seed=9)), path)
    with duckdb.connect(str(path), read_only=True) as connection:
        yield connection


def test_each_employee_carries_their_rating_as_of_the_run_date(
    con: duckdb.DuckDBPyConnection,
) -> None:
    state, _ = seed_state(con, schema_from_world(con), RUN)
    rows = state.live_employees
    ids = rows[:, EmployeeColumn.EMPLOYEE_ID].astype(np.int64)
    seeded = rows[:, EmployeeColumn.PERF_RATING]
    latest = con.execute(
        "SELECT employee_id, rating_rank FROM rating WHERE cycle_date < ? "
        "QUALIFY row_number() OVER (PARTITION BY employee_id ORDER BY cycle_date DESC) = 1",
        [RUN],
    ).df()
    rank = dict(zip(latest["employee_id"], latest["rating_rank"], strict=True))
    assert seeded.tolist() == [float(rank.get(int(i), 0)) for i in ids]
    assert 0 < (seeded > 0).mean() < 1  # most are rated; recent hires aren't yet
    # A cycle on the run date itself hasn't happened yet when the run starts.
    on_cycle = ratings_as_of(con, dt.date(2025, 3, 1), pd.Series(ids))
    after_cycle = ratings_as_of(con, dt.date(2025, 3, 2), pd.Series(ids))
    assert (on_cycle != after_cycle).any()


def test_a_rating_subset_is_filtered_in_memory(con: duckdb.DuckDBPyConnection) -> None:
    state, report = seed_state(
        con, schema_from_world(con), RUN, subset=SubsetSpec.of(perf_rating=["3", "0"])
    )
    assert set(state.live_employees[:, EmployeeColumn.PERF_RATING].tolist()) <= {0.0, 3.0}
    assert report.employee_causes["outside_subset"] > 0
    assert report.employees_seeded > 0
    # Requisitions have no rating, so the rule leaves them alone.
    _, everyone = seed_state(con, schema_from_world(con), RUN)
    assert report.openings_seeded == everyone.openings_seeded


def test_the_manager_flag_comes_from_the_stock(con: duckdb.DuckDBPyConnection) -> None:
    state, _ = seed_state(con, schema_from_world(con), RUN)
    rows = state.live_employees
    flags = con.execute(
        "SELECT employee_id, is_manager FROM workforce_txn "
        "WHERE trans_type = 'ACTIVE' AND event_date = DATE '2025-12-31'"
    ).df()
    flag = dict(zip(flags["employee_id"], flags["is_manager"], strict=True))
    expected = [float(flag[int(i)]) for i in rows[:, EmployeeColumn.EMPLOYEE_ID]]
    assert rows[:, EmployeeColumn.IS_MANAGER].tolist() == expected
    assert 0 < np.mean(expected) < 1


def test_openings_are_clipped_to_the_configured_maximum(con: duckdb.DuckDBPyConnection) -> None:
    schema = schema_from_world(con)
    state, report = seed_state(con, schema, RUN, max_openings=1)
    openings = state.live_openings[:, OpeningColumn.NUM_OPENINGS]
    assert (openings == 1).all()
    _, unclipped = seed_state(con, schema, RUN)
    assert report.openings_clipped > unclipped.openings_clipped
    with pytest.raises(SeedingError, match="at least one opening"):
        seed_state(con, schema, RUN, max_openings=0)


def test_requisition_types_are_mapped_and_unmapped_ones_pruned(
    con: duckdb.DuckDBPyConnection,
) -> None:
    schema = schema_from_world(con)
    _, everyone = seed_state(con, schema, RUN)
    without_backfill = {k: v for k, v in REQ_TYPE_MAP.items() if k != "backfill"}
    _, report = seed_state(con, schema, RUN, req_types=without_backfill)
    pruned = report.opening_causes["unknown_req_type"]
    assert pruned > 0
    assert report.openings_seeded + pruned == everyone.openings_seeded
    labels = pd.Series([" Growth", "BACKFILL", "campus", "posting", None])
    assert _mapped_types(labels, {"GROWTH": "growth", "backfill": "backfill"}).tolist() == [
        "growth",
        "backfill",
        None,
        None,
        None,
    ]
