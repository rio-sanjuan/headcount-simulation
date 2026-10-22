"""Tests for buffer capacity rules and seeding from a configuration."""

from __future__ import annotations

import datetime as dt
from collections.abc import Iterator

import duckdb
import pytest

from hcsim.engine.config import ConfigError, ForecastScope, SimulationConfig
from hcsim.engine.seeding import CapacityRule, schema_from_world, seed_for, seed_state
from hcsim.model.core.layouts import OpeningColumn
from hcsim.synth.config import preset
from hcsim.synth.world import generate_world, write_world
from tests.engine.test_config_state_seeds import config, models

RUN = dt.date(2026, 1, 1)


@pytest.fixture(scope="module")
def con(tmp_path_factory: pytest.TempPathFactory) -> Iterator[duckdb.DuckDBPyConnection]:
    path = tmp_path_factory.mktemp("world") / "w.duckdb"
    write_world(generate_world(preset("tiny", seed=9)), path)
    with duckdb.connect(str(path), read_only=True) as connection:
        yield connection


def test_capacity_grows_with_the_seed_and_the_run_s_length() -> None:
    rule = CapacityRule(employee_headroom=0.5, opening_headroom=0.25, min_spare_rows=10)
    assert rule.sizes(1_000, 40, 12) == (1_500, 40 + 375)
    assert rule.sizes(1_000, 40, 36) == (2_500, 40 + 625)
    # A run shorter than a year still gets a year's room.
    assert rule.sizes(1_000, 40, 3) == rule.sizes(1_000, 40, 12)
    # Small seeds get the floor of spare rows.
    assert CapacityRule(min_spare_rows=500).sizes(10, 2, 12) == (510, 502)
    assert CapacityRule(0.0, 0.0, 0).sizes(7, 3, 12) == (7, 3)


def _run(con: duckdb.DuckDBPyConnection, **settings: object) -> SimulationConfig:
    return SimulationConfig(
        run_date=RUN,
        schema=schema_from_world(con),
        models=models(),
        forecast_scope=ForecastScope.HEADCOUNT,
        **settings,  # type: ignore[arg-type]
    )


def test_a_configuration_seeds_its_run(con: duckdb.DuckDBPyConnection) -> None:
    run = _run(
        con,
        simulation_months=24,
        seeding_max_openings=1,
        seeding_employee_headroom=0.25,
        seeding_opening_headroom=0.1,
        seeding_min_spare_rows=0,
    )
    state, report = seed_for(con, run)
    expected = CapacityRule(0.25, 0.1, 0).sizes(state.n_employees, state.n_openings, 24)
    assert (report.employee_capacity, report.opening_capacity) == expected
    assert (state.live_openings[:, OpeningColumn.NUM_OPENINGS] == 1).all()
    assert state.ledger is None  # a headcount-only run tracks no equity
    _, direct = seed_state(con, run.schema, RUN, max_openings=1)
    assert report.employees_seeded == direct.employees_seeded


def test_a_run_can_open_every_requisition_itself(con: duckdb.DuckDBPyConnection) -> None:
    state, report = seed_for(con, _run(con, seeding_include_opening_buffer=False))
    assert state.n_openings == report.openings_seeded == 0
    assert report.opening_capacity == CapacityRule().sizes(state.n_employees, 0, 12)[1]


def test_seeding_settings_are_validated() -> None:
    with pytest.raises(ConfigError) as caught:
        config(seeding_employee_headroom=-0.1, seeding_min_spare_rows=-1)
    assert caught.value.problems == [
        "seeding headrooms must be non-negative",
        "seeding_min_spare_rows must be non-negative",
    ]
