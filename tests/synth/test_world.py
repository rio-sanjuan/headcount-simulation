"""Tests for the tiny synthetic world's structure and initial population."""

from __future__ import annotations

import json
from pathlib import Path

import duckdb
import pandas as pd
import pytest

from hcsim.synth.config import WorldConfig, preset
from hcsim.synth.world import generate_world, month_ends, write_world


@pytest.fixture(scope="module")
def tiny() -> pd.DataFrame:
    """The initial stock: ACTIVE rows at the first month-end."""
    txn = generate_world(preset("tiny", seed=1)).tables["workforce_txn"]
    first = pd.Timestamp("2021-01-31")
    stock = txn[(txn["trans_type"] == "ACTIVE") & (txn["event_date"] == first)]
    return stock[stock["account"] == "ACCT-100"]  # regular employees, not the contingent pool


def test_the_tiny_preset_has_the_planned_shape() -> None:
    world = generate_world(preset("tiny", seed=1))
    org = world.tables["dim_org"]
    assert len(org) == 10
    assert sorted(org["org_level1"].unique()) == ["ORG-A", "ORG-B"]
    assert (org["org_level2"] == org["cost_center"]).all()  # the leaf pads the deepest level
    jobs = world.tables["dim_job"]
    assert len(jobs) == 4 * 5 + 4 * 2  # every family and level, plus manager titles at 4-5
    assert jobs["job_code"].is_unique
    assert len(world.tables["dim_location"]) == len(world.tables["dim_company"]) == 1


def test_the_initial_population_is_one_active_row_per_employee(tiny: pd.DataFrame) -> None:
    assert len(tiny) == 200
    assert tiny["employee_id"].is_unique
    assert (tiny["trans_type"] == "ACTIVE").all()
    assert (tiny["event_date"] == pd.Timestamp("2021-01-31")).all()
    assert (pd.to_datetime(tiny["hire_date"]) < pd.Timestamp("2021-01-01")).all()


def test_every_role_uses_a_job_code_from_the_job_dimension(tiny: pd.DataFrame) -> None:
    jobs = generate_world(preset("tiny", seed=1)).tables["dim_job"]
    assert set(tiny["job_code"]) <= set(jobs["job_code"])
    # The fact carries job codes only: level comes from the job dimension.
    joined = tiny.merge(jobs, on="job_code", validate="many_to_one")
    assert not joined.loc[joined["job_level"] < 4, "is_manager"].any()  # managers start at 4


def test_the_same_seed_gives_identical_tables_and_another_seed_doesnt() -> None:
    a = generate_world(preset("tiny", seed=5)).tables["workforce_txn"]
    b = generate_world(preset("tiny", seed=5)).tables["workforce_txn"]
    c = generate_world(preset("tiny", seed=6)).tables["workforce_txn"]
    pd.testing.assert_frame_equal(a, b)
    assert not a.equals(c)


def test_month_ends_cover_the_configured_window() -> None:
    ends = month_ends(preset("tiny"))
    assert (ends[0], ends[-1], len(ends)) == (
        pd.Timestamp("2021-01-31"),
        pd.Timestamp("2025-12-31"),
        60,
    )


def test_invalid_configurations_list_every_problem() -> None:
    with pytest.raises(ValueError, match="level_mix needs one share per level") as error:
        WorldConfig(level_mix=(1.0,), months=0)
    assert "months and initial_employees must be positive" in str(error.value)
    with pytest.raises(ValueError, match="unknown preset"):
        preset("enormous")


def test_the_world_writes_to_duckdb_with_its_truth_manifest(tmp_path: Path) -> None:
    world = generate_world(preset("tiny", seed=1))
    path = tmp_path / "world.duckdb"
    write_world(world, path)
    write_world(world, path)  # rewriting replaces the file
    with duckdb.connect(str(path), read_only=True) as con:
        tables = {row[0] for row in con.execute("SHOW TABLES").fetchall()}
        assert {"dim_org", "dim_job", "workforce_txn", "truth_manifest"} <= tables
        (count,) = con.execute(
            "SELECT count(*) FROM workforce_txn "
            "WHERE trans_type = 'ACTIVE' AND event_date = DATE '2021-01-31' "
            "AND account = 'ACCT-100'"
        ).fetchone() or (0,)
        (manifest,) = con.execute("SELECT manifest FROM truth_manifest").fetchone() or ("",)
    assert count == 200
    assert json.loads(manifest)["config"]["seed"] == 1
