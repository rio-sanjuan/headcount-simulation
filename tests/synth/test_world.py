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


def test_the_aggregate_fact_sums_the_transactions(tmp_path: Path) -> None:
    world = generate_world(preset("tiny", seed=4))
    path = tmp_path / "w.duckdb"
    write_world(world, path)
    with duckdb.connect(str(path), read_only=True) as con:
        mismatched = con.execute(
            """
            WITH fact AS (
                SELECT last_day(t.event_date) AS period, t.cost_center, j.job_family,
                       j.job_level, t.location, t.company, t.account, t.trans_type,
                       count(*) AS headcount, sum(t.amount) AS amount
                FROM workforce_txn t JOIN dim_job j USING (job_code) GROUP BY ALL
            )
            SELECT count(*) FROM fact FULL JOIN workforce_agg a
              USING (period, cost_center, job_family, job_level, location, company, account,
                     trans_type)
            WHERE fact.headcount IS DISTINCT FROM a.headcount
               OR fact.amount IS DISTINCT FROM a.amount
            """
        ).fetchone() or (1,)
        periods = {
            row[0] for row in con.execute("SELECT DISTINCT period FROM workforce_agg").fetchall()
        }
    assert mismatched == (0,)
    assert all(p == p.replace(day=1) + pd.offsets.MonthEnd(0) for p in map(pd.Timestamp, periods))


def test_the_rating_dimension_lists_every_label() -> None:
    world = generate_world(preset("tiny", seed=4))
    ratings = world.tables["dim_rating"]
    assert ratings["rating"].tolist() == ["R1", "R2", "R3", "R4", "R5"]
    assert ratings["rating_rank"].tolist() == [1, 2, 3, 4, 5]
    assert set(world.tables["rating"]["rating"].astype(str)) <= set(ratings["rating"])
