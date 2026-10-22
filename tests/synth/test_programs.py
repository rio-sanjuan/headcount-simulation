"""Tests for termination programs."""

from __future__ import annotations

import dataclasses
import itertools

import pandas as pd
import pytest

from hcsim.synth.config import BackfillTruth, ProgramTruth, WorldConfig, preset
from hcsim.synth.world import World, generate_world


@pytest.fixture(scope="module")
def world() -> World:
    """A small world where programs are common, and every vacancy is backfilled."""
    config = dataclasses.replace(
        preset("small", seed=5),
        programs=ProgramTruth(start_probability=0.2),
        backfill=BackfillTruth(base_probability=1.0, notice_share=0.0),
    )
    return generate_world(config)


def rifs(world: World) -> pd.DataFrame:
    txn = world.tables["workforce_txn"].astype(
        {"trans_type": str, "account": str, "cost_center": str}
    )
    org = world.tables["dim_org"].set_index("cost_center")["org_level1"]
    rows = txn[txn["trans_type"] == "RIF"]
    return rows.assign(
        org=rows["cost_center"].map(org), month=pd.to_datetime(rows["event_date"]).dt.to_period("M")
    )


def test_programs_terminate_their_plan_within_their_org(world: World) -> None:
    programs = world.truth["processes"]["programs"]
    rows = rifs(world)
    assert len(programs) > 3
    assert (rows["amount"] == -1).all()
    assert set(rows["account"]) == {"ACCT-100"}
    last = pd.Period("2025-12", freq="M")
    for program in programs:
        start = pd.Period(program["start"], freq="M")
        window = [start + k for k in range(program["months"]) if start + k <= last]
        mine = rows[(rows["org"] == program["org"]) & rows["month"].isin(window)]
        if program["terminated"] == program["planned"]:
            assert len(mine) >= program["terminated"]
        per_month = mine.groupby("month").size().reindex(window, fill_value=0)
        assert per_month.max() - per_month.min() <= max(1, program["planned"] // 10)
    assert sum(p["terminated"] for p in programs) == len(rows)


def test_an_org_runs_one_program_at_a_time(world: World) -> None:
    spans: dict[str, list[tuple[pd.Period, pd.Period]]] = {}
    for p in world.truth["processes"]["programs"]:
        start = pd.Period(p["start"], freq="M")
        spans.setdefault(p["org"], []).append((start, start + p["months"] - 1))
    for windows in spans.values():
        windows.sort()
        assert all(later[0] > earlier[1] for earlier, later in itertools.pairwise(windows))


def test_programs_lean_on_low_ratings(world: World) -> None:
    """Compared with the org's staff at the time, a program cuts the low-rated more often."""
    txn = world.tables["workforce_txn"].astype(
        {"trans_type": str, "account": str, "cost_center": str}
    )
    org = world.tables["dim_org"].set_index("cost_center")["org_level1"]
    ratings = world.tables["rating"].assign(
        rating=lambda f: f["rating"].astype(str),
        cycle_date=lambda f: pd.to_datetime(f["cycle_date"]),
    )
    cut, staff = [], []
    for (unit, key), rows in rifs(world).groupby(["org", "month"]):
        month = pd.Period(str(key), freq="M")
        before = (month - 1).end_time.normalize()
        stock = txn[
            (txn["trans_type"] == "ACTIVE")
            & (pd.to_datetime(txn["event_date"]) == before)
            & (txn["account"] == "ACCT-100")
            & (txn["cost_center"].map(org) == unit)
        ]
        latest = ratings[ratings["cycle_date"] <= month.start_time].drop_duplicates(
            "employee_id", keep="last"
        )
        rating = latest.set_index("employee_id")["rating"]
        cut += rating.reindex(rows["employee_id"]).dropna().tolist()
        staff += rating.reindex(stock["employee_id"]).dropna().tolist()
    cut_s, staff_s = pd.Series(cut), pd.Series(staff)
    assert (cut_s == "R1").mean() > 2 * (staff_s == "R1").mean()
    assert (cut_s == "R5").mean() < (staff_s == "R5").mean()


def test_program_terminations_open_no_backfill(world: World) -> None:
    """With every vacancy backfilled, backfills match departures and transfer fills exactly:
    program terminations add none."""
    txn, reqs = world.tables["workforce_txn"], world.tables["req_interval"]
    vacancies = int(
        (txn["trans_type"] == "TERM").sum() + (reqs["close_reason"] == "filled_transfer").sum()
    )
    assert int(reqs.loc[reqs["req_type"] == "backfill", "req_id"].nunique()) == vacancies
    assert (txn["trans_type"] == "RIF").sum() > 0


def test_the_guidance_attrition_measure_leaves_programs_out(world: World) -> None:
    txn = world.tables["workforce_txn"]
    assert (
        not txn.loc[txn["trans_type"] == "TERM", "employee_id"]
        .isin(rifs(world)["employee_id"])
        .any()
    )


@pytest.mark.parametrize(
    ("truth", "message"),
    [
        (ProgramTruth(start_probability=1.5), "programs.start_probability"),
        (ProgramTruth(window_months=(0, 2)), "programs.window_months"),
        (ProgramTruth(share=(0.5, 0.1)), "programs.share"),
        (ProgramTruth(rating_weight=(1.0, 0.0, 1.0, 1.0, 1.0)), "programs.rating_weight"),
    ],
)
def test_a_bad_program_truth_is_refused(truth: ProgramTruth, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        WorldConfig(programs=truth)
