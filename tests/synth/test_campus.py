"""Tests for campus hiring."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from hcsim.synth.config import CampusTruth, WorldConfig
from hcsim.synth.world import World
from tests.support.stat_check import stat_check


def campus(world: World) -> pd.DataFrame:
    reqs = world.tables["req_interval"].astype({"req_type": str, "close_reason": str})
    return reqs[reqs["req_type"] == "campus"].assign(
        opened=lambda f: pd.to_datetime(f["opened_date"]),
        closed=lambda f: pd.to_datetime(f["closed_date"]),
    )


def test_campus_requisitions_open_each_october_at_entry_levels(small_world: World) -> None:
    reqs = campus(small_world)
    assert set(reqs["opened"].dt.month) == {10}
    assert set(reqs["opened"].dt.year) == {2021, 2022, 2023, 2024, 2025}
    assert set(reqs["job_level"]) <= {1, 2}
    share = (reqs["job_level"] == 1).mean()
    stat_check(float(share), 0.85, float(np.sqrt(0.85 * 0.15 / len(reqs))), label="level 1 share")


def test_openings_scale_with_headcount(small_world: World) -> None:
    txn = small_world.tables["workforce_txn"]
    regular = txn[(txn["trans_type"] == "ACTIVE") & (txn["account"] == "ACCT-100")]
    september = (
        regular[pd.to_datetime(regular["event_date"]).dt.month == 9].groupby("event_date").size()
    )
    expected = small_world.config.campus.positions_per_head * float(september.sum())
    observed = len(campus(small_world))
    stat_check(observed, expected, float(np.sqrt(expected)) * 1.2, label="campus openings")


def test_campus_fills_only_through_its_channel_in_the_start_months(small_world: World) -> None:
    reqs = campus(small_world)
    closed = reqs.dropna(subset=["closed_date"])
    assert set(closed["close_reason"]) == {"filled_campus", "unfilled"}
    assert set(closed["closed"].dt.month) <= {6, 7, 8}
    assert set(closed.loc[closed["close_reason"] == "unfilled", "closed"].dt.month) == {8}
    # Every cohort is closed by the August after it opens, except the last, still waiting.
    still_open = reqs[reqs["closed_date"].isna()]
    assert set(still_open["opened"].dt.year) == {2025}
    txn = small_world.tables["workforce_txn"]
    hires = txn[txn["trans_type"] == "HIRE"]
    filled = closed[closed["close_reason"] == "filled_campus"]
    keys = set(zip(hires["employee_id"], pd.to_datetime(hires["event_date"]), strict=True))
    assert all(
        key in keys for key in zip(filled["filled_by"].astype(int), filled["closed"], strict=True)
    )
    assert (filled["closed"].dt.weekday == 0).mean() > 0.95  # campus hires start on Mondays


def test_no_other_channel_fills_a_campus_requisition(small_world: World) -> None:
    reqs = small_world.tables["req_interval"]
    other = reqs[reqs["req_type"] != "campus"]
    assert not (other["close_reason"] == "filled_campus").any()


@pytest.mark.parametrize(
    ("truth", "message"),
    [
        (CampusTruth(open_month=13), "campus months must be calendar months"),
        (CampusTruth(open_month=6), "campus.open_month can't also be a start month"),
        (CampusTruth(fill_probability=(0.5,)), "one chance per start month"),
        (CampusTruth(level_mix={1: 0.5}), "campus.level_mix"),
        (CampusTruth(positions_per_head=-1.0), "campus.positions_per_head"),
    ],
)
def test_a_bad_campus_truth_is_refused(truth: CampusTruth, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        WorldConfig(campus=truth)
