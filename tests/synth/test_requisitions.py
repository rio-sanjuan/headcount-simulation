"""Tests for the world's requisition process."""

from __future__ import annotations

from pathlib import Path

import duckdb
import pandas as pd
import pytest

from hcsim.synth.config import preset
from hcsim.synth.world import World, generate_world, write_world


@pytest.fixture(scope="module")
def world() -> World:
    return generate_world(preset("tiny", seed=11))


def test_every_hire_fills_an_external_or_campus_requisition_in_its_role(world: World) -> None:
    txn, reqs = world.tables["workforce_txn"], world.tables["req_interval"]
    hires = txn[txn["trans_type"] == "HIRE"].set_index("employee_id")
    filled = reqs["close_reason"].isin(["filled_external", "filled_campus"])
    external = reqs[filled].set_index("filled_by")
    assert len(hires) == len(external) > 0
    joined = hires.join(external, rsuffix="_req", how="inner")
    assert len(joined) == len(hires)
    assert (joined["cost_center"] == joined["cost_center_req"]).all()
    assert (joined["job_code"] == joined["job_code_req"]).all()
    assert (joined["event_date"] == joined["closed_date"]).all()


def test_every_transfer_in_fills_a_transfer_requisition(world: World) -> None:
    txn, reqs = world.tables["workforce_txn"], world.tables["req_interval"]
    arrivals = txn[txn["trans_type"] == "XFER_IN"]
    transfers = reqs[reqs["close_reason"] == "filled_transfer"]
    assert len(arrivals) == len(transfers) > 0
    keys = set(zip(transfers["filled_by"], transfers["closed_date"], strict=True))
    assert set(zip(arrivals["employee_id"], arrivals["event_date"], strict=True)) == keys


def test_intervals_are_consistent(world: World) -> None:
    reqs = world.tables["req_interval"]
    closed = reqs[reqs["closed_date"].notna()]
    assert (pd.to_datetime(closed["closed_date"]) >= pd.to_datetime(closed["opened_date"])).all()
    assert set(closed["close_reason"]) <= {
        "filled_external",
        "filled_transfer",
        "filled_conversion",
        "filled_campus",
        "unfilled",
    }
    filled = closed["close_reason"].str.startswith("filled")
    assert closed.loc[filled, "filled_by"].notna().all()
    assert closed.loc[~filled, "filled_by"].isna().all()
    assert set(reqs["req_type"]) == {"backfill", "campus", "growth"}
    assert (pd.to_datetime(reqs["interval_start"]) >= pd.to_datetime(reqs["opened_date"])).all()


def test_each_requisition_s_intervals_chain_and_count_down(world: World) -> None:
    """Intervals follow each other without gaps; each fill takes one opening; the requisition
    ends filled from its last opening, unfilled, or still open in its last interval."""
    reqs = world.tables["req_interval"]
    for _, rows in reqs.groupby("req_id"):
        starts = pd.to_datetime(rows["interval_start"]).tolist()
        ends = pd.to_datetime(rows["closed_date"]).tolist()
        openings = rows["openings"].tolist()
        reasons = rows["close_reason"].astype(object).tolist()
        assert starts[1:] == ends[:-1]
        assert openings == list(range(openings[0], openings[0] - len(rows), -1))
        assert all(str(r).startswith("filled") for r in reasons[:-1])
        last = reasons[-1]
        assert pd.isna(last) or last == "unfilled" or openings[-1] == 1


def test_the_snapshot_view_shows_requisitions_open_at_each_month_end(
    world: World, tmp_path: Path
) -> None:
    path = tmp_path / "world.duckdb"
    write_world(world, path)
    with duckdb.connect(str(path), read_only=True) as con:
        snapshot = con.execute("SELECT * FROM req_snapshot").df()
    reqs = world.tables["req_interval"]
    for month_end in pd.to_datetime(snapshot["snapshot_date"]).drop_duplicates().head(12):
        opened = pd.to_datetime(reqs["interval_start"]) <= month_end
        closed = pd.to_datetime(reqs["closed_date"])
        still_open = closed.isna() | (closed > month_end)
        expected = set(reqs.loc[opened & still_open, "req_id"])
        shown = set(snapshot.loc[pd.to_datetime(snapshot["snapshot_date"]) == month_end, "req_id"])
        assert shown == expected


def test_a_requisition_filled_within_its_month_never_appears_in_a_snapshot(
    world: World, tmp_path: Path
) -> None:
    path = tmp_path / "world.duckdb"
    write_world(world, path)
    with duckdb.connect(str(path), read_only=True) as con:
        seen = set(con.execute("SELECT DISTINCT req_id FROM req_snapshot").df()["req_id"])
    intervals = world.tables["req_interval"]
    final = intervals.groupby("req_id").agg(
        opened=("opened_date", "min"),
        closed=("closed_date", lambda d: d.max() if d.notna().all() else pd.NaT),
    )
    final = final.dropna(subset=["closed"])
    same_month = final[
        pd.to_datetime(final["opened"]).dt.to_period("M")
        == pd.to_datetime(final["closed"]).dt.to_period("M")
    ]
    assert len(same_month) > 0
    assert not set(same_month.index) & seen
