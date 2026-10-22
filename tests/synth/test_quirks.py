"""Tests for the data quirks: each has a toggle, and each is checked here."""

from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import Any

import duckdb
import numpy as np
import pandas as pd
import pytest

from hcsim.guidance.ingest import ingest
from hcsim.synth.codes import CATALOG
from hcsim.synth.config import QuirkTruth, WorldConfig, preset
from hcsim.synth.world import World, generate_world, write_world
from tests.support.stat_check import stat_check

QUIRKS = QuirkTruth(
    backdated_share=0.2, correction_share=0.2, no_show_pair_share=1.0, pending_hire_rows=True
)


def world_with(**quirks: Any) -> World:
    config = dataclasses.replace(preset("tiny", seed=12), quirks=QuirkTruth(**quirks))
    return generate_world(config)


@pytest.fixture(scope="module")
def clean() -> World:
    return world_with()


@pytest.fixture(scope="module")
def quirky() -> World:
    """Every quirk at once."""
    return generate_world(dataclasses.replace(preset("tiny", seed=12), quirks=QUIRKS))


def txn(world: World) -> pd.DataFrame:
    return world.tables["workforce_txn"].astype(
        {"trans_type": str, "cost_center": str, "account": str}
    )


def test_every_quirk_is_off_by_default(clean: World) -> None:
    assert QuirkTruth() == WorldConfig().quirks
    codes = set(txn(clean)["trans_type"])
    assert not codes & {"HIRE_REV", "TERM_REV", "HIRE_PENDING"}
    lag = (txn(clean)["reporting_date"] - txn(clean)["event_date"]).dt.days
    assert (
        lag[txn(clean)["trans_type"] != "TERM_PENDING"].max()
        <= clean.config.dates.reporting_lag_days
    )


def test_quirks_leave_the_processes_and_the_equity_world_alone(clean: World, quirky: World) -> None:
    for table in ("req_interval", "equity_award", "equity_vest", "rating"):
        pd.testing.assert_frame_equal(clean.tables[table], quirky.tables[table])

    def stock(world: World) -> pd.DataFrame:
        rows = txn(world)
        rows = rows[rows["trans_type"] == "ACTIVE"]
        return rows.sort_values(["event_date", "employee_id"], ignore_index=True)

    pd.testing.assert_frame_equal(stock(clean), stock(quirky))


def test_backdated_rows_report_a_month_or_more_late() -> None:
    t = txn(world_with(backdated_share=QUIRKS.backdated_share))
    events = t[~t["trans_type"].isin(["ACTIVE", "TERM_PENDING", "HIRE_PENDING"])]
    late = (events["reporting_date"] - events["event_date"]).dt.days
    assert late.max() <= 90
    p = QUIRKS.backdated_share
    observed = int((late >= 30).sum())
    stat_check(
        observed,
        len(events) * p,
        float(np.sqrt(len(events) * p * (1 - p))) * 1.5,
        label="backdated",
    )


def test_corrections_record_a_wrong_day_its_reversal_and_the_right_day(clean: World) -> None:
    corrected = world_with(correction_share=QUIRKS.correction_share)
    t = txn(corrected)
    reversals = t[t["trans_type"].isin(["HIRE_REV", "TERM_REV"])]
    assert len(reversals) > 0
    assert (reversals["amount"] == np.where(reversals["trans_type"] == "HIRE_REV", -1, 1)).all()
    # Every reversal cancels a row of its code on the same person and day.
    original = reversals["trans_type"].str.removesuffix("_REV")
    keys = set(zip(t["employee_id"], t["event_date"], t["trans_type"], strict=True))
    assert all(
        (e, d, o) in keys
        for e, d, o in zip(reversals["employee_id"], reversals["event_date"], original, strict=True)
    )

    # Netted, the hires and departures are exactly the clean world's.
    def net(world: World, codes: list[str]) -> pd.Series:
        rows = txn(world)
        return rows[rows["trans_type"].isin(codes)].groupby("employee_id")["amount"].sum()

    pd.testing.assert_series_equal(
        net(clean, ["TERM"]), net(corrected, ["TERM", "TERM_REV"]), check_names=False
    )


def test_no_show_pairs_hire_and_reverse_within_the_month() -> None:
    t = txn(world_with(no_show_pair_share=QUIRKS.no_show_pair_share))
    reversed_ = t[t["trans_type"] == "HIRE_REV"]
    hires = t[t["trans_type"] == "HIRE"].set_index("employee_id")["event_date"]
    in_post = set(t.loc[t["trans_type"] == "ACTIVE", "employee_id"])
    never = reversed_[~reversed_["employee_id"].isin(in_post)]  # no-shows, not corrections
    assert len(never) > 0
    started = never["employee_id"].map(hires)
    assert (started.dt.to_period("M") == never["event_date"].dt.to_period("M")).all()
    assert (never["event_date"] >= started).all()  # same day when due on the month's last


def test_pending_hires_are_dated_on_their_future_start(quirky: World) -> None:
    t = txn(quirky)
    pending = t[t["trans_type"] == "HIRE_PENDING"]
    assert len(pending) == int(
        (quirky.tables["req_interval"]["close_reason"] == "filled_external").sum()
    )
    assert (pending["event_date"] > pending["reporting_date"]).all()
    assert (pending["amount"] == 0).all()


def test_stocks_still_agree_with_the_summed_flows(quirky: World) -> None:
    """Corrections and no-show pairs net out within their month, so regular stocks move by
    exactly the month's summed amounts."""
    t = txn(quirky)
    regular = t[t["account"] == "ACCT-100"].assign(
        month=lambda f: f["event_date"].dt.to_period("M")
    )
    stock = (
        regular[regular["trans_type"] == "ACTIVE"]
        .groupby(["month", "cost_center"])
        .size()
        .unstack(fill_value=0)
        .sort_index()
    )
    flows = regular[regular["trans_type"] != "ACTIVE"]
    net = (
        flows.groupby(["month", "cost_center"])["amount"]
        .sum()
        .unstack(fill_value=0)
        .reindex(index=stock.index, columns=stock.columns, fill_value=0)
    )
    pd.testing.assert_frame_equal(stock.diff().iloc[1:].astype(int), net.iloc[1:].astype(int))


@pytest.mark.parametrize(
    "field",
    [
        "backdated_share",
        "correction_share",
        "no_show_pair_share",
        "aggregate_only_share",
        "snapshot_gap_share",
    ],
)
def test_a_quirk_share_outside_the_unit_interval_is_refused(field: str) -> None:
    with pytest.raises(ValueError, match=f"quirks.{field}"):
        WorldConfig(quirks=QuirkTruth(**{field: 1.5}))  # type: ignore[arg-type]


def test_ignored_codes_appear_only_in_the_aggregate_fact(tmp_path: Path) -> None:
    """With the quirk on, the aggregate carries `SYS_ADJ` rows that no transaction stands
    behind. They're ignored, so the guidance measures, which select their codes, don't move."""
    plain, quirky = world_with(), world_with(aggregate_only_share=0.2)
    agg = quirky.tables["workforce_agg"].astype({"trans_type": str})
    adjustments = agg[agg["trans_type"] == "SYS_ADJ"]
    assert len(adjustments) > 0
    assert (adjustments["amount"] == 0).all()
    assert CATALOG["SYS_ADJ"][0] == "ignored"
    assert "SYS_ADJ" not in set(txn(quirky)["trans_type"])
    measured = agg[agg["trans_type"] != "SYS_ADJ"].reset_index(drop=True)
    pd.testing.assert_frame_equal(
        measured.astype({"trans_type": str}),
        plain.tables["workforce_agg"].astype({"trans_type": str}),
    )
    series = []
    for name, world in (("plain", plain), ("quirky", quirky)):
        path = tmp_path / f"{name}.duckdb"
        write_world(world, path)
        with duckdb.connect(str(path), read_only=True) as con:
            series.append(ingest(con).frame)
    pd.testing.assert_frame_equal(series[0], series[1])


def _snapshot(world: World, path: Path) -> pd.DataFrame:
    write_world(world, path)
    with duckdb.connect(str(path), read_only=True) as con:
        return con.execute("SELECT * FROM req_snapshot").df()


def test_snapshot_gaps_drop_whole_month_ends(tmp_path: Path) -> None:
    world = world_with(snapshot_gap_share=0.2)
    gaps = world.truth["quirks"]["snapshot_gaps"]
    assert len(gaps) > 0
    seen = set(
        pd.to_datetime(_snapshot(world, tmp_path / "g.duckdb")["snapshot_date"]).dt.date.astype(str)
    )
    assert not seen & set(gaps)
    full = set(
        pd.to_datetime(
            _snapshot(world_with(), tmp_path / "f.duckdb")["snapshot_date"]
        ).dt.date.astype(str)
    )
    assert full - seen == set(gaps) & full


def test_a_feed_seam_changes_how_the_state_is_spelled(tmp_path: Path) -> None:
    snapshot = _snapshot(world_with(state_seam_month="2023-07"), tmp_path / "s.duckdb")
    after = pd.to_datetime(snapshot["snapshot_date"]) >= pd.Timestamp("2023-07-01")
    assert set(snapshot.loc[~after, "state"]) == {"open"}
    assert set(snapshot.loc[after, "state"]) == {"OPEN"}
    assert set(_snapshot(world_with(), tmp_path / "p.duckdb")["state"]) == {"open"}


def test_bulk_openings_come_and_go_with_their_toggle() -> None:
    def bulk(world: World) -> int:
        reqs = world.tables["req_interval"]
        return int((reqs.groupby("req_id")["openings"].max() >= 15).sum())

    tiny = preset("tiny", seed=12)
    on = dataclasses.replace(tiny, openings=dataclasses.replace(tiny.openings, bulk_share=0.2))
    off = dataclasses.replace(tiny, openings=dataclasses.replace(tiny.openings, bulk_share=0.0))
    assert bulk(generate_world(on)) > 0
    assert bulk(generate_world(off)) == 0


def test_closures_are_right_censored_at_the_world_s_end(tmp_path: Path) -> None:
    world = world_with()
    reqs = world.tables["req_interval"]
    assert (pd.to_datetime(reqs["closed_date"].dropna()) <= pd.Timestamp("2025-12-31")).all()
    still_open = set(reqs.loc[reqs["closed_date"].isna(), "req_id"])
    assert len(still_open) > 0
    snapshot = _snapshot(world, tmp_path / "c.duckdb")
    last = snapshot[pd.to_datetime(snapshot["snapshot_date"]) == pd.Timestamp("2025-12-31")]
    # Everything open at the end shows in the last snapshot, except what opened that very day.
    opened_last_day = set(
        reqs.loc[pd.to_datetime(reqs["interval_start"]) > pd.Timestamp("2025-12-31"), "req_id"]
    )
    assert still_open - opened_last_day <= set(last["req_id"])


def test_a_bad_seam_month_is_refused() -> None:
    with pytest.raises(ValueError, match=r"quirks\.state_seam_month"):
        WorldConfig(quirks=QuirkTruth(state_seam_month="July"))
