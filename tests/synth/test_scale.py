"""Tests for what keeps big worlds within memory: the deferred ledger, the packed-key
aggregate, and the shortcuts that avoid copying the biggest tables."""

from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import cast

import duckdb
import numpy as np
import pandas as pd
import pytest

from hcsim.synth import awards, quirks
from hcsim.synth.config import QuirkTruth, preset
from hcsim.synth.offers import Offers
from hcsim.synth.world import AGG_KEYS, World, _aggregate, generate_world, month_ends, write_world


@pytest.fixture(scope="module")
def quirky() -> World:
    every = QuirkTruth(correction_share=0.2, no_show_pair_share=1.0, pending_hire_rows=True)
    return generate_world(dataclasses.replace(preset("tiny", seed=9), quirks=every))


def _grouped(txn: pd.DataFrame, jobs: pd.DataFrame) -> pd.DataFrame:
    """The aggregate as a plain pandas group-by would build it."""
    job = jobs.set_index("job_code")
    keyed = txn.astype({"job_code": object}).assign(
        period=pd.to_datetime(txn["event_date"]).dt.to_period("M").dt.end_time.dt.normalize(),
        job_family=lambda f: f["job_code"].map(job["job_family"]),
        job_level=lambda f: f["job_code"].map(job["job_level"]),
    )
    keys = {k: object for k in AGG_KEYS if k not in ("period", "job_level")}
    grouped = (
        keyed.astype(keys)
        .groupby(AGG_KEYS)
        .agg(headcount=("employee_id", "size"), amount=("amount", "sum"))
    )
    return grouped.reset_index().sort_values(AGG_KEYS, ignore_index=True)


def _plain(agg: pd.DataFrame) -> pd.DataFrame:
    labels = {k: object for k in AGG_KEYS if k not in ("period", "job_level")}
    frame = agg.astype(labels).astype({"job_level": "int64"})
    return frame.sort_values(AGG_KEYS, ignore_index=True)


def test_the_packed_aggregate_matches_a_group_by(quirky: World) -> None:
    txn, jobs = quirky.tables["workforce_txn"], quirky.tables["dim_job"]
    got = _aggregate(txn, jobs)
    assert list(got.columns) == [*AGG_KEYS, "headcount", "amount"]
    assert got["period"].dtype == "datetime64[ns]"
    assert got["job_family"].dtype == "category"
    pd.testing.assert_frame_equal(_plain(got), _grouped(txn, jobs), check_dtype=False)
    # Groups come out in the keys' order, as a sorted group-by gives them.
    order = got.assign(**{k: got[k].cat.codes for k in AGG_KEYS if got[k].dtype == "category"})
    assert order.equals(order.sort_values(AGG_KEYS))


def test_a_row_missing_a_key_belongs_to_no_group(quirky: World) -> None:
    txn = quirky.tables["workforce_txn"].head(500).copy()
    txn["job_code"] = txn["job_code"].where(txn.index != 3)  # a missing job
    got = _aggregate(txn, quirky.tables["dim_job"])
    assert int(got["headcount"].sum()) == len(txn) - 1
    assert len(_aggregate(txn.iloc[0:0], quirky.tables["dim_job"])) == 0


def test_no_row_quirks_means_no_copy_of_the_transactions(quirky: World) -> None:
    txn = quirky.tables["workforce_txn"]
    config = preset("tiny", seed=9)  # every quirk off
    same = quirks.apply(config, txn, cast(Offers, None), np.random.default_rng(0))
    assert same is txn


def test_the_ledger_is_written_in_chunks_and_read_whole(tmp_path: Path) -> None:
    world = generate_world(preset("tiny", seed=9))
    assert "equity_expense" in world.deferred
    assert "equity_expense" not in world.tables
    whole = world.table("equity_expense")
    chunks = list(world.chunks("equity_expense"))
    pd.testing.assert_frame_equal(pd.concat(chunks, ignore_index=True), whole)
    assert "equity_expense" in world.table_names
    small = list(
        awards.ledger_chunks(
            world.config,
            world.tables["equity_award"],
            world.tables["equity_vest"],
            world.tables["equity_forfeiture_estimate"],
            month_ends(world.config),
            chunk_awards=50,
        )
    )
    assert len(small) > 1
    assert sum(len(c["award_id"]) for c in small) == len(whole)
    path = tmp_path / "w.duckdb"
    write_world(world, path)
    with duckdb.connect(str(path), read_only=True) as con:
        written = con.execute("SELECT * FROM equity_expense").df()
    pd.testing.assert_frame_equal(
        written, whole.assign(period=whole["period"].astype("datetime64[us]")), check_dtype=False
    )


def test_a_world_without_awards_writes_an_empty_ledger() -> None:
    world = generate_world(preset("tiny", seed=9))
    none = world.tables["equity_award"].iloc[0:0]
    frames = list(
        awards.ledger_frames(
            world.config,
            none,
            world.tables["equity_vest"].iloc[0:0],
            world.tables["equity_forfeiture_estimate"],
            month_ends(world.config),
        )
    )
    assert len(frames) == 1
    assert frames[0].empty
    assert list(frames[0].columns) == awards.LEDGER_COLUMNS


def test_a_month_end_with_no_stock_is_empty(quirky: World) -> None:
    history = awards._History(
        quirky.tables["workforce_txn"], quirky.tables["rating"], quirky.tables["dim_job"]
    )
    assert history.stock(np.datetime64("1999-12-31"), "ACCT-100").empty
    stock = history.stock(np.datetime64("2021-01-31"), "ACCT-100")
    assert len(stock) == 200
    assert set(stock.columns) >= {"employee_id", "job_level", "hire_date"}
