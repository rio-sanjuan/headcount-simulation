"""Tests for the transaction code catalog, the `_previous` twins, and reporting dates."""

from __future__ import annotations

import datetime as dt
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
import pytest

from hcsim.engine.seeding import SeedingError, schema_from_world, seed_state
from hcsim.guidance.ingest import IngestionError, ingest
from hcsim.synth.codes import CATALOG, DISPOSITIONS
from hcsim.synth.config import preset
from hcsim.synth.workforce import TRANS_TYPES
from hcsim.synth.world import World, generate_world, write_world
from tests.support.stat_check import stat_check


def test_the_catalog_covers_every_code_with_its_disposition_and_amount(small_world: World) -> None:
    assert set(CATALOG) == set(TRANS_TYPES)
    assert {d for d, _, _ in CATALOG.values()} == set(DISPOSITIONS)
    txn = small_world.tables["workforce_txn"].astype({"trans_type": str})
    assert set(txn["trans_type"]) <= set(CATALOG)
    amounts = txn.groupby("trans_type")["amount"].unique()
    for code, values in amounts.items():
        assert values.tolist() == [CATALOG[str(code)][1]], code
    table = small_world.tables["dim_trans_type"]
    assert set(table["trans_type"]) == set(CATALOG)


def test_excluded_codes_belong_to_the_pool_and_ignored_ones_move_nothing(
    small_world: World,
) -> None:
    txn = small_world.tables["workforce_txn"].astype({"trans_type": str, "account": str})
    excluded = [c for c, (d, _, _) in CATALOG.items() if d == "excluded"]
    assert set(txn.loc[txn["trans_type"].isin(excluded), "account"]) == {"ACCT-200"}
    admin = txn[txn["trans_type"] == "ADMIN"]
    stock = int((txn["trans_type"] == "ACTIVE").sum()) - int(
        (txn["event_date"] == txn["event_date"].min()).sum()
    )
    rate = small_world.config.dates.admin_change_rate
    stat_check(len(admin), stock * rate, float(np.sqrt(stock * rate)) * 1.1, label="admin rows")


def test_previous_twins_hold_the_attributes_before_a_change(small_world: World) -> None:
    txn = small_world.tables["workforce_txn"].astype(str)
    out = txn[txn["trans_type"] == "CTG_START"]
    converted = out[out["account_previous"] == "ACCT-100"]  # conversions out, not new starts
    assert len(converted) > 0
    assert (converted["account"] == "ACCT-200").all()
    steady = txn[txn["trans_type"].isin(["ACTIVE", "HIRE", "TERM", "ADMIN"])]
    for column in ("cost_center", "job_code", "location", "company", "account", "is_manager"):
        assert (steady[column] == steady[f"{column}_previous"]).all(), column


def test_events_report_on_the_day_or_a_few_days_later(small_world: World) -> None:
    txn = small_world.tables["workforce_txn"].astype({"trans_type": str})
    lag = (pd.to_datetime(txn["reporting_date"]) - pd.to_datetime(txn["event_date"])).dt.days
    stocks = txn["trans_type"] == "ACTIVE"
    notices = txn["trans_type"] == "TERM_PENDING"
    assert (lag[stocks] == 0).all()
    assert (lag[notices] < 0).all()  # a notice is reported before the day of leaving
    events = lag[~stocks & ~notices]
    assert events.between(0, small_world.config.dates.reporting_lag_days).all()
    p = small_world.config.dates.same_day_reporting_share
    stat_check(
        int((events == 0).sum()),
        len(events) * p,
        float(np.sqrt(len(events) * p * (1 - p))),
        label="same day",
    )


@pytest.fixture(scope="module")
def unknown_code_world(tmp_path_factory: pytest.TempPathFactory) -> Path:
    path = tmp_path_factory.mktemp("codes") / "world.duckdb"
    write_world(generate_world(preset("tiny", seed=2)), path)
    with duckdb.connect(str(path)) as con:
        con.execute(
            "INSERT INTO workforce_txn SELECT * REPLACE ('MYSTERY' AS trans_type) "
            "FROM workforce_txn LIMIT 1"
        )
    return path


def test_readers_refuse_a_code_the_catalog_doesnt_define(unknown_code_world: Path) -> None:
    with duckdb.connect(str(unknown_code_world), read_only=True) as con:
        with pytest.raises(SeedingError, match=r"missing from dim_trans_type: \['MYSTERY'\]"):
            seed_state(con, schema_from_world(con), dt.date(2026, 1, 1))
        with pytest.raises(IngestionError, match="MYSTERY"):
            ingest(con)
