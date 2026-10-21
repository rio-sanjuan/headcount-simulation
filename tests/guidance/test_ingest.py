"""Tests for guidance ingestion against a synthetic world."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import duckdb
import pandas as pd
import pytest

from hcsim.guidance.ingest import IngestionError, Series, ingest
from hcsim.model.core.categories import GuidanceCategory
from hcsim.synth.config import preset
from hcsim.synth.world import generate_world, write_world

ATTRITION, HIRES = GuidanceCategory.ATTRITION, GuidanceCategory.HIRES
HEADCOUNT, OPENED = GuidanceCategory.HEADCOUNT, GuidanceCategory.REQUISITIONS_OPENED


@pytest.fixture(scope="module")
def con(tmp_path_factory: pytest.TempPathFactory) -> Iterator[duckdb.DuckDBPyConnection]:
    path = tmp_path_factory.mktemp("world") / "world.duckdb"
    write_world(generate_world(preset("tiny", seed=4)), path)
    with duckdb.connect(str(path), read_only=True) as connection:
        yield connection


def raw_counts(con: duckdb.DuckDBPyConnection, code: str) -> pd.Series:
    return (
        con.execute(
            f"SELECT cost_center, count(*) n FROM workforce_txn WHERE trans_type = '{code}' "
            "AND account = 'ACCT-100' GROUP BY ALL"
        )
        .df()
        .set_index("cost_center")["n"]
    )


def test_the_grid_is_complete_for_every_group_month_and_category(
    con: duckdb.DuckDBPyConnection,
) -> None:
    series = ingest(con)
    assert len(series.groups) == 10
    assert series.frame[["month", "group"]].drop_duplicates().shape[0] == 60 * 10
    assert len(series.frame) == 60 * 10 * 4
    assert series.snapshot == pd.Period("2025-12", freq="M")


def test_categories_sum_to_the_raw_transaction_counts(con: duckdb.DuckDBPyConnection) -> None:
    series = ingest(con)
    by_group = series.frame.groupby(["group", "category"])["value"].sum()
    for code, category in (("TERM", ATTRITION), ("HIRE", HIRES), ("ACTIVE", HEADCOUNT)):
        expected = raw_counts(con, code)
        got = by_group.xs(category.value, level="category").reindex(expected.index)
        assert (got == expected).all(), code


def test_requisitions_opened_counts_first_sightings(con: duckdb.DuckDBPyConnection) -> None:
    series = ingest(con)
    (opened,) = con.execute("SELECT sum(openings) FROM req_interval").fetchone() or (0,)
    total = series.frame.loc[series.frame["category"] == OPENED.value, "value"].sum()
    assert total == opened


def test_headcount_is_the_month_end_stock(con: duckdb.DuckDBPyConnection) -> None:
    series = ingest(con)
    stock = series.values("CC-01", HEADCOUNT)
    (first,) = con.execute(
        "SELECT count(*) FROM workforce_txn WHERE trans_type='ACTIVE' AND cost_center='CC-01' "
        "AND event_date = DATE '2021-01-31' AND account = 'ACCT-100'"
    ).fetchone() or (0,)
    assert stock.iloc[0] == first
    assert stock.index.min() == pd.Period("2021-01", freq="M")


def test_coarser_groupings_preserve_totals(con: duckdb.DuckDBPyConnection) -> None:
    leaf = ingest(con).frame.groupby(["month", "category"])["value"].sum()
    for grouping in ("org_level1", "TOPLINE"):
        coarse = ingest(con, grouping)
        assert coarse.groups == (["ORG-A", "ORG-B"] if grouping == "org_level1" else ["ALL"])
        pd.testing.assert_series_equal(
            coarse.frame.groupby(["month", "category"])["value"].sum(), leaf
        )


def test_unsupported_groupings_are_refused(con: duckdb.DuckDBPyConnection) -> None:
    with pytest.raises(IngestionError, match="unsupported grouping 'job_family'"):
        ingest(con, "job_family")


def test_an_empty_warehouse_is_refused(tmp_path: Path) -> None:
    path = tmp_path / "empty.duckdb"
    world = generate_world(preset("tiny", seed=1))
    for name in ("workforce_txn", "req_interval"):
        world.tables[name] = world.tables[name].iloc[0:0]
    write_world(world, path)
    with (
        duckdb.connect(str(path), read_only=True) as connection,
        pytest.raises(IngestionError, match="no workforce or requisition data"),
    ):
        ingest(connection)


def test_series_reads_one_group_and_category(con: duckdb.DuckDBPyConnection) -> None:
    series: Series = ingest(con)
    values = series.values("CC-02", ATTRITION)
    assert len(values) == 60
    assert values.index.is_monotonic_increasing
