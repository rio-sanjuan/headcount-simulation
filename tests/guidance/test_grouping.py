"""Tests for grouping series by the column catalog's grouping columns."""

from __future__ import annotations

from collections.abc import Iterator

import duckdb
import pandas as pd
import pytest

from hcsim.guidance.ingest import GROUPINGS, IngestionError, ingest
from hcsim.guidance.sources import TABLES, FrameSource
from hcsim.model.core.aligner import TOPLINE
from hcsim.model.core.contract import CONTRACT_TABLE
from hcsim.model.sql.catalog import COLUMNS, Use
from hcsim.synth.config import preset
from hcsim.synth.world import generate_world, write_world

#: Every grouping the catalog permits, with org levels named as a world has them.
CATALOG_GROUPINGS = sorted(n for n, c in COLUMNS.items() if Use.GROUP in c.uses)


@pytest.fixture(scope="module")
def con(tmp_path_factory: pytest.TempPathFactory) -> Iterator[duckdb.DuckDBPyConnection]:
    path = tmp_path_factory.mktemp("world") / "world.duckdb"
    write_world(generate_world(preset("tiny", seed=4)), path)
    with duckdb.connect(str(path), read_only=True) as connection:
        yield connection


def tables_of(con: duckdb.DuckDBPyConnection) -> dict[str, pd.DataFrame]:
    return {name: con.execute(f"SELECT * FROM {name}").df() for name in (*TABLES, CONTRACT_TABLE)}


def totals(frame: pd.DataFrame) -> pd.Series:
    return frame.groupby(["month", "category"])["value"].sum()


def test_every_catalog_grouping_is_supported() -> None:
    assert set(CATALOG_GROUPINGS) <= set(GROUPINGS)
    assert {"org_level<k>", TOPLINE} <= set(GROUPINGS)


@pytest.mark.parametrize("grouping", [*CATALOG_GROUPINGS, "org_level1", "org_level2"])
def test_every_grouping_partitions_the_whole_population(
    con: duckdb.DuckDBPyConnection, grouping: str
) -> None:
    whole = ingest(con, TOPLINE)
    grouped = ingest(con, grouping)
    assert grouped.ungrouped == 0.0
    assert grouped.snapshot == whole.snapshot
    assert len(grouped.groups) >= 1
    pd.testing.assert_series_equal(totals(grouped.frame), totals(whole.frame))


def test_groups_are_labels(con: duckdb.DuckDBPyConnection) -> None:
    levels = ingest(con, "job_level").groups
    expected = con.execute("SELECT DISTINCT job_level FROM workforce_agg").fetchall()
    assert set(levels) == {str(row[0]) for row in expected}
    assert all(isinstance(level, str) for level in levels)
    assert set(ingest(con, "region").groups) <= set(
        con.execute("SELECT region FROM dim_location").df()["region"]
    )


def test_rows_no_group_takes_are_counted_and_left_out(con: duckdb.DuckDBPyConnection) -> None:
    tables = tables_of(con)
    org = tables["dim_org"]
    dropped = org["cost_center"].iloc[0]
    tables["dim_org"] = org.iloc[1:]
    assert (tables["workforce_agg"]["cost_center"] == dropped).any()
    with FrameSource(tables).connect() as connection:
        levels = ingest(connection, "org_level2")
        whole = ingest(connection, TOPLINE)
    assert levels.ungrouped > 0
    every = totals(whole.frame)
    gap = every - totals(levels.frame).reindex(every.index, fill_value=0.0)
    assert 0 < float(gap.abs().sum()) <= levels.ungrouped + 1e-9


def test_a_grouping_that_takes_no_row_is_refused(con: duckdb.DuckDBPyConnection) -> None:
    tables = tables_of(con)
    tables["dim_location"] = tables["dim_location"].iloc[0:0]
    with (
        FrameSource(tables).connect() as connection,
        pytest.raises(IngestionError, match="no row falls in any 'country' group"),
    ):
        ingest(connection, "country")


def test_a_job_with_two_tech_indicators_can_t_group_by_it(
    con: duckdb.DuckDBPyConnection,
) -> None:
    tables = tables_of(con)
    jobs = tables["dim_job"]
    twin = jobs.iloc[[0]].assign(job_code="TWIN-1", tech_indicator="other")
    tables["dim_job"] = pd.concat([jobs, twin], ignore_index=True)
    with (
        FrameSource(tables).connect() as connection,
        pytest.raises(IngestionError, match="more than one tech indicator"),
    ):
        ingest(connection, "tech_indicator")
