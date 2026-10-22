"""Tests for the pipeline's sources and the consolidated measure query."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import duckdb
import pandas as pd
import pytest

from hcsim.guidance.config import GuidanceConfig
from hcsim.guidance.framework import PipelineError
from hcsim.guidance.ingest import ingest, problems
from hcsim.guidance.pipeline import run
from hcsim.guidance.sources import (
    TABLES,
    DuckDBSource,
    FileSource,
    FrameSource,
    SourceError,
    as_source,
    missing_tables,
)
from hcsim.model.core.contract import CONTRACT_TABLE, role_column
from hcsim.model.core.measures import CATEGORY_MEASURES, KEYS, MeasureError, consolidated_sql
from hcsim.synth.config import preset
from hcsim.synth.export import export_parquet
from hcsim.synth.world import generate_world, write_world


@pytest.fixture(scope="module")
def world(tmp_path_factory: pytest.TempPathFactory) -> Path:
    path = tmp_path_factory.mktemp("world") / "world.duckdb"
    write_world(generate_world(preset("tiny", seed=4)), path)
    return path


@pytest.fixture(scope="module")
def con(world: Path) -> Iterator[duckdb.DuckDBPyConnection]:
    with duckdb.connect(str(world), read_only=True) as connection:
        yield connection


@pytest.fixture(scope="module")
def exported(
    world: Path, con: duckdb.DuckDBPyConnection, tmp_path_factory: pytest.TempPathFactory
) -> Path:
    directory = tmp_path_factory.mktemp("export") / "tables"
    export_parquet(con, directory)
    return directory


def tables_of(con: duckdb.DuckDBPyConnection) -> dict[str, pd.DataFrame]:
    names = [*TABLES, CONTRACT_TABLE]
    return {name: con.execute(f"SELECT * FROM {name}").df() for name in names}


# --- the consolidated query ---------------------------------------------------------------------


def long_form(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.assign(month=pd.to_datetime(frame["month"]), value=frame["value"].astype(float))
    return frame.sort_values(["measure", "month", *KEYS], ignore_index=True)


def test_one_query_counts_what_each_measure_counts_on_its_own(
    con: duckdb.DuckDBPyConnection,
) -> None:
    measures = list(CATEGORY_MEASURES.values())
    columns = {m.table: role_column(con, m.table) for m in measures}
    together = long_form(con.execute(consolidated_sql(measures, columns)).df())
    alone = []
    for m in measures:
        frame = con.execute(m.sql(columns[m.table])).df().assign(measure=m.name)
        alone.append(frame[["month", *KEYS, "measure", "value"]])
    expected = long_form(pd.concat(alone, ignore_index=True))
    # One pass counts every measure for each month and cost center any code touched, so it
    # carries zeros a measure's own query leaves out.
    nonzero = together[together["value"] != 0].reset_index(drop=True)
    pd.testing.assert_frame_equal(nonzero, expected[expected["value"] != 0].reset_index(drop=True))
    assert set(together["measure"]) == {m.name for m in measures}


def test_the_query_refuses_no_measures_or_a_measure_twice() -> None:
    with pytest.raises(MeasureError, match="no measures"):
        consolidated_sql([])
    attrition = CATEGORY_MEASURES[next(iter(CATEGORY_MEASURES))]
    with pytest.raises(MeasureError, match="repeat"):
        consolidated_sql([attrition, attrition])


# --- sources ------------------------------------------------------------------------------------


def series_of(source: object) -> pd.DataFrame:
    with as_source(source).connect() as con:
        return ingest(con).frame


def test_every_kind_of_source_yields_the_same_series(
    world: Path, con: duckdb.DuckDBPyConnection, exported: Path
) -> None:
    expected = series_of(con)
    for source in (
        DuckDBSource(world),
        world,  # a path to a file
        FileSource(exported),
        exported,  # a path to a directory
        FrameSource(tables_of(con)),
    ):
        pd.testing.assert_frame_equal(series_of(source), expected)


def test_the_pipeline_runs_on_any_source(world: Path, exported: Path) -> None:
    config = GuidanceConfig(horizon=3)
    from_file = run(world, config)
    from_export = run(FileSource(exported), config)
    assert from_export.artifact.fingerprint() == from_file.artifact.fingerprint()
    assert from_export.report["steps"][0]["step"] == "ingest"


def test_a_source_missing_tables_is_refused_naming_them(con: duckdb.DuckDBPyConnection) -> None:
    partial = {k: v for k, v in tables_of(con).items() if k not in ("dim_org", "req_interval")}
    with FrameSource(partial).connect() as connection:
        assert missing_tables(connection) == ["req_interval", "dim_org"]
        assert problems(connection) == [
            "the source has no req_interval, dim_org: ingestion reads " + ", ".join(TABLES)
        ]
    with pytest.raises(PipelineError, match="the source has no req_interval, dim_org"):
        run(FrameSource(partial))


def test_a_file_source_reads_csv_and_refuses_two_files_for_a_table(
    con: duckdb.DuckDBPyConnection, exported: Path, tmp_path: Path
) -> None:
    for name, frame in tables_of(con).items():
        frame.to_csv(tmp_path / f"{name}.csv", index=False)
    csv = series_of(FileSource(tmp_path))
    pd.testing.assert_frame_equal(csv, series_of(exported))
    (tmp_path / "dim_org.parquet").write_bytes((exported / "dim_org.parquet").read_bytes())
    with pytest.raises(SourceError, match="two files for the table 'dim_org'"):
        FileSource(tmp_path).files()
    with pytest.raises(SourceError, match="no directory"):
        FileSource(tmp_path / "missing").files()


def test_a_duckdb_source_takes_a_path_or_a_connection(
    con: duckdb.DuckDBPyConnection, tmp_path: Path
) -> None:
    with pytest.raises(SourceError, match="a path or a connection"):
        DuckDBSource()
    with pytest.raises(SourceError, match="a path or a connection"):
        DuckDBSource(tmp_path / "w.duckdb", con)
    with pytest.raises(SourceError, match="no DuckDB file"), DuckDBSource(tmp_path / "w").connect():
        pass
    with DuckDBSource(connection=con).connect() as same:
        assert same is con
    assert DuckDBSource(connection=con).description == "duckdb:connection"
    with pytest.raises(SourceError, match="not int"):
        as_source(3)
