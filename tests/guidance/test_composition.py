"""Tests for composing categories before fitting, and for the calendar covariates."""

from __future__ import annotations

from collections.abc import Iterator

import duckdb
import numpy as np
import pandas as pd
import pytest

from hcsim.guidance.config import GuidanceConfig, GuidanceConfigError
from hcsim.guidance.covariates import CALENDAR_COLUMNS, calendar, covariates
from hcsim.guidance.framework import Context
from hcsim.guidance.ingest import DEFAULT_CATEGORIES, MONTH, counted, ingest
from hcsim.guidance.pipeline import pipeline, run
from hcsim.model.core.categories import GuidanceCategory
from hcsim.model.core.contract import role_column
from hcsim.model.core.measures import CATEGORY_MEASURES
from hcsim.synth.config import preset
from hcsim.synth.world import generate_world, write_world

C = GuidanceCategory
FILL_NODES = (C.FILLS, C.HIRES, C.EXTERNAL_HIRES, C.CONVERSION_HIRES, C.INTERNAL_STARTS)


@pytest.fixture(scope="module")
def con(tmp_path_factory: pytest.TempPathFactory) -> Iterator[duckdb.DuckDBPyConnection]:
    path = tmp_path_factory.mktemp("world") / "world.duckdb"
    write_world(generate_world(preset("tiny", seed=4)), path)
    with duckdb.connect(str(path), read_only=True) as connection:
        yield connection


def test_an_inner_node_is_counted_as_its_leaves() -> None:
    assert counted([C.HIRES]) == [C.EXTERNAL_HIRES, C.CONVERSION_HIRES]
    assert counted([C.FILLS, C.HIRES, C.ATTRITION]) == [
        C.EXTERNAL_HIRES,
        C.CONVERSION_HIRES,
        C.INTERNAL_STARTS,
        C.ATTRITION,
    ]
    assert counted(DEFAULT_CATEGORIES) == list(DEFAULT_CATEGORIES)


def by_month(con: duckdb.DuckDBPyConnection, category: GuidanceCategory) -> pd.Series:
    """`category`'s own measure, counted alone and summed per month."""
    measure = CATEGORY_MEASURES[category]
    frame = con.execute(measure.sql(role_column(con, measure.table))).df()
    frame[MONTH] = pd.to_datetime(frame["month"]).dt.to_period("M")
    return frame.groupby(MONTH)["value"].sum().astype(float)


def test_a_composed_node_is_its_leaves_sum_and_matches_its_own_measure(
    con: duckdb.DuckDBPyConnection,
) -> None:
    series = ingest(con, "TOPLINE", FILL_NODES)
    assert set(series.categories) == set(FILL_NODES)
    wide = series.frame.pivot(index=MONTH, columns="category", values="value")
    hires = wide["external_hires"] + wide["conversion_hires"]
    assert np.array_equal(wide["hires"], hires)
    assert np.array_equal(wide["fills"], hires + wide["internal_starts"])
    assert wide["hires"].sum() > wide["external_hires"].sum() > 0
    # Composing the leaves counts what the node's own measure counts.
    for node in (C.HIRES, C.FILLS):
        own = by_month(con, node).reindex(wide.index, fill_value=0.0)
        assert np.array_equal(own.to_numpy(), wide[node.value].to_numpy()), node


def test_only_the_requested_categories_are_kept(con: duckdb.DuckDBPyConnection) -> None:
    series = ingest(con, "TOPLINE", (C.HEADCOUNT, C.HIRES))
    assert series.categories == [C.HEADCOUNT, C.HIRES]
    assert set(series.frame["category"]) == {"headcount", "hires"}


def test_a_configuration_s_categories_are_checked_at_load() -> None:
    config = GuidanceConfig.from_mapping({"categories": ["headcount", "hires", "fills"]})
    assert config.categories == (C.HEADCOUNT, C.HIRES, C.FILLS)
    with pytest.raises(GuidanceConfigError) as error:
        GuidanceConfig.from_mapping({"categories": ["hires", "hires"]})
    message = str(error.value)
    assert "categories: each category is named once; ['hires'] repeat" in message
    assert "reconciliation holds the 'headcount' anchor fixed" in message
    assert GuidanceConfig.from_mapping(
        {"categories": ["hires"], "reconciliation": {"method": None}}
    ).categories == (C.HIRES,)
    with pytest.raises(GuidanceConfigError, match=r"categories\.0"):
        GuidanceConfig.from_mapping({"categories": ["nonsense"]})


def test_a_run_forecasts_composed_categories(con: duckdb.DuckDBPyConnection) -> None:
    config = GuidanceConfig.from_mapping(
        {"horizon": 3, "grouping": "TOPLINE", "categories": ["headcount", "attrition", "hires"]}
    )
    result = run(con, config)
    assert result.artifact.categories == frozenset({C.HEADCOUNT, C.ATTRITION, C.HIRES})
    ingested = [s for s in result.report["steps"] if s["step"] == "ingest"]
    assert ingested[0]["gave"] == ["series", "covariates"]


# --- calendar covariates ------------------------------------------------------------------------


def test_each_month_s_calendar() -> None:
    frame = calendar(pd.period_range("2024-01", "2024-04", freq="M"))
    assert tuple(frame.columns) == CALENDAR_COLUMNS
    assert frame["month_of_year"].tolist() == [1, 2, 3, 4]
    assert frame["quarter"].tolist() == [1, 1, 1, 2]
    assert frame["days"].tolist() == [31, 29, 31, 30]  # a leap February
    assert frame["weekdays"].tolist() == [23, 21, 21, 22]
    harmonics = frame[["sin1", "cos1", "sin2", "cos2"]].to_numpy()
    assert np.allclose(harmonics[0], [0, 1, 0, 1])  # January starts the cycle
    assert np.allclose(harmonics[3], [1, 0, 0, -1])  # April is a quarter of the way around
    with pytest.raises(ValueError, match="monthly"):
        calendar(pd.period_range("2024-01-01", periods=3, freq="D"))


def test_covariates_span_the_history_and_the_horizon(con: duckdb.DuckDBPyConnection) -> None:
    series = ingest(con, "TOPLINE")
    frame = covariates(series, 12)
    history = series.frame[MONTH].nunique()
    assert len(frame) == history + 12
    assert int(frame["future"].sum()) == 12
    assert frame.index[0] == series.frame[MONTH].min()
    assert frame.index[-1] == series.snapshot + 12
    assert not frame.loc[: series.snapshot, "future"].any()
    assert len(covariates(series, 0)) == history


def test_the_ingest_step_gives_the_covariates(con: duckdb.DuckDBPyConnection) -> None:
    config = GuidanceConfig(horizon=6, grouping="TOPLINE")
    ctx = pipeline(config).run(Context({"source": con, "config": config}))
    assert int(ctx["covariates"]["future"].sum()) == 6
