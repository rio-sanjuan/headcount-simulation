"""Tests for validation: the snapshot rule, interior gaps, all-zero series, and month ends."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import duckdb
import pandas as pd
import pytest

from hcsim.guidance.config import GuidanceConfig, GuidanceConfigError
from hcsim.guidance.framework import PipelineError
from hcsim.guidance.ingest import CATEGORY, GROUP, MONTH, VALUE, Series, problems
from hcsim.guidance.pipeline import run
from hcsim.guidance.sources import TABLES, FrameSource
from hcsim.guidance.validation import gaps, validate
from hcsim.guidance.validation import problems as validation_problems
from hcsim.model.core.categories import GuidanceCategory
from hcsim.model.core.contract import CONTRACT_TABLE
from hcsim.synth.config import preset
from hcsim.synth.world import generate_world, write_world


def series(values: dict[tuple[str, str], list[float]], observed: list[str] | None = None) -> Series:
    """Hand-made series from January 2024, observed every month unless `observed` says."""
    n = len(next(iter(values.values())))
    months = pd.period_range("2024-01", periods=n, freq="M")
    rows = [
        {MONTH: m, GROUP: g, CATEGORY: c, VALUE: v}
        for (g, c), vs in values.items()
        for m, v in zip(months, vs, strict=True)
    ]
    seen = months if observed is None else pd.PeriodIndex(observed, freq="M")
    return Series(pd.DataFrame(rows), "cost_center", months[-1], tuple(seen))


TWO = {("A", "attrition"): [1.0, 2.0, 0.0, 3.0], ("B", "attrition"): [0.0, 0.0, 0.0, 0.0]}


def test_a_fixed_snapshot_cuts_the_history_there() -> None:
    settled = validate(series(TWO), "2024-02")
    assert settled.series.snapshot == pd.Period("2024-02", "M")
    assert settled.series.frame[MONTH].max() == pd.Period("2024-02", "M")
    assert settled.series.values("A", GuidanceCategory.ATTRITION).tolist() == [1.0, 2.0]


def test_a_snapshot_past_or_before_the_data_is_refused() -> None:
    assert validation_problems(series(TWO), "2024-05", "refuse") == [
        "the snapshot 2024-05 is past the data, which ends at 2024-04"
    ]
    assert validation_problems(series(TWO), "2023-12", "refuse") == [
        "the snapshot 2023-12 is before the data, which starts at 2024-01"
    ]
    assert validation_problems(series(TWO), None, "refuse") == []


def test_an_interior_gap_is_refused_unless_it_s_declared_zeros() -> None:
    gappy = series(TWO, observed=["2024-01", "2024-02", "2024-04"])
    assert gaps(gappy, gappy.snapshot) == (pd.Period("2024-03", "M"),)
    (problem,) = validation_problems(gappy, None, "refuse")
    assert problem.startswith("1 month(s) inside the history have no rows at all: 2024-03.")
    assert validation_problems(gappy, None, "zero") == []
    settled = validate(gappy, None)
    assert settled.gaps == (pd.Period("2024-03", "M"),)
    assert settled.series.values("A", GuidanceCategory.ATTRITION).tolist()[2] == 0.0
    # A gap after a fixed snapshot is outside the history, so it doesn't count.
    assert validation_problems(gappy, "2024-02", "refuse") == []


def test_an_all_zero_series_is_dropped_and_named() -> None:
    settled = validate(series(TWO), None)
    assert settled.dropped == (("B", "attrition"),)
    assert settled.series.keys == [("A", GuidanceCategory.ATTRITION)]
    kept = validate(series(TWO), None, drop_all_zero=False)
    assert kept.dropped == ()
    assert len(kept.series.keys) == 2
    # A series zero only up to a fixed snapshot is zero for the whole history it keeps.
    assert validate(series(TWO), "2024-02").dropped == (("B", "attrition"),)


def test_the_validation_settings_are_checked_at_load() -> None:
    config = GuidanceConfig.from_mapping({"validation": {"snapshot": "2025-06", "gaps": "zero"}})
    assert (config.validation.snapshot, config.validation.gaps) == ("2025-06", "zero")
    with pytest.raises(GuidanceConfigError) as error:
        GuidanceConfig.from_mapping({"validation": {"snapshot": "2025-13", "gaps": "fill"}})
    assert "validation.snapshot" in str(error.value)
    assert "validation.gaps" in str(error.value)


# --- on a world ---------------------------------------------------------------------------------


@pytest.fixture(scope="module")
def con(tmp_path_factory: pytest.TempPathFactory) -> Iterator[duckdb.DuckDBPyConnection]:
    path = tmp_path_factory.mktemp("world") / "world.duckdb"
    write_world(generate_world(preset("tiny", seed=4)), path)
    with duckdb.connect(str(path), read_only=True) as connection:
        yield connection


def test_a_run_reports_its_validation_and_a_fixed_snapshot_moves_the_artifact(
    con: duckdb.DuckDBPyConnection,
) -> None:
    default = run(con, GuidanceConfig(horizon=3))
    assert default.report["snapshot"] == "2025-12"
    fixed = run(
        con, GuidanceConfig.from_mapping({"horizon": 3, "validation": {"snapshot": "2025-06"}})
    )
    assert fixed.artifact.snapshot_date == "2025-06-30"
    assert fixed.report["snapshot"] == "2025-06"
    assert fixed.report["validation"] == {"gaps": [], "dropped": []}
    assert default.artifact.snapshot_date == "2025-12-31"
    assert fixed.artifact.fingerprint() != default.artifact.fingerprint()


def test_a_period_that_isn_t_a_month_end_is_refused(
    con: duckdb.DuckDBPyConnection, tmp_path: Path
) -> None:
    tables = {name: con.execute(f"SELECT * FROM {name}").df() for name in (*TABLES, CONTRACT_TABLE)}
    fact = tables["workforce_agg"]
    fact.loc[fact.index[:3], "period"] = pd.Timestamp("2024-05-15")
    with FrameSource(tables).connect() as connection:
        (problem,) = problems(connection)
    assert problem.startswith("3 rows of workforce_agg have a period that isn't a month end")
    assert "2024-05-15" in problem
    with pytest.raises(PipelineError, match="isn't a month end") as error:
        run(FrameSource(tables))
    assert error.value.step == "ingest"
