"""Tests for the guidance study ladder's arms and runs."""

from __future__ import annotations

import dataclasses
from collections.abc import Iterator

import duckdb
import pandas as pd
import pytest

from hcsim.guidance.ingest import ingest
from hcsim.guidance.ladder import (
    ARMS,
    ARMS_PER_SERIES,
    SERIES,
    Arm,
    LadderError,
    LadderRun,
    config_for,
    problems,
    run_ladder,
)
from hcsim.model.core.categories import GuidanceCategory
from hcsim.synth.config import preset
from hcsim.synth.world import generate_world, write_world

C = GuidanceCategory
CUTOFF = pd.Period("2024-12", "M")


def test_each_series_has_six_arms_from_its_null_to_an_ensemble() -> None:
    assert problems(ARMS) == []
    assert [s.value for s in SERIES] == ["attrition", "hires", "internal_starts"]
    for series in SERIES:
        arms = sorted((a for a in ARMS if a.series is series), key=lambda a: a.simplicity)
        assert len(arms) == ARMS_PER_SERIES
        assert arms[0].null
        assert arms[0].models == ("native.SeasonalNaive",)
        assert not any(a.null for a in arms[1:])
        ensemble = arms[-1]
        assert ensemble.name == "ensemble"
        assert set(ensemble.models) == {m for a in arms[:-1] for m in a.models}
    own = {a.series: a.models for a in ARMS if a.simplicity == 4}
    assert own == {
        C.ATTRITION: ("native.PoissonGLM",),
        C.HIRES: ("native.NegativeBinomialGLM",),
        C.INTERNAL_STARTS: ("statsforecast.CrostonSBA",),
    }


def test_broken_arms_are_named_all_at_once() -> None:
    null = Arm("seasonal_naive", C.ATTRITION, 0, ("native.SeasonalNaive",), null=True)
    arms = [
        null,
        dataclasses.replace(null, simplicity=1),  # a repeated name, and a second null
        Arm("rich", C.ATTRITION, 1, ("native.Nope",)),  # a shared rank, and an unknown model
        Arm("bare", C.ATTRITION, 2, ()),
        Arm("first", C.HIRES, 0, ("native.Drift",)),
        Arm("null", C.HIRES, 1, ("native.SeasonalNaive",), null=True),
    ]
    found = problems(arms)
    assert "arm names repeat within a series: ['attrition/seasonal_naive']" in found
    assert "attrition needs exactly one null arm, not 2" in found
    assert "attrition's arms share simplicity ranks: [0, 1, 1, 2]" in found
    assert any(p.startswith("attrition/rich: ") and "native.Nope" in p for p in found)
    assert "attrition/bare: an arm needs at least one model" in found
    assert "hires's null arm, null, isn't its simplest" in found


def test_an_arm_is_a_guidance_configuration_fitted_through_the_cutoff() -> None:
    arm = next(a for a in ARMS if a.series is C.HIRES and a.name == "ensemble")
    config = config_for(arm, CUTOFF, "org_level1")
    assert config.validation.snapshot == "2024-12"
    assert config.categories == (C.HIRES, C.HEADCOUNT)
    assert config.models == arm.models
    assert config.combination.top_k == 3
    assert config.reconciliation.method is None
    null = next(a for a in ARMS if a.series is C.HIRES and a.null)
    assert config_for(null, CUTOFF).combination.top_k == 1


# --- a run --------------------------------------------------------------------------------------


@pytest.fixture(scope="module")
def con(tmp_path_factory: pytest.TempPathFactory) -> Iterator[duckdb.DuckDBPyConnection]:
    path = tmp_path_factory.mktemp("world") / "world.duckdb"
    write_world(generate_world(preset("tiny", seed=4)), path)
    with duckdb.connect(str(path), read_only=True) as connection:
        yield connection


@pytest.fixture(scope="module")
def ladder(con: duckdb.DuckDBPyConnection) -> LadderRun:
    return run_ladder(con, CUTOFF)


def test_every_arm_forecasts_the_year_after_the_cutoff(ladder: LadderRun) -> None:
    frame = ladder.frame()
    assert len(ladder.runs) == len(ARMS)
    months = pd.period_range("2025-01", "2025-12", freq="M")
    for run in ladder.runs:
        assert sorted(run.frame["month"].unique()) == list(months)
        assert (run.frame["forecast"] >= 0).all()
    assert set(frame.groupby(["series", "arm"]).size()) == {12 * frame["group"].nunique()}


def test_the_actuals_are_the_guidance_measures_of_those_months(
    con: duckdb.DuckDBPyConnection, ladder: LadderRun
) -> None:
    series = ingest(con, "org_level1", list(SERIES)).frame
    year = series[(series["month"] > CUTOFF) & (series["month"] <= CUTOFF + 12)]
    expected = year.groupby("category")["value"].sum()
    frame = ladder.frame()
    actual = frame[frame["arm"] == "seasonal_naive"].groupby("series")["actual"].sum()
    pd.testing.assert_series_equal(actual, expected, check_names=False)


def test_the_arms_forecast_differently(ladder: LadderRun) -> None:
    frame = ladder.frame()
    hires = frame[frame["series"] == "hires"].pivot_table(
        index=["group", "month"], columns="arm", values="forecast"
    )
    assert not hires["seasonal_naive"].equals(hires["exponential_smoothing"])
    assert all(run.unforecast == 0 for run in ladder.runs)


def test_a_cutoff_without_a_year_of_actuals_is_refused(con: duckdb.DuckDBPyConnection) -> None:
    with pytest.raises(LadderError, match="needs actuals through 2026-06"):
        run_ladder(con, "2025-06")
    with pytest.raises(LadderError, match="exactly one null arm"):
        run_ladder(con, CUTOFF, arms=[dataclasses.replace(ARMS[0], null=False)])
