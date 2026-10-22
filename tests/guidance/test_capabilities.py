"""Tests for the models' capability declarations, and the covariates they read."""

from __future__ import annotations

from collections.abc import Iterator

import duckdb
import numpy as np
import pandas as pd
import pytest

from hcsim.guidance.adapters.base import LibraryModel
from hcsim.guidance.capabilities import Capabilities, CovariateModel, declared
from hcsim.guidance.config import GuidanceConfig
from hcsim.guidance.covariates import CALENDAR_COLUMNS, calendar
from hcsim.guidance.errors import ForecastError
from hcsim.guidance.models import MODELS, capabilities
from hcsim.guidance.pipeline import run
from hcsim.guidance.selection import prepared, wape
from hcsim.model.interface import ProbabilisticOutput
from hcsim.synth.config import preset
from hcsim.synth.world import generate_world, write_world

Q = ProbabilisticOutput
COVARIATE_AWARE = {
    "statsforecast.AutoARIMA",
    "darts.LightGBMModel",
    "darts.LinearRegressionModel",
    "sktime.ReductionLinear",
    "sktime.ReductionBoosted",
    "prophet.Prophet",
}


def test_every_model_declares_its_capabilities_without_being_built() -> None:
    for name in MODELS.names:
        declared_ = capabilities(name)
        assert Q.NONE in declared_.outputs, name
        assert set(declared_.covariates) <= set(CALENDAR_COLUMNS), name
    aware = {n for n in MODELS.names if capabilities(n).covariates}
    assert aware == COVARIATE_AWARE
    assert {n for n in MODELS.names if capabilities(n).exposure} == {"native.LogitRateARIMA"}
    with pytest.raises(ForecastError, match="unknown model"):
        capabilities("native.Nothing")


def test_darts_and_sktime_declare_their_native_uncertainty() -> None:
    assert Q.QUANTILES in capabilities("darts.TFTModel").outputs
    assert Q.SAMPLE_PATHS in capabilities("darts.ExponentialSmoothing").outputs
    assert Q.QUANTILES in capabilities("sktime.ThetaForecaster").outputs
    assert Q.QUANTILES in capabilities("statsforecast.AutoETS").outputs
    assert Q.PARAMETRIC in capabilities("native.NegativeBinomialGLM").outputs
    assert not capabilities("native.SeasonalNaive").probabilistic
    assert capabilities("statsforecast.AutoETS").probabilistic


def test_a_capability_record_reads_as_plain_data() -> None:
    record = Capabilities(frozenset({Q.QUANTILES, Q.NONE}), ("weekdays",), False).as_record()
    assert record == {
        "outputs": ["none", "quantiles"],
        "covariates": ["weekdays"],
        "exposure": False,
    }
    assert declared(object) == Capabilities(frozenset({Q.NONE}), (), False)


# --- covariates ---------------------------------------------------------------------------------

MONTHS = pd.period_range("2020-01", periods=60, freq="M")
CALENDAR = calendar(MONTHS)
#: A count driven by the working days in its month.
DRIVEN = (
    3.0 * CALENDAR["weekdays"].to_numpy() + np.random.default_rng(2).normal(0, 0.5, 60)
).astype(np.float64)


def test_a_covariate_aware_model_is_given_its_columns_for_the_history_and_horizon() -> None:
    model = prepared("statsforecast.AutoARIMA", None, None, CALENDAR, months=48, horizon=12)
    assert isinstance(model, CovariateModel)
    assert isinstance(model, LibraryModel)
    past, future = model.past_covariates, model.future_covariates
    assert past is not None
    assert future is not None
    assert (past.shape, future.shape) == ((48, 1), (12, 1))
    assert np.array_equal(future[:, 0], CALENDAR["weekdays"].to_numpy()[48:])
    short = prepared("statsforecast.AutoARIMA", None, None, CALENDAR, months=55, horizon=12)
    assert isinstance(short, LibraryModel)
    assert short.future_covariates is None  # the calendar ends before the horizon does
    with pytest.raises(ForecastError, match="needs its covariates for the 12 months ahead"):
        short.fit(DRIVEN[:55]).predict(12)
    plain = prepared("native.SeasonalNaive", None, None, CALENDAR, months=48, horizon=12)
    assert not hasattr(plain, "past_covariates")


def test_working_days_improve_a_forecast_they_drive() -> None:
    history, future = DRIVEN[:48], DRIVEN[48:]
    aware = prepared("statsforecast.AutoARIMA", None, None, CALENDAR, months=48, horizon=12)
    blind = prepared("statsforecast.AutoARIMA", None, None, None)
    with_days = wape(future, aware.fit(history).predict(12))
    without = wape(future, blind.fit(history).predict(12))
    assert with_days < 0.05
    assert with_days < without


@pytest.mark.extras
@pytest.mark.parametrize(
    "name", ["darts.LinearRegressionModel", "sktime.ReductionLinear", "prophet.Prophet"]
)
def test_each_library_s_covariate_models_read_the_working_days(name: str) -> None:
    pytest.importorskip(name.split(".")[0])
    model = prepared(name, None, None, CALENDAR, months=48, horizon=12)
    forecast = model.fit(DRIVEN[:48]).predict(12)
    assert wape(DRIVEN[48:], forecast) < 0.1, name


@pytest.fixture(scope="module")
def con(tmp_path_factory: pytest.TempPathFactory) -> Iterator[duckdb.DuckDBPyConnection]:
    path = tmp_path_factory.mktemp("world") / "world.duckdb"
    write_world(generate_world(preset("tiny", seed=4)), path)
    with duckdb.connect(str(path), read_only=True) as connection:
        yield connection


def test_a_run_reports_its_suite_s_capabilities_and_feeds_the_calendar(
    con: duckdb.DuckDBPyConnection,
) -> None:
    config = GuidanceConfig.from_mapping(
        {
            "horizon": 6,
            "grouping": "TOPLINE",
            "models": ["native.SeasonalNaive", "statsforecast.AutoARIMA"],
        }
    )
    report = run(con, config).report
    assert report["capabilities"]["statsforecast.AutoARIMA"]["covariates"] == ["weekdays"]
    assert report["capabilities"]["native.SeasonalNaive"]["outputs"] == ["none"]
    scores = report["selection"]["ALL/attrition"]["scores"]
    assert np.isfinite(scores["statsforecast.AutoARIMA"])
