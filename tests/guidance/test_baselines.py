"""Tests for the fixed-form baselines, exposures in selection, and the Prophet adapter."""

from __future__ import annotations

import importlib.util
from collections.abc import Iterator

import duckdb
import numpy as np
import pytest

from hcsim.guidance.baselines import (
    BASELINES,
    LogCountARIMA,
    LogitRateARIMA,
    NegativeBinomialGLM,
    PoissonGLM,
)
from hcsim.guidance.capabilities import ExposureModel
from hcsim.guidance.config import GuidanceConfig
from hcsim.guidance.errors import ForecastError
from hcsim.guidance.models import MODELS, build
from hcsim.guidance.pipeline import run
from hcsim.guidance.selection import EvaluationConfig, SelectionConfig, forecast, score
from hcsim.model.interface import ProbabilisticOutput, TimeSeriesModel
from hcsim.synth.config import preset
from hcsim.synth.world import generate_world, write_world

MONTHS = np.arange(60)
#: A Poisson count with a known level, trend, and season.
TRUTH = np.exp(3.0 + 0.05 * MONTHS / 12 + 0.3 * np.sin(2 * np.pi * MONTHS / 12))
COUNTS = np.random.default_rng(5).poisson(TRUTH).astype(np.float64)


def test_every_baseline_is_registered_and_a_time_series_model() -> None:
    for model in BASELINES:
        assert model.NAME in MODELS
        assert isinstance(model(), TimeSeriesModel)
    assert isinstance(LogitRateARIMA(), ExposureModel)
    assert not isinstance(LogCountARIMA(), ExposureModel)
    assert ProbabilisticOutput.PARAMETRIC in NegativeBinomialGLM.OUTPUTS


def test_log_count_arima_forecasts_counts_near_their_level() -> None:
    forecast = LogCountARIMA().fit(COUNTS).predict(12)
    assert (forecast >= 0).all()
    assert abs(forecast.mean() - TRUTH[-12:].mean()) < 0.2 * TRUTH.mean()
    with pytest.raises(ForecastError, match="can't be negative"):
        LogCountARIMA().fit(np.r_[COUNTS[:-1], -1.0])
    with pytest.raises(ForecastError, match="needs at least 24 months"):
        LogCountARIMA().fit(COUNTS[:20])


def test_logit_rate_arima_forecasts_the_share_of_the_exposure() -> None:
    exposure = 1000 + 10 * MONTHS.astype(np.float64)
    flow = np.round(0.02 * exposure)
    future = 1600 + 10 * np.arange(6, dtype=np.float64)
    model = LogitRateARIMA()
    model.expose(exposure, future)
    forecast = model.fit(flow).predict(6)
    assert np.allclose(forecast / future, 0.02, atol=0.002)


def test_logit_rate_arima_needs_its_exposure() -> None:
    with pytest.raises(ForecastError, match="needs the series' exposure"):
        LogitRateARIMA().fit(COUNTS)
    model = LogitRateARIMA()
    model.expose(np.ones(10), None)
    with pytest.raises(ForecastError, match="one exposure value per month"):
        model.fit(COUNTS)
    model.expose(np.full(60, 500.0), None)
    with pytest.raises(ForecastError, match="exposure's forecast for 3 months"):
        model.fit(COUNTS).predict(3)
    model.expose(np.r_[np.full(59, 500.0), 0.0], np.ones(3))
    with pytest.raises(ForecastError, match="positive exposure"):
        model.fit(COUNTS)


def harmonic(months: np.ndarray, level: float) -> np.ndarray:
    return np.exp(level + 0.05 * months / 12 + 0.3 * np.sin(2 * np.pi * months / 12))


def test_the_glms_recover_a_harmonic_count_process() -> None:
    truth = harmonic(MONTHS, 5.3)  # about 200 a month, so five years pin the shape down
    counts = np.random.default_rng(5).poisson(truth).astype(np.float64)
    future = harmonic(np.arange(60, 72), 5.3)
    for model in (PoissonGLM(), NegativeBinomialGLM()):
        assert np.allclose(model.fit(counts).predict(12), future, rtol=0.05), model.NAME
    assert NegativeBinomialGLM().fit(counts).dispersion < 0.005  # Poisson: no overdispersion
    lumpy = np.random.default_rng(6).negative_binomial(4, 4 / (4 + truth)).astype(np.float64)
    assert NegativeBinomialGLM().fit(lumpy).dispersion > 0.1
    with pytest.raises(ForecastError, match="zero throughout"):
        PoissonGLM().fit(np.zeros(36))


# --- exposures in selection ---------------------------------------------------------------------

EVALUATION = EvaluationConfig(n_splits=3, test_months=6, step_months=6, min_train_months=24)


def test_a_share_model_scores_only_with_an_exposure() -> None:
    exposure = 1000 + 10 * MONTHS.astype(np.float64)
    flow = np.round(0.02 * exposure)
    assert score("native.LogitRateARIMA", flow, EVALUATION) == float("inf")
    assert score("native.LogitRateARIMA", flow, EVALUATION, exposure) < 0.05


def test_a_selected_model_that_can_t_refit_leaves_the_combination() -> None:
    exposure = 1000 + 10 * MONTHS.astype(np.float64)
    flow = np.round(0.02 * exposure)
    config = SelectionConfig(
        suite=("native.LogitRateARIMA", "native.HistoricalAverage"), top_k=2, evaluation=EVALUATION
    )
    both = forecast(flow, 6, config, exposure, 1600 + 10 * np.arange(6.0))
    assert set(both.weights) == {"native.LogitRateARIMA", "native.HistoricalAverage"}
    # No forecast of the exposure: the share model can't refit, and the other takes its weight.
    alone = forecast(flow, 6, config, exposure, None)
    assert alone.weights == {"native.HistoricalAverage": 1.0}
    assert np.allclose(alone.values, flow.mean())
    with pytest.raises(ForecastError, match="could refit"):
        forecast(
            flow,
            6,
            SelectionConfig(suite=("native.LogitRateARIMA",), top_k=1, evaluation=EVALUATION),
            exposure,
            None,
        )


@pytest.fixture(scope="module")
def con(tmp_path_factory: pytest.TempPathFactory) -> Iterator[duckdb.DuckDBPyConnection]:
    path = tmp_path_factory.mktemp("world") / "world.duckdb"
    write_world(generate_world(preset("tiny", seed=4)), path)
    with duckdb.connect(str(path), read_only=True) as connection:
        yield connection


def test_a_run_gives_each_flow_its_group_s_headcount_as_exposure(
    con: duckdb.DuckDBPyConnection,
) -> None:
    config = GuidanceConfig.from_mapping(
        {
            "horizon": 6,
            "grouping": "TOPLINE",
            "models": ["native.HistoricalAverage", "native.LogitRateARIMA"],
            "combination": {"top_k": 2},
        }
    )
    selection = run(con, config).report["selection"]
    # The anchor has no exposure, so only the flows can use the share model.
    assert selection["ALL/headcount"]["scores"]["native.LogitRateARIMA"] == float("inf")
    assert np.isfinite(selection["ALL/attrition"]["scores"]["native.LogitRateARIMA"])
    assert "native.LogitRateARIMA" in selection["ALL/attrition"]["weights"]


# --- Prophet ------------------------------------------------------------------------------------


def test_prophet_is_refused_without_its_extra() -> None:
    problems = MODELS.problems(["prophet.Prophet"])
    if importlib.util.find_spec("prophet") is None:
        assert problems == [
            "model 'prophet.Prophet' needs the 'prophet' extra; install it with "
            "`uv sync --extra prophet`"
        ]
    else:
        assert problems == []


@pytest.mark.extras
def test_prophet_forecasts_a_seasonal_series() -> None:
    pytest.importorskip("prophet")
    forecast = build("prophet.Prophet").fit(COUNTS).predict(12)
    future = harmonic(np.arange(60, 72), 3.0)
    assert abs(forecast.mean() / future.mean() - 1) < 0.15
    assert np.corrcoef(forecast, future)[0, 1] > 0.5  # the season's shape, roughly
    assert np.array_equal(forecast, build("prophet.Prophet").fit(COUNTS).predict(12))
