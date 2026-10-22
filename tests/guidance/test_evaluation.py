"""Tests for rolling-origin evaluation, the metrics per fold, and nested tuning."""

from __future__ import annotations

import math
from collections.abc import Iterator

import duckdb
import numpy as np
import pytest

from hcsim.guidance.config import GuidanceConfig, GuidanceConfigError
from hcsim.guidance.errors import ForecastError
from hcsim.guidance.evaluation import (
    EvaluationConfig,
    backtest,
    folds,
    mape,
    score,
    score_of,
    space_of,
    tuned,
    wape,
)
from hcsim.guidance.models import build
from hcsim.guidance.pipeline import run
from hcsim.guidance.selection import SelectionConfig, forecast
from hcsim.guidance.tuning import TuningSettings
from hcsim.synth.config import preset
from hcsim.synth.world import generate_world, write_world

#: A level that shifts up in the last year: a short window follows it, a long one lags.
SHIFTED = np.r_[np.full(48, 20.0), np.full(12, 40.0)] + np.random.default_rng(1).normal(0, 1, 60)
TUNING = TuningSettings(trials=12, seed=3, inner_splits=2, inner_min_train_months=24)


def test_the_documented_defaults() -> None:
    config = EvaluationConfig()
    assert (config.n_splits, config.test_months, config.step_months) == (5, 6, 2)
    assert (config.min_train_months, config.metric, config.min_length) == (36, "wape", 42)
    assert folds(60, config) == [(46, 52), (48, 54), (50, 56), (52, 58), (54, 60)]
    assert folds(43, config) == [(37, 43)]
    assert folds(41, config) == []


def test_mape_averages_over_the_months_with_an_actual() -> None:
    assert mape(np.array([10.0, 0.0, 20.0]), np.array([12.0, 5.0, 15.0])) == pytest.approx(0.225)
    assert math.isnan(mape(np.zeros(3), np.ones(3)))
    assert wape(np.zeros(3), np.zeros(3)) == 0.0


def test_each_fold_records_both_metrics_and_the_score_uses_the_configured_one() -> None:
    series = 10 + np.arange(60, dtype=np.float64)
    scored = backtest("native.HistoricalAverage", series, EvaluationConfig())
    assert len(scored) == 5
    assert all(f.parameters == {} for f in scored)
    first = scored[0]
    actual = series[first.train_end : first.test_end]
    predicted = np.full(6, series[: first.train_end].mean())
    assert first.wape == pytest.approx(wape(actual, predicted))
    assert first.mape == pytest.approx(mape(actual, predicted))
    by_mape = EvaluationConfig(metric="mape")
    assert score("native.HistoricalAverage", series, by_mape) == pytest.approx(
        np.mean([f.mape for f in scored])
    )
    assert score_of([], "wape") == float("inf")


def test_a_model_that_fails_every_fold_scores_infinity() -> None:
    assert score("native.LogitRateARIMA", SHIFTED, EvaluationConfig()) == float("inf")


# --- tuning -------------------------------------------------------------------------------------


def test_tunable_models_declare_their_search_space() -> None:
    assert set(space_of("statsforecast.WindowAverage")) == {"window_size"}
    assert set(space_of("native.PoissonGLM")) == {"harmonics"}
    assert space_of("native.SeasonalNaive") == {}
    with pytest.raises(ForecastError, match="has no hyperparameters \\['bogus'\\]"):
        build("statsforecast.WindowAverage", bogus=1)


def test_tuning_finds_the_window_the_series_calls_for_reproducibly() -> None:
    values = tuned("statsforecast.WindowAverage", SHIFTED, EvaluationConfig(), TUNING)
    assert values["window_size"] <= 6  # a long window would average across the shift
    assert tuned("statsforecast.WindowAverage", SHIFTED, EvaluationConfig(), TUNING) == values
    assert tuned("native.SeasonalNaive", SHIFTED, EvaluationConfig(), TUNING) == {}
    too_short = tuned("statsforecast.WindowAverage", SHIFTED[:28], EvaluationConfig(), TUNING)
    assert too_short == {}  # no inner fold fits, so the defaults stand


def test_each_fold_is_tuned_on_its_own_training_window() -> None:
    config = EvaluationConfig()
    scored = backtest("statsforecast.WindowAverage", SHIFTED, config, tuning=TUNING)
    assert len(scored) == 5
    for fold in scored:
        window = SHIFTED[: fold.train_end]
        assert fold.parameters == tuned("statsforecast.WindowAverage", window, config, TUNING)


def test_a_tuned_forecast_refits_with_the_values_the_whole_history_calls_for() -> None:
    config = SelectionConfig(
        suite=("statsforecast.WindowAverage", "native.HistoricalAverage"), top_k=1, tuning=TUNING
    )
    result = forecast(SHIFTED, 6, config)
    assert result.weights == {"statsforecast.WindowAverage": 1.0}
    chosen = result.parameters["statsforecast.WindowAverage"]
    assert chosen == tuned("statsforecast.WindowAverage", SHIFTED, config.evaluation, TUNING)
    assert np.allclose(result.values, SHIFTED[-chosen["window_size"] :].mean())
    # The score is out of sample: each fold tuned on its own window, which can choose worse
    # than the defaults when the series shifts after it, as this one does.
    assert np.isfinite(result.scores["statsforecast.WindowAverage"])


def test_the_tuning_settings_are_checked_at_load() -> None:
    config = GuidanceConfig.from_mapping(
        {
            "models": ["native.PoissonGLM", "native.SeasonalNaive"],
            "tuning": {"enabled": True, "trials": 5, "seed": 9},
            "evaluation": {"metric": "mape"},
        }
    )
    tuning = config.selection.tuning
    assert tuning is not None
    assert (tuning.trials, tuning.seed) == (5, 9)
    assert config.selection.evaluation.metric == "mape"
    assert GuidanceConfig().selection.tuning is None
    with pytest.raises(GuidanceConfigError, match="no model in the suite has a search space"):
        GuidanceConfig.from_mapping({"tuning": {"enabled": True}})
    with pytest.raises(GuidanceConfigError, match=r"evaluation\.metric"):
        GuidanceConfig.from_mapping({"evaluation": {"metric": "rmse"}})


@pytest.fixture(scope="module")
def con(tmp_path_factory: pytest.TempPathFactory) -> Iterator[duckdb.DuckDBPyConnection]:
    path = tmp_path_factory.mktemp("world") / "world.duckdb"
    write_world(generate_world(preset("tiny", seed=4)), path)
    with duckdb.connect(str(path), read_only=True) as connection:
        yield connection


def test_a_run_reports_each_fold_and_the_tuned_values(con: duckdb.DuckDBPyConnection) -> None:
    config = GuidanceConfig.from_mapping(
        {
            "horizon": 3,
            "grouping": "TOPLINE",
            "models": ["native.SeasonalNaive", "statsforecast.WindowAverage"],
            "tuning": {"enabled": True, "trials": 4},
        }
    )
    selection = run(con, config).report["selection"]["ALL/attrition"]
    assert len(selection["folds"]["native.SeasonalNaive"]) == 5
    assert set(selection["folds"]["native.SeasonalNaive"][0]) == {
        "train_end",
        "test_end",
        "wape",
        "mape",
        "parameters",
    }
    window_folds = selection["folds"]["statsforecast.WindowAverage"]
    assert all("window_size" in f["parameters"] for f in window_folds)
    if "statsforecast.WindowAverage" in selection["weights"]:
        assert "window_size" in selection["parameters"]["statsforecast.WindowAverage"]
