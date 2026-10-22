"""Tests for the benchmark forecasters and selection."""

from __future__ import annotations

import numpy as np
import pytest

from hcsim.guidance.models import (
    Drift,
    ForecastError,
    HistoricalAverage,
    LastObserved,
    SeasonalNaive,
    build,
)
from hcsim.guidance.selection import (
    EvaluationConfig,
    SelectionConfig,
    folds,
    forecast,
    inverse_error_weights,
    score,
    wape,
)
from hcsim.model.interface import ProbabilisticOutput, TimeSeriesModel

SEASONAL = np.tile(np.arange(12, dtype=np.float64), 5)  # 0..11 repeated for 60 months


def test_every_benchmark_implements_the_protocol() -> None:
    for model in (LastObserved(), SeasonalNaive(), HistoricalAverage(), Drift()):
        assert isinstance(model, TimeSeriesModel)
        assert frozenset({ProbabilisticOutput.NONE}) == model.OUTPUTS


def test_last_observed_repeats_the_last_month() -> None:
    assert LastObserved().fit(np.array([4.0, 7.0, 5.0])).predict(3).tolist() == [5.0, 5.0, 5.0]


def test_drift_extends_the_line_through_the_first_and_last_months() -> None:
    history = np.array([10.0, 3.0, 25.0, 16.0])  # an average change of 2 a month
    assert Drift().fit(history).predict(3).tolist() == [18.0, 20.0, 22.0]
    assert Drift().fit(np.array([5.0, 5.0])).predict(2).tolist() == [5.0, 5.0]


def test_seasonal_naive_repeats_the_last_season_in_phase() -> None:
    history = np.arange(1, 27, dtype=np.float64)  # 26 months: the last season is 15..26
    assert SeasonalNaive().fit(history).predict(14).tolist() == [*range(15, 27), 15, 16]


def test_historical_average_forecasts_the_mean() -> None:
    assert HistoricalAverage().fit(np.array([1.0, 2.0, 6.0])).predict(2).tolist() == [3.0, 3.0]


def test_models_refuse_too_little_history_and_unfitted_use() -> None:
    with pytest.raises(ForecastError, match="at least 12"):
        SeasonalNaive().fit(np.ones(11))
    with pytest.raises(ForecastError, match="at least one"):
        HistoricalAverage().fit(np.ones(0))
    with pytest.raises(ForecastError, match="at least one"):
        LastObserved().fit(np.ones(0))
    with pytest.raises(ForecastError, match="at least two"):
        Drift().fit(np.ones(1))
    for model in (LastObserved(), SeasonalNaive(), HistoricalAverage(), Drift()):
        with pytest.raises(ForecastError, match="fit before"):
            model.predict(1)
    with pytest.raises(ForecastError, match="season must be"):
        SeasonalNaive(season=0)
    with pytest.raises(ForecastError, match="unknown model 'prophet'"):
        build("prophet")


def test_folds_step_back_from_the_end_and_drop_short_training_windows() -> None:
    config = EvaluationConfig()
    assert folds(60, config) == [(46, 52), (48, 54), (50, 56), (52, 58), (54, 60)]
    assert folds(44, config) == [(36, 42), (38, 44)]  # three folds would train on under 36
    assert folds(41, config) == []


def test_wape_handles_all_zero_actuals() -> None:
    assert wape(np.array([2.0, 2.0]), np.array([1.0, 3.0])) == pytest.approx(0.5)
    assert wape(np.zeros(3), np.zeros(3)) == 0.0
    assert wape(np.zeros(3), np.ones(3)) == float("inf")


def test_seasonal_naive_scores_perfectly_on_a_purely_seasonal_series() -> None:
    assert score("seasonal_naive", SEASONAL, EvaluationConfig()) == 0.0
    assert score("historical_average", SEASONAL, EvaluationConfig()) > 0.0


def test_weights_favor_the_lower_error_and_handle_perfect_and_failed_scores() -> None:
    assert inverse_error_weights({"a": 0.5, "b": 1.0}, 2) == pytest.approx({"a": 2 / 3, "b": 1 / 3})
    assert inverse_error_weights({"a": 0.0, "b": 1.0}, 2) == {"a": 1.0}
    assert inverse_error_weights({"a": 0.2, "b": 0.1, "c": 9.0}, 1) == {"b": 1.0}
    inf = float("inf")
    assert inverse_error_weights({"a": inf, "b": inf}, 2) == {"a": 0.5, "b": 0.5}


def test_a_seasonal_series_is_forecast_by_seasonal_naive_alone() -> None:
    result = forecast(SEASONAL, 12, SelectionConfig())
    assert result.weights == {"seasonal_naive": 1.0}
    assert result.values.tolist() == list(range(12))


def test_a_flat_noisy_series_blends_both_models() -> None:
    series = 5.0 + np.random.default_rng(1).normal(0, 1, 60)
    result = forecast(series, 6, SelectionConfig())
    assert set(result.weights) == {"seasonal_naive", "historical_average"}
    assert sum(result.weights.values()) == pytest.approx(1.0)
    assert len(result.values) == 6


def test_a_short_series_gets_no_forecast() -> None:
    with pytest.raises(ForecastError, match="needs 42 months; this one has 30"):
        forecast(np.ones(30), 12, SelectionConfig())


def test_the_full_benchmark_suite_selects_the_rule_that_fits() -> None:
    suite = ("last_observed", "seasonal_naive", "historical_average", "drift")
    seasonal = forecast(SEASONAL, 12, SelectionConfig(suite=suite, top_k=1))
    assert seasonal.weights == {"seasonal_naive": 1.0}
    trend = forecast(np.arange(60, dtype=np.float64), 3, SelectionConfig(suite=suite, top_k=1))
    assert trend.weights == {"drift": 1.0}
    assert trend.values.tolist() == [60.0, 61.0, 62.0]
