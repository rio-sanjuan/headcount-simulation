"""Tests for generation: refitting, reweighting, and forecasting a selection."""

from __future__ import annotations

import numpy as np
import pytest

from hcsim.guidance.errors import ForecastError
from hcsim.guidance.evaluation import EvaluationConfig
from hcsim.guidance.generation import generate
from hcsim.guidance.models import build
from hcsim.guidance.selection import Selection, SelectionConfig, forecast, select

MONTHS = np.arange(60, dtype=np.float64)
SERIES = 20 + 3 * np.sin(2 * np.pi * MONTHS / 12) + 0.1 * MONTHS
CONFIG = SelectionConfig(
    suite=("native.SeasonalNaive", "native.HistoricalAverage", "native.Drift"), top_k=2
)


def test_selection_scores_and_weighs_without_refitting() -> None:
    selection = select(SERIES, CONFIG)
    assert set(selection.scores) == set(CONFIG.suite)
    assert len(selection.weights) == 2
    assert np.isclose(sum(selection.weights.values()), 1.0)
    assert selection.method == "inverse_error"
    assert all(len(folds) == 5 for folds in selection.folds.values())
    with pytest.raises(ForecastError, match="a series needs 42 months"):
        select(SERIES[:30], CONFIG)


def test_generation_is_the_weighted_sum_of_the_refitted_forecasts() -> None:
    selection = select(SERIES, CONFIG)
    result = generate(selection, SERIES, 6, CONFIG)
    expected = sum(w * build(n).fit(SERIES).predict(6) for n, w in selection.weights.items())
    assert np.allclose(result.values, expected)
    assert result.weights == selection.weights
    assert result.scores == selection.scores
    assert np.allclose(forecast(SERIES, 6, CONFIG).values, result.values)


def test_a_model_that_can_t_refit_leaves_and_the_rest_share_its_weight() -> None:
    weights = {"native.LogitRateARIMA": 0.75, "native.HistoricalAverage": 0.25}
    selection = Selection(dict.fromkeys(weights, 0.1), weights, {}, "inverse_error")
    result = generate(selection, SERIES, 6, CONFIG)  # no exposure: the share model can't refit
    assert result.weights == {"native.HistoricalAverage": 1.0}
    assert np.allclose(result.values, SERIES.mean())
    only = Selection({"native.LogitRateARIMA": 0.1}, {"native.LogitRateARIMA": 1.0}, {}, "equal")
    with pytest.raises(ForecastError, match="could refit"):
        generate(only, SERIES, 6, CONFIG)


def test_generation_refits_tuned_when_tuning_is_on() -> None:
    from hcsim.guidance.tuning import TuningSettings

    tuning = TuningSettings(trials=6, seed=1)
    config = SelectionConfig(
        suite=("statsforecast.WindowAverage",),
        top_k=1,
        evaluation=EvaluationConfig(),
        tuning=tuning,
    )
    result = generate(select(SERIES, config), SERIES, 6, config)
    window = result.parameters["statsforecast.WindowAverage"]["window_size"]
    assert np.allclose(result.values, SERIES[-window:].mean())
