"""Tests for the combination methods and their fallbacks."""

from __future__ import annotations

import math

import numpy as np
import pytest

from hcsim.guidance.combination import (
    COMBINERS,
    CombinationError,
    CombinationInputs,
    bates_granger,
    combine,
    equal,
    inverse_error,
    rank,
    ridge,
    softmax,
)
from hcsim.guidance.config import GuidanceConfig, GuidanceConfigError
from hcsim.guidance.evaluation import FoldScore
from hcsim.guidance.selection import SelectionConfig, forecast

SERIES = np.arange(40, dtype=np.float64)


def fold(start: int, end: int, predicted: list[float]) -> FoldScore:
    return FoldScore(start, end, 0.0, 0.0, {}, np.asarray(predicted, dtype=np.float64))


def inputs(scores: dict[str, float], top_k: int = 3, **kwargs: object) -> CombinationInputs:
    return CombinationInputs(scores, kwargs.pop("folds", {}), SERIES, top_k, **kwargs)  # type: ignore[arg-type]


def test_every_method_weighs_only_the_top_k() -> None:
    scores = {"a": 0.4, "b": 0.1, "c": 0.2, "d": 0.3}
    assert inputs(scores, top_k=2).candidates == ["b", "c"]
    for method in ("inverse_error", "equal", "rank", "softmax"):
        weights = COMBINERS.build(method)(inputs(scores, top_k=2))
        assert set(weights) <= {"b", "c"}, method
        assert math.isclose(sum(weights.values()), 1.0), method


def test_inverse_error_equal_and_rank_weights() -> None:
    scores = {"a": 0.1, "b": 0.2, "c": 0.4}
    assert inverse_error(inputs(scores)) == pytest.approx({"a": 4 / 7, "b": 2 / 7, "c": 1 / 7})
    assert inverse_error(inputs({"a": 0.0, "b": 0.0, "c": 0.3})) == {"a": 0.5, "b": 0.5}
    assert inverse_error(inputs({"a": math.inf, "b": math.inf})) == {"a": 0.5, "b": 0.5}
    assert equal(inputs(scores)) == pytest.approx({"a": 1 / 3, "b": 1 / 3, "c": 1 / 3})
    assert rank(inputs(scores)) == pytest.approx({"a": 3 / 6, "b": 2 / 6, "c": 1 / 6})


def test_softmax_sharpens_as_the_temperature_falls() -> None:
    scores = {"a": 0.1, "b": 0.2}
    warm = softmax(inputs(scores, temperature=0.1))
    assert warm["a"] == pytest.approx(1 / (1 + math.exp(-1)))
    cold = softmax(inputs(scores, temperature=0.01))
    assert cold["a"] > warm["a"] > 0.5
    with pytest.raises(CombinationError, match="no candidate has a finite score"):
        softmax(inputs({"a": math.inf}))


def test_bates_granger_weighs_by_inverse_backtest_error() -> None:
    truth = SERIES[30:36]
    folds = {
        "close": [fold(30, 36, list(truth + 1.0))],  # mean squared error 1
        "far": [fold(30, 36, list(truth + 2.0))],  # mean squared error 4
    }
    weights = bates_granger(inputs({"close": 0.1, "far": 0.2}, folds=folds))
    assert weights == pytest.approx({"close": 0.8, "far": 0.2})
    with pytest.raises(CombinationError, match="no candidate has a backtest error"):
        bates_granger(inputs({"close": 0.1}))


def test_ridge_learns_to_trust_the_model_that_tracks_the_actuals() -> None:
    noise = np.random.default_rng(0).normal(0, 3, 10)
    folds = {
        "tracks": [fold(20, 25, list(SERIES[20:25])), fold(25, 30, list(SERIES[25:30]))],
        "wanders": [
            fold(20, 25, list(SERIES[20:25] + noise[:5])),
            fold(25, 30, list(SERIES[25:30] + noise[5:])),
        ],
    }
    scores = {"tracks": 0.1, "wanders": 0.2}
    sharp = ridge(inputs(scores, folds=folds, ridge_alpha=0.01))
    assert sharp["tracks"] > 0.95
    shrunk = ridge(inputs(scores, folds=folds, ridge_alpha=1e6))
    assert shrunk == pytest.approx({"tracks": 0.5, "wanders": 0.5}, abs=1e-3)
    thin = {"tracks": [fold(20, 21, [20.0])], "wanders": [fold(20, 21, [21.0])]}
    with pytest.raises(CombinationError, match="at least 2 fold months"):
        ridge(inputs(scores, folds=thin))


def test_a_method_that_can_t_weigh_falls_back_in_order() -> None:
    scores = {"a": 0.1, "b": 0.2}
    weights, method = combine(inputs(scores), "ridge", ("bates_granger", "rank"))
    assert method == "rank"
    assert weights == pytest.approx({"a": 2 / 3, "b": 1 / 3})
    assert combine(inputs({"a": math.inf}), "softmax", ("rank",)) == ({"a": 1.0}, "equal")
    weights, method = combine(inputs(scores), "inverse_error")
    assert method == "inverse_error"
    assert weights == pytest.approx({"a": 2 / 3, "b": 1 / 3})


def test_a_forecast_reports_the_method_that_weighed_it() -> None:
    series = 20 + 3 * np.sin(2 * np.pi * np.arange(60) / 12) + np.arange(60) * 0.1
    suite = ("native.SeasonalNaive", "native.HistoricalAverage", "native.Drift")
    for method in ("ridge", "bates_granger", "softmax", "rank", "equal"):
        result = forecast(series, 6, SelectionConfig(suite=suite, top_k=3, combination=method))
        assert result.method == method
        assert math.isclose(sum(result.weights.values()), 1.0)
        assert result.values.shape == (6,)


def test_the_combination_settings_are_checked_at_load() -> None:
    config = GuidanceConfig.from_mapping(
        {"combination": {"method": "ridge", "fallbacks": ["bates_granger"], "ridge_alpha": 2.0}}
    )
    selection = config.selection
    assert (selection.combination, selection.fallbacks, selection.ridge_alpha) == (
        "ridge",
        ("bates_granger",),
        2.0,
    )
    with pytest.raises(GuidanceConfigError) as error:
        GuidanceConfig.from_mapping({"combination": {"fallbacks": ["median"], "temperature": 0}})
    message = str(error.value)
    assert "combination.temperature" in message
    with pytest.raises(GuidanceConfigError, match=r"combination\.fallbacks: unknown combination"):
        GuidanceConfig.from_mapping({"combination": {"fallbacks": ["median"]}})
