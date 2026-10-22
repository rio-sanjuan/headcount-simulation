"""Selection: scoring the suite, combining the best, and the combined forecast.

Every model in the suite is scored by rolling-origin evaluation (see `evaluation`). The
configured combination method turns the scores into weights, and the selected models refit on
the whole history, tuned first when tuning is on, and combine into the forecast.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from hcsim.guidance.evaluation import (
    EvaluationConfig,
    FoldScore,
    backtest,
    folds,
    mape,
    prepared,
    score,
    score_of,
    tuned,
    wape,
)
from hcsim.guidance.models import ForecastError
from hcsim.guidance.registry import Entry, Registry
from hcsim.guidance.tuning import TuningSettings

Floats = NDArray[np.float64]

__all__ = ["EvaluationConfig", "folds", "mape", "prepared", "score", "wape"]


@dataclass(frozen=True)
class SelectionConfig:
    suite: tuple[str, ...] = ("native.SeasonalNaive", "native.HistoricalAverage")
    top_k: int = 2
    combination: str = "inverse_error"
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    #: How to tune the models that declare a search space; none leaves their defaults.
    tuning: TuningSettings | None = None


class Combiner(Protocol):
    """Weights for the models that combine into a forecast, from their scores."""

    def __call__(self, scores: dict[str, float], top_k: int) -> dict[str, float]: ...


def inverse_error_weights(scores: dict[str, float], top_k: int) -> dict[str, float]:
    """Keep the `top_k` lowest scores, weighted by inverse error.

    Perfect scores share the weight equally. With no finite score, every model shares it.
    """
    ranked = sorted(scores, key=lambda name: (scores[name], name))[: max(1, top_k)]
    perfect = [name for name in ranked if scores[name] == 0.0]
    if perfect:
        return {name: 1.0 / len(perfect) for name in perfect}
    finite = [name for name in ranked if np.isfinite(scores[name])]
    if not finite:
        return {name: 1.0 / len(ranked) for name in ranked}
    inverse = {name: 1.0 / scores[name] for name in finite}
    total = sum(inverse.values())
    return {name: w / total for name, w in inverse.items()}


#: The combination methods a configuration can name.
COMBINERS: Registry[Combiner] = Registry(
    "combination method", (Entry("inverse_error", lambda: inverse_error_weights),)
)


@dataclass(frozen=True)
class SeriesForecast:
    values: Floats
    scores: dict[str, float]
    weights: dict[str, float]
    #: Each model's score on each fold.
    folds: dict[str, list[FoldScore]] = field(default_factory=dict)
    #: The tuned values each selected model refit with; empty for its defaults.
    parameters: dict[str, dict[str, Any]] = field(default_factory=dict)


def forecast(
    series: Sequence[float] | Floats,
    horizon: int,
    config: SelectionConfig,
    exposure: Floats | None = None,
    future_exposure: Floats | None = None,
    calendar: pd.DataFrame | None = None,
) -> SeriesForecast:
    """Evaluate the suite, select, refit on the full history, and combine. `exposure` is the
    series' exposure, such as its group's headcount, and `future_exposure` its forecast, for
    the models of a share. `calendar` holds the calendar covariates from the history's first
    month through the horizon, for the models that read them. A selected model that can't
    refit leaves the combination, and the rest share its weight."""
    history = np.asarray(series, dtype=np.float64)
    if len(history) < config.evaluation.min_length:
        raise ForecastError(
            f"a series needs {config.evaluation.min_length} months; this one has {len(history)}"
        )
    evaluation, tuning = config.evaluation, config.tuning
    scored = {
        name: backtest(name, history, evaluation, exposure, calendar, tuning)
        for name in config.suite
    }
    scores = {name: score_of(f, evaluation.metric) for name, f in scored.items()}
    weights = COMBINERS.build(config.combination)(scores, config.top_k)
    forecasts, parameters = {}, {}
    for name in weights:
        values = (
            {} if tuning is None else tuned(name, history, evaluation, tuning, exposure, calendar)
        )
        try:
            model = prepared(
                name, exposure, future_exposure, calendar, len(history), horizon, values
            )
            forecasts[name] = model.fit(history).predict(horizon)
        except ForecastError:
            continue
        parameters[name] = values
    if not forecasts:
        raise ForecastError(f"none of the selected models {sorted(weights)} could refit")
    total = sum(weights[name] for name in forecasts)
    weights = {name: weights[name] / total for name in forecasts}
    combined = np.zeros(horizon)
    for name, weight in weights.items():
        combined += weight * forecasts[name]
    return SeriesForecast(combined, scores, weights, scored, parameters)
