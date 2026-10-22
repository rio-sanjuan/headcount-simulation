"""Selection: scoring the suite, combining the best, and the combined forecast.

Every model in the suite is scored by rolling-origin evaluation (see `evaluation`). The
configured combination method turns the scores into weights, and the selected models refit on
the whole history, tuned first when tuning is on, and combine into the forecast.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from hcsim.guidance.combination import COMBINERS, CombinationInputs, combine, inverse_error
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
from hcsim.guidance.tuning import TuningSettings

Floats = NDArray[np.float64]

__all__ = ["COMBINERS", "EvaluationConfig", "folds", "mape", "prepared", "score", "wape"]


@dataclass(frozen=True)
class SelectionConfig:
    suite: tuple[str, ...] = ("native.SeasonalNaive", "native.HistoricalAverage")
    top_k: int = 2
    combination: str = "inverse_error"
    #: The methods tried, in order, when the combination method can't weigh the candidates.
    fallbacks: tuple[str, ...] = ("equal",)
    #: The softmax temperature, and the ridge meta-learner's penalty.
    temperature: float = 0.1
    ridge_alpha: float = 1.0
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    #: How to tune the models that declare a search space; none leaves their defaults.
    tuning: TuningSettings | None = None


def inverse_error_weights(scores: dict[str, float], top_k: int) -> dict[str, float]:
    """Inverse-error weights for the `top_k` best of `scores` (see `combination`)."""
    return inverse_error(CombinationInputs(scores, {}, np.zeros(0), top_k))


@dataclass(frozen=True)
class SeriesForecast:
    values: Floats
    scores: dict[str, float]
    weights: dict[str, float]
    #: Each model's score on each fold.
    folds: dict[str, list[FoldScore]] = field(default_factory=dict)
    #: The tuned values each selected model refit with; empty for its defaults.
    parameters: dict[str, dict[str, Any]] = field(default_factory=dict)
    #: The combination method that weighed the forecast: the configured one, or a fallback.
    method: str = "inverse_error"


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
    inputs = CombinationInputs(
        scores, scored, history, config.top_k, config.temperature, config.ridge_alpha
    )
    weights, method = combine(inputs, config.combination, config.fallbacks)
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
    return SeriesForecast(combined, scores, weights, scored, parameters, method)
