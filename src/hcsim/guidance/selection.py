"""Selection: scoring the suite and weighing the best.

Every model in the suite is scored by rolling-origin evaluation (see `evaluation`), and the
configured combination method, or a fallback, turns the scores into weights for the
candidates (see `combination`). Generation then refits the selected models and forecasts (see
`generation`); `forecast` does both for one series.
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
    #: How many sample paths a forecast draws, and from what seed; none draws none.
    uncertainty: UncertaintySettings | None = None


@dataclass(frozen=True)
class UncertaintySettings:
    paths: int = 200
    seed: int = 0


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
    #: (paths, horizon) draws from the combined predictive distribution, or none, and why.
    paths: Floats | None = None
    point_only: str = ""


@dataclass(frozen=True)
class Selection:
    """A series' suite scored and its candidates weighed, before any refit."""

    scores: dict[str, float]
    weights: dict[str, float]
    folds: dict[str, list[FoldScore]]
    #: The combination method that weighed it: the configured one, or a fallback.
    method: str


def select(
    series: Sequence[float] | Floats,
    config: SelectionConfig,
    exposure: Floats | None = None,
    calendar: pd.DataFrame | None = None,
) -> Selection:
    """Score every model in the suite by rolling-origin evaluation, and weigh the best.
    `exposure` is the series' exposure, such as its group's headcount, for the models of a
    share; `calendar` holds the calendar covariates, for the models that read them."""
    history = np.asarray(series, dtype=np.float64)
    if len(history) < config.evaluation.min_length:
        raise ForecastError(
            f"a series needs {config.evaluation.min_length} months; this one has {len(history)}"
        )
    evaluation = config.evaluation
    scored = {
        name: backtest(name, history, evaluation, exposure, calendar, config.tuning)
        for name in config.suite
    }
    scores = {name: score_of(f, evaluation.metric) for name, f in scored.items()}
    inputs = CombinationInputs(
        scores, scored, history, config.top_k, config.temperature, config.ridge_alpha
    )
    weights, method = combine(inputs, config.combination, config.fallbacks)
    return Selection(scores, weights, scored, method)


def forecast(
    series: Sequence[float] | Floats,
    horizon: int,
    config: SelectionConfig,
    exposure: Floats | None = None,
    future_exposure: Floats | None = None,
    calendar: pd.DataFrame | None = None,
) -> SeriesForecast:
    """Select, then generate (see `generation`): the whole forecast of one series."""
    from hcsim.guidance.generation import generate

    selection = select(series, config, exposure, calendar)
    history = np.asarray(series, dtype=np.float64)
    return generate(selection, history, horizon, config, exposure, future_exposure, calendar)
