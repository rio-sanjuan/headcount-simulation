"""Rolling-origin evaluation, top-k inverse-error selection, and combined forecasts.

Evaluation works backwards from the end of a series. The latest fold's test window ends at the
last observed month, each earlier fold steps back, every training window starts at the first
month, and a fold with too little training history is dropped. A model's score is its WAPE
averaged across folds. WAPE suits sparse count series, where MAPE is undefined on zeros.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

from hcsim.guidance.models import ForecastError, build

Floats = NDArray[np.float64]


@dataclass(frozen=True)
class EvaluationConfig:
    n_splits: int = 5
    test_months: int = 6
    step_months: int = 2
    min_train_months: int = 36

    @property
    def min_length(self) -> int:
        """A series shorter than this gets no forecast."""
        return self.min_train_months + self.test_months


@dataclass(frozen=True)
class SelectionConfig:
    suite: tuple[str, ...] = ("seasonal_naive", "historical_average")
    top_k: int = 2
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)


def folds(n_months: int, config: EvaluationConfig) -> list[tuple[int, int]]:
    """(train end, test end) index pairs, chronological. Ends are exclusive."""
    result = []
    for i in range(config.n_splits):
        test_end = n_months - i * config.step_months
        train_end = test_end - config.test_months
        if train_end >= config.min_train_months:
            result.append((train_end, test_end))
    return sorted(result)


def wape(actual: Floats, forecast: Floats) -> float:
    """Absolute error over absolute actuals. All-zero actuals score 0 if matched, else inf."""
    denominator = float(np.abs(actual).sum())
    error = float(np.abs(actual - forecast).sum())
    if denominator == 0.0:
        return 0.0 if error == 0.0 else float("inf")
    return error / denominator


def score(name: str, series: Floats, config: EvaluationConfig) -> float:
    """Mean WAPE across folds. A model that fails a fold loses that fold."""
    scores = []
    for train_end, test_end in folds(len(series), config):
        try:
            prediction = build(name).fit(series[:train_end]).predict(test_end - train_end)
        except ForecastError:
            continue
        scores.append(wape(series[train_end:test_end], prediction))
    return float(np.mean(scores)) if scores else float("inf")


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


@dataclass(frozen=True)
class SeriesForecast:
    values: Floats
    scores: dict[str, float]
    weights: dict[str, float]


def forecast(
    series: Sequence[float] | Floats, horizon: int, config: SelectionConfig
) -> SeriesForecast:
    """Evaluate the suite, select, refit on the full history, and combine."""
    history = np.asarray(series, dtype=np.float64)
    if len(history) < config.evaluation.min_length:
        raise ForecastError(
            f"a series needs {config.evaluation.min_length} months; this one has {len(history)}"
        )
    scores = {name: score(name, history, config.evaluation) for name in config.suite}
    weights = inverse_error_weights(scores, config.top_k)
    combined = np.zeros(horizon)
    for name, weight in weights.items():
        combined += weight * build(name).fit(history).predict(horizon)
    return SeriesForecast(combined, scores, weights)
