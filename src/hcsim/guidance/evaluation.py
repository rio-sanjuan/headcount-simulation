"""Evaluation: rolling-origin backtests, their metrics per fold, and nested tuning.

**Rolling-origin folds.** Evaluation works backwards from the end of a series. The latest
fold's test window ends at the last observed month, each earlier fold steps back, every
training window starts at the first month, and a fold with too little training history is
dropped. The defaults, which a configuration can change:

| setting | default | meaning |
|---|---|---|
| `n_splits` | 5 | folds, at most |
| `test_months` | 6 | each fold's test window |
| `step_months` | 2 | how far each earlier fold steps back |
| `min_train_months` | 36 | the shortest training window a fold keeps |

So a series needs `min_train_months + test_months` months, 42 by default, to be forecast.

**Metrics per fold.** Each fold records its WAPE and its MAPE. WAPE is absolute error over
absolute actuals, which suits sparse count series; MAPE is the mean absolute percentage error
over the months with a nonzero actual, undefined (`nan`) when there are none. A model's score
is its configured metric, WAPE by default, averaged across the folds it could forecast; a
model that fails a fold loses that fold, and one that fails every fold scores infinity.

**Nested tuning.** With tuning on, a model that declares a search space is tuned on each
fold's training window before that fold scores it (see `tuning`), and on the whole history
before its final refit.

A model of a share is scored on the exposure each fold's test months realized, and a
covariate-aware model reads the calendar of each fold's months.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Final, Literal

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from hcsim.guidance.capabilities import CovariateModel, ExposureModel
from hcsim.guidance.models import MODELS, Forecaster, ForecastError, build
from hcsim.guidance.tuning import Space, TuningSettings, tune

Floats = NDArray[np.float64]
Metric = Literal["wape", "mape"]
METRICS: Final = ("wape", "mape")


@dataclass(frozen=True)
class EvaluationConfig:
    n_splits: int = 5
    test_months: int = 6
    step_months: int = 2
    min_train_months: int = 36
    #: The metric a model's score averages across folds.
    metric: Metric = "wape"

    @property
    def min_length(self) -> int:
        """A series shorter than this gets no forecast."""
        return self.min_train_months + self.test_months


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


def mape(actual: Floats, forecast: Floats) -> float:
    """The mean absolute percentage error over the months with a nonzero actual; `nan` when
    there are none."""
    actual, forecast = np.asarray(actual, dtype=np.float64), np.asarray(forecast, np.float64)
    nonzero = actual != 0
    if not nonzero.any():
        return float("nan")
    return float(np.mean(np.abs((actual[nonzero] - forecast[nonzero]) / actual[nonzero])))


def space_of(name: str) -> Space:
    """The search space the model `name` declares; empty when it has nothing to tune."""
    factory = MODELS.factory(name)
    space: Space = getattr(factory, "SPACE", {})
    return space


def prepared(
    name: str,
    past: Floats | None,
    future: Floats | None,
    calendar: pd.DataFrame | None = None,
    months: int = 0,
    horizon: int = 0,
    parameters: dict[str, Any] | None = None,
) -> Forecaster:
    """The model `name`, built with `parameters` when it's been tuned, given the exposure when
    it models a share of one, and the calendar covariates it reads, when it reads any: the
    first `months` rows of `calendar` for its history, and the `horizon` after them for its
    forecast."""
    model = build(name, **(parameters or {}))
    if isinstance(model, ExposureModel):
        if past is None:
            raise ForecastError(f"{name} models a share of an exposure, and this series has none")
        model.expose(past, future)
    columns = list(getattr(model, "COVARIATES", ()))
    if columns and calendar is not None and isinstance(model, CovariateModel):
        values = calendar[columns].to_numpy(dtype=np.float64)
        ahead = values[months : months + horizon]
        model.provide(values[:months], ahead if len(ahead) == horizon else None)
    return model


@dataclass(frozen=True)
class FoldScore:
    train_end: int
    test_end: int
    wape: float
    mape: float
    #: The tuned values the fold scored the model with; empty for its defaults.
    parameters: dict[str, Any]
    #: The fold's forecast of its test months, which combination methods learn from.
    predicted: Floats = field(default_factory=lambda: np.zeros(0))

    def squared_error(self, series: Floats) -> float:
        """The fold's mean squared error against `series`' test months."""
        actual = series[self.train_end : self.test_end]
        return float(np.mean((actual - self.predicted) ** 2))

    def as_record(self) -> dict[str, Any]:
        return {
            "train_end": self.train_end,
            "test_end": self.test_end,
            "wape": self.wape,
            "mape": self.mape,
            "parameters": self.parameters,
        }


def _forecast(
    name: str,
    series: Floats,
    exposure: Floats | None,
    calendar: pd.DataFrame | None,
    train_end: int,
    test_end: int,
    parameters: dict[str, Any] | None,
) -> Floats:
    past = None if exposure is None else exposure[:train_end]
    future = None if exposure is None else exposure[train_end:test_end]
    model = prepared(name, past, future, calendar, train_end, test_end - train_end, parameters)
    return model.fit(series[:train_end]).predict(test_end - train_end)


def tuned(
    name: str,
    history: Floats,
    config: EvaluationConfig,
    tuning: TuningSettings,
    exposure: Floats | None = None,
    calendar: pd.DataFrame | None = None,
) -> dict[str, Any]:
    """The model's tuned values on `history`, by mean WAPE over inner folds; empty when it
    has nothing to tune or the history is too short for any inner fold."""
    space = space_of(name)
    inner = EvaluationConfig(
        n_splits=tuning.inner_splits,
        test_months=config.test_months,
        step_months=config.step_months,
        min_train_months=tuning.inner_min_train_months,
    )
    pairs = folds(len(history), inner)
    if not space or not pairs:
        return {}

    def objective(parameters: dict[str, Any]) -> float:
        scores = []
        for train_end, test_end in pairs:
            try:
                prediction = _forecast(
                    name, history, exposure, calendar, train_end, test_end, parameters
                )
            except ForecastError:
                return float("inf")
            scores.append(wape(history[train_end:test_end], prediction))
        return float(np.mean(scores))

    return tune(space, objective, tuning)


def backtest(
    name: str,
    series: Floats,
    config: EvaluationConfig,
    exposure: Floats | None = None,
    calendar: pd.DataFrame | None = None,
    tuning: TuningSettings | None = None,
) -> list[FoldScore]:
    """The model's score on each fold it could forecast, tuned on each fold's training window
    first when `tuning` is given."""
    scored = []
    for train_end, test_end in folds(len(series), config):
        parameters: dict[str, Any] = {}
        if tuning is not None:
            window = None if exposure is None else exposure[:train_end]
            parameters = tuned(name, series[:train_end], config, tuning, window, calendar)
        try:
            prediction = _forecast(
                name, series, exposure, calendar, train_end, test_end, parameters
            )
        except ForecastError:
            continue
        actual = series[train_end:test_end]
        scored.append(
            FoldScore(
                train_end,
                test_end,
                wape(actual, prediction),
                mape(actual, prediction),
                parameters,
                np.asarray(prediction, dtype=np.float64),
            )
        )
    return scored


def score_of(scored: list[FoldScore], metric: Metric = "wape") -> float:
    """The mean of `metric` across the folds where it's defined; infinite when none is. A
    fold the model missed entirely, as WAPE marks a forecast against all-zero actuals, keeps
    the mean infinite."""
    values = [getattr(f, metric) for f in scored if not np.isnan(getattr(f, metric))]
    return float(np.mean(values)) if values else float("inf")


def score(
    name: str,
    series: Floats,
    config: EvaluationConfig,
    exposure: Floats | None = None,
    calendar: pd.DataFrame | None = None,
    tuning: TuningSettings | None = None,
) -> float:
    """The model's score: its configured metric averaged across folds."""
    return score_of(backtest(name, series, config, exposure, calendar, tuning), config.metric)
