"""Generation: refitting the selected models, weighting them, and forecasting.

Selection (see `selection`) scores the suite and weighs the candidates. Generation then:

1. **refits** each model that carries weight on the series' whole history: tuned on it first
   when tuning is on, given the exposure's history and forecast when it models a share, and
   the calendar for the history and horizon when it reads covariates;
2. **reweights**: a model that can't refit, as one can that forecast every fold but fails on
   the full history, leaves the combination, and the rest share its weight in proportion;
3. **forecasts** the weighted sum of the refitted models' forecasts.

A series none of whose selected models can refit gets no forecast, and the run report says
why.

**Uncertainty.** When the configuration asks for sample paths, a series whose weighted models
can all sample them (see `capabilities`) draws them from the combined predictive distribution:
a mixture of the models' own, the paths split among the models in proportion to their weights.
Each series draws from its own seeded stream, so the paths are reproducible and independent of
which other series the run forecasts. A series with any weighted model that can't sample gets
no paths, since the pipeline never fabricates uncertainty, and the report says which model
kept them from it.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from hcsim.guidance.capabilities import SamplingModel
from hcsim.guidance.evaluation import prepared, tuned
from hcsim.guidance.models import ForecastError, capabilities
from hcsim.guidance.selection import (
    Selection,
    SelectionConfig,
    SeriesForecast,
    UncertaintySettings,
)
from hcsim.model.interface import ProbabilisticOutput

Floats = NDArray[np.float64]


def generate(
    selection: Selection,
    history: Floats,
    horizon: int,
    config: SelectionConfig,
    exposure: Floats | None = None,
    future_exposure: Floats | None = None,
    calendar: pd.DataFrame | None = None,
    stream: int = 0,
) -> SeriesForecast:
    """Refit the selected models on `history`, reweight, and forecast `horizon` months, with
    sample paths drawn from the series' `stream` when the configuration asks for them."""
    values = np.asarray(history, dtype=np.float64)
    forecasts: dict[str, Floats] = {}
    fitted: dict[str, Any] = {}
    parameters: dict[str, dict[str, Any]] = {}
    for name in selection.weights:
        tuning = config.tuning
        chosen = (
            {}
            if tuning is None
            else tuned(name, values, config.evaluation, tuning, exposure, calendar)
        )
        try:
            model = prepared(
                name, exposure, future_exposure, calendar, len(values), horizon, chosen
            )
            forecasts[name] = model.fit(values).predict(horizon)
        except ForecastError:
            continue
        fitted[name] = model
        parameters[name] = chosen
    if not forecasts:
        raise ForecastError(f"none of the selected models {sorted(selection.weights)} could refit")
    total = sum(selection.weights[name] for name in forecasts)
    weights = {name: selection.weights[name] / total for name in forecasts}
    combined = np.zeros(horizon)
    for name, weight in weights.items():
        combined += weight * forecasts[name]
    paths, point_only = None, ""
    if config.uncertainty is not None:
        paths, point_only = _paths(fitted, weights, horizon, config.uncertainty, stream)
    return SeriesForecast(
        combined,
        selection.scores,
        weights,
        selection.folds,
        parameters,
        selection.method,
        paths,
        point_only,
        _residuals(selection, weights, values),
    )


def _residuals(selection: Selection, weights: dict[str, float], history: Floats) -> Floats | None:
    """The combined forecast's backtest errors: the actuals less the weighted fold forecasts,
    over the folds every weighted model forecast, in fold order."""
    by_window = {
        name: {(f.train_end, f.test_end): f.predicted for f in selection.folds.get(name, ())}
        for name in weights
    }
    shared = set.intersection(*(set(w) for w in by_window.values())) if by_window else set()
    errors = [
        history[start:end] - sum(weights[n] * by_window[n][(start, end)] for n in weights)
        for start, end in sorted(shared)
        if all(len(by_window[n][(start, end)]) == end - start for n in weights)
    ]
    return np.concatenate(errors) if errors else None


def _paths(
    fitted: dict[str, Any],
    weights: dict[str, float],
    horizon: int,
    settings: UncertaintySettings,
    stream: int,
) -> tuple[Floats | None, str]:
    """Paths from the weighted mixture of the models' predictive distributions, or none and
    the model that can't sample."""
    for name in weights:
        declared = ProbabilisticOutput.SAMPLE_PATHS in capabilities(name).outputs
        if not declared or not isinstance(fitted[name], SamplingModel):
            return None, f"{name} can't sample paths"
    rng = np.random.default_rng(np.random.SeedSequence([settings.seed, stream]))
    names = list(weights)
    counts = rng.multinomial(settings.paths, [weights[n] for n in names])
    draws = []
    for name, count in zip(names, counts.tolist(), strict=True):
        if count:
            try:
                draws.append(np.asarray(fitted[name].sample(horizon, count, rng), np.float64))
            except ForecastError as error:
                return None, str(error)
    return np.concatenate(draws), ""
