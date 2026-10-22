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
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from hcsim.guidance.evaluation import prepared, tuned
from hcsim.guidance.models import ForecastError
from hcsim.guidance.selection import Selection, SelectionConfig, SeriesForecast

Floats = NDArray[np.float64]


def generate(
    selection: Selection,
    history: Floats,
    horizon: int,
    config: SelectionConfig,
    exposure: Floats | None = None,
    future_exposure: Floats | None = None,
    calendar: pd.DataFrame | None = None,
) -> SeriesForecast:
    """Refit the selected models on `history`, reweight, and forecast `horizon` months."""
    values = np.asarray(history, dtype=np.float64)
    forecasts: dict[str, Floats] = {}
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
        parameters[name] = chosen
    if not forecasts:
        raise ForecastError(f"none of the selected models {sorted(selection.weights)} could refit")
    total = sum(selection.weights[name] for name in forecasts)
    weights = {name: selection.weights[name] / total for name in forecasts}
    combined = np.zeros(horizon)
    for name, weight in weights.items():
        combined += weight * forecasts[name]
    return SeriesForecast(
        combined, selection.scores, weights, selection.folds, parameters, selection.method
    )
