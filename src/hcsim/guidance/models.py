"""The model registry, and the native benchmark forecasters.

The registry is namespaced: a model is named `<namespace>.<Name>`, the namespace being its
library's (`statsforecast`, `darts`, `sktime`, or `prophet`) or `native` for the models this
package implements. A configuration's names are checked against it when it loads.

The native benchmarks are simple rules any richer model has to beat:

- `native.LastObserved` repeats the last month;
- `native.SeasonalNaive` repeats the last season, in phase;
- `native.HistoricalAverage` forecasts the mean of the history;
- `native.Drift` extends the line from the first month through the last, as a random walk
  with drift forecasts it.

Each implements `TimeSeriesModel` and emits point forecasts only. The native fixed-form
baselines are in `baselines`.
"""

from __future__ import annotations

from typing import ClassVar, Self

import numpy as np
from numpy.typing import NDArray

from hcsim.guidance.adapters import darts, prophet, sktime, statsforecast
from hcsim.guidance.baselines import BASELINES
from hcsim.guidance.errors import ForecastError
from hcsim.guidance.registry import Entry, Registry, RegistryError
from hcsim.model.interface import ProbabilisticOutput, TimeSeriesModel

Floats = NDArray[np.float64]


class SeasonalNaive:
    """Each forecast month repeats the same month one season earlier."""

    NAME: ClassVar[str] = "native.SeasonalNaive"
    OUTPUTS: ClassVar[frozenset[ProbabilisticOutput]] = frozenset({ProbabilisticOutput.NONE})

    def __init__(self, season: int = 12) -> None:
        if season < 1:
            raise ForecastError("season must be at least 1")
        self.season = season
        self._last_season: Floats | None = None

    def fit(self, history: Floats) -> Self:
        values = np.asarray(history, dtype=np.float64)
        if len(values) < self.season:
            raise ForecastError(f"{self.NAME} needs at least {self.season} months")
        self._last_season = values[-self.season :]
        return self

    def predict(self, horizon: int) -> Floats:
        if self._last_season is None:
            raise ForecastError("fit before predicting")
        return np.resize(self._last_season, horizon).astype(np.float64)


class HistoricalAverage:
    """Every forecast month is the mean of the history."""

    NAME: ClassVar[str] = "native.HistoricalAverage"
    OUTPUTS: ClassVar[frozenset[ProbabilisticOutput]] = frozenset({ProbabilisticOutput.NONE})

    def __init__(self) -> None:
        self._mean: float | None = None

    def fit(self, history: Floats) -> Self:
        values = np.asarray(history, dtype=np.float64)
        if len(values) == 0:
            raise ForecastError(f"{self.NAME} needs at least one month")
        self._mean = float(values.mean())
        return self

    def predict(self, horizon: int) -> Floats:
        if self._mean is None:
            raise ForecastError("fit before predicting")
        return np.full(horizon, self._mean)


class LastObserved:
    """Every forecast month repeats the last observed month."""

    NAME: ClassVar[str] = "native.LastObserved"
    OUTPUTS: ClassVar[frozenset[ProbabilisticOutput]] = frozenset({ProbabilisticOutput.NONE})

    def __init__(self) -> None:
        self._last: float | None = None

    def fit(self, history: Floats) -> Self:
        values = np.asarray(history, dtype=np.float64)
        if len(values) == 0:
            raise ForecastError(f"{self.NAME} needs at least one month")
        self._last = float(values[-1])
        return self

    def predict(self, horizon: int) -> Floats:
        if self._last is None:
            raise ForecastError("fit before predicting")
        return np.full(horizon, self._last)


class Drift:
    """The last month plus the history's average change, once per month ahead."""

    NAME: ClassVar[str] = "native.Drift"
    OUTPUTS: ClassVar[frozenset[ProbabilisticOutput]] = frozenset({ProbabilisticOutput.NONE})

    def __init__(self) -> None:
        self._last: float | None = None
        self._slope = 0.0

    def fit(self, history: Floats) -> Self:
        values = np.asarray(history, dtype=np.float64)
        if len(values) < 2:
            raise ForecastError(f"{self.NAME} needs at least two months")
        self._last = float(values[-1])
        self._slope = float(values[-1] - values[0]) / (len(values) - 1)
        return self

    def predict(self, horizon: int) -> Floats:
        if self._last is None:
            raise ForecastError("fit before predicting")
        return self._last + self._slope * np.arange(1, horizon + 1, dtype=np.float64)


Forecaster = TimeSeriesModel

#: The registered suite: the native benchmarks, and each adapter's models under its library's
#: namespace. A configuration names models from here.
MODELS: Registry[Forecaster] = Registry(
    "model",
    (
        Entry(LastObserved.NAME, LastObserved),
        Entry(SeasonalNaive.NAME, SeasonalNaive),
        Entry(HistoricalAverage.NAME, HistoricalAverage),
        Entry(Drift.NAME, Drift),
        *(Entry(model.NAME, model) for model in BASELINES),
        *statsforecast.ENTRIES,
        *darts.ENTRIES,
        *sktime.ENTRIES,
        *prophet.ENTRIES,
    ),
    namespaced=True,
)


def build(name: str) -> Forecaster:
    try:
        return MODELS.build(name)
    except RegistryError as error:
        raise ForecastError(str(error)) from None
