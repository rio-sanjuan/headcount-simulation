"""Benchmark forecasters. Both implement TimeSeriesModel and emit point forecasts only."""

from __future__ import annotations

from typing import ClassVar, Self

import numpy as np
from numpy.typing import NDArray

from hcsim.guidance.registry import Entry, Registry, RegistryError
from hcsim.model.interface import ProbabilisticOutput

Floats = NDArray[np.float64]


class ForecastError(ValueError):
    """A forecaster asked to work without the history it needs."""


class SeasonalNaive:
    """Each forecast month repeats the same month one season earlier."""

    NAME: ClassVar[str] = "seasonal_naive"
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

    NAME: ClassVar[str] = "historical_average"
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


Forecaster = SeasonalNaive | HistoricalAverage

#: The registered suite. A configuration names models from here.
MODELS: Registry[Forecaster] = Registry(
    "model",
    (Entry(SeasonalNaive.NAME, SeasonalNaive), Entry(HistoricalAverage.NAME, HistoricalAverage)),
)


def build(name: str) -> Forecaster:
    try:
        return MODELS.build(name)
    except RegistryError as error:
        raise ForecastError(str(error)) from None
