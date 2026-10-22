"""The `statsforecast` adapter: the library's statistical models as guidance forecasters.

Each model registers as `statsforecast.<Name>`, built with a monthly season where it takes one:

- the automatic families: `AutoARIMA`, `AutoETS`, `AutoTheta`, and `AutoCES`;
- the theta methods: `Theta`, `OptimizedTheta`, and `DynamicOptimizedTheta`;
- the intermittent-demand methods, for sparse count series: `CrostonClassic`,
  `CrostonOptimized`, `CrostonSBA`, `ADIDA`, `IMAPA`, and `TSB`;
- the window averages: `WindowAverage` and `SeasonalWindowAverage`;
- the library's own benchmarks: `Naive`, `SeasonalNaive`, `HistoricAverage`, and
  `RandomWalkWithDrift`, the twins of the native ones.

A model that fails on a series, as one can on a short or flat history, raises `ForecastError`,
so selection scores it out of that fold rather than stopping the run. Warnings the library
raises while fitting, such as a convergence notice, are kept on the model, not raised.
"""

from __future__ import annotations

import importlib
import warnings
from collections.abc import Mapping
from types import MappingProxyType
from typing import Any, ClassVar, Final, Self

import numpy as np
from numpy.typing import NDArray

from hcsim.guidance.errors import ForecastError
from hcsim.guidance.registry import Entry
from hcsim.model.interface import ProbabilisticOutput

Floats = NDArray[np.float64]

NAMESPACE: Final = "statsforecast"
SEASON: Final = 12
_POINT = frozenset({ProbabilisticOutput.NONE})
_INTERVALS = frozenset({ProbabilisticOutput.NONE, ProbabilisticOutput.QUANTILES})

#: Each model's constructor arguments, and what it can emit beyond a point forecast. The
#: intermittent methods and window averages give intervals only through conformal calibration,
#: which the adapter doesn't set up.
MODELS: Final[Mapping[str, tuple[Mapping[str, Any], frozenset[ProbabilisticOutput]]]] = (
    MappingProxyType(
        {
            "AutoARIMA": ({"season_length": SEASON}, _INTERVALS),
            "AutoETS": ({"season_length": SEASON}, _INTERVALS),
            "AutoTheta": ({"season_length": SEASON}, _INTERVALS),
            "AutoCES": ({"season_length": SEASON}, _INTERVALS),
            "Theta": ({"season_length": SEASON}, _INTERVALS),
            "OptimizedTheta": ({"season_length": SEASON}, _INTERVALS),
            "DynamicOptimizedTheta": ({"season_length": SEASON}, _INTERVALS),
            "CrostonClassic": ({}, _POINT),
            "CrostonOptimized": ({}, _POINT),
            "CrostonSBA": ({}, _POINT),
            "ADIDA": ({}, _POINT),
            "IMAPA": ({}, _POINT),
            "TSB": ({"alpha_d": 0.2, "alpha_p": 0.2}, _POINT),
            "WindowAverage": ({"window_size": SEASON}, _POINT),
            "SeasonalWindowAverage": ({"season_length": SEASON, "window_size": 3}, _POINT),
            "Naive": ({}, _INTERVALS),
            "SeasonalNaive": ({"season_length": SEASON}, _INTERVALS),
            "HistoricAverage": ({}, _INTERVALS),
            "RandomWalkWithDrift": ({}, _INTERVALS),
        }
    )
)


class StatsForecastModel:
    """One statsforecast model, fitted to one monthly series. Each adapted model is its own
    subclass (`ADAPTERS`), with its name and outputs."""

    NAME: ClassVar[str] = NAMESPACE
    OUTPUTS: ClassVar[frozenset[ProbabilisticOutput]] = _POINT
    #: The library's class name for the model.
    MODEL: ClassVar[str] = ""

    def __init__(self) -> None:
        self.warnings: list[str] = []
        self._fitted: Any = None

    def _build(self) -> Any:
        library = importlib.import_module("statsforecast.models")
        return getattr(library, self.MODEL)(**MODELS[self.MODEL][0])

    def fit(self, history: Floats) -> Self:
        values = np.asarray(history, dtype=np.float64)
        if len(values) < 2:
            raise ForecastError(f"{self.NAME} needs at least two months")
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            try:
                self._fitted = self._build().fit(y=values)
            except Exception as error:  # the library's failure on this series
                raise ForecastError(f"{self.NAME} can't fit this series: {error}") from error
        self.warnings = [str(w.message) for w in caught]
        return self

    def predict(self, horizon: int) -> Floats:
        if self._fitted is None:
            raise ForecastError("fit before predicting")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                mean = self._fitted.predict(h=horizon)["mean"]
            except Exception as error:
                raise ForecastError(f"{self.NAME} can't forecast: {error}") from error
        result = np.asarray(mean, dtype=np.float64)
        if result.shape != (horizon,) or not np.isfinite(result).all():
            raise ForecastError(f"{self.NAME} gave no finite forecast for {horizon} months")
        return result


def _adapter(model: str) -> type[StatsForecastModel]:
    namespace = {"NAME": f"{NAMESPACE}.{model}", "OUTPUTS": MODELS[model][1], "MODEL": model}
    return type(f"StatsForecast{model}", (StatsForecastModel,), namespace)


#: Each adapted model's class, by the library's name for it.
ADAPTERS: Final[Mapping[str, type[StatsForecastModel]]] = MappingProxyType(
    {model: _adapter(model) for model in MODELS}
)
#: The adapter's registry entries, by namespaced name.
ENTRIES: Final = tuple(Entry(cls.NAME, cls, module="statsforecast") for cls in ADAPTERS.values())
