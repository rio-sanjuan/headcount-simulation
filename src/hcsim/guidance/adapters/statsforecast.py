"""The `statsforecast` adapter: the library's statistical models as guidance forecasters.

Each model registers as `statsforecast.<Name>`, built with a monthly season where it takes one:

- the automatic families: `AutoARIMA`, `AutoETS`, `AutoTheta`, and `AutoCES`;
- the theta methods: `Theta`, `OptimizedTheta`, and `DynamicOptimizedTheta`;
- the intermittent-demand methods, for sparse count series: `CrostonClassic`,
  `CrostonOptimized`, `CrostonSBA`, `ADIDA`, `IMAPA`, and `TSB`;
- the window averages: `WindowAverage` and `SeasonalWindowAverage`;
- the library's own benchmarks: `Naive`, `SeasonalNaive`, `HistoricAverage`, and
  `RandomWalkWithDrift`, the twins of the native ones.

Fitting, forecasting, and failing work as for every adapter (see `base`).
"""

from __future__ import annotations

import importlib
from collections.abc import Mapping
from types import MappingProxyType
from typing import Any, Final

import numpy as np
from numpy.typing import NDArray

from hcsim.guidance.adapters.base import LibraryModel, adapted
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


class StatsForecastModel(LibraryModel):
    """One statsforecast model, fitted to one monthly series."""

    def _fit(self, values: Floats) -> Any:
        library = importlib.import_module("statsforecast.models")
        return getattr(library, self.MODEL)(**self.ARGUMENTS).fit(y=values)

    def _predict(self, fitted: Any, horizon: int) -> Any:
        return fitted.predict(h=horizon)["mean"]


#: Each adapted model's class, by the library's name for it.
ADAPTERS: Final[Mapping[str, type[StatsForecastModel]]] = MappingProxyType(
    {
        model: adapted(StatsForecastModel, NAMESPACE, model, arguments, outputs)
        for model, (arguments, outputs) in MODELS.items()
    }
)
#: The adapter's registry entries, by namespaced name.
ENTRIES: Final = tuple(Entry(cls.NAME, cls, module="statsforecast") for cls in ADAPTERS.values())
