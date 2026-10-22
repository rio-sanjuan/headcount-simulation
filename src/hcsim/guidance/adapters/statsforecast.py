"""The `statsforecast` adapter: the library's statistical models as guidance forecasters.

Each model registers as `statsforecast.<Name>`, built with a monthly season where it takes one:

- the automatic families: `AutoARIMA`, `AutoETS`, `AutoTheta`, and `AutoCES`;
- the theta methods: `Theta`, `OptimizedTheta`, and `DynamicOptimizedTheta`;
- the intermittent-demand methods, for sparse count series: `CrostonClassic`,
  `CrostonOptimized`, `CrostonSBA`, `ADIDA`, `IMAPA`, and `TSB`;
- the window averages: `WindowAverage` and `SeasonalWindowAverage`;
- the library's own benchmarks: `Naive`, `SeasonalNaive`, `HistoricAverage`, and
  `RandomWalkWithDrift`, the twins of the native ones.

`AutoARIMA` reads the working-day count (`weekdays`) as an exogenous regressor when the
pipeline gives it the calendar. A model with prediction intervals samples paths from a normal
around its forecast, at the spread its own 80% interval implies.

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
from hcsim.guidance.errors import ForecastError
from hcsim.guidance.registry import Entry
from hcsim.guidance.tuning import FloatParam, IntParam, Space
from hcsim.model.interface import ProbabilisticOutput

Floats = NDArray[np.float64]

NAMESPACE: Final = "statsforecast"
SEASON: Final = 12
_POINT = frozenset({ProbabilisticOutput.NONE})
_INTERVALS = frozenset(
    {ProbabilisticOutput.NONE, ProbabilisticOutput.QUANTILES, ProbabilisticOutput.SAMPLE_PATHS}
)
#: The prediction-interval level a path's spread is read from, and its normal quantile.
LEVEL: Final = 80
Z: Final = 1.2815515655446004

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


#: The models that read calendar covariates, as exogenous regressors, and the ones they read.
COVARIATES: Final[Mapping[str, tuple[str, ...]]] = MappingProxyType({"AutoARIMA": ("weekdays",)})
#: The models a tuner may tune, and their search spaces.
SPACES: Final[Mapping[str, Space]] = MappingProxyType(
    {
        "WindowAverage": {"window_size": IntParam(3, 24)},
        "SeasonalWindowAverage": {"window_size": IntParam(1, 5)},
        "TSB": {"alpha_d": FloatParam(0.05, 0.5), "alpha_p": FloatParam(0.05, 0.5)},
    }
)


class StatsForecastModel(LibraryModel):
    """One statsforecast model, fitted to one monthly series."""

    def _fit(self, values: Floats) -> Any:
        library = importlib.import_module("statsforecast.models")
        model = getattr(library, self.MODEL)(**self.arguments)
        if self.covariates_given:
            return model.fit(y=values, X=self.past_covariates)
        return model.fit(y=values)

    def _predict(self, fitted: Any, horizon: int) -> Any:
        if not self.covariates_given:
            return fitted.predict(h=horizon)["mean"]
        future = self.future_covariates
        if future is None or len(future) < horizon:
            raise ForecastError(f"{self.NAME} needs its covariates for the {horizon} months ahead")
        return fitted.predict(h=horizon, X=future[:horizon])["mean"]

    def sample(self, horizon: int, paths: int, rng: np.random.Generator) -> Floats:
        """Normal draws around the forecast, at the spread the model's own 80% prediction
        interval implies, month by month."""
        if ProbabilisticOutput.SAMPLE_PATHS not in self.OUTPUTS:
            raise ForecastError(f"{self.NAME} gives no prediction intervals to sample from")
        if self._fitted is None:
            raise ForecastError("fit before sampling")
        extra = {}
        if self.covariates_given and self.future_covariates is not None:
            extra["X"] = self.future_covariates[:horizon]
        try:
            band = self._fitted.predict(h=horizon, level=[LEVEL], **extra)
        except Exception as error:
            raise ForecastError(f"{self.NAME} can't give intervals: {error}") from error
        mean = np.asarray(band["mean"], dtype=np.float64)
        spread = (np.asarray(band[f"hi-{LEVEL}"]) - np.asarray(band[f"lo-{LEVEL}"])) / (2 * Z)
        if not (np.isfinite(mean).all() and np.isfinite(spread).all()):
            raise ForecastError(f"{self.NAME} gave no finite intervals")
        return rng.normal(mean, np.maximum(spread, 0.0), size=(paths, horizon))


#: Each adapted model's class, by the library's name for it.
ADAPTERS: Final[Mapping[str, type[StatsForecastModel]]] = MappingProxyType(
    {
        model: adapted(
            StatsForecastModel,
            NAMESPACE,
            model,
            arguments,
            outputs,
            covariates=COVARIATES.get(model, ()),
            space=SPACES.get(model),
        )
        for model, (arguments, outputs) in MODELS.items()
    }
)
#: The adapter's registry entries, by namespaced name.
ENTRIES: Final = tuple(Entry(cls.NAME, cls, module="statsforecast") for cls in ADAPTERS.values())
