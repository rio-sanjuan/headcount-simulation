"""The `darts` adapter: the library's local, regression, and deep models as guidance
forecasters, from the optional `darts` and `darts-deep` extras.

Each model registers as `darts.<Name>`:

- the statistical models: `ExponentialSmoothing`, with a monthly season, `Theta`, `FFT`, and
  `KalmanForecaster`;
- the regression models on the last year's lags: `LightGBMModel` and
  `LinearRegressionModel`. They're the covariate-aware options;
- the deep models, `NBEATSModel` and `TFTModel`, which need PyTorch and so come only with the
  `darts-deep` extra, the one place PyTorch enters. They train on CPU, seeded, without
  progress bars or loggers.

A configuration that names a model whose extra isn't installed is refused at load, with the
install command. Fitting, forecasting, and failing work as for every adapter (see `base`).
"""

from __future__ import annotations

import importlib
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, ClassVar, Final

import numpy as np

from hcsim.guidance.adapters.base import Floats, LibraryModel, adapted
from hcsim.guidance.registry import Entry
from hcsim.model.interface import ProbabilisticOutput

NAMESPACE: Final = "darts"
SEASON: Final = 12
SEED: Final = 0
_POINT = frozenset({ProbabilisticOutput.NONE})
_PATHS = frozenset({ProbabilisticOutput.NONE, ProbabilisticOutput.SAMPLE_PATHS})
_QUANTILES = frozenset(
    {ProbabilisticOutput.NONE, ProbabilisticOutput.QUANTILES, ProbabilisticOutput.SAMPLE_PATHS}
)
#: How the deep models train: briefly, on CPU, seeded, and quietly.
_TRAINER: Final = MappingProxyType(
    {
        "accelerator": "cpu",
        "enable_progress_bar": False,
        "enable_model_summary": False,
        "logger": False,
    }
)
_DEEP: Final = MappingProxyType(
    {
        "input_chunk_length": 24,
        "output_chunk_length": SEASON,
        "n_epochs": 20,
        "random_state": SEED,
        "pl_trainer_kwargs": dict(_TRAINER),
    }
)


@dataclass(frozen=True)
class Spec:
    #: The model's constructor arguments.
    arguments: Mapping[str, Any]
    #: What it emits beyond a point forecast.
    outputs: frozenset[ProbabilisticOutput]
    #: The fewest months it fits.
    min_months: int
    #: The module its extra provides.
    module: str = "darts"
    #: Whether it trains on the series scaled to [0, 1], as the deep models need.
    scaled: bool = False
    #: The samples whose median is its point forecast; one for a model that forecasts a mean.
    samples: int = 1


#: Each adapted model's spec, by the library's name for it.
MODELS: Final[Mapping[str, Spec]] = MappingProxyType(
    {
        "ExponentialSmoothing": Spec({"seasonal_periods": SEASON}, _PATHS, 2 * SEASON),
        "Theta": Spec({"theta": 2, "seasonality_period": SEASON}, _POINT, 2 * SEASON),
        "FFT": Spec({"nr_freqs_to_keep": 10}, _POINT, SEASON),
        "KalmanForecaster": Spec({"dim_x": 4}, _PATHS, SEASON),
        "LightGBMModel": Spec(
            {"lags": SEASON, "random_state": SEED, "verbose": -1}, _POINT, 2 * SEASON, "lightgbm"
        ),
        "LinearRegressionModel": Spec({"lags": SEASON}, _POINT, 2 * SEASON),
        "NBEATSModel": Spec(dict(_DEEP), _POINT, 36, "torch", scaled=True),
        "TFTModel": Spec(
            {**_DEEP, "add_relative_index": True}, _QUANTILES, 36, "torch", True, samples=200
        ),
    }
)
#: The extra that provides each module.
EXTRAS: Final = MappingProxyType({"darts": "darts", "lightgbm": "darts", "torch": "darts-deep"})


class DartsModel(LibraryModel):
    """One darts model, fitted to one monthly series."""

    SPEC: ClassVar[Spec]

    def _fit(self, values: Floats) -> Any:
        series = importlib.import_module("darts").TimeSeries.from_values(values)
        scaler = None
        if self.SPEC.scaled:
            transformers = importlib.import_module("darts.dataprocessing.transformers")
            scaler = transformers.Scaler()
            series = scaler.fit_transform(series)
        model = getattr(importlib.import_module("darts.models"), self.MODEL)(**self.ARGUMENTS)
        model.fit(series)
        return model, scaler

    def _predict(self, fitted: Any, horizon: int) -> Any:
        model, scaler = fitted
        samples = self.SPEC.samples
        forecast = (
            model.predict(horizon, num_samples=samples) if samples > 1 else model.predict(horizon)
        )
        if scaler is not None:
            forecast = scaler.inverse_transform(forecast)
        if samples > 1:
            forecast = forecast.quantile(0.5)
        return np.asarray(forecast.values()).ravel()


def _adapter(model: str, spec: Spec) -> type[DartsModel]:
    cls = adapted(DartsModel, NAMESPACE, model, spec.arguments, spec.outputs, spec.min_months)
    cls.SPEC = spec
    return cls


#: Each adapted model's class, by the library's name for it.
ADAPTERS: Final[Mapping[str, type[DartsModel]]] = MappingProxyType(
    {model: _adapter(model, spec) for model, spec in MODELS.items()}
)
#: The adapter's registry entries: each refused, with the extra to install, when its module
#: isn't installed.
ENTRIES: Final = tuple(
    Entry(ADAPTERS[model].NAME, ADAPTERS[model], extra=EXTRAS[spec.module], module=spec.module)
    for model, spec in MODELS.items()
)
