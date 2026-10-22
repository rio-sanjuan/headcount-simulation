"""The `sktime` adapter: the library's forecasters, reductions, and composed pipelines as
guidance forecasters, from the optional `sktime` extra.

Each model registers as `sktime.<Name>`:

- `ThetaForecaster` and `TBATS`, with a monthly season. TBATS fits trend and Box-Cox off and
  ARMA errors off, which keeps a fit to about a second;
- `ExponentialSmoothing` and `AutoETS`, statsmodels' exponential smoothing through sktime;
- the reduction forecasters, which turn forecasting into regression on the last year's lags,
  applied recursively: `ReductionLinear`, on a linear regression, and `ReductionBoosted`, on
  scikit-learn's histogram gradient boosting, seeded;
- `DeseasonalizedReduction`, a composed pipeline: the series deseasonalized, forecast by the
  linear reduction, and reseasonalized, as sktime's composition builds it.

A configuration that names a model when the extra isn't installed is refused at load, with the
install command. Fitting, forecasting, and failing work as for every adapter (see `base`).
"""

from __future__ import annotations

import importlib
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, ClassVar, Final

import numpy as np
import pandas as pd

from hcsim.guidance.adapters.base import Floats, LibraryModel, adapted
from hcsim.guidance.registry import Entry
from hcsim.model.interface import ProbabilisticOutput

NAMESPACE: Final = "sktime"
EXTRA: Final = "sktime"
SEASON: Final = 12
SEED: Final = 0
_POINT = frozenset({ProbabilisticOutput.NONE})
_INTERVALS = frozenset({ProbabilisticOutput.NONE, ProbabilisticOutput.QUANTILES})


def _forecaster(path: str, **arguments: Any) -> Any:
    module, name = path.rsplit(".", 1)
    return getattr(importlib.import_module(module), name)(**arguments)


def _reduction(regressor: Any) -> Any:
    compose = importlib.import_module("sktime.forecasting.compose")
    return compose.make_reduction(regressor, window_length=SEASON, strategy="recursive")


def _linear() -> Any:
    return _reduction(_forecaster("sklearn.linear_model.LinearRegression"))


def _boosted() -> Any:
    regressor = _forecaster("sklearn.ensemble.HistGradientBoostingRegressor", random_state=SEED)
    return _reduction(regressor)


def _deseasonalized() -> Any:
    compose = importlib.import_module("sktime.forecasting.compose")
    deseasonalize = _forecaster("sktime.transformations.series.detrend.Deseasonalizer", sp=SEASON)
    return compose.TransformedTargetForecaster(
        [("deseasonalize", deseasonalize), ("forecast", _linear())]
    )


@dataclass(frozen=True)
class Spec:
    #: Builds the forecaster, importing the library as it does.
    build: Callable[[], Any]
    #: What it emits beyond a point forecast.
    outputs: frozenset[ProbabilisticOutput]
    #: The fewest months it fits.
    min_months: int
    #: The module the extra provides that the model imports.
    module: str = "sktime"


#: Each adapted model's spec, by name.
MODELS: Final[Mapping[str, Spec]] = MappingProxyType(
    {
        "ThetaForecaster": Spec(
            lambda: _forecaster("sktime.forecasting.theta.ThetaForecaster", sp=SEASON),
            _INTERVALS,
            2 * SEASON,
        ),
        "TBATS": Spec(
            lambda: _forecaster(
                "sktime.forecasting.tbats.TBATS",
                sp=SEASON,
                use_box_cox=False,
                use_trend=False,
                use_damped_trend=False,
                use_arma_errors=False,
                n_jobs=1,
            ),
            _INTERVALS,
            2 * SEASON,
            "tbats",
        ),
        "ExponentialSmoothing": Spec(
            lambda: _forecaster(
                "sktime.forecasting.exp_smoothing.ExponentialSmoothing",
                trend="add",
                seasonal="add",
                sp=SEASON,
            ),
            _POINT,
            2 * SEASON,
        ),
        "AutoETS": Spec(
            lambda: _forecaster("sktime.forecasting.ets.AutoETS", auto=True, sp=SEASON),
            _INTERVALS,
            2 * SEASON,
        ),
        "ReductionLinear": Spec(_linear, _POINT, 2 * SEASON),
        "ReductionBoosted": Spec(_boosted, _POINT, 2 * SEASON),
        "DeseasonalizedReduction": Spec(_deseasonalized, _POINT, 3 * SEASON),
    }
)


class SktimeModel(LibraryModel):
    """One sktime forecaster, fitted to one monthly series."""

    SPEC: ClassVar[Spec]

    def _fit(self, values: Floats) -> Any:
        forecaster = self.SPEC.build()
        forecaster.set_config(remember_data=False)
        forecaster.fit(pd.Series(values))
        return forecaster

    def _predict(self, fitted: Any, horizon: int) -> Any:
        return np.asarray(fitted.predict(np.arange(1, horizon + 1))).ravel()


def _adapter(model: str, spec: Spec) -> type[SktimeModel]:
    cls = adapted(SktimeModel, NAMESPACE, model, {}, spec.outputs, spec.min_months)
    cls.SPEC = spec
    return cls


#: Each adapted model's class, by name.
ADAPTERS: Final[Mapping[str, type[SktimeModel]]] = MappingProxyType(
    {model: _adapter(model, spec) for model, spec in MODELS.items()}
)
#: The adapter's registry entries, each refused with the install command without the extra.
ENTRIES: Final = tuple(
    Entry(ADAPTERS[model].NAME, ADAPTERS[model], extra=EXTRA, module=spec.module)
    for model, spec in MODELS.items()
)
