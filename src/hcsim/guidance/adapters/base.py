"""What every library adapter shares: fitting and forecasting one series, safely.

An adapted model fits one monthly series and forecasts it. The base class does the parts every
library needs the same way:

- it refuses a history too short for the model, and a forecast before a fit;
- a library that can't be loaded, or that fails on a series, raises `ForecastError`, so
  selection scores the model out of that fold rather than stopping the run;
- warnings the library raises while fitting are kept on the model, not raised;
- a forecast must be finite, one value per month of the horizon.

Each library's subclass builds and drives its own model, and each adapted model is its own
class, with its name and outputs as class variables, as `TimeSeriesModel` declares them.
"""

from __future__ import annotations

import warnings
from collections.abc import Mapping
from typing import Any, ClassVar, Self

import numpy as np
from numpy.typing import ArrayLike, NDArray

from hcsim.guidance.errors import ForecastError
from hcsim.model.interface import ProbabilisticOutput

Floats = NDArray[np.float64]


class LibraryModel:
    NAME: ClassVar[str] = ""
    OUTPUTS: ClassVar[frozenset[ProbabilisticOutput]] = frozenset({ProbabilisticOutput.NONE})
    #: The library's class name for the model, and the arguments it's built with.
    MODEL: ClassVar[str] = ""
    ARGUMENTS: ClassVar[Mapping[str, Any]] = {}
    #: The fewest months the model fits.
    MIN_MONTHS: ClassVar[int] = 2

    def __init__(self) -> None:
        self.warnings: list[str] = []
        self._fitted: Any = None

    def _fit(self, values: Floats) -> Any:
        """The library's fitted model for `values`."""
        raise NotImplementedError

    def _predict(self, fitted: Any, horizon: int) -> ArrayLike:
        """The fitted model's point forecast for `horizon` months."""
        raise NotImplementedError

    def fit(self, history: Floats) -> Self:
        values = np.asarray(history, dtype=np.float64)
        if len(values) < self.MIN_MONTHS:
            raise ForecastError(f"{self.NAME} needs at least {self.MIN_MONTHS} months")
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            try:
                self._fitted = self._fit(values)
            except Exception as error:  # the library's failure, to load or on this series
                raise ForecastError(f"{self.NAME} can't fit this series: {error}") from error
        self.warnings = [str(w.message) for w in caught]
        return self

    def predict(self, horizon: int) -> Floats:
        if self._fitted is None:
            raise ForecastError("fit before predicting")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                forecast = self._predict(self._fitted, horizon)
            except Exception as error:
                raise ForecastError(f"{self.NAME} can't forecast: {error}") from error
        result = np.asarray(forecast, dtype=np.float64).ravel()
        if result.shape != (horizon,) or not np.isfinite(result).all():
            raise ForecastError(f"{self.NAME} gave no finite forecast for {horizon} months")
        return result


def adapted[T: LibraryModel](
    base: type[T],
    namespace: str,
    model: str,
    arguments: Mapping[str, Any],
    outputs: frozenset[ProbabilisticOutput],
    min_months: int = 2,
) -> type[T]:
    """The class adapting one library model, named `<namespace>.<model>`."""
    attributes = {
        "NAME": f"{namespace}.{model}",
        "OUTPUTS": outputs,
        "MODEL": model,
        "ARGUMENTS": arguments,
        "MIN_MONTHS": min_months,
    }
    return type(f"{base.__name__}{model}", (base,), attributes)
