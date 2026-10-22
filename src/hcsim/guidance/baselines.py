"""Fixed-form baselines: models whose form is chosen, not searched for.

- **Log-count ARIMA** (`log_count_arima`) fits ARIMA(1,0,0)(1,0,0)[12] with a mean to
  `log(1 + count)` and forecasts back through the inverse, so a forecast is never negative.
- **Logit-rate ARIMA** (`logit_rate_arima`) models a flow as a share of its exposure, as
  attrition is a share of headcount. It fits the same ARIMA to the logit of the share, clipped
  away from 0 and 1, and forecasts the share times the exposure's forecast. It's an
  `ExposureModel`: it needs the exposure's history to fit and its forecast to predict.
- **Poisson and negative-binomial GLMs with harmonics** (`poisson_glm`,
  `negative_binomial_glm`) regress the count, through a log link, on a level, a linear trend,
  and the first two annual harmonics. Both harmonics' sine and cosine are fitted, so the
  season's phase is learned whatever month the series starts in. The negative binomial
  estimates its dispersion, so it fits overdispersed counts the Poisson can't.

The ARIMAs run on statsforecast and the GLMs on statsmodels, both in the core install and
imported when a model first fits. Fitting, forecasting, and failing work as for every adapter
(see `adapters.base`).
"""

from __future__ import annotations

import importlib
from typing import Any, ClassVar, Final, Protocol, Self, runtime_checkable

import numpy as np

from hcsim.guidance.adapters.base import Floats, LibraryModel
from hcsim.guidance.errors import ForecastError
from hcsim.model.interface import ProbabilisticOutput

SEASON: Final = 12
#: How close to 0 or 1 a share may come before its logit.
EPSILON: Final = 1e-4
_POINT = frozenset({ProbabilisticOutput.NONE})
_PARAMETRIC = frozenset({ProbabilisticOutput.NONE, ProbabilisticOutput.PARAMETRIC})


@runtime_checkable
class ExposureModel(Protocol):
    """A model of a share of an exposure. Before it fits, it's given the exposure's history,
    as long as the series', and the exposure's values for the months it will forecast."""

    def expose(self, past: Floats, future: Floats | None) -> None: ...


def _arima(values: Floats) -> Any:
    models = importlib.import_module("statsforecast.models")
    model = models.ARIMA(
        order=(1, 0, 0), seasonal_order=(1, 0, 0), season_length=SEASON, include_mean=True
    )
    return model.fit(y=values)


class LogCountARIMA(LibraryModel):
    NAME: ClassVar[str] = "log_count_arima"
    OUTPUTS: ClassVar[frozenset[ProbabilisticOutput]] = _POINT
    MIN_MONTHS: ClassVar[int] = 2 * SEASON

    def _fit(self, values: Floats) -> Any:
        if (values < 0).any():
            raise ForecastError(f"{self.NAME} fits counts, which can't be negative")
        return _arima(np.log1p(values))

    def _predict(self, fitted: Any, horizon: int) -> Any:
        return np.maximum(np.expm1(fitted.predict(h=horizon)["mean"]), 0.0)


class LogitRateARIMA(LibraryModel):
    NAME: ClassVar[str] = "logit_rate_arima"
    OUTPUTS: ClassVar[frozenset[ProbabilisticOutput]] = _POINT
    MIN_MONTHS: ClassVar[int] = 2 * SEASON

    def __init__(self) -> None:
        super().__init__()
        self._past: Floats | None = None
        self._future: Floats | None = None

    def expose(self, past: Floats, future: Floats | None) -> None:
        self._past = np.asarray(past, dtype=np.float64)
        self._future = None if future is None else np.asarray(future, dtype=np.float64)

    def _fit(self, values: Floats) -> Any:
        if self._past is None:
            raise ForecastError(f"{self.NAME} needs the series' exposure, and has none")
        if self._past.shape != values.shape:
            raise ForecastError(f"{self.NAME} needs one exposure value per month of history")
        if (self._past <= 0).any():
            raise ForecastError(f"{self.NAME} needs a positive exposure in every month")
        share = np.clip(values / self._past, EPSILON, 1 - EPSILON)
        return _arima(np.log(share / (1 - share)))

    def _predict(self, fitted: Any, horizon: int) -> Any:
        if self._future is None or len(self._future) < horizon:
            raise ForecastError(f"{self.NAME} needs the exposure's forecast for {horizon} months")
        logit = np.asarray(fitted.predict(h=horizon)["mean"], dtype=np.float64)
        return self._future[:horizon] / (1 + np.exp(-logit))


def _design(start: int, months: int) -> Floats:
    """The level, the trend in years, and the first two annual harmonics, by month."""
    t = np.arange(start, start + months, dtype=np.float64)
    angle = 2 * np.pi * t / SEASON
    columns = [np.ones(months), t / SEASON]
    columns += [np.sin(angle), np.cos(angle), np.sin(2 * angle), np.cos(2 * angle)]
    return np.column_stack(columns)


class _HarmonicGLM(LibraryModel):
    OUTPUTS: ClassVar[frozenset[ProbabilisticOutput]] = _PARAMETRIC
    MIN_MONTHS: ClassVar[int] = 2 * SEASON

    def __init__(self) -> None:
        super().__init__()
        self._months = 0

    def _model(self, counts: Floats, design: Floats) -> Any:
        raise NotImplementedError

    def fit(self, history: Floats) -> Self:
        self._months = len(history)
        return super().fit(history)

    def _fit(self, values: Floats) -> Any:
        if (values < 0).any():
            raise ForecastError(f"{self.NAME} fits counts, which can't be negative")
        if not values.any():
            raise ForecastError(f"{self.NAME} can't fit a series that's zero throughout")
        return self._model(values, _design(0, len(values)))

    def _predict(self, fitted: Any, horizon: int) -> Any:
        design = _design(self._months, horizon)
        return np.exp(design @ np.asarray(fitted.params, dtype=np.float64)[: design.shape[1]])


class PoissonGLM(_HarmonicGLM):
    NAME: ClassVar[str] = "poisson_glm"

    def _model(self, counts: Floats, design: Floats) -> Any:
        sm = importlib.import_module("statsmodels.api")
        return sm.GLM(counts, design, family=sm.families.Poisson()).fit()


class NegativeBinomialGLM(_HarmonicGLM):
    NAME: ClassVar[str] = "negative_binomial_glm"

    def _model(self, counts: Floats, design: Floats) -> Any:
        sm = importlib.import_module("statsmodels.api")
        return sm.NegativeBinomial(counts, design).fit(disp=0)

    @property
    def dispersion(self) -> float:
        """The fitted NB2 dispersion `alpha`: the variance is `mu + alpha * mu**2`."""
        if self._fitted is None:
            raise ForecastError("fit before reading the dispersion")
        return float(np.asarray(self._fitted.params)[-1])


#: Every fixed-form baseline.
BASELINES: Final = (LogCountARIMA, LogitRateARIMA, PoissonGLM, NegativeBinomialGLM)
