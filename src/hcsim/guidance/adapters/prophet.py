"""The Prophet adapter, from the optional `prophet` extra.

`prophet.Prophet` fits Prophet's additive model to a monthly series: a piecewise-linear trend
and yearly seasonality, with weekly and daily seasonality off, as a monthly series has
neither. Prophet needs dates, so the series is laid on consecutive month starts; the season's
phase is learned from the data, so which month the dates start on doesn't matter. Given the
calendar, it reads the working-day count (`weekdays`) as an extra regressor. Its Stan backend
is silenced.

A configuration that names it without the extra is refused at load, with the install command.
Fitting, forecasting, and failing work as for every adapter (see `base`).
"""

from __future__ import annotations

import importlib
import logging
from typing import Any, ClassVar, Final

import pandas as pd

from hcsim.guidance.adapters.base import Floats, LibraryModel
from hcsim.guidance.errors import ForecastError
from hcsim.guidance.registry import Entry
from hcsim.model.interface import ProbabilisticOutput

NAMESPACE: Final = "prophet"
EXTRA: Final = "prophet"
#: Where the series' dates start. Any month start would do.
ORIGIN: Final = "2000-01-01"


class ProphetModel(LibraryModel):
    NAME: ClassVar[str] = f"{NAMESPACE}.Prophet"
    OUTPUTS: ClassVar[frozenset[ProbabilisticOutput]] = frozenset(
        {ProbabilisticOutput.NONE, ProbabilisticOutput.QUANTILES, ProbabilisticOutput.SAMPLE_PATHS}
    )
    MODEL: ClassVar[str] = "Prophet"
    MIN_MONTHS: ClassVar[int] = 24
    COVARIATES: ClassVar[tuple[str, ...]] = ("weekdays",)

    def _fit(self, values: Floats) -> Any:
        for name in ("cmdstanpy", "prophet"):
            logging.getLogger(name).setLevel(logging.ERROR)
        library = importlib.import_module("prophet")
        frame = pd.DataFrame(
            {"ds": pd.date_range(ORIGIN, periods=len(values), freq="MS"), "y": values}
        )
        model = library.Prophet(
            yearly_seasonality=True, weekly_seasonality=False, daily_seasonality=False
        )
        if self.covariates_given:
            assert self.past_covariates is not None
            for k, column in enumerate(self.COVARIATES):
                model.add_regressor(column)
                frame[column] = self.past_covariates[:, k]
        return model.fit(frame)

    def _predict(self, fitted: Any, horizon: int) -> Any:
        future = fitted.make_future_dataframe(periods=horizon, freq="MS", include_history=False)
        if self.covariates_given:
            ahead = self.future_covariates
            if ahead is None or len(ahead) < horizon:
                raise ForecastError(
                    f"{self.NAME} needs its covariates for the {horizon} months ahead"
                )
            for k, column in enumerate(self.COVARIATES):
                future[column] = ahead[:horizon, k]
        return fitted.predict(future)["yhat"].to_numpy()


ENTRIES: Final = (Entry(ProphetModel.NAME, ProphetModel, extra=EXTRA, module="prophet"),)
