"""Parity: the same model family, adapted from different libraries, forecasts alike.

On a synthetic series with a level, a trend, an additive season, and a little noise, each
family's forecasts must agree within a tolerance fixed before the first run:

- the benchmarks, native and statsforecast's twins, exactly;
- exponential smoothing (statsforecast's AutoETS, darts' and sktime's Holt-Winters, and
  sktime's AutoETS) within `TOLERANCE` of each other, relative to the forecast's mean;
- the theta method (statsforecast's, darts', and sktime's) within the same tolerance.

The cross-library families need the darts and sktime extras, so they run in the extras job.
"""

from __future__ import annotations

import itertools
import logging
from collections.abc import Iterator

import numpy as np
import pytest

from hcsim.guidance.models import build

#: The most two forecasts of one family may differ, as the mean absolute difference over the
#: horizon relative to their mean level.
TOLERANCE = 0.05
HORIZON = 12

MONTHS = np.arange(72, dtype=np.float64)
SERIES = 100 + 0.5 * MONTHS + 10 * np.sin(2 * np.pi * MONTHS / 12)
SERIES = SERIES + np.random.default_rng(11).normal(0, 1, len(MONTHS))

FAMILIES = {
    "exponential smoothing": (
        "statsforecast.AutoETS",
        "darts.ExponentialSmoothing",
        "sktime.ExponentialSmoothing",
        "sktime.AutoETS",
    ),
    "theta": ("statsforecast.Theta", "darts.Theta", "sktime.ThetaForecaster"),
}
TWINS = (
    ("native.LastObserved", "statsforecast.Naive"),
    ("native.SeasonalNaive", "statsforecast.SeasonalNaive"),
    ("native.HistoricalAverage", "statsforecast.HistoricAverage"),
    ("native.Drift", "statsforecast.RandomWalkWithDrift"),
)


@pytest.fixture(autouse=True)
def quiet() -> Iterator[None]:
    logging.disable(logging.WARNING)
    yield
    logging.disable(logging.NOTSET)


def gap(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.abs(a - b).mean() / np.abs((a + b) / 2).mean())


@pytest.mark.parametrize(("native", "twin"), TWINS)
def test_the_benchmarks_and_their_twins_forecast_the_same(native: str, twin: str) -> None:
    ours = build(native).fit(SERIES).predict(HORIZON)
    theirs = build(twin).fit(SERIES).predict(HORIZON)
    assert np.allclose(ours, theirs)


@pytest.mark.extras
@pytest.mark.parametrize("family", sorted(FAMILIES))
def test_a_family_forecasts_alike_across_libraries(family: str) -> None:
    pytest.importorskip("darts")
    pytest.importorskip("sktime")
    forecasts = {name: build(name).fit(SERIES).predict(HORIZON) for name in FAMILIES[family]}
    for (a, fa), (b, fb) in itertools.combinations(forecasts.items(), 2):
        assert gap(fa, fb) < TOLERANCE, f"{family}: {a} and {b} differ by {gap(fa, fb):.3f}"
    # And each tracks where the series is heading.
    truth = (
        100 + 0.5 * np.arange(72, 72 + HORIZON) + 10 * np.sin(2 * np.pi * np.arange(72, 84) / 12)
    )
    for name, forecast in forecasts.items():
        assert gap(forecast, truth) < TOLERANCE, name
