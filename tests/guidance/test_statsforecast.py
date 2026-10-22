"""Tests for the statsforecast adapter."""

from __future__ import annotations

import subprocess
import sys
from collections.abc import Iterator

import duckdb
import numpy as np
import pytest

from hcsim.guidance.adapters.statsforecast import ADAPTERS, MODELS, NAMESPACE
from hcsim.guidance.config import GuidanceConfig
from hcsim.guidance.errors import ForecastError
from hcsim.guidance.models import MODELS as REGISTRY
from hcsim.guidance.models import build
from hcsim.guidance.pipeline import run
from hcsim.model.interface import ProbabilisticOutput, TimeSeriesModel
from hcsim.synth.config import preset
from hcsim.synth.world import generate_world, write_world

RNG = np.random.default_rng(3)
#: Four years of a seasonal count series with noise.
SEASONAL = (20 + 5 * np.sin(2 * np.pi * np.arange(48) / 12) + RNG.poisson(2, 48)).astype(np.float64)


def test_every_adapted_model_is_registered_under_the_library_s_namespace() -> None:
    names = {f"{NAMESPACE}.{model}" for model in MODELS}
    assert names <= set(REGISTRY.names)
    assert REGISTRY.problems(sorted(names)) == []
    for model, cls in ADAPTERS.items():
        assert f"statsforecast.{model}" == cls.NAME
        assert isinstance(cls(), TimeSeriesModel)
        assert ProbabilisticOutput.NONE in cls.OUTPUTS
    assert ProbabilisticOutput.QUANTILES in ADAPTERS["AutoETS"].OUTPUTS
    assert frozenset({ProbabilisticOutput.NONE}) == ADAPTERS["CrostonClassic"].OUTPUTS


@pytest.mark.parametrize("model", sorted(MODELS))
def test_every_adapted_model_forecasts_a_monthly_series(model: str) -> None:
    forecast = build(f"statsforecast.{model}").fit(SEASONAL).predict(12)
    assert forecast.shape == (12,)
    assert np.isfinite(forecast).all()


def test_the_library_s_benchmarks_match_the_native_ones() -> None:
    for twin, native in (
        ("Naive", "last_observed"),
        ("SeasonalNaive", "seasonal_naive"),
        ("HistoricAverage", "historical_average"),
        ("RandomWalkWithDrift", "drift"),
    ):
        adapted = build(f"statsforecast.{twin}").fit(SEASONAL).predict(12)
        assert np.allclose(adapted, build(native).fit(SEASONAL).predict(12)), twin


def test_a_flat_series_forecasts_flat() -> None:
    forecast = build("statsforecast.AutoETS").fit(np.full(36, 7.0)).predict(6)
    assert np.allclose(forecast, 7.0)


def test_a_model_that_can_t_forecast_raises_a_forecast_error() -> None:
    model = build("statsforecast.AutoETS")
    with pytest.raises(ForecastError, match="fit before"):
        model.predict(3)
    with pytest.raises(ForecastError, match="at least 2 months"):
        model.fit(np.ones(1))
    broken = np.r_[SEASONAL[:24], np.nan, SEASONAL[25:]]
    with pytest.raises(ForecastError, match=r"statsforecast\.AutoETS"):
        model.fit(broken).predict(3)


def test_naming_a_model_doesn_t_import_the_library() -> None:
    code = (
        "import sys, hcsim.guidance.models as m; m.build('statsforecast.AutoETS'); "
        "print('statsforecast' in sys.modules)"
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert out.stdout.strip() == "False"


@pytest.fixture(scope="module")
def con(tmp_path_factory: pytest.TempPathFactory) -> Iterator[duckdb.DuckDBPyConnection]:
    path = tmp_path_factory.mktemp("world") / "world.duckdb"
    write_world(generate_world(preset("tiny", seed=4)), path)
    with duckdb.connect(str(path), read_only=True) as connection:
        yield connection


def test_a_run_can_forecast_with_statsforecast_models(con: duckdb.DuckDBPyConnection) -> None:
    config = GuidanceConfig.from_mapping(
        {
            "horizon": 6,
            "grouping": "TOPLINE",
            "models": ["statsforecast.AutoETS", "statsforecast.Theta", "seasonal_naive"],
        }
    )
    result = run(con, config)
    weights = {m for s in result.report["selection"].values() for m in s["weights"]}
    assert weights <= {"statsforecast.AutoETS", "statsforecast.Theta", "seasonal_naive"}
    assert weights & {"statsforecast.AutoETS", "statsforecast.Theta"}
    assert result.artifact.horizon == 6
