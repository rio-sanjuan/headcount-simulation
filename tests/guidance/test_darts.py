"""Tests for the darts adapter. They need the `darts` extra, and the deep models `darts-deep`;
the extras CI job installs `darts`, and they skip where it isn't installed."""

from __future__ import annotations

import importlib.util
import logging
from collections.abc import Iterator

import numpy as np
import pytest

from hcsim.guidance.adapters.darts import ADAPTERS, EXTRAS, MODELS, NAMESPACE
from hcsim.guidance.errors import ForecastError
from hcsim.guidance.models import MODELS as REGISTRY
from hcsim.guidance.models import build
from hcsim.model.interface import ProbabilisticOutput, TimeSeriesModel

RNG = np.random.default_rng(3)
SEASONAL = (20 + 5 * np.sin(2 * np.pi * np.arange(48) / 12) + RNG.poisson(2, 48)).astype(np.float64)


def installed(module: str) -> bool:
    return importlib.util.find_spec(module) is not None


def loads(module: str) -> bool:
    """Whether `module` imports here, as LightGBM doesn't without its OpenMP runtime."""
    try:
        importlib.import_module(module)
    except (ImportError, OSError):
        return False
    return True


def test_the_models_are_registered_and_refused_without_their_extra() -> None:
    for model, spec in MODELS.items():
        name = f"{NAMESPACE}.{model}"
        assert name == ADAPTERS[model].NAME
        assert name in REGISTRY
        problems = REGISTRY.problems([name])
        if installed(spec.module):
            assert problems == []
        else:
            extra = EXTRAS[spec.module]
            assert problems == [
                f"model {name!r} needs the {extra!r} extra; install it with "
                f"`uv sync --extra {extra}`"
            ]
    assert EXTRAS["torch"] == "darts-deep"
    assert ProbabilisticOutput.SAMPLE_PATHS in ADAPTERS["ExponentialSmoothing"].OUTPUTS
    assert ProbabilisticOutput.QUANTILES in ADAPTERS["TFTModel"].OUTPUTS


@pytest.fixture(autouse=True)
def quiet() -> Iterator[None]:
    logging.disable(logging.WARNING)
    yield
    logging.disable(logging.NOTSET)


LOCAL = [m for m, spec in MODELS.items() if spec.module == "darts"]


@pytest.mark.extras
@pytest.mark.parametrize("model", LOCAL)
def test_each_local_model_forecasts_reproducibly(model: str) -> None:
    pytest.importorskip("darts")
    first = build(f"darts.{model}").fit(SEASONAL).predict(12)
    again = build(f"darts.{model}").fit(SEASONAL).predict(12)
    assert isinstance(build(f"darts.{model}"), TimeSeriesModel)
    assert first.shape == (12,)
    assert np.isfinite(first).all()
    assert np.array_equal(first, again)
    assert 5 < first.mean() < 40  # the series' level, roughly


@pytest.mark.extras
def test_the_gradient_boosted_model_forecasts_where_lightgbm_loads() -> None:
    pytest.importorskip("darts")
    if not loads("lightgbm"):
        with pytest.raises(ForecastError, match=r"darts\.LightGBMModel can't fit"):
            build("darts.LightGBMModel").fit(SEASONAL)
        pytest.skip("LightGBM can't load here, so the model refuses with a forecast error")
    forecast = build("darts.LightGBMModel").fit(SEASONAL).predict(12)
    assert np.isfinite(forecast).all()
    assert np.array_equal(forecast, build("darts.LightGBMModel").fit(SEASONAL).predict(12))


@pytest.mark.extras
@pytest.mark.parametrize("model", ["NBEATSModel", "TFTModel"])
def test_the_deep_models_train_seeded_on_the_scaled_series(model: str) -> None:
    pytest.importorskip("darts")
    pytest.importorskip("torch")
    first = build(f"darts.{model}").fit(SEASONAL).predict(12)
    assert np.array_equal(first, build(f"darts.{model}").fit(SEASONAL).predict(12))
    assert 5 < first.mean() < 40


@pytest.mark.extras
def test_too_short_a_history_is_refused_before_the_library_runs() -> None:
    pytest.importorskip("darts")
    with pytest.raises(ForecastError, match="needs at least 24 months"):
        build("darts.ExponentialSmoothing").fit(SEASONAL[:20])
