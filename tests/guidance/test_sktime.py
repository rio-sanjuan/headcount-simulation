"""Tests for the sktime adapter. They need the `sktime` extra, which the extras CI job
installs, and skip where it isn't installed."""

from __future__ import annotations

import importlib.util

import numpy as np
import pytest

from hcsim.guidance.adapters.sktime import ADAPTERS, EXTRA, MODELS, NAMESPACE
from hcsim.guidance.errors import ForecastError
from hcsim.guidance.models import MODELS as REGISTRY
from hcsim.guidance.models import build
from hcsim.model.interface import ProbabilisticOutput, TimeSeriesModel

RNG = np.random.default_rng(3)
SEASONAL = (20 + 5 * np.sin(2 * np.pi * np.arange(48) / 12) + RNG.poisson(2, 48)).astype(np.float64)


def test_the_models_are_registered_and_refused_without_the_extra() -> None:
    for model, spec in MODELS.items():
        name = f"{NAMESPACE}.{model}"
        assert name == ADAPTERS[model].NAME
        problems = REGISTRY.problems([name])
        if importlib.util.find_spec(spec.module) is not None:
            assert problems == []
        else:
            assert problems == [
                f"model {name!r} needs the {EXTRA!r} extra; install it with "
                f"`uv sync --extra {EXTRA}`"
            ]
    assert ProbabilisticOutput.QUANTILES in ADAPTERS["ThetaForecaster"].OUTPUTS
    assert frozenset({ProbabilisticOutput.NONE}) == ADAPTERS["ReductionLinear"].OUTPUTS


@pytest.mark.extras
@pytest.mark.parametrize("model", sorted(MODELS))
def test_each_model_forecasts_reproducibly(model: str) -> None:
    pytest.importorskip("sktime")
    pytest.importorskip(MODELS[model].module)
    first = build(f"sktime.{model}").fit(SEASONAL).predict(12)
    assert isinstance(build(f"sktime.{model}"), TimeSeriesModel)
    assert first.shape == (12,)
    assert np.isfinite(first).all()
    assert np.array_equal(first, build(f"sktime.{model}").fit(SEASONAL).predict(12))
    assert 5 < first.mean() < 40


@pytest.mark.extras
def test_the_composed_pipeline_differs_from_its_plain_reduction() -> None:
    pytest.importorskip("sktime")
    plain = build("sktime.ReductionLinear").fit(SEASONAL).predict(12)
    composed = build("sktime.DeseasonalizedReduction").fit(SEASONAL).predict(12)
    assert not np.allclose(plain, composed)


@pytest.mark.extras
def test_too_short_a_history_is_refused_before_the_library_runs() -> None:
    pytest.importorskip("sktime")
    with pytest.raises(ForecastError, match="needs at least 36 months"):
        build("sktime.DeseasonalizedReduction").fit(SEASONAL[:30])
