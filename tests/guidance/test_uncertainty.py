"""Tests for sample paths, the quantiles read from them, and the artifacts that carry them."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import duckdb
import numpy as np
import pytest

from hcsim.guidance.adapters.statsforecast import StatsForecastModel
from hcsim.guidance.baselines import NegativeBinomialGLM
from hcsim.guidance.capabilities import SamplingModel
from hcsim.guidance.config import GuidanceConfig, GuidanceConfigError
from hcsim.guidance.generation import generate
from hcsim.guidance.models import MODELS, build, capabilities
from hcsim.guidance.pipeline import run
from hcsim.guidance.selection import Selection, SelectionConfig, UncertaintySettings
from hcsim.model.core.guidance import GuidanceForecast
from hcsim.model.interface import ProbabilisticOutput
from hcsim.synth.config import preset
from hcsim.synth.world import generate_world, write_world

PATHS = ProbabilisticOutput.SAMPLE_PATHS
MONTHS = np.arange(60, dtype=np.float64)
LEVEL = np.exp(4.0 + 0.2 * np.sin(2 * np.pi * MONTHS / 12))
COUNTS = np.random.default_rng(4).poisson(LEVEL).astype(np.float64)


def test_every_model_that_declares_sample_paths_can_draw_them() -> None:
    for name in MODELS.names:
        if PATHS in capabilities(name).outputs:
            assert callable(getattr(MODELS.factory(name), "sample", None)), name


def test_the_glms_draw_from_their_fitted_distribution() -> None:
    rng = np.random.default_rng(0)
    poisson = build("native.PoissonGLM").fit(COUNTS)
    assert isinstance(poisson, SamplingModel)
    draws = poisson.sample(6, 4000, rng)
    mean = poisson.predict(6)
    assert draws.shape == (4000, 6)
    assert np.allclose(draws.mean(axis=0), mean, rtol=0.03)
    assert np.allclose(draws.var(axis=0), mean, rtol=0.1)  # Poisson: variance is the mean
    lumpy = np.random.default_rng(5).negative_binomial(5, 5 / (5 + LEVEL)).astype(np.float64)
    nb = build("native.NegativeBinomialGLM").fit(lumpy)
    assert isinstance(nb, NegativeBinomialGLM)
    alpha, mu = nb.dispersion, nb.predict(6)
    variance = nb.sample(6, 8000, rng).var(axis=0)
    assert np.allclose(variance, mu + alpha * mu**2, rtol=0.15)


def test_an_interval_model_draws_the_spread_its_interval_implies() -> None:
    model = build("statsforecast.AutoETS").fit(COUNTS)
    assert isinstance(model, StatsForecastModel)
    band = model._fitted.predict(h=6, level=[80])
    draws = model.sample(6, 20000, np.random.default_rng(1))
    low, high = np.quantile(draws, [0.1, 0.9], axis=0)
    width = np.asarray(band["hi-80"]) - np.asarray(band["lo-80"])
    assert np.allclose(high - low, width, rtol=0.05)


CONFIG = SelectionConfig(
    suite=("native.NegativeBinomialGLM", "statsforecast.AutoETS"),
    uncertainty=UncertaintySettings(paths=200, seed=7),
)
BOTH = Selection({}, {"native.NegativeBinomialGLM": 0.75, "statsforecast.AutoETS": 0.25}, {}, "x")


def test_paths_mix_the_models_in_proportion_to_their_weights_reproducibly() -> None:
    first = generate(BOTH, COUNTS, 6, CONFIG, stream=3)
    assert first.paths is not None
    assert first.paths.shape == (200, 6)
    assert np.allclose(first.paths.mean(axis=0), first.values, rtol=0.1)
    again = generate(BOTH, COUNTS, 6, CONFIG, stream=3)
    assert again.paths is not None
    assert np.array_equal(first.paths, again.paths)
    other = generate(BOTH, COUNTS, 6, CONFIG, stream=4)
    assert other.paths is not None
    assert not np.array_equal(first.paths, other.paths)
    plain = generate(BOTH, COUNTS, 6, SelectionConfig(suite=CONFIG.suite))
    assert plain.paths is None
    assert np.array_equal(plain.values, first.values)


def test_a_model_that_can_t_sample_leaves_its_series_point_only() -> None:
    mixed = Selection({}, {"native.NegativeBinomialGLM": 0.5, "native.SeasonalNaive": 0.5}, {}, "x")
    result = generate(mixed, COUNTS, 6, CONFIG)
    assert result.paths is None
    assert result.point_only == "native.SeasonalNaive can't sample paths"


def test_the_uncertainty_settings_are_checked_at_load() -> None:
    config = GuidanceConfig.from_mapping(
        {
            "models": ["native.PoissonGLM"],
            "uncertainty": {"enabled": True, "paths": 50, "quantiles": [5, 50, 95]},
        }
    )
    settings = config.selection.uncertainty
    assert settings is not None
    assert settings.paths == 50
    assert config.uncertainty.quantiles == (5, 50, 95)
    assert GuidanceConfig().selection.uncertainty is None
    with pytest.raises(GuidanceConfigError) as error:
        GuidanceConfig.from_mapping({"uncertainty": {"enabled": True, "quantiles": [90, 10]}})
    message = str(error.value)
    assert "levels must be distinct, ascending, and in 1..99" in message
    assert "no model in the suite samples them" in message


# --- in a run ------------------------------------------------------------------------------------


@pytest.fixture(scope="module")
def con(tmp_path_factory: pytest.TempPathFactory) -> Iterator[duckdb.DuckDBPyConnection]:
    path = tmp_path_factory.mktemp("world") / "world.duckdb"
    write_world(generate_world(preset("tiny", seed=4)), path)
    with duckdb.connect(str(path), read_only=True) as connection:
        yield connection


def configured(models: list[str], **uncertainty: object) -> GuidanceConfig:
    return GuidanceConfig.from_mapping(
        {
            "horizon": 6,
            "grouping": "TOPLINE",
            "models": models,
            "combination": {"top_k": len(models)},
            "uncertainty": {"enabled": True, "paths": 100, **uncertainty},
        }
    )


def test_a_run_writes_paths_and_quantiles_into_the_artifact(
    con: duckdb.DuckDBPyConnection, tmp_path: Path
) -> None:
    result = run(con, configured(["native.NegativeBinomialGLM", "statsforecast.AutoETS"]))
    artifact = result.artifact
    assert {ProbabilisticOutput.QUANTILES, PATHS} <= artifact.outputs
    assert artifact.quantile_levels == (10, 50, 90)
    assert artifact.sample_paths is not None
    assert artifact.sample_paths.shape == (4, 100, 6)  # four categories, one group
    assert set(artifact.path_index) == {("ALL", c.value) for c in artifact.categories}
    q = artifact.frame[["q10", "q50", "q90"]].to_numpy()
    assert (np.diff(q, axis=1) >= 0).all()
    assert result.report["uncertainty"]["series_with_paths"] == 4
    assert result.report["uncertainty"]["point_only"] == []
    artifact.save(tmp_path / "g")
    loaded = GuidanceForecast.load(tmp_path / "g")
    assert loaded.fingerprint() == artifact.fingerprint()
    assert loaded.sample_paths is not None
    assert np.array_equal(loaded.sample_paths, artifact.sample_paths)


def test_a_run_with_a_point_only_series_writes_paths_but_no_quantiles(
    con: duckdb.DuckDBPyConnection,
) -> None:
    result = run(con, configured(["native.NegativeBinomialGLM", "native.SeasonalNaive"]))
    artifact, report = result.artifact, result.report["uncertainty"]
    assert ProbabilisticOutput.QUANTILES not in artifact.outputs
    assert report["quantiles"] == []
    assert report["point_only"]
    assert all(entry["reason"].endswith("can't sample paths") for entry in report["point_only"])
    assert report["series_with_paths"] == len(artifact.path_index) < 4


@pytest.mark.extras
@pytest.mark.parametrize(
    "name", ["darts.ExponentialSmoothing", "sktime.ThetaForecaster", "prophet.Prophet"]
)
def test_each_library_s_probabilistic_models_sample_reproducibly(name: str) -> None:
    pytest.importorskip(name.split(".")[0])
    model = build(name).fit(COUNTS)
    assert isinstance(model, SamplingModel)
    state = np.random.get_state()  # noqa: NPY002 - checking it's left as it was
    first = model.sample(6, 300, np.random.default_rng(2))
    assert np.array_equal(first, model.sample(6, 300, np.random.default_rng(2)))
    assert first.shape == (300, 6)
    assert np.allclose(first.mean(axis=0), model.predict(6), rtol=0.15)
    after = np.random.get_state()  # noqa: NPY002
    assert after[0] == state[0]
    assert np.array_equal(after[1], state[1])
