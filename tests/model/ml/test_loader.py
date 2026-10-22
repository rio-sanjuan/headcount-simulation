"""Tests for loading a run's models: conformance and probes, with failures per domain."""

from __future__ import annotations

from pathlib import Path
from typing import ClassVar

import numpy as np

from hcsim.model.ml.arms import DEFAULT_MATRIX
from hcsim.model.ml.classifiers import LevelOnlyModel
from hcsim.model.ml.entity import ConstantModel
from hcsim.model.ml.generative import ConditionalCategorical, EmpiricalAttributeGenerator
from hcsim.model.ml.group import NegativeBinomialGrowthModel
from hcsim.model.ml.loader import Probe, conformance, load_models, probe
from hcsim.model.ml.persistence import save_model

FEATURES = np.column_stack([np.arange(6) % 3, np.zeros(6)]).astype(float)
PROBES = {
    "attrition": Probe(features=FEATURES),
    "growth": Probe(groups=np.array([0, 1, -1])),
    "job_family": Probe(contexts=np.array([[0], [1], [-1]]), n_categories=3),
}


class Broken:
    """Claims to be a classifier, but scores out of range."""

    ALGORITHM_NAME: ClassVar[str] = "level_only"

    def compute_probability(self, features: np.ndarray) -> np.ndarray:
        return np.full(len(features), 1.5)


class Raises(Broken):
    def compute_probability(self, features: np.ndarray) -> np.ndarray:
        raise RuntimeError("no data")


def saved(tmp_path: Path) -> dict[str, str]:
    rng = np.random.default_rng(0)
    labels = (rng.random(600) < 0.1).astype(float)
    level = LevelOnlyModel(0).fit(np.column_stack([rng.integers(0, 3, 600), np.zeros(600)]), labels)
    growth = NegativeBinomialGrowthModel(2).fit(rng.poisson(2.0, size=(12, 2)).astype(float))
    families = ConditionalCategorical(3, min_count=1).fit(
        np.array([[0], [1], [1]]), np.array([0, 1, 2])
    )
    return {
        "attrition": save_model(level, tmp_path, "attrition", {}).version,
        "growth": save_model(growth, tmp_path, "growth", {}).version,
        "job_family": save_model(families, tmp_path, "job_family", {}).version,
    }


def test_conforming_models_load_and_pass_their_probes(tmp_path: Path) -> None:
    report = load_models(tmp_path, saved(tmp_path), DEFAULT_MATRIX, PROBES)
    assert report.ok
    assert sorted(report.models) == ["attrition", "growth", "job_family"]
    model, artifact = report.models["growth"]
    assert artifact.domain == "growth"
    assert model.ALGORITHM_NAME == "negative_binomial_growth"  # type: ignore[attr-defined]


def test_every_failure_is_recorded_against_its_domain(tmp_path: Path) -> None:
    chosen = saved(tmp_path)
    broken = save_model(Broken(), tmp_path, "promotion", {}).version
    raises = save_model(Raises(), tmp_path, "cancellation", {}).version
    wrong_family = save_model(ConstantModel(0.1), tmp_path, "growth", {}).version
    report = load_models(
        tmp_path,
        chosen
        | {
            "promotion": broken,
            "cancellation": raises,
            "growth": wrong_family,
            "external_fill": "v1-missing",
            "reorg": "v1-x",
            "job_family": chosen["job_family"],
        },
        DEFAULT_MATRIX,
        PROBES | {"promotion": Probe(features=FEATURES), "cancellation": Probe(features=FEATURES)},
    )
    assert not report.ok
    assert report.failures["promotion"] == ["probabilities must be finite and lie in [0, 1]"]
    assert report.failures["cancellation"] == ["the probe raised RuntimeError: no data"]
    assert "'constant' isn't a count algorithm" in report.failures["growth"]
    assert "isn't a model artifact" in report.failures["external_fill"][0]
    assert "no arms for domain 'reorg'" in report.failures["reorg"][0]
    assert sorted(report.models) == ["attrition", "job_family"]  # the good ones still load


def test_a_domain_without_a_probe_fails(tmp_path: Path) -> None:
    chosen = saved(tmp_path)
    report = load_models(tmp_path, {"attrition": chosen["attrition"]}, DEFAULT_MATRIX, {})
    assert report.failures == {"attrition": ["no probe input for this domain"]}


def test_conformance_names_what_each_family_needs() -> None:
    assert conformance(object(), "classifier") == [
        "the model has no ALGORITHM_NAME",
        "a classifier must score rows: it needs compute_probability",
    ]
    assert "a count model needs rate_for and sample" in conformance(ConstantModel(0.1), "count")
    assert "a generator needs sample" in conformance(ConstantModel(0.1), "generator")
    assert "no model family 'tree'" in conformance(ConstantModel(0.1), "tree")


def test_probes_check_shape_and_range() -> None:
    assert probe(ConstantModel(0.1), "classifier", Probe()) == ["a classifier probe needs features"]
    assert probe(ConstantModel(0.1), "count", Probe()) == ["a count probe needs groups"]
    assert "needs contexts" in probe(ConstantModel(0.1), "generator", Probe())[0]

    class Short(Broken):
        def compute_probability(self, features: np.ndarray) -> np.ndarray:
            return np.zeros(1)

    assert probe(Short(), "classifier", Probe(features=FEATURES)) == ["scored (1,) for 6 rows"]

    class BadCounts:
        def rate_for(self, group: int) -> float:
            return -1.0

        def sample(self, groups: np.ndarray, rng: np.random.Generator) -> np.ndarray:
            return np.array([-1])

    problems = probe(BadCounts(), "count", Probe(groups=np.array([0, 1])))
    assert problems == [
        "rates must be finite and non-negative",
        "counts must be non-negative, one per group",
    ]

    class BadDraws:
        def __init__(self, out: np.ndarray) -> None:
            self.out = out

        def sample(self, contexts: np.ndarray, rng: np.random.Generator) -> np.ndarray:
            return self.out

    contexts = Probe(contexts=np.zeros((2, 1), np.int64), n_categories=3)
    assert probe(BadDraws(np.zeros(1)), "generator", contexts) == ["drew (1,) for 2 contexts"]
    assert probe(BadDraws(np.array([0, 5])), "generator", contexts) == ["outcomes must lie in 0..2"]


def test_a_generator_of_whole_roles_draws_one_per_cost_center() -> None:
    roles = np.array([[0, 1, 2, 0, 0, 1], [1, 0, 1, 1, 1, 0]], dtype=np.int64)
    generator = EmpiricalAttributeGenerator(min_rows=1).fit(np.array([0, 1]), roles)
    contexts = np.array([[0, 5], [1, 5], [7, 5]])
    assert probe(generator, "generator", Probe(contexts=contexts, n_categories=3)) == []
    # Its job family must lie within the probe's categories.
    assert probe(generator, "generator", Probe(contexts=contexts, n_categories=2)) == [
        "roles must carry codes, with a job family in 0..1"
    ]
    np.testing.assert_allclose(generator.distribution(0, 2, 3), [0, 0, 1])
    np.testing.assert_allclose(generator.distribution(7, 2, 3), [0, 0.5, 0.5])


def test_a_generator_of_roles_of_the_wrong_width_fails_its_probe() -> None:
    class Narrow(EmpiricalAttributeGenerator):
        def sample(self, cost_center: int, n: int, rng: np.random.Generator) -> np.ndarray:
            return np.zeros((n, 2), dtype=np.int64)

    narrow = Narrow().fit(np.zeros(1, np.int64), np.zeros((1, 6), np.int64))
    contexts = np.array([[0], [0]])
    assert probe(narrow, "generator", Probe(contexts=contexts, n_categories=3)) == [
        "drew (2, 2) for 2 contexts"
    ]
