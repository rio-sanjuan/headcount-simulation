"""Protocol conformance: every algorithm the arm matrix allows has a model that serves its
family, and fitted models of each pass the loader's probe."""

from __future__ import annotations

import numpy as np
import pytest

from hcsim.model.core.layouts import EmployeeColumn
from hcsim.model.ml.arms import ALGORITHMS
from hcsim.model.ml.classifiers import BoostedModel, FeatureMask, LevelOnlyModel, LogisticModel
from hcsim.model.ml.entity import ConstantModel, NaiveRateModel
from hcsim.model.ml.generative import ConditionalCategorical, EmpiricalAttributeGenerator
from hcsim.model.ml.group import BackfillRateModel, NegativeBinomialGrowthModel, PoissonGrowthModel
from hcsim.model.ml.loader import Probe, conformance, probe
from hcsim.model.ml.markov import MarkovModel

RNG = np.random.default_rng(0)
FEATURES = np.column_stack(
    [RNG.integers(0, 3, 400), RNG.normal(size=400)] + [RNG.integers(0, 4, 400)] * 6
).astype(float)
LABELS = (RNG.random(400) < 0.2).astype(float)
MASK = FeatureMask((0, 1), categorical=(0,))

FITTED: dict[str, object] = {
    "constant": ConstantModel(0.1),
    "level_only": LevelOnlyModel(0).fit(FEATURES, LABELS),
    "naive_rate": NaiveRateModel([0]).fit(FEATURES, LABELS),
    "logistic": LogisticModel(MASK).fit(FEATURES, LABELS),
    "boosted": BoostedModel(MASK, max_iter=10).fit(FEATURES, LABELS),
    "backfill_rate": BackfillRateModel(
        4, min_departures=5, group_column=EmployeeColumn.COST_CENTER
    ).fit(FEATURES[:, EmployeeColumn.COST_CENTER].astype(np.int64), LABELS.astype(bool)),
    "poisson_growth": PoissonGrowthModel(2).fit(RNG.poisson(2.0, (12, 2)).astype(float)),
    "negative_binomial_growth": NegativeBinomialGrowthModel(2).fit(
        RNG.poisson(2.0, (12, 2)).astype(float)
    ),
    "conditional_categorical": ConditionalCategorical(3, min_count=1).fit(
        np.array([[0], [1], [1]]), np.array([0, 1, 2])
    ),
    "markov": MarkovModel(3).fit(np.array([0, 1, 2]), np.array([1, 2, 0]), np.array([0, 1])),
}
PROBES = {
    "classifier": Probe(features=FEATURES[:20]),
    "count": Probe(groups=np.array([0, 1, -1])),
    "generator": Probe(contexts=np.array([[0], [1], [-1]]), n_categories=3),
}
CASES = [(family, algorithm) for family, names in ALGORITHMS.items() for algorithm in sorted(names)]


def test_every_allowed_algorithm_has_a_model() -> None:
    missing = sorted({a for _, a in CASES} - set(FITTED) - {"empirical_attributes", "markov"})
    assert missing == []
    for name, model in FITTED.items():
        assert name == model.ALGORITHM_NAME  # type: ignore[attr-defined]


@pytest.mark.parametrize(("family", "algorithm"), [c for c in CASES if c[1] in FITTED])
def test_each_model_conforms_and_passes_its_probe(family: str, algorithm: str) -> None:
    model = FITTED[algorithm]
    assert conformance(model, family) == []
    assert probe(model, family, PROBES[family]) == []


def test_the_generators_that_draw_whole_rows_or_states_conform_too() -> None:
    """The empirical generator draws attribute rows for a cost center, and the Markov model
    draws next states: both conform as generators, and their draws are probed on their own."""
    empirical = EmpiricalAttributeGenerator(min_rows=1).fit(
        np.zeros(5, np.int64), np.zeros((5, 6), np.int64)
    )
    assert conformance(empirical, "generator") == []
    assert conformance(FITTED["markov"], "generator") == []
    markov = FITTED["markov"]
    drawn = markov.next_states(np.array([0, 1, 2]), np.random.default_rng(1))  # type: ignore[attr-defined]
    assert ((drawn >= 0) & (drawn < 3)).all()
