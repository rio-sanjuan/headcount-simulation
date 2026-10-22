"""Tests for the conditional categorical generator."""

from __future__ import annotations

import numpy as np
import pytest

from hcsim.model.ml.entity import ModelNotFittedError
from hcsim.model.ml.generative import ConditionalCategorical
from tests.support.stat_check import stat_check


def hierarchy() -> tuple[np.ndarray, np.ndarray]:
    """Two levels: cost centers 0-2 under orgs 0-1. Cost center 0 is all category 0, cost
    center 1 all category 1, and cost center 2 is thin: 3 rows of category 2."""
    contexts = np.array([[0, 0]] * 40 + [[1, 0]] * 40 + [[2, 1]] * 3 + [[3, 1]] * 30)
    outcomes = np.array([0] * 40 + [1] * 40 + [2] * 3 + [2] * 10 + [0] * 20)
    return contexts, outcomes


def test_a_context_reads_its_finest_well_observed_level() -> None:
    contexts, outcomes = hierarchy()
    model = ConditionalCategorical(3, min_count=20).fit(contexts, outcomes)
    assert model.level_used(np.array([0, 0])) == 0
    assert model.distribution(np.array([0, 0])).tolist() == [1.0, 0.0, 0.0]
    assert model.level_used(np.array([2, 1])) == 1  # thin cost center: its org
    np.testing.assert_allclose(model.distribution(np.array([2, 1])), [20 / 33, 0.0, 13 / 33])
    assert model.level_used(np.array([-1, 0])) == 1  # an unknown leaf skips to its org
    assert model.level_used(np.array([9, 9])) is None  # nothing known: the overall mix
    np.testing.assert_allclose(model.distribution(np.array([9, 9])), [60 / 113, 40 / 113, 13 / 113])


def test_smoothing_blends_a_cell_toward_its_parent() -> None:
    contexts, outcomes = hierarchy()
    sharp = ConditionalCategorical(3, min_count=20).fit(contexts, outcomes)
    smooth = ConditionalCategorical(3, min_count=20, smoothing=40.0).fit(contexts, outcomes)
    org = sharp.distribution(np.array([9, 0]))  # org 0 alone: half 0, half 1
    np.testing.assert_allclose(org, [0.5, 0.5, 0.0])
    blended = smooth.distribution(np.array([0, 0]))
    np.testing.assert_allclose(
        blended, (np.array([40.0, 0, 0]) + 40 * smooth.distribution(np.array([9, 0]))) / 80
    )
    assert 0.5 < blended[0] < 1.0


def test_draws_follow_each_context_s_distribution() -> None:
    contexts, outcomes = hierarchy()
    model = ConditionalCategorical(3, min_count=20).fit(contexts, outcomes)
    rng = np.random.default_rng(13)
    asked = np.repeat(np.array([[0, 0], [2, 1]]), 5000, axis=0)
    drawn = model.sample(asked, rng)
    assert (drawn[:5000] == 0).all()
    share = float((drawn[5000:] == 2).mean())
    stat_check(share, 13 / 33, float(np.sqrt((13 / 33) * (20 / 33) / 5000)), label="org draw")
    assert model.sample(np.zeros((0, 2), np.int64), rng).size == 0


def test_bad_generator_inputs_are_refused() -> None:
    with pytest.raises(ValueError, match="at least one category"):
        ConditionalCategorical(0)
    with pytest.raises(ValueError, match="min_count"):
        ConditionalCategorical(2, smoothing=-1)
    model = ConditionalCategorical(2)
    with pytest.raises(ModelNotFittedError):
        model.distribution(np.array([0]))
    with pytest.raises(ValueError, match="one outcome per row"):
        model.fit(np.zeros((3, 1), np.int64), np.zeros(2, np.int64))
    with pytest.raises(ValueError, match="zero rows"):
        model.fit(np.zeros((0, 1), np.int64), np.zeros(0, np.int64))
    with pytest.raises(ValueError, match=r"outcomes must lie in 0\.\.1"):
        model.fit(np.zeros((1, 1), np.int64), np.array([2]))
    model.fit(np.zeros((2, 1), np.int64), np.array([0, 1]))
    with pytest.raises(ValueError, match="needs 1 levels, not 2"):
        model.distribution(np.array([0, 0]))
    with pytest.raises(ValueError, match="\\(rows, levels\\)"):
        model.sample(np.zeros(3, np.int64), np.random.default_rng())
