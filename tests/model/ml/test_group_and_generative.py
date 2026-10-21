"""Tests for the group models and the attribute generator."""

from __future__ import annotations

import numpy as np
import pytest

from hcsim.model.interface import GroupRateModel
from hcsim.model.ml.entity import ModelNotFittedError
from hcsim.model.ml.generative import ATTRIBUTES, EmpiricalAttributeGenerator
from hcsim.model.ml.group import BackfillRateModel, PoissonGrowthModel
from tests.support.stat_check import mean_se, stat_check


def rng(seed: int = 3) -> np.random.Generator:
    return np.random.default_rng(seed)


# --- backfill -----------------------------------------------------------------------------------


def test_backfill_rates_fall_back_to_the_pooled_rate_for_thin_cost_centers() -> None:
    groups = np.array([0] * 20 + [1] * 4, dtype=np.int64)
    flags = np.array([True] * 10 + [False] * 10 + [True] * 4)
    model = BackfillRateModel(3, min_departures=20).fit(groups, flags)
    assert isinstance(model, GroupRateModel)
    pooled = 14 / 24
    assert model.rate_for(0) == pytest.approx(0.5)
    assert model.rate_for(1) == pytest.approx(pooled)  # 4 departures: too few
    assert model.rate_for(2) == pytest.approx(pooled)  # none at all
    assert model.rate_for(-1) == pytest.approx(pooled)
    assert model.rates(np.array([0, 1, 9])).tolist() == pytest.approx([0.5, pooled, pooled])


def test_a_constant_backfill_rate_serves_every_group() -> None:
    model = BackfillRateModel.constant(0.4, 5)
    assert model.rate_for(3) == model.pooled == 0.4
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        BackfillRateModel.constant(1.4, 5)


def test_backfill_refuses_bad_inputs() -> None:
    with pytest.raises(ValueError, match="one backfill flag"):
        BackfillRateModel(2).fit(np.array([0]), np.array([True, False]))
    with pytest.raises(ValueError, match="zero departures"):
        BackfillRateModel(2).fit(np.array([], dtype=np.int64), np.array([], dtype=bool))
    with pytest.raises(ModelNotFittedError):
        BackfillRateModel(2).rate_for(0)


# --- growth -------------------------------------------------------------------------------------


def test_growth_means_need_enough_observed_months() -> None:
    counts = np.array([[2, 0], [4, 1], [0, 0], [2, 3]], dtype=np.float64)
    observed = np.array([[True, False], [True, False], [True, True], [True, True]])
    model = PoissonGrowthModel(2, min_months=3).fit(counts, observed)
    assert model.rate_for(0) == pytest.approx(2.0)
    pooled = (2 + 4 + 0 + 2 + 0 + 3) / 6
    assert model.rate_for(1) == pytest.approx(pooled)  # 2 observed months: too few


def test_growth_draws_average_their_rate() -> None:
    model = PoissonGrowthModel(1, min_months=1).fit(np.full((12, 1), 1.5))
    draws = model.sample(np.zeros(4_000, dtype=np.int64), rng(5)).astype(float)
    stat_check(float(draws.mean()), 1.5, mean_se(list(draws)), label="Poisson mean")


def test_growth_refuses_bad_inputs() -> None:
    with pytest.raises(ValueError, match=r"\(months, 2\)"):
        PoissonGrowthModel(2).fit(np.zeros((3, 3)))
    with pytest.raises(ValueError, match="non-negative"):
        PoissonGrowthModel(1).fit(np.full((3, 1), -1.0))
    with pytest.raises(ValueError, match="zero observed months"):
        PoissonGrowthModel(1).fit(np.zeros((2, 1)), np.zeros((2, 1), dtype=bool))


# --- attributes ---------------------------------------------------------------------------------


def observed_rows() -> tuple[np.ndarray, np.ndarray]:
    # Cost center 0 has six rows of one profile; cost center 1 has two of another.
    profile_a, profile_b = [1, 0, 2, 0, 0, 0], [3, 0, 1, 2, 1, 0]
    rows = np.array([profile_a] * 6 + [profile_b] * 2, dtype=np.int64)
    return np.array([0] * 6 + [1] * 2, dtype=np.int64), rows


def test_generated_rows_come_from_the_cost_centers_own_observations() -> None:
    generator = EmpiricalAttributeGenerator(min_rows=5).fit(*observed_rows())
    sample = generator.sample(0, 50, rng())
    assert sample.shape == (50, len(ATTRIBUTES))
    assert (sample == [1, 0, 2, 0, 0, 0]).all()
    assert generator.covers(0)


def test_thin_cost_centers_draw_from_every_observed_row_and_never_invent_one() -> None:
    generator = EmpiricalAttributeGenerator(min_rows=5).fit(*observed_rows())
    assert not generator.covers(1)
    seen = {tuple(row) for row in observed_rows()[1]}
    assert {tuple(row) for row in generator.sample(1, 200, rng())} <= seen
    assert {tuple(row) for row in generator.sample(7, 200, rng())} <= seen


def test_the_generator_refuses_bad_inputs() -> None:
    with pytest.raises(ValueError, match=r"\(N, 6\)"):
        EmpiricalAttributeGenerator().fit(
            np.zeros(2, dtype=np.int64), np.zeros((2, 3), dtype=np.int64)
        )
    with pytest.raises(ValueError, match="one cost center"):
        EmpiricalAttributeGenerator().fit(
            np.zeros(1, dtype=np.int64), np.zeros((2, 6), dtype=np.int64)
        )
    with pytest.raises(ValueError, match="zero rows"):
        EmpiricalAttributeGenerator().fit(
            np.zeros(0, dtype=np.int64), np.zeros((0, 6), dtype=np.int64)
        )
    with pytest.raises(ModelNotFittedError):
        EmpiricalAttributeGenerator().sample(0, 1, rng())
