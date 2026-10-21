"""Tests for the samplers."""

from __future__ import annotations

import numpy as np
import pytest

from hcsim.model.interface import SamplingStrategy
from hcsim.model.ml.sampling import (
    BernoulliSampling,
    QuotaSampling,
    SamplingError,
    apportion,
    bernoulli,
    quota_select,
)
from tests.support.stat_check import bernoulli_sum_se, stat_check


def rng(seed: int = 7) -> np.random.Generator:
    return np.random.default_rng(seed)


# --- apportionment ------------------------------------------------------------------------------


def test_apportionment_preserves_the_rounded_total() -> None:
    assert apportion([0.4, 0.4, 0.4]).tolist() == [1, 0, 0]  # 1.2 rounds to 1
    assert apportion([2.6, 1.3, 0.1]).tolist() == [3, 1, 0]
    assert apportion([0.5, 0.5]).tolist() == [1, 0]  # 1.0: ties go to the lower index


def test_independent_rounding_would_lose_these_events() -> None:
    quota = [0.3] * 10
    assert int(np.round(quota).sum()) == 0
    assert int(apportion(quota).sum()) == 3


def test_apportionment_refuses_negative_or_non_finite_quotas() -> None:
    for bad in ([-0.1], [float("nan")], [[1.0]]):
        with pytest.raises(SamplingError):
            apportion(bad)


# --- Bernoulli ----------------------------------------------------------------------------------


def test_bernoulli_honors_zero_and_one() -> None:
    mask = bernoulli([0.0, 1.0, 0.0, 1.0], rng())
    assert mask.tolist() == [False, True, False, True]


def test_bernoulli_refuses_probabilities_outside_the_unit_interval() -> None:
    with pytest.raises(SamplingError, match=r"\[0, 1\]"):
        bernoulli([1.2], rng())


def test_bernoulli_counts_match_their_expectation() -> None:
    p = np.linspace(0.01, 0.5, 4_000)
    count = int(bernoulli(p, rng(11)).sum())
    stat_check(count, float(p.sum()), bernoulli_sum_se(p), label="Bernoulli count")


# --- quota sampling -----------------------------------------------------------------------------


def test_quota_sampling_realizes_exactly_min_of_quota_and_eligible() -> None:
    p = np.array([0.1, 0.9, 0.5, 0.0, 0.2, 0.3, 0.4])
    groups = np.array([0, 0, 0, 0, 1, 1, -1])
    result = quota_select(p, groups, np.array([2, 5]), rng())
    assert result.eligible.tolist() == [3, 2]  # the p = 0 row and the -1 row aren't eligible
    assert result.realized.tolist() == [2, 2]
    assert result.shortfall.tolist() == [0, 3]
    assert not result.mask[3]  # probability 0 is never selected
    assert not result.mask[6]  # group -1 is never selected


def test_a_zero_quota_selects_nothing() -> None:
    result = quota_select(np.full(5, 0.9), np.zeros(5, dtype=np.int64), np.array([0]), rng())
    assert not result.mask.any()


def test_the_same_seed_gives_the_same_selection() -> None:
    p, groups, quota = np.full(50, 0.3), np.arange(50) % 3, np.array([4, 5, 6])
    first = quota_select(p, groups, quota, rng(3)).mask
    assert (quota_select(p, groups, quota, rng(3)).mask == first).all()


def test_a_single_draw_picks_each_row_in_proportion_to_its_probability() -> None:
    # Gumbel-top-1 is exact: row i wins with probability p_i / sum(p).
    p = np.array([0.1, 0.2, 0.3, 0.4])
    groups = np.zeros(4, dtype=np.int64)
    generator = rng(19)
    n = 6_000
    wins = np.zeros(4)
    for _ in range(n):
        wins += quota_select(p, groups, np.array([1]), generator).mask
    share = p / p.sum()
    for i in range(4):
        se = float(np.sqrt(n * share[i] * (1 - share[i])))
        stat_check(wins[i], n * share[i], se, label=f"row {i} wins")


@pytest.mark.parametrize(
    ("groups", "quota", "match"),
    [
        (np.zeros(3, dtype=np.int64), np.array([1.5]), "non-negative integers"),
        (np.zeros(3, dtype=np.int64), np.array([-1]), "non-negative integers"),
        (np.array([0, 2, 0]), np.array([1]), r"-1\.\.0"),
        (np.zeros(2, dtype=np.int64), np.array([1]), "one entry per row"),
    ],
)
def test_quota_sampling_refuses_malformed_inputs(
    groups: np.ndarray, quota: np.ndarray, match: str
) -> None:
    with pytest.raises(SamplingError, match=match):
        quota_select(np.full(3, 0.5), groups, quota, rng())


def test_both_strategies_conform_to_the_protocol() -> None:
    assert isinstance(BernoulliSampling(), SamplingStrategy)
    assert isinstance(QuotaSampling(), SamplingStrategy)
    mask = QuotaSampling().select(
        np.full(4, 0.5), rng(), groups=np.zeros(4, dtype=np.int64), quota=np.array([2])
    )
    assert int(mask.sum()) == 2
    with pytest.raises(SamplingError, match="needs groups and a quota"):
        QuotaSampling().select(np.full(4, 0.5), rng())
    assert BernoulliSampling().select(np.ones(3), rng()).all()
