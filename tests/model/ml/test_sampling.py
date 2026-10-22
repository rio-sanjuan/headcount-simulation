"""Tests for the samplers."""

from __future__ import annotations

import numpy as np
import pytest

from hcsim.model.interface import SamplingStrategy
from hcsim.model.ml.sampling import (
    BernoulliSampling,
    CandidatePool,
    QuotaSampling,
    SamplingError,
    SeedBasis,
    apportion,
    assign,
    bernoulli,
    quota_select,
    rake_quota,
    seed_base,
    sequential,
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


# --- capacity-constrained assignment -------------------------------------------------------------


def test_each_column_gets_its_quota_and_no_row_its_capacity_over() -> None:
    weights = rng(1).uniform(0.1, 1.0, size=(40, 3))
    capacity = rng(2).integers(1, 4, size=40)
    result = assign(weights, np.array([10, 5, 7]), rng(3), capacity=capacity)
    assert result.realized.tolist() == [10, 5, 7]
    assert result.shortfall.tolist() == [0, 0, 0]
    assert (result.counts.sum(axis=1) <= capacity).all()
    assert result.counts.shape == (40, 3)


def test_a_zero_weight_never_assigns_and_supply_can_bind() -> None:
    weights = np.array([[1.0, 0.0], [1.0, 0.0], [0.0, 0.0]])
    result = assign(weights, np.array([5, 1]), rng(4))
    assert result.counts.tolist() == [[1, 0], [1, 0], [0, 0]]
    assert result.shortfall.tolist() == [3, 1]
    assert assign(np.zeros((0, 2)), np.array([1, 1]), rng(5)).shortfall.tolist() == [1, 1]


def test_a_row_with_several_openings_can_fill_several_times() -> None:
    result = assign(np.array([[1.0], [1e-9]]), np.array([3]), rng(6), capacity=np.array([3, 3]))
    assert result.counts[:, 0].tolist() == [3, 0]


def test_rows_lean_to_the_column_they_weigh_most_in() -> None:
    weights = np.tile([[9.0, 1.0]], (200, 1))
    weights[100:] = [1.0, 9.0]
    result = assign(weights, np.array([100, 100]), rng(7))
    first = int(result.counts[:100, 0].sum())
    assert first > 70  # mostly the rows that favor column 0
    assert result.realized.tolist() == [100, 100]


def test_one_column_is_successive_sampling() -> None:
    """With one column and unit capacities, assignment and quota sampling are the same draw:
    a Gumbel-top-k on log-weights. Their inclusion rates agree."""
    p = np.array([0.9, 0.5, 0.5, 0.2, 0.1, 0.05])
    runs = 4000
    assigned = np.zeros(len(p))
    sampled = np.zeros(len(p))
    for seed in range(runs):
        assigned += assign(p[:, None], np.array([2]), rng(seed)).counts[:, 0]
        sampled += quota_select(p, np.zeros(len(p), np.int64), np.array([2]), rng(seed + runs)).mask
    for i in range(len(p)):
        rate = sampled[i] / runs
        se = np.sqrt(2 * max(rate * (1 - rate), 1e-6) / runs)
        stat_check(assigned[i] / runs, rate, se, label=f"row {i}")


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"weights": np.ones(3)}, "weights must be \\(rows, columns\\)"),
        ({"weights": -np.ones((2, 1))}, "finite and non-negative"),
        ({"quota": np.array([1, 1])}, "one non-negative integer per column"),
        ({"quota": np.array([1.5])}, "one non-negative integer per column"),
        ({"capacity": np.array([1])}, "one non-negative integer per row"),
        ({"capacity": np.array([-1, 1])}, "one non-negative integer per row"),
    ],
)
def test_bad_assignment_inputs_are_refused(kwargs: dict[str, object], message: str) -> None:
    args: dict[str, object] = {"weights": np.ones((2, 1)), "quota": np.array([1])} | kwargs
    with pytest.raises(SamplingError, match=message):
        assign(args["weights"], args["quota"], rng(), capacity=args.get("capacity"))  # type: ignore[arg-type]


# --- pools, sequential draws, seed bases, and raked quotas ------------------------------------


def test_sequential_draws_never_select_a_row_twice() -> None:
    n = 50
    groups = np.repeat([0, 1], 25)
    p = np.full(n, 0.5)
    first, second = sequential(
        [(p, groups, np.array([20, 20])), (p, groups, np.array([10, 10]))], rng(8)
    )
    assert first.realized.tolist() == [20, 20]
    assert second.realized.tolist() == [5, 5]  # only what the first left
    assert second.shortfall.tolist() == [5, 5]
    assert not (first.mask & second.mask).any()
    assert sequential([], rng()) == []


def test_a_pool_starts_from_its_eligible_rows_and_refuses_double_takes() -> None:
    pool = CandidatePool([True, True, False, True])
    assert len(pool) == 3
    picked = pool.select(np.ones(4), np.zeros(4, np.int64), np.array([5]), rng(9))
    assert picked.mask.tolist() == [True, True, False, True]
    assert len(pool) == 0
    with pytest.raises(SamplingError, match="no longer has"):
        pool.take(np.array([True, False, False, False]))
    with pytest.raises(SamplingError, match="one entry per pooled row"):
        pool.take(np.array([True]))
    (only,) = sequential(
        [(np.ones(3), np.zeros(3, np.int64), np.array([3]))], rng(1), eligible=[True, False, True]
    )
    assert only.mask.tolist() == [True, False, True]


def test_seed_bases_count_rows_probability_or_presence() -> None:
    a, b = np.array([0, 0, 1, 1, -1]), np.array([0, 1, 1, 1, 0])
    p = np.array([0.5, 0.25, 0.1, 0.2, 0.9])
    assert seed_base(SeedBasis.ROWS, a, b, (2, 2)).tolist() == [[1, 1], [0, 2]]
    np.testing.assert_allclose(
        seed_base(SeedBasis.PROBABILITY, a, b, (2, 2), p), [[0.5, 0.25], [0.0, 0.3]]
    )
    assert seed_base(SeedBasis.UNIFORM, a, b, (2, 2)).tolist() == [[1, 1], [0, 1]]
    with pytest.raises(SamplingError, match="needs the probabilities"):
        seed_base(SeedBasis.PROBABILITY, a, b, (2, 2))
    with pytest.raises(SamplingError, match="inside the table's shape"):
        seed_base(SeedBasis.ROWS, a, b, (1, 2))
    with pytest.raises(SamplingError, match="one group in each grouping"):
        seed_base(SeedBasis.ROWS, a, b[:2], (2, 2))


def test_raked_quotas_meet_the_rows_exactly_and_the_columns_within_rounding() -> None:
    seed = np.array([[4.0, 1.0, 1.0], [1.0, 2.0, 3.0]])
    table = rake_quota(seed, [10, 20], [9, 8, 13])
    assert table.dtype == np.int64
    assert table.sum(axis=1).tolist() == [10, 20]
    assert np.abs(table.sum(axis=0) - np.array([9, 8, 13])).max() <= 1
    with pytest.raises(SamplingError, match="disagree on the total"):
        rake_quota(seed, [10, 20], [1, 1, 1])
    assert rake_quota(np.zeros((0, 2)), [], [0, 0]).shape == (0, 2)
