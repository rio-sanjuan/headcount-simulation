"""The gap between quota sampling's inclusion and conditional-Bernoulli inclusion.

Quota sampling is successive sampling: it includes a row with a probability that isn't the
model's rate conditioned on the count. These tests compute both exactly for small groups,
check the sampler against the exact successive-sampling values, and pin down the gap in the
regimes that matter.
"""

from __future__ import annotations

import itertools

import numpy as np
import pytest

from hcsim.model.ml.inclusion import (
    InclusionError,
    conditional_bernoulli_inclusion,
    inclusion_gap,
    successive_inclusion,
)
from hcsim.model.ml.sampling import quota_select
from tests.support.stat_check import stat_check


def brute_force_conditional_bernoulli(p: np.ndarray, k: int) -> np.ndarray:
    weight = {
        s: np.prod([p[i] if i in s else 1 - p[i] for i in range(len(p))])
        for s in itertools.combinations(range(len(p)), k)
    }
    total = sum(weight.values())
    return np.array([sum(w for s, w in weight.items() if i in s) / total for i in range(len(p))])


P = np.array([0.9, 0.5, 0.5, 0.2, 0.1, 0.05])


def test_conditional_bernoulli_matches_enumeration() -> None:
    for k in range(len(P) + 1):
        exact = conditional_bernoulli_inclusion(P, k)
        np.testing.assert_allclose(exact, brute_force_conditional_bernoulli(P, k), atol=1e-12)
        assert exact.sum() == pytest.approx(k)


def test_successive_inclusion_sums_to_the_draws() -> None:
    for k in range(len(P) + 1):
        assert successive_inclusion(P, k).sum() == pytest.approx(k)
    assert successive_inclusion([0.0, 0.5, 0.5], 2).tolist() == pytest.approx([0.0, 1.0, 1.0])


def test_the_sampler_is_exact_successive_sampling() -> None:
    k, runs = 2, 20_000
    exact = successive_inclusion(P, k)
    seen = np.zeros(len(P))
    groups, quota = np.zeros(len(P), np.int64), np.array([k])
    rng = np.random.default_rng(2024)
    for _ in range(runs):
        seen += quota_select(P, groups, quota, rng).mask
    for i, rate in enumerate(exact):
        stat_check(seen[i] / runs, rate, float(np.sqrt(rate * (1 - rate) / runs)), label=f"row {i}")


def test_equal_rates_have_no_gap() -> None:
    assert inclusion_gap([0.3] * 6, 2) == pytest.approx(0.0, abs=1e-12)


def test_rare_events_have_a_small_gap() -> None:
    """Monthly hazards like attrition's sit at a few percent, where the gap is under 1%."""
    rng = np.random.default_rng(7)
    worst = max(
        inclusion_gap(rng.uniform(0.01, 0.1, size=8), int(rng.integers(1, 3))) for _ in range(30)
    )
    assert worst < 0.01


def test_spread_out_high_rates_have_a_large_gap() -> None:
    """Successive sampling under-includes the likeliest rows: here the 0.9 row by about 0.23."""
    gap = successive_inclusion(P, 2) - conditional_bernoulli_inclusion(P, 2)
    assert gap[0] == pytest.approx(-0.2253, abs=1e-4)
    assert inclusion_gap(P, 2) == pytest.approx(0.2253, abs=1e-4)
    assert (gap[3:] > 0).all()  # and over-includes the unlikely ones


@pytest.mark.parametrize(
    ("call", "message"),
    [
        (lambda: successive_inclusion(np.full(11, 0.1), 1), "at most 10 rows"),
        (lambda: successive_inclusion([0.5, 0.0], 2), "fewer rows with positive"),
        (lambda: successive_inclusion([0.5], 2), "k must lie in 0..1"),
        (lambda: successive_inclusion([1.5], 1), "must lie in \\[0, 1\\]"),
        (lambda: conditional_bernoulli_inclusion([1.0, 0.5], 1), "must lie in \\(0, 1\\)"),
    ],
)
def test_bad_inputs_are_refused(call, message: str) -> None:  # type: ignore[no-untyped-def]
    with pytest.raises(InclusionError, match=message):
        call()


def test_zero_draws_include_no_one() -> None:
    assert conditional_bernoulli_inclusion(P, 0).tolist() == [0.0] * len(P)
