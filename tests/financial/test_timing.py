"""Tests for award timing: lag distributions, lags per calendar month, and suppressed months."""

from __future__ import annotations

import numpy as np
import pytest

from hcsim.financial.calculator import SizingError
from hcsim.financial.timing import MAX_LAG, AwardTiming, LagDistribution


def rng(seed: int = 0) -> np.random.Generator:
    return np.random.default_rng(seed)


def test_a_lag_distribution_draws_each_lag_at_its_chance() -> None:
    lags = LagDistribution((0.5, 0.3, 0.2)).draw(rng(), 20_000)
    shares = np.bincount(lags, minlength=3) / len(lags)
    assert shares == pytest.approx([0.5, 0.3, 0.2], abs=0.015)
    assert LagDistribution.fixed(2).draw(rng(), 4).tolist() == [2, 2, 2, 2]
    assert LagDistribution().draw(rng(), 3).tolist() == [0, 0, 0]


@pytest.mark.parametrize(
    ("probabilities", "message"),
    [
        ((0.5, 0.4), "sum to 1, not 0.9"),
        ((1.2, -0.2), "finite and non-negative"),
        ((), f"lags 0 to at most {MAX_LAG}"),
        ((0.0,) * (MAX_LAG + 1) + (1.0,), f"lags 0 to at most {MAX_LAG}"),
    ],
)
def test_a_lag_distribution_is_checked(probabilities: tuple[float, ...], message: str) -> None:
    with pytest.raises(SizingError, match=message):
        LagDistribution(probabilities)


def test_a_dispatch_month_can_have_lags_of_its_own() -> None:
    timing = AwardTiming(LagDistribution.fixed(1), by_month={12: LagDistribution.fixed(2)})
    assert timing.award_steps(10, 11, 2, rng()).tolist() == [11, 11]
    assert timing.award_steps(11, 12, 2, rng()).tolist() == [13, 13]  # December waits two


def test_an_award_due_in_a_suppressed_month_slides_to_the_next_open_one() -> None:
    timing = AwardTiming(LagDistribution.fixed(1), suppressed=frozenset({12, 1}))
    # Dispatched in November, due in December: December and January are suppressed.
    assert timing.award_steps(5, 11, 1, rng()).tolist() == [8]
    # Dispatched in September, due in October: open.
    assert timing.award_steps(5, 9, 1, rng()).tolist() == [6]
    # A zero lag in a suppressed month slides too.
    assert AwardTiming(suppressed=frozenset({3})).award_steps(0, 3, 1, rng()).tolist() == [1]


def test_timing_is_reproducible_and_immediate_only_without_lags_or_blackouts() -> None:
    timing = AwardTiming(LagDistribution((0.2, 0.8)))
    first = timing.award_steps(0, 1, 50, rng(4))
    assert np.array_equal(first, timing.award_steps(0, 1, 50, rng(4)))
    assert not timing.immediate
    assert AwardTiming().immediate
    assert not AwardTiming(suppressed=frozenset({6})).immediate


def test_award_timing_is_checked() -> None:
    with pytest.raises(SizingError) as error:
        AwardTiming(by_month={13: LagDistribution()}, suppressed=frozenset({0}))
    assert "calendar months run 1-12, not [0, 13]" in str(error.value)
    with pytest.raises(SizingError, match="every calendar month is suppressed"):
        AwardTiming(suppressed=frozenset(range(1, 13)))
    with pytest.raises(SizingError, match="a calendar month runs 1-12"):
        AwardTiming().award_steps(0, 0, 1, rng())
