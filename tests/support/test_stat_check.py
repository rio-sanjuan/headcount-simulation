"""Tests for the statistical check helper."""

import math

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st

from tests.support.stat_check import (
    DEFAULT_ALPHA,
    StatCheckError,
    bernoulli_sum_se,
    critical_z,
    mean_se,
    stat_check,
    suite_false_failure_rate,
)


def test_default_bound_is_about_3_89_standard_errors() -> None:
    assert critical_z() == pytest.approx(3.8906, abs=1e-4)


def test_a_suite_of_100_checks_fails_by_chance_under_1_percent() -> None:
    assert suite_false_failure_rate(100) < 0.01
    assert suite_false_failure_rate(0) == 0.0


def test_inside_the_bound_passes_and_returns_z() -> None:
    assert stat_check(103.0, 100.0, 1.0) == pytest.approx(3.0)


def test_outside_the_bound_fails_with_the_label() -> None:
    with pytest.raises(StatCheckError, match="attrition count"):
        stat_check(104.0, 100.0, 1.0, label="attrition count")


def test_zero_standard_error_demands_an_exact_match() -> None:
    assert stat_check(5.0, 5.0, 0.0) == 0.0
    with pytest.raises(StatCheckError, match="exactly"):
        stat_check(5.0, 6.0, 0.0)


@pytest.mark.parametrize(
    ("observed", "expected", "se"),
    [(math.nan, 0.0, 1.0), (0.0, math.inf, 1.0), (0.0, 0.0, -1.0), (0.0, 0.0, math.nan)],
)
def test_invalid_inputs_are_refused(observed: float, expected: float, se: float) -> None:
    with pytest.raises(ValueError, match="must be"):
        stat_check(observed, expected, se)


@pytest.mark.parametrize("alpha", [0.0, 1.0, -0.1, 2.0])
def test_alpha_outside_the_unit_interval_is_refused(alpha: float) -> None:
    with pytest.raises(ValueError, match="alpha"):
        critical_z(alpha)


def test_bernoulli_sum_se_matches_the_poisson_binomial_variance() -> None:
    probabilities = [0.1, 0.5, 0.9, 0.0, 1.0]
    expected = math.sqrt(0.1 * 0.9 + 0.25 + 0.9 * 0.1)
    assert bernoulli_sum_se(probabilities) == pytest.approx(expected)
    with pytest.raises(ValueError, match="probabilities"):
        bernoulli_sum_se([1.5])


def test_mean_se_matches_numpy() -> None:
    samples = [1.0, 2.0, 4.0, 8.0, 16.0]
    expected = float(np.std(samples, ddof=1) / math.sqrt(len(samples)))
    assert mean_se(samples) == pytest.approx(expected)
    with pytest.raises(ValueError, match="at least 2"):
        mean_se([1.0])


def test_false_failure_rate_matches_alpha_on_correct_checks() -> None:
    # At alpha = 0.05 on standard normal draws, about 5% of correct checks fail. The fraction
    # itself is checked with stat_check, at the default alpha.
    rng = np.random.default_rng(20261021)
    n, alpha = 20_000, 0.05
    failures = 0
    for draw in rng.standard_normal(n):
        try:
            stat_check(float(draw), 0.0, 1.0, alpha=alpha)
        except StatCheckError:
            failures += 1
    stat_check(failures, n * alpha, bernoulli_sum_se([alpha] * n), label="false failures")


@given(
    expected=st.floats(-1e6, 1e6),
    se=st.floats(1e-3, 1e3),
    z=st.floats(-10.0, 10.0),
)
def test_a_check_passes_exactly_when_z_is_within_the_bound(
    expected: float, se: float, z: float
) -> None:
    observed = expected + z * se
    realized = (observed - expected) / se
    bound = critical_z(DEFAULT_ALPHA)
    if abs(realized) <= bound:
        assert stat_check(observed, expected, se) == pytest.approx(realized)
    else:
        with pytest.raises(StatCheckError):
            stat_check(observed, expected, se)
