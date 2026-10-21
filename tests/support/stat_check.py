"""Statistical checks with a suite-wide false-failure budget.

A statistical check passes when an observed value lies within a two-sided bound of its
expectation, measured in standard errors. The default false-failure rate per check is 1e-4,
about 3.89 standard errors, so a suite of 100 independent checks fails by chance less than 1% of
the time.

Checks run with fixed seeds, so for a given version of the code a check either always passes or
always fails. The budget matters when a code change reshuffles the random streams: every check
then faces a fresh draw. Never retry a failed check with another seed. A failure is either a
defect or one of the rare chance failures the budget allows, and only investigation tells them
apart.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence
from statistics import NormalDist

DEFAULT_ALPHA = 1e-4


class StatCheckError(AssertionError):
    """A statistical check landed outside its bound."""


def critical_z(alpha: float = DEFAULT_ALPHA) -> float:
    """The two-sided bound, in standard errors, for a false-failure rate of `alpha`."""
    if not 0.0 < alpha < 1.0:
        raise ValueError(f"alpha must lie in (0, 1), got {alpha!r}")
    return NormalDist().inv_cdf(1.0 - alpha / 2.0)


def suite_false_failure_rate(n_checks: int, alpha: float = DEFAULT_ALPHA) -> float:
    """The chance that at least one of `n_checks` independent, correct checks fails."""
    if n_checks < 0:
        raise ValueError(f"n_checks must be non-negative, got {n_checks!r}")
    critical_z(alpha)  # validates alpha
    return 1.0 - (1.0 - alpha) ** n_checks


def stat_check(
    observed: float,
    expected: float,
    se: float,
    *,
    alpha: float = DEFAULT_ALPHA,
    label: str = "statistic",
) -> float:
    """Assert that `observed` lies within the bound around `expected`, and return its z-score.

    A zero standard error demands an exact match, which suits quantities fixed by construction,
    such as a quota-sampled count.
    """
    for name, value in (("observed", observed), ("expected", expected), ("se", se)):
        if not math.isfinite(value):
            raise ValueError(f"{name} must be finite, got {value!r}")
    if se < 0.0:
        raise ValueError(f"se must be non-negative, got {se!r}")
    bound = critical_z(alpha)
    difference = observed - expected
    if se == 0.0:
        if difference != 0.0:
            raise StatCheckError(
                f"{label}: expected exactly {expected:.6g} with zero standard error, "
                f"observed {observed:.6g}"
            )
        return 0.0
    z = difference / se
    if abs(z) > bound:
        raise StatCheckError(
            f"{label}: observed {observed:.6g}, expected {expected:.6g}, standard error "
            f"{se:.3g}, so z = {z:+.2f}, outside the bound of ±{bound:.2f} (alpha = {alpha:g})"
        )
    return z


def bernoulli_sum_se(probabilities: Iterable[float]) -> float:
    """The standard error of a count of independent Bernoulli draws with these probabilities."""
    variance = 0.0
    for p in probabilities:
        if not 0.0 <= p <= 1.0:
            raise ValueError(f"probabilities must lie in [0, 1], got {p!r}")
        variance += p * (1.0 - p)
    return math.sqrt(variance)


def mean_se(samples: Sequence[float]) -> float:
    """The standard error of the mean of independent samples, with Bessel's correction."""
    n = len(samples)
    if n < 2:
        raise ValueError(f"need at least 2 samples, got {n}")
    mean = math.fsum(samples) / n
    variance = math.fsum((x - mean) ** 2 for x in samples) / (n - 1)
    return math.sqrt(variance / n)
