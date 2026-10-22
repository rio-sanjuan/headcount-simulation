"""Exact inclusion probabilities, to measure what quota sampling does to a model's rates.

Quota sampling selects exactly `k` of a group's rows by successive sampling: one row at a time,
each in proportion to its probability among those left (the Gumbel-top-k of `quota_select`).
The rows a model rates higher are included more often, but not in exact proportion: a row's
inclusion probability isn't its probability rescaled.

The natural reference is conditional Bernoulli: independent draws at the model's probabilities,
conditioned on exactly `k` successes. That's what the rates mean, given the count. Its
inclusion probabilities follow from the odds `w = p / (1 - p)` and their elementary symmetric
polynomials `e`: row `i` is included with probability `w_i e_{k-1}(w without i) / e_k(w)`.

The gap between the two is a property of the sampler, measured exactly here for small groups.
"""

from __future__ import annotations

import itertools
from typing import Final

import numpy as np
from numpy.typing import ArrayLike, NDArray

Floats = NDArray[np.float64]
#: Exact successive-sampling inclusion enumerates ordered draws, so groups stay small.
MAX_ROWS: Final = 10


class InclusionError(ValueError):
    """Probabilities or a count the exact computation can't take."""


def _checked(p: ArrayLike, k: int, *, open_interval: bool) -> Floats:
    values = np.asarray(p, dtype=np.float64)
    if values.ndim != 1 or not 0 <= k <= len(values):
        raise InclusionError(f"k must lie in 0..{values.size}, for one-dimensional probabilities")
    low_ok = (values > 0).all() if open_interval else (values >= 0).all()
    high_ok = (values < 1).all() if open_interval else (values <= 1).all()
    if not (np.isfinite(values).all() and low_ok and high_ok):
        interval = "(0, 1)" if open_interval else "[0, 1]"
        raise InclusionError(f"probabilities must lie in {interval}")
    return values


def successive_inclusion(p: ArrayLike, k: int) -> Floats:
    """Each row's chance of being among `k` drawn one at a time in proportion to `p`."""
    values = _checked(p, k, open_interval=False)
    if values.size > MAX_ROWS:
        raise InclusionError(f"exact inclusion enumerates draws; at most {MAX_ROWS} rows")
    if k and (values > 0).sum() < k:
        raise InclusionError("fewer rows with positive probability than draws")
    inclusion = np.zeros(values.size)
    for order in itertools.permutations(range(values.size), k):
        left = values.sum()
        chance = 1.0
        for row in order:
            if left <= 0:
                chance = 0.0
                break
            chance *= values[row] / left
            left -= values[row]
        inclusion[list(order)] += chance
    return inclusion


def _elementary(w: Floats, k: int) -> float:
    """e_k(w), the sum of products of every k-subset, by the standard recurrence."""
    e = np.zeros(k + 1)
    e[0] = 1.0
    for x in w:
        e[1:] = e[1:] + x * e[:-1]
    return float(e[k])


def conditional_bernoulli_inclusion(p: ArrayLike, k: int) -> Floats:
    """Each row's chance of success, given independent draws at `p` and exactly `k` successes."""
    values = _checked(p, k, open_interval=True)
    odds = values / (1 - values)
    total = _elementary(odds, k)
    if k == 0:
        return np.zeros(values.size)
    return np.array(
        [odds[i] * _elementary(np.delete(odds, i), k - 1) / total for i in range(values.size)]
    )


def inclusion_gap(p: ArrayLike, k: int) -> float:
    """The largest difference between the two, over rows."""
    gap = successive_inclusion(p, k) - conditional_bernoulli_inclusion(p, k)
    return float(np.abs(gap).max(initial=0.0))
