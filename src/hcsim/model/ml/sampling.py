"""Samplers: independent Bernoulli draws, and exact-count quota sampling.

Quota sampling realizes exactly `min(K, eligible)` events per group: the count comes from the
guidance forecast, and only who experiences the event is random. Selection is a segmented
Gumbel-top-k on `log(p) + Gumbel`, which is successive sampling in proportion to `p` within each
group. Rows in group UNATTRIBUTED (-1), and rows with probability 0, are never selected.

Fractional quotas become integers by largest-remainder apportionment, which preserves the total.
Rounding each group independently would zero every quota below 0.5 and silently lose events.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from hcsim.model.interface import Mask, Probabilities

Ints = NDArray[np.int64]


class SamplingError(ValueError):
    """Inputs a sampler can't use."""


def _probabilities(p: ArrayLike) -> Probabilities:
    array = np.asarray(p, dtype=np.float64)
    if array.ndim != 1:
        raise SamplingError("probabilities must be one-dimensional")
    if not np.isfinite(array).all() or (array < 0).any() or (array > 1).any():
        raise SamplingError("probabilities must lie in [0, 1]")
    return array


def apportion(quota: ArrayLike) -> Ints:
    """Largest-remainder apportionment of fractional quotas to integers.

    The integers sum to the total rounded half up, and each lies within 1 of its quota. Ties in
    the fractional part go to the lower group index, so the result is deterministic.
    """
    q = np.asarray(quota, dtype=np.float64)
    if q.ndim != 1:
        raise SamplingError("quota must be one-dimensional")
    if not np.isfinite(q).all() or (q < 0).any():
        raise SamplingError("quota must be finite and non-negative")
    base = np.floor(q).astype(np.int64)
    remainder = int(np.floor(q.sum() + 0.5)) - int(base.sum())
    if remainder > 0:
        order = np.lexsort((np.arange(len(q)), -(q - base)))
        base[order[:remainder]] += 1
    return base


def bernoulli(p: ArrayLike, rng: np.random.Generator) -> Mask:
    """One independent draw per row. The count is an output."""
    probabilities = _probabilities(p)
    return rng.random(len(probabilities)) < probabilities


@dataclass(frozen=True)
class QuotaSelection:
    """A quota draw and its accounting, per group."""

    mask: Mask
    quota: Ints
    eligible: Ints
    realized: Ints

    @property
    def shortfall(self) -> Ints:
        """Quota the group couldn't supply: positive only when eligibility binds."""
        return self.quota - self.realized


def quota_select(
    p: ArrayLike, groups: ArrayLike, quota: ArrayLike, rng: np.random.Generator
) -> QuotaSelection:
    """Select exactly `min(quota[g], eligible[g])` rows in each group `g`.

    `groups` holds a group index per row, with -1 for rows no group carries. `quota` holds one
    non-negative integer per group. A row is eligible when it's in a group and `p > 0`.
    """
    probabilities = _probabilities(p)
    group = np.asarray(groups, dtype=np.int64)
    k = np.asarray(quota)
    if group.shape != probabilities.shape:
        raise SamplingError("groups must have one entry per row")
    if k.ndim != 1 or not np.issubdtype(k.dtype, np.integer) or (k < 0).any():
        raise SamplingError("quota must be a one-dimensional array of non-negative integers")
    k = k.astype(np.int64)
    if group.size and (group.min() < -1 or group.max() >= len(k)):
        raise SamplingError(f"groups must lie in -1..{len(k) - 1}")

    # Draw a key for every row, so the stream doesn't depend on which rows are eligible.
    with np.errstate(divide="ignore"):
        keys = np.log(probabilities) + rng.gumbel(size=len(probabilities))
    eligible_rows = np.flatnonzero((group >= 0) & (probabilities > 0))
    order = eligible_rows[np.lexsort((-keys[eligible_rows], group[eligible_rows]))]
    ordered_groups = group[order]
    starts = np.flatnonzero(np.r_[True, ordered_groups[1:] != ordered_groups[:-1]])
    lengths = np.diff(np.r_[starts, len(order)])
    rank = np.arange(len(order)) - np.repeat(starts, lengths)
    chosen = order[rank < k[ordered_groups]]

    mask = np.zeros(len(probabilities), dtype=bool)
    mask[chosen] = True
    eligible = np.bincount(ordered_groups, minlength=len(k)).astype(np.int64)
    realized = np.bincount(group[chosen], minlength=len(k)).astype(np.int64)
    return QuotaSelection(mask, k, eligible, realized)


class BernoulliSampling:
    """`SamplingStrategy` for independent draws. A quota, if given, is ignored."""

    def select(
        self,
        probabilities: Probabilities,
        rng: np.random.Generator,
        *,
        groups: Ints | None = None,
        quota: Ints | None = None,
    ) -> Mask:
        return bernoulli(probabilities, rng)


class QuotaSampling:
    """`SamplingStrategy` for exact-count draws. It needs both groups and a quota."""

    def select(
        self,
        probabilities: Probabilities,
        rng: np.random.Generator,
        *,
        groups: Ints | None = None,
        quota: Ints | None = None,
    ) -> Mask:
        if groups is None or quota is None:
            raise SamplingError("quota sampling needs groups and a quota")
        return quota_select(probabilities, groups, quota, rng).mask
