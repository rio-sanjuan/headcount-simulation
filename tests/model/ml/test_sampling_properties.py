"""Property tests for the samplers.

Each property holds for every input Hypothesis generates: arbitrary probabilities (zeros
included), arbitrary group assignments (-1 included), and arbitrary quotas.
"""

from __future__ import annotations

import math

import numpy as np
from hypothesis import given
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays

from hcsim.model.ml.sampling import apportion, quota_select

MAX_GROUPS = 6


@st.composite
def draws(draw: st.DrawFn) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    n = draw(st.integers(0, 60))
    n_groups = draw(st.integers(1, MAX_GROUPS))
    # Mix exact zeros and ones in with interior probabilities.
    p = draw(
        arrays(np.float64, n, elements=st.one_of(st.just(0.0), st.just(1.0), st.floats(1e-6, 1.0)))
    )
    groups = draw(arrays(np.int64, n, elements=st.integers(-1, n_groups - 1)))
    quota = draw(arrays(np.int64, n_groups, elements=st.integers(0, 12)))
    seed = draw(st.integers(0, 2**32 - 1))
    return p, groups, quota, seed


@given(draws())
def test_each_group_realizes_exactly_min_of_quota_and_eligible(
    case: tuple[np.ndarray, np.ndarray, np.ndarray, int],
) -> None:
    p, groups, quota, seed = case
    result = quota_select(p, groups, quota, np.random.default_rng(seed))
    eligible = np.bincount(groups[(groups >= 0) & (p > 0)], minlength=len(quota))
    assert result.eligible.tolist() == eligible.tolist()
    assert result.realized.tolist() == np.minimum(quota, eligible).tolist()
    assert (result.shortfall >= 0).all()


@given(draws())
def test_unattributed_and_zero_probability_rows_are_never_selected(
    case: tuple[np.ndarray, np.ndarray, np.ndarray, int],
) -> None:
    p, groups, quota, seed = case
    mask = quota_select(p, groups, quota, np.random.default_rng(seed)).mask
    assert not mask[groups == -1].any()
    assert not mask[p == 0].any()


@given(draws())
def test_the_mask_agrees_with_the_reported_counts(
    case: tuple[np.ndarray, np.ndarray, np.ndarray, int],
) -> None:
    p, groups, quota, seed = case
    result = quota_select(p, groups, quota, np.random.default_rng(seed))
    selected = np.bincount(groups[result.mask], minlength=len(quota))
    assert selected.tolist() == result.realized.tolist()


@given(draws())
def test_a_quota_at_or_above_eligibility_selects_every_eligible_row(
    case: tuple[np.ndarray, np.ndarray, np.ndarray, int],
) -> None:
    p, groups, _, seed = case
    generous = np.full(MAX_GROUPS, len(p), dtype=np.int64)
    mask = quota_select(p, groups, generous, np.random.default_rng(seed)).mask
    assert (mask == ((groups >= 0) & (p > 0))).all()


@given(arrays(np.float64, st.integers(0, 40), elements=st.floats(0.0, 50.0)))
def test_apportionment_preserves_the_total_and_stays_within_one(quota: np.ndarray) -> None:
    result = apportion(quota)
    assert int(result.sum()) == math.floor(float(quota.sum()) + 0.5)
    assert (result >= 0).all()
    assert (np.abs(result - quota) < 1.0 + 1e-9).all()
