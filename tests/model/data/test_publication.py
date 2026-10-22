"""Tests for the SBC publication rule for subsets."""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays

from hcsim.model.data.publication import PublicationError, membership, publish

# Two holders over four months: the first moves out after month 1, the second in at month 2.
INSIDE = np.array([[True, True, False, False], [False, False, True, True]])


def test_a_tranche_is_published_only_while_its_holder_is_inside() -> None:
    expense = np.array([[10.0, 10.0, 10.0, 10.0], [5.0, 5.0, 5.0, 5.0], [1.0, 2.0, 3.0, 4.0]])
    holder = [0, 1, 0]
    split = publish(expense, holder, INSIDE)
    assert split.published.tolist() == [[10, 10, 0, 0], [0, 0, 5, 5], [1, 2, 0, 0]]
    assert split.scratch.tolist() == [[0, 0, 10, 10], [5, 5, 0, 0], [0, 0, 3, 4]]
    np.testing.assert_array_equal(split.total, expense)
    assert membership(INSIDE, holder).shape == (3, 4)


@given(
    arrays(np.float64, (5, 6), elements=st.floats(-100, 100)),
    arrays(np.bool_, (3, 6)),
    st.lists(st.integers(0, 2), min_size=5, max_size=5),
)
def test_nothing_is_lost_between_published_and_scratch(
    expense: np.ndarray, inside: np.ndarray, holder: list[int]
) -> None:
    split = publish(expense, holder, inside)
    np.testing.assert_array_equal(split.total, expense)
    assert not ((split.published != 0) & (split.scratch != 0)).any()


def test_misaligned_inputs_are_refused() -> None:
    with pytest.raises(PublicationError, match=r"holders must lie in 0\.\.1"):
        publish(np.zeros((1, 4)), [2], INSIDE)
    with pytest.raises(PublicationError, match=r"expense is \(1, 3\)"):
        publish(np.zeros((1, 3)), [0], INSIDE)
    with pytest.raises(PublicationError, match="inside is \\(holders, months\\)"):
        membership(np.zeros(4, dtype=bool), [0])
