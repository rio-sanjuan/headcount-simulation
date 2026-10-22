"""Tests for quota derivations, share bases, disaggregation, and raking."""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from hcsim.model.data.grain import GrainRelation, Placement
from hcsim.model.data.quotas import (
    Derivation,
    QuotaError,
    QuotaRefused,
    RakingMarginals,
    Refusal,
    check_derivation,
    derive,
    disaggregate,
    inside_shares,
    rake,
)

D = Derivation
GROUPS = ("ORG-A", "ORG-B", "ORG-C")
PLACED = Placement(GrainRelation.CROSSING, ("ORG-A",), ("ORG-B",), ("ORG-C",))
COVERED = Placement(GrainRelation.COVERED, ("ORG-A",), (), ("ORG-B", "ORG-C"))
UNREACHABLE = Placement(GrainRelation.UNREACHABLE, (), (), GROUPS)


def test_each_derivation_gives_the_subset_its_quota() -> None:
    quota = np.array([10.0, 20.0, 30.0])
    assert derive(quota, GROUPS, None, D.UNCHANGED).tolist() == [10, 20, 30]  # type: ignore[union-attr]
    assert derive(quota, GROUPS, COVERED, D.RESTRICT).tolist() == [10, 0, 0]  # type: ignore[union-attr]
    reshared = derive(quota, GROUPS, PLACED, D.RESHARE, shares=[1.0, 0.25, 1.0])
    assert reshared is not None
    assert reshared.tolist() == [10, 5, 0]  # outside groups get nothing, whatever their share
    assert derive(quota, GROUPS, PLACED, D.UNGUIDED) is None
    assert derive(quota, GROUPS, PLACED, D.REFIT) is None


def test_derivations_work_step_by_step() -> None:
    quota = np.array([[10.0, 20.0, 30.0], [1.0, 2.0, 3.0]])
    got = derive(quota, GROUPS, PLACED, D.RESHARE, shares=[1.0, 0.5, 0.0])
    assert got is not None
    assert got.tolist() == [[10, 10, 0], [1, 1, 0]]


def test_bad_derivation_inputs_are_refused() -> None:
    with pytest.raises(QuotaError, match="quotas have 2 groups, the artifact 3"):
        derive([1.0, 2.0], GROUPS, COVERED, D.RESTRICT)
    with pytest.raises(QuotaError, match="non-negative"):
        derive([1.0, -2.0, 0.0], GROUPS, COVERED, D.RESTRICT)
    with pytest.raises(QuotaError, match="needs each group's share"):
        derive([1.0, 2.0, 3.0], GROUPS, PLACED, D.RESHARE)
    with pytest.raises(QuotaError, match="one value in \\[0, 1\\] per group"):
        derive([1.0, 2.0, 3.0], GROUPS, PLACED, D.RESHARE, shares=[1.0, 2.0, 0.0])
    with pytest.raises(QuotaError, match="use disaggregate"):
        derive([1.0, 2.0, 3.0], GROUPS, PLACED, D.DISAGGREGATE)


def test_shares_come_from_a_basis_counted_per_group() -> None:
    assert inside_shares([5, 0, 3], [10, 0, 3]).tolist() == [0.5, 0.0, 1.0]
    with pytest.raises(QuotaError, match="more inside the subset than in all"):
        inside_shares([5], [4])
    with pytest.raises(QuotaError, match="one entry per group"):
        inside_shares([5, 1], [5])


def test_disaggregation_splits_by_basis_and_equally_without_one() -> None:
    # Two coarse groups: the first over fine groups 0 and 1, the second over 2 and 3.
    got = disaggregate([10.0, 6.0], parent=[0, 0, 1, 1], basis=[3.0, 1.0, 0.0, 0.0])
    assert got.tolist() == [7.5, 2.5, 3.0, 3.0]
    steps = disaggregate([[10.0, 6.0], [2.0, 2.0]], [0, 0, 1, 1], [1.0, 1.0, 1.0, 3.0])
    assert steps.tolist() == [[5.0, 5.0, 1.5, 4.5], [1.0, 1.0, 0.5, 1.5]]
    with pytest.raises(QuotaError, match=r"coarse groups \[1\] have no finer groups"):
        disaggregate([10.0, 6.0], [0, 0], [1.0, 1.0])
    with pytest.raises(QuotaError, match=r"parents must lie in 0\.\.1"):
        disaggregate([10.0, 6.0], [0, 2], [1.0, 1.0])
    with pytest.raises(QuotaError, match="one entry per fine group"):
        disaggregate([10.0], [0, 0], [1.0])


@settings(max_examples=50)
@given(
    st.lists(st.floats(0, 100), min_size=1, max_size=4),
    st.lists(st.floats(0.1, 10), min_size=4, max_size=4),
)
def test_disaggregation_preserves_each_group_s_total(
    quota: list[float], basis: list[float]
) -> None:
    parent = [i % len(quota) for i in range(max(4, len(quota)))]
    weights = (basis * 2)[: len(parent)]
    got = disaggregate(quota, parent, weights)
    totals = np.bincount(parent, weights=got, minlength=len(quota))
    np.testing.assert_allclose(totals, quota, atol=1e-9)


def test_raking_matches_both_margins() -> None:
    seed = np.array([[1.0, 2.0], [3.0, 4.0]])
    margins = RakingMarginals(np.array([10.0, 20.0]), np.array([12.0, 18.0]))
    table = rake(seed, margins)
    np.testing.assert_allclose(table.sum(axis=1), [10, 20])
    np.testing.assert_allclose(table.sum(axis=0), [12, 18])
    # Raking keeps the seed's odds ratio.
    assert table[0, 0] * table[1, 1] / (table[0, 1] * table[1, 0]) == pytest.approx(4 / 6)


def test_raking_refuses_inconsistent_or_impossible_margins() -> None:
    with pytest.raises(QuotaError, match="disagree on the total: 30 by row, 31 by column"):
        RakingMarginals(np.array([10.0, 20.0]), np.array([12.0, 19.0]))
    with pytest.raises(QuotaError, match="one-dimensional"):
        RakingMarginals(np.ones((2, 2)), np.ones((2, 2)))
    margins = RakingMarginals(np.array([10.0, 20.0]), np.array([15.0, 15.0]))
    with pytest.raises(QuotaError, match=r"empty where the margins need mass: rows \[1\]"):
        rake(np.array([[1.0, 1.0], [0.0, 0.0]]), margins)
    with pytest.raises(QuotaError, match="the seed is \\(1, 2\\)"):
        rake(np.ones((1, 2)), margins)
    with pytest.raises(QuotaError, match="didn't converge in 1 iterations"):
        rake(np.array([[1.0, 2.0], [3.0, 4.0]]), margins, max_iterations=1)


@pytest.mark.parametrize(
    ("derivation", "placement", "refusal", "hint"),
    [
        (D.UNCHANGED, COVERED, Refusal.UNCHANGED_FOR_SUBSET, "use restrict, reshare"),
        (D.RESTRICT, None, Refusal.SUBSET_DERIVATION_WITHOUT_SUBSET, "use unchanged"),
        (D.RESHARE, None, Refusal.SUBSET_DERIVATION_WITHOUT_SUBSET, "use unchanged"),
        (D.RESTRICT, PLACED, Refusal.RESTRICT_CROSSING, "groups ['ORG-B'] cross"),
        (D.RESHARE, UNREACHABLE, Refusal.NO_GROUP_REACHES, "use refit or unguided"),
        (D.RESTRICT, UNREACHABLE, Refusal.NO_GROUP_REACHES, "nothing to read"),
        (D.DISAGGREGATE, None, Refusal.DISAGGREGATE_NOT_COARSER, "nothing to split"),
    ],
)
def test_refusals_are_named(
    derivation: Derivation, placement: Placement | None, refusal: Refusal, hint: str
) -> None:
    with pytest.raises(QuotaRefused, match=f"^{refusal.value}: ") as error:
        check_derivation(derivation, placement)
    assert error.value.refusal is refusal
    assert hint in str(error.value)


def test_what_every_run_can_do() -> None:
    for placement in (None, COVERED, PLACED, UNREACHABLE):
        for derivation in (D.REFIT, D.UNGUIDED):
            check_derivation(derivation, placement)
    check_derivation(D.UNCHANGED, None)
    check_derivation(D.RESHARE, COVERED)  # resharing a covered subset keeps whole groups
    check_derivation(D.DISAGGREGATE, None, coarser=True)
    check_derivation(D.DISAGGREGATE, COVERED, coarser=True)


def test_derive_refuses_through_the_same_checks() -> None:
    with pytest.raises(QuotaRefused, match=r"^restrict_crossing: "):
        derive([1.0, 2.0, 3.0], GROUPS, PLACED, D.RESTRICT)
