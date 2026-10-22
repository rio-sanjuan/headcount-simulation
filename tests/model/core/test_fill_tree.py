"""Tests for the fill tree and its structure checks."""

from __future__ import annotations

import pytest

from hcsim.model.core import fill_tree
from hcsim.model.core.categories import GuidanceCategory
from hcsim.model.core.fill_tree import (
    FILL_TREE,
    ancestors,
    check,
    cover,
    identity_terms,
    is_leaf,
    leaves,
    nodes,
    parent,
    roots,
)

C = GuidanceCategory


def test_the_tree_nests_fills_over_hires_and_internal_starts() -> None:
    assert check(FILL_TREE) == []
    assert roots() == [C.FILLS]
    assert nodes() == {C.FILLS, C.HIRES, C.INTERNAL_STARTS, C.EXTERNAL_HIRES, C.CONVERSION_HIRES}
    assert leaves(C.FILLS) == [C.EXTERNAL_HIRES, C.CONVERSION_HIRES, C.INTERNAL_STARTS]
    assert leaves(C.HIRES) == [C.EXTERNAL_HIRES, C.CONVERSION_HIRES]
    assert leaves(C.INTERNAL_STARTS) == [C.INTERNAL_STARTS]
    assert parent(C.EXTERNAL_HIRES) is C.HIRES
    assert parent(C.FILLS) is None
    assert ancestors(C.CONVERSION_HIRES) == [C.HIRES, C.FILLS]
    assert is_leaf(C.EXTERNAL_HIRES)
    assert not is_leaf(C.HIRES)
    assert not is_leaf(C.ATTRITION)  # not in the tree at all


@pytest.mark.parametrize(
    ("tree", "problem"),
    [
        (
            {C.FILLS: (C.HIRES, C.INTERNAL_STARTS), C.HIRES: (C.INTERNAL_STARTS, C.EXTERNAL_HIRES)},
            "more than one parent: ['internal_starts']",
        ),
        (
            {C.FILLS: (C.INTERNAL_STARTS,), C.HIRES: (C.EXTERNAL_HIRES,)},
            "needs one root, not ['fills', 'hires']",
        ),
        ({C.FILLS: (C.HIRES,), C.HIRES: (C.FILLS,)}, "a cycle through"),
        (
            {
                C.FILLS: (C.HIRES, C.REQUISITIONS_OPENED),
                C.HIRES: (C.EXTERNAL_HIRES, C.CONVERSION_HIRES),
            },
            "'requisitions_opened' has requisition grain",
        ),
        ({C.ATTRITION: (C.EXTERNAL_HIRES,)}, "inner node 'attrition' is a term of the identity"),
        ({C.HIRES: (C.EXTERNAL_HIRES,)}, "'hires' has symbol 'F_ext + F_conv', not 'F_ext'"),
        ({C.HIRES: (C.EXTERNAL_HIRES,)}, "'hires''s measure isn't the sum of its children's"),
        (
            {C.FILLS: (C.HIRES, C.INTERNAL_STARTS), C.HIRES: (C.EXTERNAL_HIRES, C.CONVERSIONS_OUT)},
            "the leaves mix inflows and outflows",
        ),
        ({C.HIRES: (C.EXTERNAL_HIRES, C.PROMOTIONS)}, "leaf 'promotions' isn't a term"),
        ({C.HIRES: (C.REQUISITIONS_OPENED,)}, "must all be transaction measures"),
    ],
)
def test_a_broken_tree_is_named(
    tree: dict[GuidanceCategory, tuple[GuidanceCategory, ...]], problem: str
) -> None:
    assert any(problem in found for found in check(tree)), check(tree)


def test_the_tree_is_checked_when_imported() -> None:
    assert fill_tree._problems == []


# --- the identity's terms: the deepest carried cover ---------------------------------------------


@pytest.mark.parametrize(
    ("carried", "terms", "missing"),
    [
        # Leaves, when they cover their parent, are preferred to it.
        (
            {C.FILLS, C.HIRES, C.EXTERNAL_HIRES, C.CONVERSION_HIRES, C.INTERNAL_STARTS},
            {C.EXTERNAL_HIRES, C.CONVERSION_HIRES, C.INTERNAL_STARTS},
            set(),
        ),
        # A parent stands in for children that don't cover it.
        ({C.HIRES, C.EXTERNAL_HIRES, C.INTERNAL_STARTS}, {C.HIRES, C.INTERNAL_STARTS}, set()),
        ({C.FILLS, C.EXTERNAL_HIRES}, {C.FILLS}, set()),
        # With no carried parent, the children cover what they can.
        ({C.EXTERNAL_HIRES}, {C.EXTERNAL_HIRES}, {C.CONVERSION_HIRES, C.INTERNAL_STARTS}),
        (set(), set(), {C.EXTERNAL_HIRES, C.CONVERSION_HIRES, C.INTERNAL_STARTS}),
    ],
)
def test_the_fill_tree_is_covered_by_its_deepest_carried_nodes(
    carried: set[GuidanceCategory], terms: set[GuidanceCategory], missing: set[GuidanceCategory]
) -> None:
    found, gaps = cover(C.FILLS, carried)
    assert set(found) == terms
    assert set(gaps) == missing
    # Every leaf is covered once, or missing.
    covered = [leaf for node in found for leaf in leaves(node)]
    assert len(covered) == len(set(covered))
    assert set(covered) | set(gaps) == set(leaves(C.FILLS))


def test_the_identity_takes_the_cover_and_every_other_carried_flow() -> None:
    terms, missing = identity_terms(
        {C.HEADCOUNT, C.ATTRITION, C.HIRES, C.INTERNAL_STARTS, C.PROMOTIONS}
    )
    assert terms == {C.HIRES: 1, C.INTERNAL_STARTS: 1, C.ATTRITION: -1}
    assert missing == [C.CONVERSIONS_OUT]
    # The skeleton's categories, as the lite pipeline reconciled them.
    terms, missing = identity_terms({C.HEADCOUNT, C.ATTRITION, C.EXTERNAL_HIRES})
    assert terms == {C.EXTERNAL_HIRES: 1, C.ATTRITION: -1}
    assert set(missing) == {C.CONVERSION_HIRES, C.INTERNAL_STARTS, C.CONVERSIONS_OUT}
    # A requisition flow and a promotion are never terms: they move no one in or out.
    terms, _ = identity_terms({C.REQUISITIONS_OPENED, C.PROMOTIONS})
    assert terms == {}
