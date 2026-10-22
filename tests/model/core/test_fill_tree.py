"""Tests for the fill tree and its structure checks."""

from __future__ import annotations

import pytest

from hcsim.model.core import fill_tree
from hcsim.model.core.categories import GuidanceCategory
from hcsim.model.core.fill_tree import (
    FILL_TREE,
    ancestors,
    check,
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
