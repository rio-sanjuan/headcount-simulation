"""Tests for the guidance categories."""

from __future__ import annotations

import pytest

from hcsim.model.core.categories import (
    ANCHOR,
    CATEGORIES,
    Grain,
    GuidanceCategory,
    UnknownCategoryError,
    category_for_consumer,
    coherency_terms,
    parse_category,
)
from hcsim.model.core.measures import CATEGORY_MEASURES

C = GuidanceCategory


def test_there_are_eleven_categories_each_with_a_spec_and_a_measure() -> None:
    assert len(GuidanceCategory) == 11
    assert set(CATEGORIES) == set(GuidanceCategory) == set(CATEGORY_MEASURES)
    for category, measure in CATEGORY_MEASURES.items():
        assert measure.name == category.value
        assert measure.grain is CATEGORIES[category].grain


def test_grains_place_each_category_in_its_identity() -> None:
    by_grain = {g: {c for c, s in CATEGORIES.items() if s.grain is g} for g in Grain}
    assert by_grain[Grain.STOCK] == {ANCHOR}
    assert by_grain[Grain.REQUISITION] == {C.REQUISITIONS_OPENED, C.REQUISITIONS_CLOSED_UNFILLED}
    assert len(by_grain[Grain.HEADCOUNT]) == 8


def test_the_coherency_identity_holds_the_leaf_flows() -> None:
    assert coherency_terms() == {
        C.ATTRITION: -1,
        C.EXTERNAL_HIRES: +1,
        C.CONVERSION_HIRES: +1,
        C.INTERNAL_STARTS: +1,
        C.CONVERSIONS_OUT: -1,
    }
    # The fill tree's inner nodes enter through their parts, and promotions move no one.
    for category in (C.HIRES, C.FILLS, C.PROMOTIONS, C.REQUISITIONS_OPENED, C.HEADCOUNT):
        assert category not in coherency_terms()
    assert CATEGORIES[ANCHOR].coherency_sign == 0


def test_consumers_resolve_to_their_category() -> None:
    assert category_for_consumer("attrition") is C.ATTRITION
    assert category_for_consumer("external_fill") is C.EXTERNAL_HIRES
    assert category_for_consumer("transfer_fill") is C.INTERNAL_STARTS
    assert category_for_consumer("backfill") is C.REQUISITIONS_OPENED
    assert category_for_consumer("requisition_generation") is C.REQUISITIONS_OPENED
    assert category_for_consumer("cancellation") is C.REQUISITIONS_CLOSED_UNFILLED
    assert category_for_consumer("reorganization") is None  # unguided by declaration
    consumers = [d for s in CATEGORIES.values() for d in s.consumers]
    assert len(consumers) == len(set(consumers))  # one category per domain output


def test_unknown_names_are_refused_with_the_known_ones() -> None:
    assert parse_category("hires") is C.HIRES
    with pytest.raises(UnknownCategoryError, match="known: headcount, attrition, external_hires"):
        parse_category("terms")
