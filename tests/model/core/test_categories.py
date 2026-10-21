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


def test_every_category_has_a_spec() -> None:
    assert set(CATEGORIES) == set(GuidanceCategory)


def test_grains_place_each_category_in_its_identity() -> None:
    assert CATEGORIES[GuidanceCategory.HIRES].grain is Grain.HEADCOUNT
    assert CATEGORIES[GuidanceCategory.REQUISITIONS_OPENED].grain is Grain.REQUISITION
    assert CATEGORIES[ANCHOR].grain is Grain.STOCK


def test_the_coherency_identity_has_hires_in_and_attrition_out() -> None:
    assert coherency_terms() == {GuidanceCategory.ATTRITION: -1, GuidanceCategory.HIRES: +1}


def test_requisitions_opened_is_an_inflow_outside_the_coherency_identity() -> None:
    assert GuidanceCategory.REQUISITIONS_OPENED not in coherency_terms()


def test_consumers_resolve_to_their_category() -> None:
    assert category_for_consumer("attrition") is GuidanceCategory.ATTRITION
    assert category_for_consumer("backfill") is GuidanceCategory.REQUISITIONS_OPENED
    assert category_for_consumer("requisition_generation") is GuidanceCategory.REQUISITIONS_OPENED
    assert category_for_consumer("cancellation") is None  # unguided by declaration


def test_unknown_names_are_refused_with_the_known_ones() -> None:
    assert parse_category("hires") is GuidanceCategory.HIRES
    with pytest.raises(UnknownCategoryError, match="known: attrition, hires"):
        parse_category("terms")
