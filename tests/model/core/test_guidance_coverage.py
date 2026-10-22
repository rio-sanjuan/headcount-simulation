"""Tests for guidance coverage states and mapping fidelity."""

from __future__ import annotations

import pytest

from hcsim.model.core.categories import GuidanceCategory
from hcsim.model.core.guidance_coverage import (
    FIDELITY,
    CoverageError,
    CoverageState,
    Fidelity,
    coverage,
)

C = GuidanceCategory
S = CoverageState
EVERY_DOMAIN = sorted(FIDELITY)
SKELETON = ["attrition", "external_fill", "backfill", "requisition_generation", "cancellation"]


def states(**kwargs: object) -> dict[GuidanceCategory, CoverageState]:
    return {c: v.state for c, v in coverage(**kwargs).items()}  # type: ignore[arg-type]


def test_a_full_forecast_guides_every_domain() -> None:
    got = coverage(set(GuidanceCategory), EVERY_DOMAIN)
    for leaf in (
        C.ATTRITION,
        C.EXTERNAL_HIRES,
        C.CONVERSION_HIRES,
        C.INTERNAL_STARTS,
        C.CONVERSIONS_OUT,
        C.PROMOTIONS,
        C.REQUISITIONS_OPENED,
        C.REQUISITIONS_CLOSED_UNFILLED,
    ):
        assert got[leaf].state is S.COMPOSED
        assert got[leaf].source is leaf
        assert got[leaf].guided
    for internal in (C.HEADCOUNT, C.HIRES, C.FILLS):
        assert got[internal].state is S.FORECAST_INTERNAL
        assert not got[internal].guided


def test_the_skeleton_forecast_leaves_the_rest_unfitted() -> None:
    carried = {C.ATTRITION, C.EXTERNAL_HIRES, C.HEADCOUNT, C.REQUISITIONS_OPENED}
    got = states(carried=carried, domains=SKELETON)
    assert got[C.ATTRITION] is S.COMPOSED
    assert got[C.EXTERNAL_HIRES] is S.COMPOSED
    assert got[C.REQUISITIONS_OPENED] is S.COMPOSED
    assert got[C.HEADCOUNT] is S.FORECAST_INTERNAL
    for missing in (
        C.CONVERSION_HIRES,
        C.INTERNAL_STARTS,
        C.HIRES,
        C.FILLS,
        C.REQUISITIONS_CLOSED_UNFILLED,
        C.PROMOTIONS,
        C.CONVERSIONS_OUT,
    ):
        assert got[missing] is S.UNFITTED, missing


def test_a_parent_alone_is_shared_by_the_domains_below_it() -> None:
    got = coverage({C.HIRES}, EVERY_DOMAIN)
    assert got[C.HIRES].state is S.UNCOMPOSED
    for leaf in (C.EXTERNAL_HIRES, C.CONVERSION_HIRES):
        assert (got[leaf].state, got[leaf].source) == (S.UNCOMPOSED, C.HIRES)
    assert got[C.INTERNAL_STARTS].state is S.UNFITTED
    assert got[C.FILLS].state is S.UNFITTED
    nearest = coverage({C.FILLS, C.HIRES}, EVERY_DOMAIN)
    assert nearest[C.EXTERNAL_HIRES].source is C.HIRES  # the nearest carried ancestor
    assert nearest[C.INTERNAL_STARTS].source is C.FILLS


def test_a_parent_can_be_composed_from_its_parts() -> None:
    got = coverage({C.EXTERNAL_HIRES, C.CONVERSION_HIRES}, EVERY_DOMAIN)
    assert (got[C.HIRES].state, got[C.HIRES].source) == (S.COMPOSED, None)
    assert got[C.FILLS].state is S.UNFITTED  # internal starts are missing


def test_declared_and_unmapped_domains() -> None:
    got = states(
        carried={C.ATTRITION, C.PROMOTIONS, C.HIRES},
        domains=["attrition", "external_fill"],
        unguided=["attrition", "external_fill"],
    )
    assert got[C.ATTRITION] is S.DECLARED_UNGUIDED
    assert got[C.EXTERNAL_HIRES] is S.DECLARED_UNGUIDED
    assert got[C.PROMOTIONS] is S.UNMAPPED  # no promotion domain in this run
    assert got[C.CONVERSION_HIRES] is S.UNMAPPED  # its share of hires has no domain


def test_fidelity_marks_domains_that_realize_part_of_a_series() -> None:
    got = coverage({C.EXTERNAL_HIRES, C.ATTRITION}, EVERY_DOMAIN)
    assert dict(got[C.EXTERNAL_HIRES].consumers) == {"external_fill": Fidelity.NARROWER}
    assert dict(got[C.ATTRITION].consumers) == {"attrition": Fidelity.EXACT}
    opened = got[C.REQUISITIONS_OPENED].consumers
    assert set(opened.values()) == {Fidelity.NARROWER}
    assert coverage(set(), ["attrition"])[C.EXTERNAL_HIRES].consumers == {}


def test_unknown_domains_are_refused() -> None:
    with pytest.raises(CoverageError, match=r"no category is consumed by \['reorganization'\]"):
        coverage(set(), ["attrition", "reorganization"])
    with pytest.raises(
        CoverageError, match=r"declared unguided but not in the run: \['attrition'\]"
    ):
        coverage(set(), ["external_fill"], unguided=["attrition"])
