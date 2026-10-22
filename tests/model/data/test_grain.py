"""Tests for grain relations between a subset and an artifact's grouping."""

from __future__ import annotations

import pytest

from hcsim.model.core.schema import COST_CENTER, JOB_FAMILY, SimulationSchema
from hcsim.model.data.grain import GrainError, GrainRelation, Placement, relate
from hcsim.model.data.subset import SubsetSpec

R = GrainRelation


def schema() -> SimulationSchema:
    return SimulationSchema.build(
        {COST_CENTER: ["CC-01", "CC-02", "CC-03", "CC-04"], JOB_FAMILY: ["Data", "Sales"]},
        org_levels={
            "org_level1": {"CC-01": "ORG-A", "CC-02": "ORG-A", "CC-03": "ORG-B", "CC-04": "ORG-B"}
        },
    )


ORGS = ["ORG-A", "ORG-B"]
LEAVES = ["CC-01", "CC-02", "CC-03", "CC-04"]


def placed(grouping: str, groups: list[str], **rules: list[str]) -> Placement:
    spec = SubsetSpec({column: frozenset(values) for column, values in rules.items()})
    return relate(spec.lower(schema()), grouping, groups)


def test_a_union_of_whole_groups_is_covered() -> None:
    p = placed("org_level1", ORGS, org_level1=["ORG-A"])
    assert (p.relation, p.inside, p.crossing, p.outside) == (R.COVERED, ("ORG-A",), (), ("ORG-B",))
    leaf = placed("cost_center", LEAVES, org_level1=["ORG-B"])
    assert (leaf.relation, leaf.inside) == (R.COVERED, ("CC-03", "CC-04"))


def test_a_group_partly_inside_crosses_the_boundary() -> None:
    p = placed("org_level1", ORGS, cost_center=["CC-01", "CC-03", "CC-04"])
    assert (p.relation, p.inside, p.crossing) == (R.CROSSING, ("ORG-B",), ("ORG-A",))
    # A rule on another attribute cuts through every group it reaches.
    family = placed("org_level1", ORGS, org_level1=["ORG-A"], job_family=["Data"])
    assert (family.relation, family.crossing, family.outside) == (
        R.CROSSING,
        ("ORG-A",),
        ("ORG-B",),
    )


def test_attribute_groupings_compare_codes() -> None:
    p = placed("job_family", ["Data", "Sales"], job_family=["Sales"])
    assert (p.relation, p.inside, p.outside) == (R.COVERED, ("Sales",), ("Data",))
    cut = placed("job_family", ["Data", "Sales"], cost_center=["CC-01"])
    assert (cut.relation, cut.crossing) == (R.CROSSING, ("Data", "Sales"))


def test_topline_is_covered_only_by_the_whole_population() -> None:
    assert placed("TOPLINE", ["ALL"]).relation is R.COVERED
    assert placed("TOPLINE", ["ALL"], org_level1=["ORG-A"]).relation is R.CROSSING


def test_a_subset_no_group_reaches_is_unreachable() -> None:
    p = placed("cost_center", ["CC-01", "CC-02"], org_level1=["ORG-B"])
    assert (p.relation, p.outside) == (R.UNREACHABLE, ("CC-01", "CC-02"))
    empty = placed("org_level1", ORGS, org_level1=["ORG-A"], cost_center=["CC-03"])
    assert (empty.relation, empty.outside) == (R.UNREACHABLE, tuple(ORGS))


def test_groupings_the_subset_can_t_meet_are_refused() -> None:
    with pytest.raises(GrainError, match="no rollup 'org_level2'"):
        placed("org_level2", ["X"], org_level1=["ORG-A"])
    with pytest.raises(GrainError, match="grouping by 'region'"):
        placed("region", ["R"], org_level1=["ORG-A"])
