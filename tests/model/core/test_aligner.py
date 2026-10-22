"""Tests for the group aligner."""

from __future__ import annotations

import numpy as np
import pytest

from hcsim.model.core.aligner import TOPLINE, GroupAligner
from hcsim.model.core.layouts import EMPLOYEE, OPENING, EmployeeColumn, OpeningColumn
from hcsim.model.core.schema import (
    COST_CENTER,
    JOB_FAMILY,
    JOB_LEVEL,
    UNATTRIBUTED,
    SchemaError,
    SimulationSchema,
)


def schema() -> SimulationSchema:
    return SimulationSchema.build(
        {COST_CENTER: ["CC-01", "CC-02", "CC-03"], JOB_LEVEL: [1, 2, 3]},
        org_levels={"org_level1": {"CC-01": "ORG-A", "CC-02": "ORG-B", "CC-03": "ORG-A"}},
    )


def employees(cost_centers: list[int]) -> np.ndarray:
    buffer = EMPLOYEE.empty(len(cost_centers))
    buffer[:, EmployeeColumn.COST_CENTER] = cost_centers
    return buffer


def test_rows_map_to_the_artifacts_group_order_and_uncarried_groups_get_minus_one() -> None:
    aligner = GroupAligner(schema(), COST_CENTER, ["CC-03", "CC-01"])  # CC-02 not carried
    assert aligner.groups(employees([0, 1, 2, 0]), EMPLOYEE).tolist() == [1, UNATTRIBUTED, 0, 1]


def test_the_same_aligner_reads_both_buffer_layouts() -> None:
    aligner = GroupAligner(schema(), COST_CENTER, ["CC-01", "CC-02", "CC-03"])
    openings = OPENING.empty(2)
    openings[:, OpeningColumn.COST_CENTER] = [2, 1]
    assert aligner.groups(openings, OPENING).tolist() == [2, 1]


def test_rollups_group_by_the_org_node() -> None:
    aligner = GroupAligner(schema(), "org_level1", ["ORG-A", "ORG-B"])
    assert aligner.groups(employees([0, 1, 2]), EMPLOYEE).tolist() == [0, 1, 0]


def test_topline_puts_everyone_in_one_group() -> None:
    aligner = GroupAligner(schema(), TOPLINE, ["ALL"])
    assert aligner.groups(employees([0, 2, 1]), EMPLOYEE).tolist() == [0, 0, 0]
    with pytest.raises(SchemaError, match="exactly"):
        GroupAligner(schema(), TOPLINE, ["EVERYONE"])


def test_a_label_the_schema_cant_resolve_fails_at_build() -> None:
    with pytest.raises(SchemaError, match="not in the 'cost_center' vocabulary"):
        GroupAligner(schema(), COST_CENTER, ["CC-01", "CC-99"])
    with pytest.raises(SchemaError, match="duplicate"):
        GroupAligner(schema(), COST_CENTER, ["CC-01", "CC-01"])


def test_a_level_grouping_reads_the_level_column() -> None:
    buffer = EMPLOYEE.empty(3)
    buffer[:, EmployeeColumn.LEVEL] = [2, 0, 1]
    aligner = GroupAligner(schema(), JOB_LEVEL, ["1", "2", "3"])
    assert aligner.groups(buffer, EMPLOYEE).tolist() == [2, 0, 1]


def test_counts_ignore_unattributed_rows() -> None:
    aligner = GroupAligner(schema(), COST_CENTER, ["CC-01", "CC-02"])
    groups = np.array([0, 1, 1, UNATTRIBUTED], dtype=np.int64)
    assert aligner.counts(groups).tolist() == [1, 2]
    assert aligner.counts(groups, np.array([1.0, 3.0, 2.0, 9.0])).tolist() == [1, 5]


def composite_schema() -> SimulationSchema:
    return SimulationSchema.build(
        {
            COST_CENTER: ["CC-01", "CC-02", "CC-03"],
            JOB_FAMILY: ["Data", "Sales"],
            JOB_LEVEL: [1, 2],
        },
        org_levels={"org_level1": {"CC-01": "ORG-A", "CC-02": "ORG-B", "CC-03": "ORG-A"}},
    )


def test_a_composite_key_groups_by_several_columns_and_rollups() -> None:
    groups = [("ORG-B", "Data"), ("ORG-A", "Sales"), ("ORG-A", "Data")]  # ORG-B Sales not carried
    aligner = GroupAligner(composite_schema(), ["org_level1", "job_family"], groups)
    assert aligner.column == "org_level1+job_family"
    assert aligner.group_labels == tuple(groups)
    buffer = employees([0, 1, 2, 1, 0])
    buffer[:, EmployeeColumn.JOB_FAMILY] = [1, 0, 0, 1, 0]
    assert aligner.groups(buffer, EMPLOYEE).tolist() == [1, 0, 2, UNATTRIBUTED, 2]
    openings = OPENING.empty(2)
    openings[:, OpeningColumn.COST_CENTER] = [2, 1]
    openings[:, OpeningColumn.JOB_FAMILY] = [1, 0]
    assert aligner.groups(openings, OPENING).tolist() == [1, 0]
    assert aligner.counts(aligner.groups(buffer, EMPLOYEE)).tolist() == [1, 1, 2]


def test_composite_codes_with_any_gap_are_unattributed() -> None:
    aligner = GroupAligner(composite_schema(), ["cost_center", "job_level"], [("CC-01", "2")])
    codes = np.array([[0, 1], [0, 0], [UNATTRIBUTED, 1], [0, UNATTRIBUTED]])
    assert aligner.groups_of_codes(codes).tolist() == [0, UNATTRIBUTED, UNATTRIBUTED, UNATTRIBUTED]
    with pytest.raises(SchemaError, match="codes need 2 columns, not 1"):
        aligner.groups_of_codes(np.zeros((1, 1), dtype=np.int64))


def test_the_first_column_is_the_most_significant() -> None:
    labels = [("CC-02", "1"), ("CC-01", "2")]
    a = GroupAligner(composite_schema(), ["cost_center", "job_level"], labels)
    b = GroupAligner(composite_schema(), ["job_level", "cost_center"], [(y, x) for x, y in labels])
    codes = np.array([[1, 0], [0, 1]])
    assert a.groups_of_codes(codes).tolist() == b.groups_of_codes(codes[:, ::-1]).tolist() == [0, 1]


def test_bad_composite_groupings_are_refused() -> None:
    s = composite_schema()
    with pytest.raises(SchemaError, match="at least one column"):
        GroupAligner(s, [], [])
    with pytest.raises(SchemaError, match="can't combine"):
        GroupAligner(s, [TOPLINE, "job_family"], [("ALL", "Data")])
    with pytest.raises(SchemaError, match="columns repeat"):
        GroupAligner(s, ["job_family", "job_family"], [("Data", "Data")])
    with pytest.raises(SchemaError, match="need 2 parts"):
        GroupAligner(s, ["org_level1", "job_family"], [("ORG-A",)])
    with pytest.raises(SchemaError, match="not in the 'org_level1' vocabulary") as error:
        GroupAligner(s, ["org_level1", "job_family"], [("ORG-Z", "Legal")])
    assert "not in the 'job_family' vocabulary: ['Legal']" in str(error.value)


def test_topline_codes_are_one_group_and_huge_keys_are_refused() -> None:
    topline = GroupAligner(schema(), TOPLINE, ["ALL"])
    assert topline.groups_of_codes(np.array([3, -1, 0])).tolist() == [0, 0, 0]
    names = [f"v{k}" for k in range(5)]
    wide = SimulationSchema.build(
        {COST_CENTER: ["CC-01"], **{n: [f"{n}-{i:05d}" for i in range(10_000)] for n in names}}
    )
    with pytest.raises(SchemaError, match="too many combinations"):
        GroupAligner(wide, names, [tuple(f"{n}-00000" for n in names)])
