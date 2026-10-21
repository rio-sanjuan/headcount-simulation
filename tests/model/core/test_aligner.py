"""Tests for the group aligner."""

from __future__ import annotations

import numpy as np
import pytest

from hcsim.model.core.aligner import TOPLINE, GroupAligner
from hcsim.model.core.layouts import EMPLOYEE, OPENING, EmployeeColumn, OpeningColumn
from hcsim.model.core.schema import (
    COST_CENTER,
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
