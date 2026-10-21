"""Tests for the buffer layouts."""

from __future__ import annotations

import numpy as np
import pytest

from hcsim.model.core.layouts import (
    EMPLOYEE,
    FILL_COLUMN_MAP,
    OPENING,
    EmployeeColumn,
    OpeningColumn,
    copy_role,
)
from hcsim.model.core.schema import COST_CENTER, REQ_TYPE


def test_widths_match_the_declared_columns() -> None:
    assert (EMPLOYEE.width, OPENING.width) == (12, 11)
    assert [c.value for c in EmployeeColumn] == list(range(12))
    assert [c.value for c in OpeningColumn] == list(range(11))


def test_the_layouts_differ_so_columns_are_found_by_vocabulary() -> None:
    assert EMPLOYEE.column(COST_CENTER) is EmployeeColumn.COST_CENTER
    assert int(EmployeeColumn.COST_CENTER) == 7
    assert OPENING.column(COST_CENTER) is OpeningColumn.COST_CENTER
    assert int(OpeningColumn.COST_CENTER) == 6
    assert OPENING.column(REQ_TYPE) == OpeningColumn.REQ_TYPE
    with pytest.raises(KeyError, match="no column for vocabulary"):
        EMPLOYEE.column(REQ_TYPE)


def test_empty_buffers_are_zeroed_float64() -> None:
    buffer = OPENING.empty(5)
    assert buffer.shape == (5, 11)
    assert buffer.dtype == np.float64
    assert not buffer.any()
    with pytest.raises(ValueError, match="non-negative"):
        EMPLOYEE.empty(-1)


def test_a_fill_copies_every_role_column_and_nothing_else() -> None:
    openings = OPENING.empty(2)
    openings[:, :] = np.arange(1, 12)  # each column holds its own index + 1
    employees = EMPLOYEE.empty(2)
    copy_role(openings, employees)
    for source, target in FILL_COLUMN_MAP.items():
        assert (employees[:, target] == source + 1).all()
    untouched = sorted(set(EmployeeColumn) - set(FILL_COLUMN_MAP.values()))
    assert {EmployeeColumn.EMPLOYEE_ID, EmployeeColumn.HIRE_STEP} <= set(untouched)
    assert not employees[:, untouched].any()


def test_copy_role_needs_matching_row_counts() -> None:
    with pytest.raises(ValueError, match="one employee row per opening row"):
        copy_role(OPENING.empty(2), EMPLOYEE.empty(3))
