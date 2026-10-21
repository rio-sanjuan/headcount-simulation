"""Buffer layouts: the declared column order of the employee and opening buffers.

Each trajectory holds two preallocated float64 arrays, one row per active employee and one row
per open requisition. The two layouts don't share a column order, so every consumer that reads
both translates through these declarations, never through bare indices.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from enum import IntEnum
from types import MappingProxyType
from typing import Final

import numpy as np
from numpy.typing import NDArray

from hcsim.model.core.schema import (
    COMPANY,
    COST_CENTER,
    JOB_FAMILY,
    JOB_LEVEL,
    LOCATION,
    REQ_TYPE,
    TECH_INDICATOR,
)

Buffer = NDArray[np.float64]

# Bulk placeholder requisitions can carry hundreds of openings and would absorb every fill, so
# seeding and generation both cap a requisition's openings.
MAX_OPENINGS_PER_REQUISITION: Final = 10


class EmployeeColumn(IntEnum):
    EMPLOYEE_ID = 0
    LEVEL = 1
    LOCATION = 2
    JOB_FAMILY = 3
    TECH_INDICATOR = 4
    IS_MANAGER = 5
    PERF_RATING = 6  # 0 means not rated
    COST_CENTER = 7
    BASE_SALARY = 8  # written by award sizing
    HIRE_STEP = 9  # step - HIRE_STEP is tenure in months
    VEST_INDEX = 10  # the employee's slot in the RSU tracker
    COMPANY = 11


class OpeningColumn(IntEnum):
    REQ_ID = 0
    LEVEL = 1
    LOCATION = 2
    JOB_FAMILY = 3
    TECH_INDICATOR = 4
    IS_MANAGER = 5
    COST_CENTER = 6
    CREATION_STEP = 7  # step - CREATION_STEP is the requisition's age
    REQ_TYPE = 8
    COMPANY = 9
    NUM_OPENINGS = 10  # remaining fillable openings


@dataclass(frozen=True)
class BufferLayout:
    """A buffer's columns, and the vocabulary each encoded column's codes belong to."""

    name: str
    columns: type[IntEnum]
    vocabularies: Mapping[IntEnum, str]

    @property
    def width(self) -> int:
        return len(self.columns)

    def column(self, vocabulary: str) -> IntEnum:
        """The column holding codes from `vocabulary`."""
        for column, name in self.vocabularies.items():
            if name == vocabulary:
                return column
        raise KeyError(f"{self.name} buffer has no column for vocabulary {vocabulary!r}")

    def empty(self, capacity: int) -> Buffer:
        """A zeroed buffer. Zero is a valid code in every encoded column, so only the live
        row count separates padding from real rows."""
        if capacity < 0:
            raise ValueError("capacity must be non-negative")
        return np.zeros((capacity, self.width), dtype=np.float64)


_SHARED = {
    "LEVEL": JOB_LEVEL,
    "LOCATION": LOCATION,
    "JOB_FAMILY": JOB_FAMILY,
    "TECH_INDICATOR": TECH_INDICATOR,
    "COST_CENTER": COST_CENTER,
    "COMPANY": COMPANY,
}

EMPLOYEE: Final = BufferLayout(
    "employee",
    EmployeeColumn,
    MappingProxyType({EmployeeColumn[name]: vocab for name, vocab in _SHARED.items()}),
)
_OPENING_VOCABULARIES: dict[IntEnum, str] = {
    OpeningColumn[name]: vocab for name, vocab in _SHARED.items()
}
_OPENING_VOCABULARIES[OpeningColumn.REQ_TYPE] = REQ_TYPE
OPENING: Final = BufferLayout("opening", OpeningColumn, MappingProxyType(_OPENING_VOCABULARIES))

# A fill copies the requisition's role into the employee row it creates or moves.
FILL_COLUMN_MAP: Final[Mapping[OpeningColumn, EmployeeColumn]] = MappingProxyType(
    {OpeningColumn[name]: EmployeeColumn[name] for name in (*_SHARED, "IS_MANAGER")}
)


def copy_role(openings: Buffer, employees: Buffer) -> None:
    """Write each opening row's role columns into the matching employee row, in place."""
    if openings.shape[0] != employees.shape[0]:
        raise ValueError("copy_role needs one employee row per opening row")
    for source, target in FILL_COLUMN_MAP.items():
        employees[:, target] = openings[:, source]
