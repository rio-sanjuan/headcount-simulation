"""Compensation ranges: an exhaustive lookup by job family, level, and location.

Every gap-fill strategy starts from a range. Coverage must be complete: the table needs a range
for every family, level, and location in the schema, executive levels included, because a
recipient whose cell is missing would otherwise receive a silent zero grant. Nothing is
imputed. A table that falls short is refused when it's loaded, with every problem at once:

- every missing cell, by name, grouped by family and level;
- cells listed more than once;
- figures that aren't finite, or are negative;
- a minimum above its maximum, for base or target;
- a base adjustment that isn't positive, or differs between rows.

Rows for a family, level, or location the schema doesn't know are left out: they can't size
anyone the run holds.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from enum import IntEnum
from typing import Any, Final

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from hcsim.model.core.schema import JOB_FAMILY, JOB_LEVEL, LOCATION, SimulationSchema

AXES: Final = (JOB_FAMILY, JOB_LEVEL, LOCATION)
#: The warehouse table ranges load from.
TABLE: Final = "comp_range"


class RangeField(IntEnum):
    MIN_BASE = 0
    MAX_BASE = 1
    MIN_TARGET = 2
    MAX_TARGET = 3
    BASE_ADJ = 4


FIELDS: Final = tuple(f.name.lower() for f in RangeField)


class RangeError(ValueError):
    """Compensation ranges that can't size every recipient."""


def _cells(cells: Iterable[tuple[str, str, str]]) -> str:
    """Cells as `family level L at loc, loc`, grouped by family and level."""
    grouped: dict[tuple[str, str], list[str]] = {}
    for family, level, location in cells:
        grouped.setdefault((family, level), []).append(location)
    return "; ".join(
        f"{family} level {level} at {', '.join(locations)}"
        for (family, level), locations in grouped.items()
    )


@dataclass(frozen=True)
class RangeTable:
    #: Shape (families, levels, locations, fields), in the schema's code order.
    values: NDArray[np.float64]

    @classmethod
    def from_frame(cls, frame: pd.DataFrame, schema: SimulationSchema) -> RangeTable:
        """The table for `schema`, refusing, with every problem named, ranges that don't cover
        it or don't hold together."""
        absent = [c for c in (*AXES, *FIELDS) if c not in frame.columns]
        if absent:
            raise RangeError(f"compensation ranges need the columns {absent}")
        vocabs = [schema.vocabulary(a) for a in AXES]
        codes = [
            v.encode(frame[a], unknown="unattributed") for v, a in zip(vocabs, AXES, strict=True)
        ]
        known = np.all([c >= 0 for c in codes], axis=0)
        rows = frame[known].reset_index(drop=True)
        cell = [c[known] for c in codes]
        problems = []

        def label(index: Sequence[int] | NDArray[np.int64]) -> tuple[str, str, str]:
            names = (v.label(int(i)) for v, i in zip(vocabs, index, strict=True))
            family, level, location = names
            return family, level, location

        keys = pd.Series(list(zip(*cell, strict=True)), dtype=object)
        repeated = sorted(set(keys[keys.duplicated()]))
        if repeated:
            problems.append(
                f"{len(repeated)} cells are listed more than once: {_cells(map(label, repeated))}"
            )
        fields = rows[list(FIELDS)].to_numpy(dtype=np.float64)
        bad = ~np.isfinite(fields).all(axis=1) | (fields < 0).any(axis=1)
        if bad.any():
            listed = (label(k) for k in keys[bad])
            problems.append(
                f"{int(bad.sum())} cells have figures that aren't finite and non-negative: "
                + _cells(listed)
            )
        inverted = (fields[:, RangeField.MIN_BASE] > fields[:, RangeField.MAX_BASE]) | (
            fields[:, RangeField.MIN_TARGET] > fields[:, RangeField.MAX_TARGET]
        )
        if inverted.any():
            problems.append(
                f"{int(inverted.sum())} cells have a minimum above their maximum: "
                + _cells(label(k) for k in keys[inverted])
            )
        adjustments = np.unique(
            fields[np.isfinite(fields[:, RangeField.BASE_ADJ]), RangeField.BASE_ADJ]
        )
        if len(adjustments) > 1 or (len(adjustments) and adjustments[0] <= 0):
            problems.append(
                f"base_adj must be one positive multiplier on every row, not {adjustments.tolist()}"
            )

        values = np.full((*(len(v) for v in vocabs), len(RangeField)), np.nan)
        values[cell[0], cell[1], cell[2]] = fields
        present = np.zeros(values.shape[:-1], dtype=bool)
        present[cell[0], cell[1], cell[2]] = True
        missing = np.argwhere(~present)
        if len(missing):
            problems.append(
                f"{len(missing)} cells have no range, and nothing is imputed: "
                + _cells(label(m) for m in missing)
            )
        if problems:
            raise RangeError(
                "compensation ranges can't size every recipient: " + "; and ".join(problems)
            )
        return cls(values)

    @classmethod
    def load(cls, con: Any, schema: SimulationSchema) -> RangeTable:
        """The warehouse's ranges for `schema`, through a DuckDB connection."""
        columns = ", ".join((*AXES, *FIELDS))
        return cls.from_frame(con.execute(f"SELECT {columns} FROM {TABLE}").df(), schema)

    def lookup(
        self, families: NDArray[np.int64], levels: NDArray[np.int64], locations: NDArray[np.int64]
    ) -> NDArray[np.float64]:
        """(N, fields) ranges for N recipients. An unattributed code is an error."""
        for name, codes in (("family", families), ("level", levels), ("location", locations)):
            if codes.size and codes.min() < 0:
                raise RangeError(f"recipients with an unattributed {name} can't be sized")
        result: NDArray[np.float64] = self.values[families, levels, locations]
        return result
