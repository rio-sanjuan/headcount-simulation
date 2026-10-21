"""Compensation ranges: an exhaustive lookup by job family, level, and location.

Every gap-fill strategy starts from a range. Coverage must be complete: a missing cell is an
error that names the cells, and nothing is imputed. A recipient whose cell is missing would
otherwise receive a silent zero grant.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from hcsim.model.core.schema import JOB_FAMILY, JOB_LEVEL, LOCATION, SimulationSchema


class RangeField(IntEnum):
    MIN_BASE = 0
    MAX_BASE = 1
    MIN_TARGET = 2
    MAX_TARGET = 3
    BASE_ADJ = 4


class RangeError(ValueError):
    """Compensation ranges that can't size every recipient."""


@dataclass(frozen=True)
class RangeTable:
    #: Shape (families, levels, locations, fields), in the schema's code order.
    values: NDArray[np.float64]

    @classmethod
    def from_frame(cls, frame: pd.DataFrame, schema: SimulationSchema) -> RangeTable:
        axes = (JOB_FAMILY, JOB_LEVEL, LOCATION)
        if frame.duplicated(list(axes)).any():
            raise RangeError("duplicate ranges for one family, level, and location")
        vocabs = [schema.vocabulary(a) for a in axes]
        values = np.full((*(len(v) for v in vocabs), len(RangeField)), np.nan)
        codes = [
            v.encode(frame[a], unknown="unattributed") for v, a in zip(vocabs, axes, strict=True)
        ]
        known = np.all([c >= 0 for c in codes], axis=0)
        fields = frame[[f.name.lower() for f in RangeField]].to_numpy(dtype=np.float64)
        values[codes[0][known], codes[1][known], codes[2][known]] = fields[known]
        missing = np.argwhere(np.isnan(values[..., 0]))
        if len(missing):
            cells = [
                tuple(v.label(int(i)) for v, i in zip(vocabs, cell, strict=True))
                for cell in missing[:10]
            ]
            raise RangeError(f"{len(missing)} missing compensation ranges, such as {cells}")
        bad = (values[..., RangeField.MIN_BASE] > values[..., RangeField.MAX_BASE]) | (
            values[..., RangeField.MIN_TARGET] > values[..., RangeField.MAX_TARGET]
        )
        if bad.any():
            raise RangeError(f"{int(bad.sum())} ranges have a minimum above their maximum")
        return cls(values)

    def lookup(
        self, families: NDArray[np.int64], levels: NDArray[np.int64], locations: NDArray[np.int64]
    ) -> NDArray[np.float64]:
        """(N, fields) ranges for N recipients. An unattributed code is an error."""
        for name, codes in (("family", families), ("level", levels), ("location", locations)):
            if codes.size and codes.min() < 0:
                raise RangeError(f"recipients with an unattributed {name} can't be sized")
        result: NDArray[np.float64] = self.values[families, levels, locations]
        return result
