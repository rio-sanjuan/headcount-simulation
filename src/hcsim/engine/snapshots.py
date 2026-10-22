"""Buffer snapshots: a trajectory's live rows at the steps a run asks to keep.

A run keeps a snapshot for each step in `results_snapshot_steps`. Snapshot `s` holds the
buffers at the start of step `s`, before any of its domains run, so it agrees with row `s` of
the stock histories; snapshot `simulation_months` holds the buffers at the end of the run. A
snapshot copies the live rows, so the run's later writes leave it as it was.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd

from hcsim.engine.events import decode
from hcsim.engine.state import SimulationState
from hcsim.model.core.layouts import EMPLOYEE, OPENING, Buffer, BufferLayout
from hcsim.model.core.schema import SimulationSchema

LAYOUTS = {EMPLOYEE.name: EMPLOYEE, OPENING.name: OPENING}


@dataclass(frozen=True)
class Snapshot:
    employees: Buffer
    openings: Buffer

    @classmethod
    def of(cls, state: SimulationState) -> Snapshot:
        return cls(state.live_employees.copy(), state.live_openings.copy())

    def rows(self, buffer: str) -> Buffer:
        return self.employees if buffer == EMPLOYEE.name else self.openings


def snapshot_frame(
    snapshots: Sequence[tuple[int, Snapshot]],
    buffer: str,
    schema: SimulationSchema,
    *,
    decoded: bool = False,
) -> pd.DataFrame:
    """One row per trajectory and live row of `buffer`, from each trajectory's index and
    snapshot, with a column per layout column. The buffer's values as they're held, or with
    `decoded`, its encoded columns as their labels."""
    if buffer not in LAYOUTS:
        raise ValueError(f"no buffer {buffer!r}; expected one of {sorted(LAYOUTS)}")
    layout = LAYOUTS[buffer]
    columns = [c.name.lower() for c in layout.columns]
    parts = [
        pd.DataFrame(s.rows(buffer), columns=columns).assign(trajectory=j) for j, s in snapshots
    ]
    empty = pd.DataFrame(columns=["trajectory", *columns])
    frame = pd.concat(parts, ignore_index=True) if parts else empty
    frame = frame[["trajectory", *columns]]
    if decoded:
        _decode(frame, layout, schema)
    return frame


def _decode(frame: pd.DataFrame, layout: BufferLayout, schema: SimulationSchema) -> None:
    for column, vocabulary in layout.vocabularies.items():
        if vocabulary in schema.vocabularies:
            name = column.name.lower()
            codes = frame[name].to_numpy(dtype=np.float64).astype(np.int64)
            frame[name] = decode(schema, vocabulary, codes)
