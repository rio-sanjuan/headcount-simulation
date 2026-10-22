"""Event cells: what each domain did, counted by the attributes of the rows it touched.

Each domain's result is captured just before the apply step writes it, so a dropped row is
counted as it was, a written row both as it was (`write_from`) and as it's written
(`write_to`), and an appended row as it arrives. A cell is one combination of:

- the step and the domain;
- the buffer (`employee` or `opening`) and the change (`drop`, `write_from`, `write_to`, or
  `append`);
- the row's cost center, level, and job family.

Each cell counts its rows and, in the opening buffer, the openings they hold. A trajectory
keeps its cells as one integer array, which `cells_frame` turns into a frame, code-keyed or
decoded to labels. A code the schema doesn't carry, `UNATTRIBUTED`, decodes to a missing label.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Final, cast

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from hcsim.engine.results import DomainResult
from hcsim.engine.state import SimulationState
from hcsim.model.core.layouts import Buffer, EmployeeColumn, OpeningColumn
from hcsim.model.core.schema import COST_CENTER, JOB_FAMILY, JOB_LEVEL, SimulationSchema

Ints = NDArray[np.int64]

BUFFERS: Final = ("employee", "opening")
EMPLOYEES, OPENINGS = range(2)
CHANGES: Final = ("drop", "write_from", "write_to", "append")
#: The attributes a cell counts by, as vocabularies.
ATTRIBUTES: Final = (COST_CENTER, JOB_LEVEL, JOB_FAMILY)
#: Each buffer's columns for the attributes.
_KEYS: Final = (
    [EmployeeColumn.COST_CENTER, EmployeeColumn.LEVEL, EmployeeColumn.JOB_FAMILY],
    [OpeningColumn.COST_CENTER, OpeningColumn.LEVEL, OpeningColumn.JOB_FAMILY],
)
#: The columns of a cell frame.
CELL_COLUMNS: Final = (
    "trajectory",
    "step",
    "domain",
    "buffer",
    "change",
    *ATTRIBUTES,
    "rows",
    "openings",
)
#: The width of a trajectory's cell array: every frame column but the trajectory.
WIDTH: Final = len(CELL_COLUMNS) - 1


class EventCells:
    """One trajectory's cells, gathered step by step."""

    def __init__(self, schema: SimulationSchema) -> None:
        # One more than each vocabulary's size, so UNATTRIBUTED shifts to code 0.
        self._radix = tuple(len(schema.vocabulary(v)) + 1 for v in ATTRIBUTES)
        self._chunks: list[Ints] = []

    def capture(self, step: int, domain: int, result: DomainResult, state: SimulationState) -> None:
        """Count `result`'s rows by cell, before it's applied to `state`."""
        changes = (
            (result.employees, state.live_employees),
            (result.openings, state.live_openings),
        )
        for buffer, (change, live) in enumerate(changes):
            parts = (live[change.drop], live[change.write_index], change.write_rows, change.append)
            for kind, rows in enumerate(parts):
                if len(rows):
                    self._add(step, domain, buffer, kind, rows)

    def _add(self, step: int, domain: int, buffer: int, kind: int, rows: Buffer) -> None:
        keys = rows[:, _KEYS[buffer]].astype(np.int64) + 1
        flat = np.ravel_multi_index(tuple(keys.T), self._radix)
        cells, inverse, counts = np.unique(flat, return_inverse=True, return_counts=True)
        if buffer == OPENINGS:
            held = rows[:, OpeningColumn.NUM_OPENINGS]
            openings = np.bincount(inverse, weights=held, minlength=len(cells)).astype(np.int64)
        else:
            openings = np.zeros(len(cells), dtype=np.int64)
        attributes = np.stack(np.unravel_index(cells, self._radix), axis=1) - 1
        fixed = np.tile(np.array([step, domain, buffer, kind], dtype=np.int64), (len(cells), 1))
        self._chunks.append(np.column_stack([fixed, attributes, counts, openings]).astype(np.int64))

    def array(self) -> Ints:
        """(cells, `WIDTH`): the step, domain, buffer, change, attributes, rows, and openings."""
        if not self._chunks:
            return empty()
        return np.concatenate(self._chunks)


def empty() -> Ints:
    return np.zeros((0, WIDTH), dtype=np.int64)


def cells_frame(
    cells: Sequence[tuple[int, Ints]],
    domains: Sequence[str],
    schema: SimulationSchema,
    *,
    decoded: bool = False,
) -> pd.DataFrame:
    """One row per trajectory and cell, from each trajectory's index and cell array. The
    domain, buffer, and change are categories; the attributes are codes, or with `decoded`,
    their labels."""
    parts = [np.column_stack([np.full(len(a), j, dtype=np.int64), a]) for j, a in cells]
    array = np.concatenate(parts) if parts else np.zeros((0, len(CELL_COLUMNS)), dtype=np.int64)
    frame = pd.DataFrame(array, columns=list(CELL_COLUMNS))
    for column, names in (("domain", domains), ("buffer", BUFFERS), ("change", CHANGES)):
        codes = cast("Sequence[int]", frame[column].to_numpy())
        frame[column] = pd.Categorical.from_codes(codes, categories=pd.Index(list(names)))
    if decoded:
        for vocabulary in ATTRIBUTES:
            frame[vocabulary] = decode(schema, vocabulary, frame[vocabulary].to_numpy())
    return frame


def decode(schema: SimulationSchema, vocabulary: str, codes: Ints) -> NDArray[np.object_]:
    """Labels for codes, with UNATTRIBUTED as a missing label."""
    labels = np.full(len(codes), None, dtype=object)
    known = codes >= 0
    labels[known] = schema.decode(vocabulary, codes[known])
    return labels
