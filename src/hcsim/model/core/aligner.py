"""The group aligner: maps buffer rows to the groups of a guidance artifact.

A guidance artifact groups by plain labels and never sees the simulation schema. Building an
aligner validates every group label against the schema, so a label the schema can't resolve
fails at load rather than reading as a silent zero. The aligner then runs every step, because
group membership changes as employees move. A row whose code the artifact doesn't carry gets
group UNATTRIBUTED (-1): a coverage gap, not an error.

This is the single-column aligner. Multi-column keys arrive in M3.6.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Final

import numpy as np
from numpy.typing import NDArray

from hcsim.model.core.layouts import Buffer, BufferLayout
from hcsim.model.core.schema import (
    COST_CENTER,
    UNATTRIBUTED,
    Codes,
    SchemaError,
    SimulationSchema,
)

#: Groups the whole population as one group, labelled `ALL`.
TOPLINE: Final = "TOPLINE"
TOPLINE_LABEL: Final = "ALL"


class GroupAligner:
    """Maps rows to group indices in an artifact's sorted group order."""

    def __init__(self, schema: SimulationSchema, column: str, group_labels: Sequence[str]) -> None:
        if len(set(group_labels)) != len(group_labels):
            raise SchemaError(f"duplicate group labels for {column!r}")
        self.schema = schema
        self.column = column
        self.group_labels = tuple(group_labels)
        if column == TOPLINE:
            if self.group_labels != (TOPLINE_LABEL,):
                raise SchemaError(f"{TOPLINE} groups must be exactly [{TOPLINE_LABEL!r}]")
            self._lookup: Codes = np.zeros(1, dtype=np.int64)
            return
        vocab = schema.vocabulary(column)
        missing = [label for label in self.group_labels if label not in vocab]
        if missing:
            raise SchemaError(f"group labels not in the {column!r} vocabulary: {missing[:10]}")
        self._lookup = np.full(len(vocab), UNATTRIBUTED, dtype=np.int64)
        self._lookup[vocab.encode(self.group_labels)] = np.arange(len(self.group_labels))

    @property
    def n_groups(self) -> int:
        return len(self.group_labels)

    def groups_of_codes(self, codes: NDArray[np.int64]) -> Codes:
        """Group index per row, from the rows' codes in the grouping column's vocabulary."""
        if self.column == TOPLINE:
            return np.zeros(len(codes), dtype=np.int64)
        out = np.full(codes.shape, UNATTRIBUTED, dtype=np.int64)
        valid = codes >= 0
        out[valid] = self._lookup[codes[valid]]
        return out

    def groups(self, buffer: Buffer, layout: BufferLayout) -> Codes:
        """Group index per buffer row. Rollups such as `org_level1` roll the cost center up."""
        if self.column == TOPLINE:
            return np.zeros(len(buffer), dtype=np.int64)
        if self.column in self.schema.rollups:
            leaf = buffer[:, layout.column(COST_CENTER)].astype(np.int64)
            return self.groups_of_codes(self.schema.rollup(self.column, leaf))
        return self.groups_of_codes(buffer[:, layout.column(self.column)].astype(np.int64))

    def counts(
        self, groups: Codes, weights: NDArray[np.float64] | None = None
    ) -> NDArray[np.int64]:
        """Rows (or summed integer weights) per group, ignoring UNATTRIBUTED rows."""
        valid = groups >= 0
        w = None if weights is None else weights[valid]
        return np.bincount(groups[valid], weights=w, minlength=self.n_groups).astype(np.int64)
