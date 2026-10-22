"""The group aligner: maps buffer rows to the groups of a guidance artifact.

A guidance artifact groups by plain labels and never sees the simulation schema. Building an
aligner validates every group label against the schema, so a label the schema can't resolve
fails at load rather than reading as a silent zero. The aligner then runs every step, because
group membership changes as employees move. A row whose code the artifact doesn't carry gets
group UNATTRIBUTED (-1): a coverage gap, not an error.

A grouping can span several columns, such as `org_level1` and `job_family`. Each row's codes
combine into one mixed-radix key, the first column most significant, and groups are found by
searching the sorted keys of the artifact's groups, so the key space never needs a table of its
own. A rollup, such as an org level, rolls the row's cost center up first. The same aligner
reads both buffers, each through its own layout.
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
    """Maps rows to group indices in an artifact's group order."""

    def __init__(
        self,
        schema: SimulationSchema,
        columns: str | Sequence[str],
        group_labels: Sequence[str | Sequence[str]],
    ) -> None:
        names = (columns,) if isinstance(columns, str) else tuple(columns)
        if not names:
            raise SchemaError("a grouping needs at least one column")
        if TOPLINE in names and len(names) > 1:
            raise SchemaError(f"{TOPLINE} can't combine with other columns")
        if len(set(names)) != len(names):
            raise SchemaError(f"grouping columns repeat: {list(names)}")
        labels = [(g,) if isinstance(g, str) else tuple(str(x) for x in g) for g in group_labels]
        wrong = [g for g in labels if len(g) != len(names)]
        if wrong:
            raise SchemaError(f"group labels need {len(names)} parts, one per column: {wrong[:5]}")
        if len(set(labels)) != len(labels):
            raise SchemaError(f"duplicate group labels for {list(names)}")
        self.schema = schema
        self.columns = names
        self.group_labels: tuple[str | tuple[str, ...], ...] = tuple(
            g[0] if len(names) == 1 else g for g in labels
        )
        if names == (TOPLINE,):
            if labels != [(TOPLINE_LABEL,)]:
                raise SchemaError(f"{TOPLINE} groups must be exactly [{TOPLINE_LABEL!r}]")
            self._keys: Codes = np.zeros(1, dtype=np.int64)
            self._order: Codes = np.zeros(1, dtype=np.int64)
            self._strides: Codes = np.zeros(1, dtype=np.int64)
            return
        vocabs = [schema.vocabulary(name) for name in names]
        problems = []
        for k, (name, vocab) in enumerate(zip(names, vocabs, strict=True)):
            missing = sorted({g[k] for g in labels if g[k] not in vocab})
            if missing:
                problems.append(f"group labels not in the {name!r} vocabulary: {missing[:10]}")
        if problems:
            raise SchemaError("; ".join(problems))
        radices = [len(v) for v in vocabs]
        if np.prod([float(r) for r in radices]) >= 2.0**62:
            raise SchemaError(f"the grouping {list(names)} has too many combinations to key")
        strides = np.ones(len(names), dtype=np.int64)
        for k in range(len(names) - 2, -1, -1):
            strides[k] = strides[k + 1] * radices[k + 1]
        codes = np.array(
            [[v.code(part) for v, part in zip(vocabs, g, strict=True)] for g in labels],
            dtype=np.int64,
        ).reshape(len(labels), len(names))
        keys = codes @ strides
        self._order = np.argsort(keys, kind="stable")
        self._keys = keys[self._order]
        self._strides = strides

    @property
    def column(self) -> str:
        """The grouping's name: its column, or its columns joined with `+`."""
        return "+".join(self.columns)

    @property
    def n_groups(self) -> int:
        return len(self.group_labels)

    def groups_of_codes(self, codes: NDArray[np.int64]) -> Codes:
        """Group index per row, from the rows' codes: one column of codes per grouping column,
        or a flat array for a single column."""
        array = np.asarray(codes, dtype=np.int64)
        if self.columns == (TOPLINE,):
            return np.zeros(len(array), dtype=np.int64)
        if array.ndim == 1:
            array = array[:, None]
        if array.shape[1] != len(self.columns):
            raise SchemaError(f"codes need {len(self.columns)} columns, not {array.shape[1]}")
        out = np.full(len(array), UNATTRIBUTED, dtype=np.int64)
        valid = np.all(array >= 0, axis=1)
        key = array[valid] @ self._strides
        position = np.searchsorted(self._keys, key)
        found = position < len(self._keys)
        found[found] = self._keys[position[found]] == key[found]
        rows = np.flatnonzero(valid)
        out[rows[found]] = self._order[position[found]]
        return out

    def codes(self, buffer: Buffer, layout: BufferLayout) -> NDArray[np.int64]:
        """Each row's code in every grouping column. Rollups roll the cost center up."""
        columns = []
        for name in self.columns:
            if name in self.schema.rollups:
                leaf = buffer[:, layout.column(COST_CENTER)].astype(np.int64)
                columns.append(self.schema.rollup(name, leaf))
            else:
                columns.append(buffer[:, layout.column(name)].astype(np.int64))
        return np.stack(columns, axis=1) if columns else np.zeros((len(buffer), 0), np.int64)

    def groups(self, buffer: Buffer, layout: BufferLayout) -> Codes:
        """Group index per buffer row."""
        if self.columns == (TOPLINE,):
            return np.zeros(len(buffer), dtype=np.int64)
        return self.groups_of_codes(self.codes(buffer, layout))

    def counts(
        self, groups: Codes, weights: NDArray[np.float64] | None = None
    ) -> NDArray[np.int64]:
        """Rows (or summed integer weights) per group, ignoring UNATTRIBUTED rows."""
        valid = groups >= 0
        w = None if weights is None else weights[valid]
        return np.bincount(groups[valid], weights=w, minlength=self.n_groups).astype(np.int64)
