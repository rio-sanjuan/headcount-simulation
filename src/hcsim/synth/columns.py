"""Columnar building blocks for the world: growable tables and an append-only event log.

The world holds every person and requisition as one row in a set of NumPy columns, coded as
integers, and records events as column chunks. Labels are attached once, when the tables are
emitted, so the monthly loop works on arrays of any size.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import cast

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike, DTypeLike, NDArray

Index = NDArray[np.int64]


def _empty_value(dtype: np.dtype) -> object:
    if np.issubdtype(dtype, np.datetime64):
        return np.array("NaT", dtype=dtype)[()]  # NaT in the column's own unit
    return 0


class Table:
    """Columns that share one length and grow by doubling.

    `column` returns a view of the live rows. Writes through it land in the table, but a view
    taken before an append may point at storage the append replaced, so take views afresh.
    """

    def __init__(self, dtypes: Mapping[str, DTypeLike], capacity: int = 1024) -> None:
        self._dtypes = {name: np.dtype(d) for name, d in dtypes.items()}
        self._data = {
            name: np.full(capacity, _empty_value(d), dtype=d) for name, d in self._dtypes.items()
        }
        self._n = 0

    def __len__(self) -> int:
        return self._n

    def column(self, name: str) -> np.ndarray:
        return self._data[name][: self._n]

    def append(self, size: int, **values: ArrayLike) -> Index:
        """Add `size` rows and return their indices. Unnamed columns start empty: NaT for
        dates and 0 otherwise. Scalars broadcast."""
        unknown = set(values) - set(self._dtypes)
        if unknown:
            raise KeyError(f"no columns {sorted(unknown)}")
        start, end = self._n, self._n + size
        capacity = len(next(iter(self._data.values())))
        if end > capacity:
            grown = max(end, 2 * capacity)
            for name, d in self._dtypes.items():
                column = np.full(grown, _empty_value(d), dtype=d)
                column[:start] = self._data[name][:start]
                self._data[name] = column
        for name, value in values.items():
            self._data[name][start:end] = value
        self._n = end
        return np.arange(start, end, dtype=np.int64)


class EventLog:
    """Rows recorded as column chunks and concatenated once."""

    def __init__(self, columns: Sequence[str]) -> None:
        self.columns = tuple(columns)
        self._chunks: list[dict[str, np.ndarray]] = []

    def add(self, size: int, **values: ArrayLike) -> None:
        if set(values) != set(self.columns):
            missing = sorted(set(self.columns) - set(values))
            extra = sorted(set(values) - set(self.columns))
            raise KeyError(f"event columns differ: missing {missing}, unknown {extra}")
        if size:
            self._chunks.append(
                {name: np.broadcast_to(np.asarray(v), (size,)) for name, v in values.items()}
            )

    def __len__(self) -> int:
        return sum(len(chunk[self.columns[0]]) for chunk in self._chunks)

    def collect(
        self, dtypes: Mapping[str, DTypeLike], release: bool = False
    ) -> dict[str, np.ndarray]:
        """Every column, concatenated, in the given dtypes. With `release`, the log empties
        column by column as it goes, so the chunks and the columns are never held twice."""
        result: dict[str, np.ndarray] = {}
        for name in self.columns:
            parts = [chunk.pop(name) if release else chunk[name] for chunk in self._chunks]
            result[name] = (
                np.concatenate(parts).astype(dtypes[name], copy=False)
                if parts
                else np.zeros(0, dtype=dtypes[name])
            )
            del parts
        if release:
            self._chunks = []
        return result


def categorical(codes: NDArray[np.integer], labels: Sequence[str]) -> pd.Categorical:
    """Labels from integer codes, with -1 for missing. Building from codes never makes a string
    per row, which at scale would dwarf the table."""
    # The stubs want a Sequence; an integer array is the fast path pandas takes anyway.
    return pd.Categorical.from_codes(
        cast("Sequence[int]", np.asarray(codes)), categories=pd.Index(list(labels))
    )
