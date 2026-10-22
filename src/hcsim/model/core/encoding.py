"""Frame encoding: label columns to codes, through the schema, with coverage reported.

Every training frame and every set of known movements encodes through the run's schema. A label
the schema doesn't carry becomes UNATTRIBUTED (-1) rather than an error, because a frame can
legitimately reach past the schema, such as a cost center that closed before the run. Every gap
is counted per column and raised as one `CoverageWarning`, so a gap is never silent.

A column finds its vocabulary by name: the vocabulary itself, its `_previous` twin (where a
row's attribute was before its event), or a `to_` or `from_` prefix on a movement's destination
or origin. A movement's destination encodes through the same vocabulary as its origin, so the
two compare code for code.
"""

from __future__ import annotations

import warnings
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

from hcsim.model.core.schema import UNATTRIBUTED, Codes, SchemaError, SimulationSchema

#: Distinct unknown labels a coverage record keeps, for a readable warning.
EXAMPLES = 5


class CoverageWarning(UserWarning):
    """Rows whose labels the schema doesn't carry, encoded as UNATTRIBUTED."""


def vocabulary_for(schema: SimulationSchema, column: str) -> str:
    """The vocabulary a column encodes through."""
    candidates = [column, column.removesuffix("_previous")]
    candidates += [column.removeprefix(prefix) for prefix in ("to_", "from_")]
    for name in candidates:
        if name in schema.vocabularies:
            return name
    known = ", ".join(sorted(schema.vocabularies))
    raise SchemaError(f"no vocabulary for column {column!r}; known: {known}")


@dataclass(frozen=True)
class Coverage:
    """How much of one column the schema carries."""

    column: str
    vocabulary: str
    rows: int
    unattributed: int
    #: A few of the labels the schema lacks, sorted.
    examples: tuple[str, ...]

    @property
    def share(self) -> float:
        """The share of rows encoded, 1.0 for an empty column."""
        return 1.0 - self.unattributed / self.rows if self.rows else 1.0


@dataclass(frozen=True)
class EncodedFrame:
    codes: Mapping[str, Codes]
    coverage: tuple[Coverage, ...]

    def array(self, columns: Sequence[str] | None = None) -> NDArray[np.int64]:
        """The codes as an (N, k) array, in `columns` order."""
        names = list(self.codes) if columns is None else list(columns)
        if not names:
            return np.zeros((0, 0), dtype=np.int64)
        return np.stack([self.codes[name] for name in names], axis=1)

    @property
    def complete(self) -> NDArray[np.bool_]:
        """Rows with every column encoded."""
        if not self.codes:
            return np.zeros(0, dtype=bool)
        return np.all(self.array() != UNATTRIBUTED, axis=1)

    @property
    def gaps(self) -> tuple[Coverage, ...]:
        return tuple(c for c in self.coverage if c.unattributed)


def encode_frame(
    schema: SimulationSchema, frame: Any, columns: Sequence[str], *, warn: bool = True
) -> EncodedFrame:
    """Encode `columns` of `frame` (anything indexable by column name, such as a DataFrame).
    Unknown labels become UNATTRIBUTED, and any gap raises one `CoverageWarning`."""
    if len(set(columns)) != len(columns):
        raise SchemaError(f"columns repeat: {list(columns)}")
    missing = [c for c in columns if c not in frame.columns]
    if missing:
        raise SchemaError(f"the frame has no columns {missing}")
    codes: dict[str, Codes] = {}
    coverage = []
    for column in columns:
        name = vocabulary_for(schema, column)
        values = list(frame[column])
        encoded = schema.encode(name, values, unknown="unattributed")
        lost = encoded == UNATTRIBUTED
        unknown = sorted({str(v) for v, gone in zip(values, lost, strict=True) if gone})
        codes[column] = encoded
        coverage.append(
            Coverage(column, name, len(values), int(lost.sum()), tuple(unknown[:EXAMPLES]))
        )
    result = EncodedFrame(codes, tuple(coverage))
    if warn and result.gaps:
        detail = "; ".join(
            f"{c.column}: {c.unattributed} of {c.rows} rows, such as {list(c.examples)}"
            for c in result.gaps
        )
        warnings.warn(
            f"labels the schema lacks, encoded as -1: {detail}", CoverageWarning, stacklevel=2
        )
    return result


@dataclass(frozen=True)
class MovementEncoding:
    """Each movement's origin and destination codes, per attribute."""

    origin: Mapping[str, Codes]
    destination: Mapping[str, Codes]
    coverage: tuple[Coverage, ...]

    @property
    def placeable(self) -> NDArray[np.bool_]:
        """Movements whose every destination attribute the schema carries. Only these can be
        applied: a destination of -1 would place someone nowhere."""
        columns = list(self.destination.values())
        if not columns:
            return np.zeros(0, dtype=bool)
        return np.all(np.stack(columns, axis=1) != UNATTRIBUTED, axis=1)

    def changes(self, attribute: str) -> NDArray[np.bool_]:
        """Movements that change `attribute`, among those whose both ends are known."""
        origin, destination = self.origin[attribute], self.destination[attribute]
        known = (origin != UNATTRIBUTED) & (destination != UNATTRIBUTED)
        moved: NDArray[np.bool_] = known & (origin != destination)
        return moved


def encode_movements(
    schema: SimulationSchema,
    frame: Any,
    attributes: Sequence[str],
    *,
    origin: str = "{}_previous",
    destination: str = "{}",
    warn: bool = True,
) -> MovementEncoding:
    """Encode movements whose attributes sit in paired columns. By default the destination is
    the attribute's own column and the origin its `_previous` twin, as in the transactions;
    `origin="from_{}"` and `destination="to_{}"` read a known-movement frame."""
    pairs = [(origin.format(a), destination.format(a)) for a in attributes]
    columns = [c for pair in pairs for c in pair]
    encoded = encode_frame(schema, frame, columns, warn=warn)
    for attribute, (start, _) in zip(attributes, pairs, strict=True):
        if vocabulary_for(schema, start) != attribute:
            raise SchemaError(f"column {start!r} doesn't encode through {attribute!r}")
    return MovementEncoding(
        {a: encoded.codes[start] for a, (start, _) in zip(attributes, pairs, strict=True)},
        {a: encoded.codes[end] for a, (_, end) in zip(attributes, pairs, strict=True)},
        encoded.coverage,
    )
