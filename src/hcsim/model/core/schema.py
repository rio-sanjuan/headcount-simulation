"""The simulation schema: the single authority for encoding categorical values.

A vocabulary is a sorted set of labels, and a label's code is its zero-based position in it.
A code means nothing without the vocabulary that assigned it, so training frames, buffers, and
artifacts all encode through one schema. Levels sort numerically, everything else lexically.
The value `UNATTRIBUTED` (-1) marks a row whose label a vocabulary doesn't carry. Samplers never
select a row in group -1.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Final, Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray

UNATTRIBUTED: Final = -1

# Vocabulary names, which are also the column names the data contract uses.
COST_CENTER: Final = "cost_center"
JOB_FAMILY: Final = "job_family"
JOB_LEVEL: Final = "job_level"
LOCATION: Final = "location"
COMPANY: Final = "company"
TECH_INDICATOR: Final = "tech_indicator"
REQ_TYPE: Final = "req_type"
AWARD_TYPE: Final = "award_type"

NUMERIC_VOCABULARIES: Final = frozenset({JOB_LEVEL})

Codes = NDArray[np.int64]


class SchemaError(ValueError):
    """A label, code, or vocabulary the schema can't resolve."""


def _numeric_key(label: str) -> float:
    try:
        return float(label)
    except ValueError:
        raise SchemaError(f"numeric vocabulary got a non-numeric label: {label!r}") from None


@dataclass(frozen=True)
class Vocabulary:
    """An ordered set of labels. A label's code is its position."""

    name: str
    labels: tuple[str, ...]
    _index: Mapping[str, int] = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        if len(set(self.labels)) != len(self.labels):
            raise SchemaError(f"vocabulary {self.name!r} has duplicate labels")
        object.__setattr__(
            self, "_index", MappingProxyType({label: i for i, label in enumerate(self.labels)})
        )

    @classmethod
    def build(
        cls, name: str, labels: Iterable[object], *, numeric: bool | None = None
    ) -> Vocabulary:
        """Sort and deduplicate `labels`. Numeric vocabularies sort by value, others lexically."""
        unique = {str(label) for label in labels}
        is_numeric = name in NUMERIC_VOCABULARIES if numeric is None else numeric
        ordered = sorted(unique, key=_numeric_key) if is_numeric else sorted(unique)
        return cls(name, tuple(ordered))

    def __len__(self) -> int:
        return len(self.labels)

    def __contains__(self, label: object) -> bool:
        return str(label) in self._index

    def code(self, label: object) -> int:
        try:
            return self._index[str(label)]
        except KeyError:
            raise SchemaError(f"{self.name!r} has no label {label!r}") from None

    def label(self, code: int) -> str:
        if not 0 <= code < len(self.labels):
            raise SchemaError(f"{self.name!r} has no code {code}")
        return self.labels[code]

    def encode(
        self, labels: Iterable[object], *, unknown: Literal["raise", "unattributed"] = "raise"
    ) -> Codes:
        """Encode labels to codes. Unknown labels raise, or become UNATTRIBUTED."""
        if unknown == "raise":
            return np.fromiter((self.code(x) for x in labels), dtype=np.int64)
        index = self._index
        return np.fromiter((index.get(str(x), UNATTRIBUTED) for x in labels), dtype=np.int64)

    def decode(self, codes: ArrayLike) -> NDArray[np.object_]:
        """Decode codes to labels. UNATTRIBUTED and out-of-range codes raise."""
        array = np.asarray(codes, dtype=np.int64)
        if array.size and (array.min() < 0 or array.max() >= len(self.labels)):
            raise SchemaError(f"{self.name!r} can't decode codes outside 0..{len(self) - 1}")
        return np.asarray(self.labels, dtype=object)[array]


@dataclass(frozen=True)
class SimulationSchema:
    """The vocabularies a run encodes through, and the org rollups of cost centers.

    `rollups` maps a rollup name, such as `org_level1`, to an array that gives each leaf cost
    center's code in that rollup's own vocabulary, so rolling a row up is one array lookup.
    """

    vocabularies: Mapping[str, Vocabulary]
    rollups: Mapping[str, Codes] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "vocabularies", MappingProxyType(dict(self.vocabularies)))
        frozen = {}
        for name, mapping in self.rollups.items():
            if name not in self.vocabularies:
                raise SchemaError(f"rollup {name!r} has no vocabulary")
            array = np.asarray(mapping, dtype=np.int64).copy()
            array.setflags(write=False)
            if len(array) != len(self.vocabulary(COST_CENTER)):
                raise SchemaError(f"rollup {name!r} must map every cost center")
            frozen[name] = array
        object.__setattr__(self, "rollups", MappingProxyType(frozen))

    @classmethod
    def build(
        cls,
        labels: Mapping[str, Iterable[object]],
        *,
        org_levels: Mapping[str, Mapping[str, str]] | None = None,
    ) -> SimulationSchema:
        """Build vocabularies from labels, and rollups from {level: {cost_center: node}}."""
        vocabularies = {name: Vocabulary.build(name, values) for name, values in labels.items()}
        if COST_CENTER not in vocabularies:
            raise SchemaError("a schema needs a cost_center vocabulary")
        rollups: dict[str, Codes] = {}
        for level, parent_of in (org_levels or {}).items():
            missing = [cc for cc in vocabularies[COST_CENTER].labels if cc not in parent_of]
            if missing:
                raise SchemaError(f"rollup {level!r} doesn't place cost centers {missing[:5]}")
            vocab = Vocabulary.build(level, parent_of.values())
            vocabularies[level] = vocab
            rollups[level] = vocab.encode(parent_of[cc] for cc in vocabularies[COST_CENTER].labels)
        return cls(vocabularies, rollups)

    def vocabulary(self, name: str) -> Vocabulary:
        try:
            return self.vocabularies[name]
        except KeyError:
            known = ", ".join(sorted(self.vocabularies))
            raise SchemaError(f"no vocabulary {name!r}; known: {known}") from None

    def encode(
        self,
        name: str,
        labels: Iterable[object],
        *,
        unknown: Literal["raise", "unattributed"] = "raise",
    ) -> Codes:
        return self.vocabulary(name).encode(labels, unknown=unknown)

    def decode(self, name: str, codes: ArrayLike) -> NDArray[np.object_]:
        return self.vocabulary(name).decode(codes)

    def rollup(self, name: str, cost_center_codes: ArrayLike) -> Codes:
        """Map leaf cost-center codes to codes at rollup `name`. UNATTRIBUTED stays -1."""
        if name not in self.rollups:
            raise SchemaError(f"no rollup {name!r}")
        codes = np.asarray(cost_center_codes, dtype=np.int64)
        out = np.full(codes.shape, UNATTRIBUTED, dtype=np.int64)
        valid = codes >= 0
        out[valid] = self.rollups[name][codes[valid]]
        return out

    def to_dict(self) -> dict[str, Any]:
        return {
            "vocabularies": {name: list(v.labels) for name, v in self.vocabularies.items()},
            "rollups": {name: array.tolist() for name, array in self.rollups.items()},
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> SimulationSchema:
        vocabularies = {
            name: Vocabulary(name, tuple(labels)) for name, labels in data["vocabularies"].items()
        }
        return cls(vocabularies, {k: np.asarray(v) for k, v in data.get("rollups", {}).items()})

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), sort_keys=True)

    @classmethod
    def from_json(cls, text: str) -> SimulationSchema:
        return cls.from_dict(json.loads(text))

    def fingerprint(self) -> str:
        """`sc1-` and 16 hex characters of a SHA-256 over the canonical JSON."""
        return "sc1-" + hashlib.sha256(self.to_json().encode()).hexdigest()[:16]

    def __eq__(self, other: object) -> bool:
        return isinstance(other, SimulationSchema) and self.to_dict() == other.to_dict()

    def __hash__(self) -> int:
        return hash(self.to_json())


def check_codes(name: str, codes: Sequence[int] | Codes, size: int) -> None:
    """Raise unless every code is UNATTRIBUTED or lies in 0..size-1."""
    array = np.asarray(codes, dtype=np.int64)
    if array.size and (array.min() < UNATTRIBUTED or array.max() >= size):
        raise SchemaError(f"{name} codes must lie in -1..{size - 1}")
