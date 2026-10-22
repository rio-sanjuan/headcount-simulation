"""The simulation schema: the single authority for encoding categorical values.

A vocabulary is a sorted set of labels, and a label's code is its zero-based position in it.
A code means nothing without the vocabulary that assigned it, so training frames, buffers, and
artifacts all encode through one schema. Levels sort numerically, everything else lexically,
and a numeric label has one spelling: `3`, `3.0`, and `"3"` are the same level. The value
`UNATTRIBUTED` (-1) marks a row whose label a vocabulary doesn't carry. Samplers never select a
row in group -1.

Each org level is a rollup: an array giving every leaf cost center's code at that level, so
rolling rows up is one lookup. The levels must nest, each node with one parent.

A schema persists as canonical JSON, with a format version and a fingerprint, and loading
refuses a file whose vocabularies aren't in canonical order or whose fingerprint doesn't match.
"""

from __future__ import annotations

import hashlib
import itertools
import json
import math
import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
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
#: Org levels are named `org_level1` (the top) down to the deepest.
ORG_LEVEL: Final = re.compile(r"org_level(\d+)")
SCHEMA_FILE: Final = "schema.json"
FORMAT_VERSION: Final = 2

Codes = NDArray[np.int64]


class SchemaError(ValueError):
    """A label, code, or vocabulary the schema can't resolve."""


def _numeric_key(label: str) -> float:
    try:
        value = float(label)
    except ValueError:
        raise SchemaError(f"numeric vocabulary got a non-numeric label: {label!r}") from None
    if not math.isfinite(value):
        raise SchemaError(f"numeric vocabulary got a non-finite label: {label!r}")
    return value


def canonical_numeric(label: object) -> str:
    """A numeric label's one spelling: an integral value without a fraction, `3` not `3.0`."""
    value = _numeric_key(str(label))
    return str(int(value)) if value.is_integer() else repr(value)


def _canonical_order(labels: Iterable[str], numeric: bool) -> list[str]:
    return sorted(labels, key=_numeric_key) if numeric else sorted(labels)


@dataclass(frozen=True)
class Vocabulary:
    """An ordered set of labels. A label's code is its position. A numeric vocabulary reads
    any spelling of a number as its canonical label."""

    name: str
    labels: tuple[str, ...]
    numeric: bool = False
    _index: Mapping[str, int] = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        if len(set(self.labels)) != len(self.labels):
            raise SchemaError(f"vocabulary {self.name!r} has duplicate labels")
        if self.numeric:
            spelled = [label for label in self.labels if canonical_numeric(label) != label]
            if spelled:
                raise SchemaError(f"vocabulary {self.name!r} has non-canonical labels {spelled}")
        object.__setattr__(
            self, "_index", MappingProxyType({label: i for i, label in enumerate(self.labels)})
        )

    @classmethod
    def build(
        cls, name: str, labels: Iterable[object], *, numeric: bool | None = None
    ) -> Vocabulary:
        """Sort and deduplicate `labels`. Numeric vocabularies sort by value, others lexically."""
        is_numeric = name in NUMERIC_VOCABULARIES if numeric is None else numeric
        unique = {canonical_numeric(x) if is_numeric else str(x) for x in labels}
        return cls(name, tuple(_canonical_order(unique, is_numeric)), is_numeric)

    @property
    def is_canonical(self) -> bool:
        """Whether the labels are in the order `build` gives them."""
        return list(self.labels) == _canonical_order(self.labels, self.numeric)

    def _key(self, label: object) -> str:
        if not self.numeric:
            return str(label)
        try:
            return canonical_numeric(label)
        except SchemaError:
            return str(label)

    def __len__(self) -> int:
        return len(self.labels)

    def __contains__(self, label: object) -> bool:
        return self._key(label) in self._index

    def code(self, label: object) -> int:
        try:
            return self._index[self._key(label)]
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
        index, key = self._index, self._key
        return np.fromiter((index.get(key(x), UNATTRIBUTED) for x in labels), dtype=np.int64)

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
        problems = self._nesting_problems()
        if problems:
            raise SchemaError("org levels don't nest: " + "; ".join(problems))

    def _nesting_problems(self) -> list[str]:
        """Every org-level node with more than one parent at the level above."""
        levels = sorted(
            (int(m.group(1)), name)
            for name in self.rollups
            if (m := ORG_LEVEL.fullmatch(name)) is not None
        )
        problems = []
        for (_, upper), (_, lower) in itertools.pairwise(levels):
            pairs = np.unique(np.stack([self.rollups[lower], self.rollups[upper]], axis=1), axis=0)
            children, counts = np.unique(pairs[:, 0], return_counts=True)
            for child in children[counts > 1]:
                parents = self.decode(upper, pairs[pairs[:, 0] == child, 1]).tolist()
                label = self.vocabulary(lower).label(int(child))
                problems.append(f"{lower} {label!r} has parents {parents} in {upper}")
        return problems

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

    @property
    def org_levels(self) -> list[str]:
        """The org-level rollups, top level first."""
        found = [(int(m.group(1)), n) for n in self.rollups if (m := ORG_LEVEL.fullmatch(n))]
        return [name for _, name in sorted(found)]

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
        """The canonical form: every vocabulary's labels in code order, which vocabularies are
        numeric, and every rollup array."""
        return {
            "vocabularies": {name: list(v.labels) for name, v in self.vocabularies.items()},
            "numeric": sorted(name for name, v in self.vocabularies.items() if v.numeric),
            "rollups": {name: array.tolist() for name, array in self.rollups.items()},
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> SimulationSchema:
        """Rebuild a schema, refusing vocabularies out of canonical order: their codes would
        mean something other than positions in the sorted labels."""
        numeric = set(data.get("numeric", NUMERIC_VOCABULARIES))
        vocabularies = {
            name: Vocabulary(name, tuple(str(x) for x in labels), name in numeric)
            for name, labels in data["vocabularies"].items()
        }
        unsorted = sorted(name for name, v in vocabularies.items() if not v.is_canonical)
        if unsorted:
            raise SchemaError(f"vocabularies not in canonical order: {unsorted}")
        return cls(vocabularies, {k: np.asarray(v) for k, v in data.get("rollups", {}).items()})

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), sort_keys=True)

    @classmethod
    def from_json(cls, text: str) -> SimulationSchema:
        return cls.from_dict(json.loads(text))

    def fingerprint(self) -> str:
        """`sc2-` and 16 hex characters of a SHA-256 over the canonical JSON."""
        return "sc2-" + hashlib.sha256(self.to_json().encode()).hexdigest()[:16]

    def save(self, path: Path) -> None:
        """Write the canonical JSON, with its format version and fingerprint."""
        path.parent.mkdir(parents=True, exist_ok=True)
        document = self.to_dict() | {
            "format_version": FORMAT_VERSION,
            "fingerprint": self.fingerprint(),
        }
        path.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n")

    @classmethod
    def load(cls, path: Path) -> SimulationSchema:
        """Read a saved schema, refusing another format or a fingerprint that doesn't match."""
        if not path.is_file():
            raise SchemaError(f"no schema at {path}")
        document = json.loads(path.read_text())
        version = document.get("format_version")
        if version != FORMAT_VERSION:
            raise SchemaError(f"schema format {version!r} isn't {FORMAT_VERSION}; rebuild it")
        schema = cls.from_dict(document)
        if schema.fingerprint() != document.get("fingerprint"):
            raise SchemaError(f"{path} doesn't match its recorded fingerprint")
        return schema

    def __eq__(self, other: object) -> bool:
        return isinstance(other, SimulationSchema) and self.to_dict() == other.to_dict()

    def __hash__(self) -> int:
        return hash(self.to_json())


def org_levels_from_frame(frame: Any, cost_center: str = COST_CENTER) -> dict[str, dict[str, str]]:
    """`{level: {cost_center: node}}` for every `org_level<k>` column of an org dimension,
    top level first. `frame` is anything with columns, such as a DataFrame."""
    columns = [str(c) for c in frame.columns if ORG_LEVEL.fullmatch(str(c))]
    if not columns:
        raise SchemaError("the org dimension has no org_level columns")
    ordered = sorted(columns, key=lambda c: int(c.removeprefix("org_level")))
    leaves = [str(x) for x in frame[cost_center]]
    return {
        level: dict(zip(leaves, (str(x) for x in frame[level]), strict=True)) for level in ordered
    }


def check_codes(name: str, codes: Sequence[int] | Codes, size: int) -> None:
    """Raise unless every code is UNATTRIBUTED or lies in 0..size-1."""
    array = np.asarray(codes, dtype=np.int64)
    if array.size and (array.min() < UNATTRIBUTED or array.max() >= size):
        raise SchemaError(f"{name} codes must lie in -1..{size - 1}")
