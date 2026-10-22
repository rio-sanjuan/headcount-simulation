"""Populations: the named set of employees and cost centers a run, a model, and an artifact
must agree on.

A `PopulationSpec` is a set of membership rules, each a catalog column and the values it
admits: `account_group in (regular)` is regular headcount. A row is in the population when
every rule admits it. A rule may only use a column the column catalog lets define a population.

A spec's identity is its rules. Its digest is a hash of their canonical form, so the same
rules written in another order are the same population. A spec without a name is named after
its digest. The registry holds the built-in synthetic populations.

Whatever is built for a population carries its stamp, the name and digest. Before a run uses
an artifact, a guard compares stamps, and a model or forecast built for another population is
refused: its counts and rates describe different people.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Final

from hcsim.model.sql.catalog import CatalogError, Use, require


class PopulationError(ValueError):
    """A population that can't be declared, or one that doesn't match."""


@dataclass(frozen=True)
class PopulationStamp:
    """What an artifact records of the population it was built for."""

    name: str
    digest: str

    def __str__(self) -> str:
        return f"{self.name}@{self.digest}"

    @classmethod
    def parse(cls, text: str) -> PopulationStamp:
        name, sep, digest = text.partition("@")
        if not sep or not name or not digest:
            raise PopulationError(f"a population stamp reads name@digest, not {text!r}")
        return cls(name, digest)


@dataclass(frozen=True)
class PopulationSpec:
    rules: Mapping[str, frozenset[str]]
    name: str = ""
    description: str = ""
    _canonical: str = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        problems = []
        frozen: dict[str, frozenset[str]] = {}
        for column, values in self.rules.items():
            try:
                require(column, Use.POPULATION)
            except CatalogError as error:
                problems.append(str(error))
                continue
            admitted = frozenset(str(v) for v in values)
            if not admitted:
                problems.append(f"the rule on {column!r} admits nothing")
            frozen[column] = admitted
        if problems:
            raise PopulationError("bad population: " + "; ".join(problems))
        rules = MappingProxyType(dict(sorted(frozen.items())))
        canonical = json.dumps({c: sorted(v) for c, v in rules.items()}, sort_keys=True)
        object.__setattr__(self, "rules", rules)
        object.__setattr__(self, "_canonical", canonical)
        if not self.name:
            object.__setattr__(self, "name", f"pop-{self.digest[:12]}")

    @classmethod
    def of(cls, name: str = "", description: str = "", **rules: Iterable[str]) -> PopulationSpec:
        return cls(
            {column: frozenset(values) for column, values in rules.items()}, name, description
        )

    @property
    def digest(self) -> str:
        """`pd1-` and 16 hex characters of a SHA-256 over the canonical rules."""
        return "pd1-" + hashlib.sha256(self._canonical.encode()).hexdigest()[:16]

    @property
    def stamp(self) -> PopulationStamp:
        return PopulationStamp(self.name, self.digest)

    def admits(self, row: Mapping[str, object]) -> bool:
        """Whether a row, given as {column: value}, is in the population. A row missing a
        rule's column is refused, not admitted."""
        missing = sorted(c for c in self.rules if c not in row)
        if missing:
            raise PopulationError(f"the row has no {missing} to test")
        return all(str(row[c]) in values for c, values in self.rules.items())

    def to_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "digest": self.digest,
            "rules": {c: sorted(v) for c, v in self.rules.items()},
        }


#: The built-in populations of the synthetic world.
REGULAR: Final = PopulationSpec.of(
    "regular", "regular employees: the headcount every measure counts", account_group=["regular"]
)
REGULAR_AND_CONTINGENT: Final = PopulationSpec.of(
    "regular_and_contingent",
    "regular employees and the contingent pool",
    account_group=["regular", "contingent"],
)
EVERY_ACCOUNT: Final = PopulationSpec.of(
    "every_account",
    "every account group",
    account_group=["regular", "contingent", "temporary"],
)

REGISTRY: Final[Mapping[str, PopulationSpec]] = MappingProxyType(
    {spec.name: spec for spec in (REGULAR, REGULAR_AND_CONTINGENT, EVERY_ACCOUNT)}
)
DEFAULT: Final = REGULAR


def population(name: str) -> PopulationSpec:
    """A built-in population by name."""
    try:
        return REGISTRY[name]
    except KeyError:
        known = ", ".join(sorted(REGISTRY))
        raise PopulationError(f"no built-in population {name!r}; known: {known}") from None


def guard(expected: PopulationSpec, recorded: PopulationStamp | str | None, what: str) -> None:
    """Refuse an artifact built for another population. An artifact that records none was
    built before stamps, and is taken to be for the default population."""
    stamp = (
        DEFAULT.stamp
        if recorded is None
        else recorded
        if isinstance(recorded, PopulationStamp)
        else PopulationStamp.parse(recorded)
    )
    if stamp.digest != expected.digest:
        raise PopulationError(
            f"the {what} was built for population {stamp}, but this run is for {expected.stamp}"
        )
