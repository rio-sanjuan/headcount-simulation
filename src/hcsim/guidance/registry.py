"""Registries: the named components a guidance configuration can choose from.

Each registry maps names to factories, for one kind of component: forecasting models,
combination methods, or reconciliation methods. A configuration names its components, and the
registries check every name when the configuration loads, not when a step first needs it:

- a name no registry entry carries is refused, with the closest names it does carry, and
  either every name or, for a large namespaced registry, its namespaces or the names in the
  namespace asked for;
- a name whose entry comes from an optional extra that isn't installed is refused, with the
  command that installs it.
"""

from __future__ import annotations

import difflib
import importlib.util
import re
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import Final

#: A namespaced entry's name: a lowercase namespace, a dot, and the model's own name.
_NAMESPACED: Final = re.compile(r"[a-z][a-z0-9_]*\.[A-Za-z][A-Za-z0-9_]*")
#: The most entries an unknown name's message lists; a bigger registry names its namespaces.
_LISTED: Final = 12


class RegistryError(ValueError):
    """A name a registry can't build."""


@dataclass(frozen=True)
class Entry[T]:
    name: str
    factory: Callable[[], T]
    #: The optional extra that provides the entry, if the core install doesn't.
    extra: str | None = None
    #: The module the entry imports; when it's missing, the extra isn't installed.
    module: str | None = None

    @property
    def available(self) -> bool:
        return self.module is None or importlib.util.find_spec(self.module) is not None


class Registry[T]:
    """Named factories for one kind of component. A namespaced registry names each entry
    `<namespace>.<Name>`, the namespace being its library's, or `native`."""

    def __init__(
        self, kind: str, entries: Iterable[Entry[T]] = (), *, namespaced: bool = False
    ) -> None:
        self.kind = kind
        self.namespaced = namespaced
        self._entries: dict[str, Entry[T]] = {}
        for entry in entries:
            self.register(entry)

    def register(self, entry: Entry[T]) -> None:
        if entry.name in self._entries:
            raise RegistryError(f"the {self.kind} registry already has {entry.name!r}")
        if self.namespaced and _NAMESPACED.fullmatch(entry.name) is None:
            raise RegistryError(
                f"the {self.kind} registry names entries <namespace>.<Name>, not {entry.name!r}"
            )
        self._entries[entry.name] = entry

    @property
    def namespaces(self) -> tuple[str, ...]:
        return tuple(sorted({name.split(".", 1)[0] for name in self._entries if "." in name}))

    def in_namespace(self, namespace: str) -> tuple[str, ...]:
        """The names under `namespace`."""
        return tuple(n for n in self.names if n.split(".", 1)[0] == namespace)

    def _unknown(self, name: str) -> str:
        found = [f"unknown {self.kind} {name!r}"]
        folded = [n for n in self.names if n.lower() == name.lower()]
        close = folded + [
            c
            for c in difflib.get_close_matches(name, self.names, n=3, cutoff=0.6)
            if c not in folded
        ]
        if close:
            found.append("did you mean " + " or ".join(repr(c) for c in close[:3]))
        namespace = name.split(".", 1)[0]
        if not self.namespaced or len(self._entries) <= _LISTED:
            found.append(f"known: {', '.join(self.names) or 'none'}")
        elif namespace in self.namespaces:
            found.append(f"{namespace} has {', '.join(self.in_namespace(namespace))}")
        else:
            found.append(f"known namespaces: {', '.join(self.namespaces)}")
        return "; ".join(found)

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(sorted(self._entries))

    def __contains__(self, name: object) -> bool:
        return name in self._entries

    def problems(self, names: Iterable[str]) -> list[str]:
        """Why each of `names` can't be built, if it can't."""
        found = []
        for name in names:
            entry = self._entries.get(name)
            if entry is None:
                found.append(self._unknown(name))
            elif not entry.available:
                found.append(
                    f"{self.kind} {name!r} needs the {entry.extra!r} extra; "
                    f"install it with `uv sync --extra {entry.extra}`"
                )
        return found

    def build(self, name: str) -> T:
        problems = self.problems([name])
        if problems:
            raise RegistryError(problems[0])
        return self._entries[name].factory()
