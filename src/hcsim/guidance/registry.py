"""Registries: the named components a guidance configuration can choose from.

Each registry maps names to factories, for one kind of component: forecasting models,
combination methods, or reconciliation methods. A configuration names its components, and the
registries check every name when the configuration loads, not when a step first needs it:

- a name no registry entry carries is refused, with the names that are known;
- a name whose entry comes from an optional extra that isn't installed is refused, with the
  command that installs it.
"""

from __future__ import annotations

import importlib.util
from collections.abc import Callable, Iterable
from dataclasses import dataclass


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
    """Named factories for one kind of component."""

    def __init__(self, kind: str, entries: Iterable[Entry[T]] = ()) -> None:
        self.kind = kind
        self._entries: dict[str, Entry[T]] = {}
        for entry in entries:
            self.register(entry)

    def register(self, entry: Entry[T]) -> None:
        if entry.name in self._entries:
            raise RegistryError(f"the {self.kind} registry already has {entry.name!r}")
        self._entries[entry.name] = entry

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
                known = ", ".join(self.names) or "none"
                found.append(f"unknown {self.kind} {name!r}; known: {known}")
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
