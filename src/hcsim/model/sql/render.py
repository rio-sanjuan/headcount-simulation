"""The SQL template renderer: one pass, named slots, and no silent gaps.

A template marks a required slot `{{name}}` and an optional one `{{name?}}`. Rendering fills
every slot in a single pass: a value is inserted as it is and never scanned again, so a value
that happens to contain `{{x}}` can't fill a second slot. Rendering refuses, naming every
problem at once, a required slot left unfilled, a value no slot asks for, and a malformed
placeholder. An optional slot with no value renders as nothing.

Slot values are SQL fragments, not data: data enters a query as a bound parameter, or through
`literal`, which quotes it.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from importlib import resources
from typing import Final

#: `{{name}}` or `{{name?}}`, with optional spaces inside the braces.
_SLOT: Final = re.compile(r"\{\{\s*([A-Za-z_][A-Za-z0-9_]*)\s*(\?)?\s*\}\}")
_OPEN: Final = re.compile(r"\{\{")


class TemplateError(ValueError):
    """A template that can't render as asked."""


@dataclass(frozen=True)
class Template:
    text: str
    name: str = "template"
    required: frozenset[str] = field(init=False)
    optional: frozenset[str] = field(init=False)

    def __post_init__(self) -> None:
        required: set[str] = set()
        optional: set[str] = set()
        problems: list[str] = []
        spans = []
        for match in _SLOT.finditer(self.text):
            (optional if match.group(2) else required).add(match.group(1))
            spans.append(match.span())
        both = sorted(required & optional)
        if both:
            problems.append(f"slots both required and optional: {both}")
        # Any `{{` that doesn't open a well-formed slot is a typo that would ship as SQL.
        covered = {start for start, _ in spans}
        stray = [m.start() for m in _OPEN.finditer(self.text) if m.start() not in covered]
        if stray:
            lines = sorted({self.text.count("\n", 0, at) + 1 for at in stray})
            problems.append(f"malformed placeholders on lines {lines}")
        if problems:
            raise TemplateError(f"{self.name}: " + "; ".join(problems))
        object.__setattr__(self, "required", frozenset(required))
        object.__setattr__(self, "optional", frozenset(optional))

    @property
    def slots(self) -> frozenset[str]:
        return self.required | self.optional

    def render(self, values: Mapping[str, str] | None = None, /, **more: str) -> str:
        given = {**(values or {}), **more}
        problems = []
        missing = sorted(self.required - set(given))
        if missing:
            problems.append(f"unfilled slots {missing}")
        unknown = sorted(set(given) - self.slots)
        if unknown:
            problems.append(f"no slots for {unknown}")
        wrong = sorted(k for k, v in given.items() if not isinstance(v, str))
        if wrong:
            problems.append(f"values must be SQL text: {wrong}")
        if problems:
            raise TemplateError(f"{self.name}: " + "; ".join(problems))
        return _SLOT.sub(lambda m: given.get(m.group(1), ""), self.text)


def literal(value: object) -> str:
    """A SQL literal: a quoted string, with quotes doubled, or a plain number or boolean."""
    if isinstance(value, bool):
        return "TRUE" if value else "FALSE"
    if isinstance(value, int | float):
        return repr(value)
    return "'" + str(value).replace("'", "''") + "'"


def identifier(name: str) -> str:
    """A quoted SQL identifier."""
    return '"' + name.replace('"', '""') + '"'


def load(name: str) -> Template:
    """A packaged template from `hcsim/model/sql/templates/<name>.sql`."""
    path = resources.files("hcsim.model.sql").joinpath("templates", f"{name}.sql")
    if not path.is_file():
        raise TemplateError(f"no template named {name!r}")
    return Template(path.read_text(), name)
