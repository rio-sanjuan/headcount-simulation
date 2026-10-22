"""Resolving a known-event plan against the schema, and checking that every event fired.

**Resolution.** A movement or a pending start names where it puts someone in raw labels. Each
label is resolved to its code through the schema, so the event writes the employee buffer
directly when it fires:

- `cost_center`, `job_family`, `job_level`, `location`, `company`, and `tech_indicator` resolve
  through their vocabularies;
- `is_manager` reads `true` or `false`;
- `job_code` expands through the warehouse's jobs to the family, level, and tech indicator;
- `boundary` marks a reorganization across the run's boundary. `out` takes someone out of the
  run and sets nothing else; `in` brings someone in, and must set a whole role: the cost center,
  family, level, tech indicator, location, and company.

Every problem in every event is collected: a column a destination can't set, a label the
schema lacks, or a job the warehouse doesn't list. The resolved plan has a digest over its
events and their codes, so a run record can name exactly the known events it ran with.

**Firing.** Every known event must fire, because one that doesn't is a declaration the run
silently ignored. Domains record each event they fire in a `FiringLog`, and `check_fired`
names every event that didn't.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Final

import duckdb

from hcsim.engine.known_events import EventKind, KnownEvent, KnownEventError
from hcsim.engine.known_plan import BOUNDARY, KnownEventPlan
from hcsim.model.core.layouts import EmployeeColumn
from hcsim.model.core.schema import (
    COMPANY,
    COST_CENTER,
    JOB_FAMILY,
    JOB_LEVEL,
    LOCATION,
    TECH_INDICATOR,
    SimulationSchema,
)

DIGEST_PREFIX: Final = "kr1-"
#: The columns a destination may set, by label, and the buffer column each writes.
DESTINATION_COLUMNS: Final[Mapping[str, EmployeeColumn]] = MappingProxyType(
    {
        COST_CENTER: EmployeeColumn.COST_CENTER,
        JOB_FAMILY: EmployeeColumn.JOB_FAMILY,
        JOB_LEVEL: EmployeeColumn.LEVEL,
        LOCATION: EmployeeColumn.LOCATION,
        COMPANY: EmployeeColumn.COMPANY,
        TECH_INDICATOR: EmployeeColumn.TECH_INDICATOR,
        "is_manager": EmployeeColumn.IS_MANAGER,
    }
)
_FLAGS: Final = {"true": 1, "false": 0}
#: A role's attributes, as a job code implies them.
JOB_ATTRIBUTES: Final = (JOB_FAMILY, JOB_LEVEL, TECH_INDICATOR)


def jobs_from_world(con: duckdb.DuckDBPyConnection) -> dict[str, dict[str, str]]:
    """Each job code's family, level, and tech indicator, as labels."""
    frame = con.execute(f"SELECT job_code, {', '.join(JOB_ATTRIBUTES)} FROM dim_job").df()
    return {
        str(row.job_code): {a: str(getattr(row, a)) for a in JOB_ATTRIBUTES}
        for row in frame.itertuples(index=False)
    }


#: The columns a boundary-in reorganization must set: a whole role.
WHOLE_ROLE: Final = (
    EmployeeColumn.COST_CENTER,
    EmployeeColumn.JOB_FAMILY,
    EmployeeColumn.LEVEL,
    EmployeeColumn.TECH_INDICATOR,
    EmployeeColumn.LOCATION,
    EmployeeColumn.COMPANY,
)


@dataclass(frozen=True)
class ResolvedEvent:
    event: KnownEvent
    #: The buffer column and code each destination label resolved to.
    codes: Mapping[EmployeeColumn, int] = field(default_factory=dict)
    #: `in` or `out` for a reorganization across the run's boundary.
    boundary: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "codes", MappingProxyType(dict(sorted(self.codes.items()))))

    def to_dict(self) -> dict[str, Any]:
        codes = {c.name: v for c, v in self.codes.items()}
        return self.event.to_dict() | {"codes": codes, "boundary": self.boundary}


@dataclass(frozen=True)
class ResolvedPlan:
    plan: KnownEventPlan
    events: tuple[ResolvedEvent, ...]

    def digest(self) -> str:
        text = json.dumps([e.to_dict() for e in self.events], sort_keys=True)
        return DIGEST_PREFIX + hashlib.sha256(text.encode()).hexdigest()[:16]

    def at(self, step: int, kind: EventKind | None = None) -> list[ResolvedEvent]:
        return [
            e
            for e in self.events
            if e.event.step == step and (kind is None or e.event.kind is kind)
        ]


def _resolve(
    event: KnownEvent,
    schema: SimulationSchema,
    jobs: Mapping[str, Mapping[str, str]],
) -> tuple[dict[EmployeeColumn, int], str | None, list[str]]:
    labels = dict(event.destination)
    problems = []
    boundary = labels.pop(BOUNDARY, None)
    if boundary is not None and event.kind is not EventKind.REORGANIZATION:
        problems.append("only a reorganization crosses the run's boundary")
    elif boundary == "out" and labels:
        problems.append("a reorganization out of the run sets nothing else")
    elif boundary not in (None, "in", "out"):
        problems.append(f"a boundary is in or out, not {boundary!r}")
    job = labels.pop("job_code", None)
    if job is not None:
        if job in jobs:
            labels = {**jobs[job], **labels}
        else:
            problems.append(f"no job {job!r}")
    codes = {}
    for column, label in sorted(labels.items()):
        target = DESTINATION_COLUMNS.get(column)
        if target is None:
            problems.append(f"a destination can't set {column!r}")
        elif column == "is_manager":
            flag = str(label).lower()
            if flag in _FLAGS:
                codes[target] = _FLAGS[flag]
            else:
                problems.append(f"is_manager reads true or false, not {label!r}")
        elif column not in schema.vocabularies:
            problems.append(f"the schema has no {column} vocabulary")
        elif label in schema.vocabulary(column):
            codes[target] = schema.vocabulary(column).code(label)
        else:
            problems.append(f"{column} has no label {label!r}")
    if boundary == "in" and not problems:
        missing = [c.name.lower() for c in WHOLE_ROLE if c not in codes]
        if missing:
            problems.append(f"a reorganization into the run needs a whole role, not {missing}")
    return codes, boundary, problems


def resolve_plan(
    plan: KnownEventPlan,
    schema: SimulationSchema,
    jobs: Mapping[str, Mapping[str, str]] | None = None,
) -> ResolvedPlan:
    """Resolve every event's destination, or report every problem."""
    problems = []
    resolved = []
    for e in plan.events.events:
        codes, boundary, found = _resolve(e, schema, jobs or {})
        who = f"{e.kind.value} of {e.employee_id} at step {e.step}"
        problems += [f"{who}: {p}" for p in found]
        resolved.append(ResolvedEvent(e, codes, boundary))
    if problems:
        raise KnownEventError(problems)
    return ResolvedPlan(plan, tuple(resolved))


@dataclass
class FiringLog:
    """The known events a run fired, by their keys, and the people it left out: a replayed
    hire with no requisition, when the run makes no boundary fills, never joins, so none of
    their later events can fire, and none is expected to."""

    fired: set[tuple[int, str, int, int, str]] = field(default_factory=set)
    left_out: set[int] = field(default_factory=set)

    def record(self, event: KnownEvent) -> None:
        self.fired.add(event.key)

    def leave_out(self, employee_id: int) -> None:
        self.left_out.add(employee_id)

    def unfired(
        self, resolved: ResolvedPlan, kinds: Iterable[EventKind] | None = None
    ) -> list[KnownEvent]:
        """The events not fired, of `kinds` (every kind by default), but those of people the
        run left out."""
        wanted = None if kinds is None else set(kinds)
        return [
            e.event
            for e in resolved.events
            if e.event.key not in self.fired
            and e.event.employee_id not in self.left_out
            and (wanted is None or e.event.kind in wanted)
        ]


def check_fired(
    resolved: ResolvedPlan, log: FiringLog, kinds: Iterable[EventKind] | None = None
) -> None:
    """Refuse a run that didn't fire every known event of `kinds`, naming each one."""
    missed = log.unfired(resolved, kinds)
    if missed:
        raise KnownEventError(
            [
                f"{e.kind.value} of {e.employee_id} at step {e.step} ({e.source.value}) didn't fire"
                for e in missed
            ]
        )
