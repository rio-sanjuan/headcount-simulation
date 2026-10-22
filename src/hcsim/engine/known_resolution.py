"""Resolving a known-event plan against the schema, and checking that every event fired.

**Resolution.** A movement or a pending start names where it puts someone in raw labels. Each
label is resolved to its code through the schema, so the event writes the employee buffer
directly when it fires:

- `cost_center`, `job_family`, `job_level`, `location`, `company`, and `tech_indicator` resolve
  through their vocabularies;
- `is_manager` reads `true` or `false`;
- `job_code` expands through the warehouse's jobs to the family, level, and tech indicator.

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
from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Final

import duckdb

from hcsim.engine.known_events import EventKind, KnownEvent, KnownEventError
from hcsim.engine.known_plan import KnownEventPlan
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


@dataclass(frozen=True)
class ResolvedEvent:
    event: KnownEvent
    #: The buffer column and code each destination label resolved to.
    codes: Mapping[EmployeeColumn, int] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "codes", MappingProxyType(dict(sorted(self.codes.items()))))

    def to_dict(self) -> dict[str, Any]:
        return self.event.to_dict() | {"codes": {c.name: v for c, v in self.codes.items()}}


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
) -> tuple[dict[EmployeeColumn, int], list[str]]:
    labels = dict(event.destination)
    problems = []
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
        elif label in schema.vocabulary(column):
            codes[target] = schema.vocabulary(column).code(label)
        else:
            problems.append(f"{column} has no label {label!r}")
    return codes, problems


def resolve_plan(
    plan: KnownEventPlan,
    schema: SimulationSchema,
    jobs: Mapping[str, Mapping[str, str]] | None = None,
) -> ResolvedPlan:
    """Resolve every event's destination, or report every problem."""
    problems = []
    resolved = []
    for e in plan.events.events:
        codes, found = _resolve(e, schema, jobs or {})
        who = f"{e.kind.value} of {e.employee_id} at step {e.step}"
        problems += [f"{who}: {p}" for p in found]
        resolved.append(ResolvedEvent(e, codes))
    if problems:
        raise KnownEventError(problems)
    return ResolvedPlan(plan, tuple(resolved))


@dataclass
class FiringLog:
    """The known events a run fired, by their keys."""

    fired: set[tuple[int, str, int, int, str]] = field(default_factory=set)

    def record(self, event: KnownEvent) -> None:
        self.fired.add(event.key)

    def unfired(self, resolved: ResolvedPlan) -> list[KnownEvent]:
        return [e.event for e in resolved.events if e.event.key not in self.fired]


def check_fired(resolved: ResolvedPlan, log: FiringLog) -> None:
    """Refuse a run that didn't fire every known event, naming each one."""
    missed = log.unfired(resolved)
    if missed:
        raise KnownEventError(
            [
                f"{e.kind.value} of {e.employee_id} at step {e.step} ({e.source.value}) didn't fire"
                for e in missed
            ]
        )
