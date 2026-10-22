"""Known events: what's decided before a run starts, for named employees and requisitions.

Known events are a capability of every run mode (ADR 0006): within a domain, a month's known
events fire first, and sampling runs on the rows left. They come from five sources:

- **history**: what the warehouse already records at the snapshot for a later date. A notice
  of departure (`TERM_PENDING`) reported by the snapshot becomes a termination in the month the
  person leaves;
- **pending starts**: offers accepted by the snapshot, which start in a later month and fill
  their requisition;
- **declared movements**: reorganizations and conversions out, entered for named employees in
  a month, with a destination in raw labels;
- **termination programs**: planned terminations of named employees over a window of months,
  spread evenly in the order given;
- **replay**: every realized movement, for the run mode that pins them (M4.16).

An event fires at a *step*: the months from the run's first month to the event's month. An
event outside the run's window can't fire, so it's left out and counted. `KnownEvents` holds
the rest in a fixed order, with a digest, so two runs with the same known events agree.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from types import MappingProxyType
from typing import Any, Final

import duckdb
import pandas as pd

from hcsim.model.core.contract import role_column

DIGEST_PREFIX: Final = "ke1-"


class KnownEventError(ValueError):
    """Known events that can't be declared or collected, every problem listed."""

    def __init__(self, problems: list[str]) -> None:
        self.problems = problems
        super().__init__("bad known events: " + "; ".join(problems))


class EventKind(StrEnum):
    TERMINATION = "termination"
    CONVERSION_OUT = "conversion_out"
    REORGANIZATION = "reorganization"
    TRANSFER = "transfer"
    PROMOTION = "promotion"
    OTHER_MOVEMENT = "other_movement"
    PENDING_START = "pending_start"


class Source(StrEnum):
    HISTORY = "history"
    PENDING_START = "pending_start"
    DECLARED = "declared"
    PROGRAM = "program"
    REPLAY = "replay"


#: The kinds that move an employee somewhere, and so need a destination.
MOVEMENTS: Final = frozenset(
    {EventKind.REORGANIZATION, EventKind.TRANSFER, EventKind.PROMOTION, EventKind.OTHER_MOVEMENT}
)
#: The kinds a declaration may name.
DECLARABLE: Final = frozenset({EventKind.REORGANIZATION, EventKind.CONVERSION_OUT})


@dataclass(frozen=True)
class KnownEvent:
    kind: EventKind
    step: int
    source: Source
    employee_id: int | None = None
    #: The requisition a pending start fills.
    req_id: int | None = None
    #: Where a movement or start puts the person, in raw labels.
    destination: Mapping[str, str] = field(default_factory=dict)
    reason: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "destination", MappingProxyType(dict(self.destination)))

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind.value,
            "step": self.step,
            "source": self.source.value,
            "employee_id": self.employee_id,
            "req_id": self.req_id,
            "destination": dict(sorted(self.destination.items())),
            "reason": self.reason,
        }

    @property
    def key(self) -> tuple[int, str, int, int, str]:
        return (
            self.step,
            self.kind.value,
            -1 if self.employee_id is None else self.employee_id,
            -1 if self.req_id is None else self.req_id,
            self.source.value,
        )


@dataclass(frozen=True)
class KnownEvents:
    """Known events in a fixed order, and what was left out of the window, by source."""

    events: tuple[KnownEvent, ...] = ()
    outside_window: Mapping[str, int] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "events", tuple(sorted(self.events, key=lambda e: e.key)))
        object.__setattr__(self, "outside_window", MappingProxyType(dict(self.outside_window)))

    def __len__(self) -> int:
        return len(self.events)

    def at(self, step: int, kind: EventKind | None = None) -> list[KnownEvent]:
        return [e for e in self.events if e.step == step and (kind is None or e.kind is kind)]

    def counts(self) -> dict[str, int]:
        """Events by kind."""
        found: dict[str, int] = {}
        for e in self.events:
            found[e.kind.value] = found.get(e.kind.value, 0) + 1
        return found

    def digest(self) -> str:
        text = json.dumps([e.to_dict() for e in self.events], sort_keys=True)
        return DIGEST_PREFIX + hashlib.sha256(text.encode()).hexdigest()[:16]

    def frame(self) -> pd.DataFrame:
        return pd.DataFrame([e.to_dict() for e in self.events])


def step_of(run_date: dt.date, when: dt.date | str) -> int:
    """The step of the month `when` falls in: 0 for the run's first month."""
    month = pd.Period(when, freq="M")
    return int((month - pd.Period(run_date, freq="M")).n)


def collect(
    run_date: dt.date, months: int, sources: Iterable[tuple[Source, Iterable[KnownEvent]]]
) -> KnownEvents:
    """Every event that fires inside the run's `months`, the rest counted by source."""
    kept: list[KnownEvent] = []
    outside: dict[str, int] = {}
    for source, events in sources:
        for event in events:
            if 0 <= event.step < months:
                kept.append(event)
            else:
                outside[source.value] = outside.get(source.value, 0) + 1
    return KnownEvents(tuple(kept), outside)


# --- history and pending starts ----------------------------------------------------------------


def from_history(con: duckdb.DuckDBPyConnection, run_date: dt.date) -> list[KnownEvent]:
    """Notices of departure the warehouse has before the run, for exits on or after it."""
    notices = con.execute(
        "SELECT employee_id, event_date FROM workforce_txn "
        "WHERE trans_type = 'TERM_PENDING' AND reporting_date < $run AND event_date >= $run "
        "ORDER BY employee_id",
        {"run": run_date},
    ).df()
    return [
        KnownEvent(
            EventKind.TERMINATION,
            step_of(run_date, day),
            Source.HISTORY,
            employee_id=int(employee),
            reason="notice given",
        )
        for employee, day in zip(notices["employee_id"], notices["event_date"], strict=True)
    ]


def from_pending_starts(con: duckdb.DuckDBPyConnection, run_date: dt.date) -> list[KnownEvent]:
    """Offers accepted by the last snapshot before the run, starting on or after it."""
    cost_center = role_column(con, "pending_start")
    starts = con.execute(
        f"SELECT employee_id, req_id, {cost_center} AS cost_center, job_code, location, company, "
        "start_date FROM pending_start "
        "WHERE snapshot_date = (SELECT max(snapshot_date) FROM pending_start "
        "                       WHERE snapshot_date < $run) "
        "AND start_date >= $run ORDER BY employee_id",
        {"run": run_date},
    ).df()
    columns = ["employee_id", "req_id", "cost_center", "job_code", "location", "company"]
    rows = zip(*(starts[c] for c in columns), starts["start_date"], strict=True)
    return [
        KnownEvent(
            EventKind.PENDING_START,
            step_of(run_date, start),
            Source.PENDING_START,
            employee_id=int(employee),
            req_id=int(req),
            destination={
                "cost_center": str(cc),
                "job_code": str(job),
                "location": str(location),
                "company": str(company),
            },
            reason="accepted offer",
        )
        for employee, req, cc, job, location, company, start in rows
    ]


# --- declarations ------------------------------------------------------------------------------


@dataclass(frozen=True)
class DeclaredMovement:
    """A reorganization or conversion out, for named employees in one month (`YYYY-MM`)."""

    kind: EventKind
    month: str
    employees: tuple[int, ...]
    destination: Mapping[str, str] = field(default_factory=dict)
    reason: str = ""

    def problems(self) -> list[str]:
        found = []
        if self.kind not in DECLARABLE:
            known = ", ".join(sorted(k.value for k in DECLARABLE))
            found.append(f"a declaration can't move people by {self.kind.value!r}; it can {known}")
        try:
            pd.Period(self.month, freq="M")
        except ValueError:
            found.append(f"{self.month!r} isn't a month")
        if not self.employees:
            found.append("a declaration names at least one employee")
        if self.kind is EventKind.REORGANIZATION and not self.destination:
            found.append("a reorganization needs a destination")
        if self.kind is EventKind.CONVERSION_OUT and self.destination:
            found.append("a conversion out leaves the workforce, so it takes no destination")
        return found


@dataclass(frozen=True)
class TerminationProgram:
    """Planned terminations of named employees, spread evenly over `months` from `start`."""

    name: str
    start: str
    months: int
    employees: tuple[int, ...]

    def problems(self) -> list[str]:
        found = []
        try:
            pd.Period(self.start, freq="M")
        except ValueError:
            found.append(f"program {self.name!r}: {self.start!r} isn't a month")
        if self.months < 1:
            found.append(f"program {self.name!r} must run at least one month")
        if not self.employees:
            found.append(f"program {self.name!r} names no employees")
        return found

    def schedule(self) -> list[tuple[int, str]]:
        """Each employee and the month they leave: the k-th of n in month
        `start + floor(k * months / n)`."""
        first = pd.Period(self.start, freq="M")
        n = len(self.employees)
        return [(e, str(first + (k * self.months) // n)) for k, e in enumerate(self.employees)]


def _check(items: Sequence[DeclaredMovement | TerminationProgram]) -> None:
    problems = []
    for k, item in enumerate(items):
        problems += [f"item {k + 1}: {p}" for p in item.problems()]
    if problems:
        raise KnownEventError(problems)


def from_declarations(movements: Sequence[DeclaredMovement], run_date: dt.date) -> list[KnownEvent]:
    _check(movements)
    return [
        KnownEvent(
            m.kind,
            step_of(run_date, m.month),
            Source.DECLARED,
            employee_id=int(e),
            destination=m.destination,
            reason=m.reason,
        )
        for m in movements
        for e in m.employees
    ]


def from_programs(programs: Sequence[TerminationProgram], run_date: dt.date) -> list[KnownEvent]:
    _check(programs)
    return [
        KnownEvent(
            EventKind.TERMINATION,
            step_of(run_date, month),
            Source.PROGRAM,
            employee_id=int(e),
            reason=f"program {p.name}",
        )
        for p in programs
        for e, month in p.schedule()
    ]


def known_events(
    con: duckdb.DuckDBPyConnection,
    run_date: dt.date,
    months: int,
    *,
    movements: Sequence[DeclaredMovement] = (),
    programs: Sequence[TerminationProgram] = (),
    history: bool = True,
    pending_starts: bool = True,
) -> KnownEvents:
    """Every known event of a run from the warehouse and the declarations."""
    sources: list[tuple[Source, Iterable[KnownEvent]]] = []
    if history:
        sources.append((Source.HISTORY, from_history(con, run_date)))
    if pending_starts:
        sources.append((Source.PENDING_START, from_pending_starts(con, run_date)))
    sources.append((Source.DECLARED, from_declarations(movements, run_date)))
    sources.append((Source.PROGRAM, from_programs(programs, run_date)))
    return collect(run_date, months, sources)
