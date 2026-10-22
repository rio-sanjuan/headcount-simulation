"""Known events: what's decided before a run starts, for named employees and requisitions.

Known events are a capability of every run mode (ADR 0006): within a domain, a month's known
events fire first, and sampling runs on the rows left. They come from five sources:

- **history**: what the warehouse already records at the snapshot for a later date. A notice
  of departure (`TERM_PENDING`) reported by the snapshot becomes a termination in the month the
  person leaves;
- **pending starts**: offers accepted by the snapshot, which start in a later month. Each
  filled its requisition on the day it was accepted, so it takes none of the run's openings;
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
    #: The requisition a pending start fills in the run: none for an offer accepted before the
    #: run, whose opening history closed before the seed's snapshot.
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


def from_pending_starts(
    con: duckdb.DuckDBPyConnection, run_date: dt.date, staleness_days: int = 0
) -> list[KnownEvent]:
    """Offers accepted by the last snapshot before the run, starting on or after it. An offer
    whose start date passed less than `staleness_days` before the run is still expected, and
    starts in the run's first month; an older one is stale and left out.

    Every one of them was accepted before the run, and history closed its opening that day, so
    the seed never holds it. Its start carries no requisition: it joins as a boundary fill (see
    `pending`), in the role its offer names, with its requisition's manager flag."""
    cost_center = role_column(con, "pending_start")
    starts = con.execute(
        f"SELECT p.employee_id, p.{cost_center} AS cost_center, p.job_code, p.location, "
        "p.company, coalesce(r.is_manager, false) AS is_manager, p.start_date "
        "FROM pending_start p "
        "LEFT JOIN req_interval r ON r.req_id = p.req_id AND r.interval_number = 1 "
        "WHERE p.snapshot_date = (SELECT max(snapshot_date) FROM pending_start "
        "                         WHERE snapshot_date < $run) "
        "AND p.start_date >= $earliest ORDER BY p.employee_id",
        {"run": run_date, "earliest": run_date - dt.timedelta(days=staleness_days)},
    ).df()
    columns = ["employee_id", "cost_center", "job_code", "location", "company", "is_manager"]
    rows = zip(*(starts[c] for c in columns), starts["start_date"], strict=True)
    return [
        KnownEvent(
            EventKind.PENDING_START,
            max(0, step_of(run_date, start)),
            Source.PENDING_START,
            employee_id=int(employee),
            destination={
                "cost_center": str(cc),
                "job_code": str(job),
                "location": str(location),
                "company": str(company),
                "is_manager": str(bool(manager)).lower(),
            },
            reason="accepted offer",
        )
        for employee, cc, job, location, company, manager, start in rows
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
    staleness_days: int = 0,
) -> KnownEvents:
    """Every known event of a run from the warehouse and the declarations."""
    sources: list[tuple[Source, Iterable[KnownEvent]]] = []
    if history:
        sources.append((Source.HISTORY, from_history(con, run_date)))
    if pending_starts:
        starts = from_pending_starts(con, run_date, staleness_days)
        sources.append((Source.PENDING_START, starts))
    sources.append((Source.DECLARED, from_declarations(movements, run_date)))
    sources.append((Source.PROGRAM, from_programs(programs, run_date)))
    return collect(run_date, months, sources)


# --- replay ------------------------------------------------------------------------------------

#: Each realized code a replay routes, and the kind of event it becomes.
REPLAYED: Final[Mapping[str, EventKind]] = MappingProxyType(
    {
        "TERM": EventKind.TERMINATION,
        "RIF": EventKind.TERMINATION,
        "CONV_OUT": EventKind.CONVERSION_OUT,
        "REORG_IN": EventKind.REORGANIZATION,
        "PROMO": EventKind.PROMOTION,
        "XFER_IN": EventKind.TRANSFER,
        "RECLASS": EventKind.OTHER_MOVEMENT,
        "HIRE": EventKind.PENDING_START,
        "CONV_IN": EventKind.PENDING_START,
    }
)


def from_replay(con: duckdb.DuckDBPyConnection, run_date: dt.date) -> list[KnownEvent]:
    """Every realized movement of a regular employee on or after the run date, as a known
    event: terminations (net of reversals), reductions, conversions out, reorganizations,
    promotions, transfers, reclassifications, and hires. A hire or transfer carries the
    requisition it filled, the one with that channel's close nearest its date, when that close
    falls in the run; a hire with none is a boundary fill. So is a hire whose offer was accepted
    before the run: its opening had closed by the snapshot the seed reads, so the run never
    holds it (see `replay`). Movements carry their new role, in raw labels."""
    codes = ", ".join(f"'{c}'" for c in REPLAYED)
    rows = con.execute(
        f"""
        WITH moves AS (
            SELECT t.employee_id, t.event_date, t.trans_type, t.cost_center, t.job_code,
                   t.location, t.company, t.is_manager
            FROM workforce_txn t
            JOIN dim_account a ON a.account = t.account
            WHERE t.trans_type IN ({codes}) AND t.event_date >= $run
              AND a.account_group = 'regular'
              AND NOT (t.trans_type = 'TERM' AND EXISTS (
                  SELECT 1 FROM workforce_txn r
                  WHERE r.trans_type = 'TERM_REV' AND r.employee_id = t.employee_id
                    AND r.event_date >= t.event_date))
        ),
        fills AS (
            SELECT req_id, filled_by AS employee_id, closed_date, close_reason
            FROM req_interval WHERE filled_by IS NOT NULL
        )
        SELECT m.*, (
            SELECT CASE WHEN f.closed_date >= $run THEN f.req_id END FROM fills f
            WHERE f.employee_id = m.employee_id
              AND list_contains(
                  CASE m.trans_type
                      WHEN 'HIRE' THEN ['filled_external', 'filled_campus']
                      WHEN 'CONV_IN' THEN ['filled_conversion']
                      WHEN 'XFER_IN' THEN ['filled_transfer']
                      ELSE []::VARCHAR[] END,
                  f.close_reason)
            ORDER BY abs(datediff('day', f.closed_date, m.event_date)), f.req_id
            LIMIT 1
        ) AS req_id
        FROM moves m
        ORDER BY m.event_date, m.employee_id, m.trans_type
        """,
        {"run": run_date},
    ).df()
    events = []
    records: list[dict[Any, Any]] = rows.to_dict("records")
    for row in records:
        kind = REPLAYED[str(row["trans_type"])]
        destination: dict[str, str] = {}
        if kind is EventKind.PROMOTION:
            destination = {"job_code": str(row["job_code"])}
        elif kind in MOVEMENTS or kind is EventKind.PENDING_START:
            destination = {
                "cost_center": str(row["cost_center"]),
                "job_code": str(row["job_code"]),
                "location": str(row["location"]),
                "company": str(row["company"]),
                "is_manager": str(bool(row["is_manager"])).lower(),
            }
        req = None if pd.isna(row["req_id"]) else int(row["req_id"])
        events.append(
            KnownEvent(
                kind,
                step_of(run_date, row["event_date"]),
                Source.REPLAY,
                employee_id=int(row["employee_id"]),
                req_id=req,
                destination=destination,
                reason=str(row["trans_type"]),
            )
        )
    return events
