"""A run's known-event plan: known events sanitized against the seed, with the exclusions and
reservations every domain reads.

**Sanitizing.** Known events come from several sources and can disagree, so each employee's
events are put in order and checked against the seed state. Each dropped event is counted under
the first cause that drops it:

1. `not_in_seed`: an employee event for someone the seed doesn't hold. A pending start is for
   someone joining, so it's exempt, and so is a reorganization, which brings someone in across
   the run's boundary: it's kept, marked `boundary: in`, and must carry a whole role;
2. `duplicate`: the same kind of event for the same person in the same month as one already kept;
3. `outranked`: another event for the same person in the same month ranks above it. Events rank
   by their *class*, from an exit (termination, then conversion out) through the movements
   (reorganization, transfer, other movement) to promotion, and then by source, from declared
   down to replay;
4. `after_exit`: the person has already left by an earlier event.

**Exclusions.** An employee with a known event still to come is excluded from every sampled
draw up to it, so no one is sampled to leave, move, or be promoted while a known event waits
for them (ADR 0006). The exclusion covers the event's own month too, since a domain that runs
before the event's own domain in that month mustn't take the person first. From the month
after their last known event, they're sampled again.

**Reservations.** A pending start whose requisition is in the seed reserves one of its openings
up to and including the month it starts, so fills can't take it. A pending start whose
requisition isn't in the seed reserves nothing; pending starts (M4.10) decide what becomes of
it.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Final

import numpy as np
from numpy.typing import NDArray

from hcsim.engine.known_events import EventKind, KnownEvent, KnownEvents, Source
from hcsim.model.core.layouts import Buffer, EmployeeColumn, OpeningColumn

#: Event classes, highest first: an exit outranks a movement, which outranks a promotion.
CLASS_ORDER: Final = (
    EventKind.TERMINATION,
    EventKind.CONVERSION_OUT,
    EventKind.REORGANIZATION,
    EventKind.TRANSFER,
    EventKind.OTHER_MOVEMENT,
    EventKind.PROMOTION,
    EventKind.PENDING_START,
)
#: Sources, highest first, to break a tie between events of one class.
SOURCE_ORDER: Final = (
    Source.DECLARED,
    Source.PROGRAM,
    Source.HISTORY,
    Source.PENDING_START,
    Source.REPLAY,
)
#: The kinds that take someone out of the workforce.
EXITS: Final = frozenset({EventKind.TERMINATION, EventKind.CONVERSION_OUT})
CAUSES: Final = ("not_in_seed", "duplicate", "outranked", "after_exit")
#: The destination key that marks a reorganization across the run's boundary: `in` brings
#: someone into the run, and `out` takes them out of it.
BOUNDARY: Final = "boundary"


def rank(event: KnownEvent) -> tuple[int, int]:
    """Lower ranks first."""
    return CLASS_ORDER.index(event.kind), SOURCE_ORDER.index(event.source)


@dataclass(frozen=True)
class KnownEventPlan:
    events: KnownEvents
    #: Events dropped, per cause.
    dropped: Mapping[str, int] = field(default_factory=dict)
    #: Requisition id to the steps its reserved openings start, one per pending start.
    reservations: Mapping[int, tuple[int, ...]] = field(default_factory=dict)
    _ids: NDArray[np.int64] = field(init=False, repr=False, compare=False)
    _last: NDArray[np.int64] = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "dropped", MappingProxyType(dict(self.dropped)))
        object.__setattr__(self, "reservations", MappingProxyType(dict(self.reservations)))
        last: dict[int, int] = {}
        for e in self.events.events:
            if e.employee_id is not None and e.kind is not EventKind.PENDING_START:
                last[e.employee_id] = max(last.get(e.employee_id, -1), e.step)
        ids = np.array(sorted(last), dtype=np.int64)
        object.__setattr__(self, "_ids", ids)
        object.__setattr__(self, "_last", np.array([last[i] for i in ids], dtype=np.int64))

    def excluded(self, step: int) -> NDArray[np.int64]:
        """The employees with a known event at `step` or later, sorted."""
        ids: NDArray[np.int64] = self._ids[self._last >= step]
        return ids

    def sampled(self, rows: Buffer, step: int) -> NDArray[np.bool_]:
        """Which employee rows sampled draws may select at `step`: everyone but those
        waiting on a known event."""
        ids = rows[:, EmployeeColumn.EMPLOYEE_ID].astype(np.int64)
        result: NDArray[np.bool_] = ~np.isin(ids, self.excluded(step))
        return result

    def reserved(self, rows: Buffer, step: int) -> NDArray[np.int64]:
        """Openings held back on each requisition row at `step`, for starts at `step` or
        later, never more than the row has."""
        ids = rows[:, OpeningColumn.REQ_ID].astype(np.int64)
        held = np.array(
            [sum(s >= step for s in self.reservations.get(int(i), ())) for i in ids],
            dtype=np.int64,
        )
        result: NDArray[np.int64] = np.minimum(held, rows[:, OpeningColumn.NUM_OPENINGS])
        return result.astype(np.int64)

    def fillable(self, rows: Buffer, step: int) -> NDArray[np.int64]:
        """The openings fills may take on each requisition row at `step`."""
        total = rows[:, OpeningColumn.NUM_OPENINGS].astype(np.int64)
        result: NDArray[np.int64] = total - self.reserved(rows, step)
        return result


def sanitize(
    events: KnownEvents, employees: Iterable[int], requisitions: Iterable[int]
) -> KnownEventPlan:
    """Order and check `events` against the seed's employee and requisition ids."""
    in_seed = set(employees)
    reqs = set(requisitions)
    dropped = dict.fromkeys(CAUSES, 0)
    by_person: dict[int, list[KnownEvent]] = {}
    kept: list[KnownEvent] = []
    for e in events.events:
        if e.employee_id is None or e.kind is EventKind.PENDING_START:
            kept.append(e)
        elif e.employee_id not in in_seed and e.kind is EventKind.REORGANIZATION:
            kept.append(dataclasses.replace(e, destination={**e.destination, BOUNDARY: "in"}))
        elif e.employee_id not in in_seed:
            dropped["not_in_seed"] += 1
        else:
            by_person.setdefault(e.employee_id, []).append(e)
    for person in by_person.values():
        exited_at: int | None = None
        seen: set[tuple[int, EventKind]] = set()
        winner: dict[int, tuple[int, int]] = {}
        for e in sorted(person, key=lambda e: (e.step, rank(e))):
            if (e.step, e.kind) in seen:
                dropped["duplicate"] += 1
            elif e.step in winner:
                dropped["outranked"] += 1
            elif exited_at is not None and e.step > exited_at:
                dropped["after_exit"] += 1
            else:
                kept.append(e)
                winner[e.step] = rank(e)
                if e.kind in EXITS:
                    exited_at = e.step
            seen.add((e.step, e.kind))
    reservations: dict[int, list[int]] = {}
    for e in kept:
        if e.kind is EventKind.PENDING_START and e.req_id in reqs:
            reservations.setdefault(int(e.req_id), []).append(e.step)
    return KnownEventPlan(
        KnownEvents(tuple(kept), events.outside_window),
        {cause: n for cause, n in dropped.items() if n},
        {req: tuple(sorted(steps)) for req, steps in sorted(reservations.items())},
    )
