"""Tests for sanitizing known events, and the exclusions and reservations they imply."""

from __future__ import annotations

import datetime as dt
from pathlib import Path

import duckdb
import numpy as np

from hcsim.engine.known_events import EventKind, KnownEvent, KnownEvents, Source, known_events
from hcsim.engine.known_plan import CLASS_ORDER, SOURCE_ORDER, rank, sanitize
from hcsim.engine.seeding import schema_from_world, seed_state
from hcsim.model.core.layouts import EMPLOYEE, OPENING, EmployeeColumn, OpeningColumn
from hcsim.synth.config import preset
from hcsim.synth.world import generate_world, write_world

T, C, R, P = (
    EventKind.TERMINATION,
    EventKind.CONVERSION_OUT,
    EventKind.REORGANIZATION,
    EventKind.PROMOTION,
)


def event(kind: EventKind, step: int, person: int, source: Source = Source.DECLARED) -> KnownEvent:
    return KnownEvent(kind, step, source, employee_id=person)


def start(step: int, person: int, req: int) -> KnownEvent:
    return KnownEvent(EventKind.PENDING_START, step, Source.PENDING_START, person, req)


def test_classes_rank_exits_over_movements_over_promotions() -> None:
    assert CLASS_ORDER[:2] == (T, C)
    assert rank(event(T, 0, 1)) < rank(event(R, 0, 1)) < rank(event(P, 0, 1))
    assert rank(event(T, 0, 1, Source.DECLARED)) < rank(event(T, 0, 1, Source.HISTORY))
    assert SOURCE_ORDER[-1] is Source.REPLAY


def test_each_dropped_event_is_counted_under_its_first_cause() -> None:
    events = KnownEvents(
        (
            event(T, 3, 1, Source.HISTORY),
            event(T, 3, 1, Source.DECLARED),  # duplicate of the same exit
            event(R, 3, 1),  # outranked by the exit in the same month
            event(P, 5, 1),  # after the exit
            event(P, 1, 1),  # before the exit: kept
            event(C, 2, 2),
            event(T, 4, 2),  # after the conversion out
            event(P, 0, 99),  # not in the seed
        )
    )
    plan = sanitize(events, employees=[1, 2], requisitions=[])
    kept = [(e.kind, e.step, e.employee_id, e.source) for e in plan.events.events]
    assert kept == [
        (P, 1, 1, Source.DECLARED),
        (C, 2, 2, Source.DECLARED),
        (T, 3, 1, Source.DECLARED),  # the declared exit outranks the notice
    ]
    assert plan.dropped == {"not_in_seed": 1, "duplicate": 1, "outranked": 1, "after_exit": 2}


def test_employees_waiting_on_a_known_event_are_excluded_until_it_fires() -> None:
    plan = sanitize(
        KnownEvents((event(P, 1, 1), event(R, 4, 1), event(T, 2, 2))),
        employees=[1, 2, 3],
        requisitions=[],
    )
    assert plan.excluded(0).tolist() == [1, 2]
    assert plan.excluded(2).tolist() == [1, 2]  # 2's own month still excludes them
    assert plan.excluded(3).tolist() == [1]
    assert plan.excluded(5).tolist() == []
    rows = EMPLOYEE.empty(3)
    rows[:, EmployeeColumn.EMPLOYEE_ID] = [1, 2, 3]
    assert plan.sampled(rows, 3).tolist() == [False, True, True]


def test_pending_starts_reserve_openings_on_seeded_requisitions() -> None:
    plan = sanitize(
        KnownEvents((start(2, 500, 10), start(4, 501, 10), start(1, 502, 77), start(1, 503, 11))),
        employees=[],
        requisitions=[10, 11],
    )
    # A start for a requisition the seed doesn't hold reserves nothing, but stays a known event.
    assert dict(plan.reservations) == {10: (2, 4), 11: (1,)}
    assert len(plan.events) == 4
    reqs = OPENING.empty(3)
    reqs[:, OpeningColumn.REQ_ID] = [10, 11, 12]
    reqs[:, OpeningColumn.NUM_OPENINGS] = [3, 1, 2]
    assert plan.reserved(reqs, 0).tolist() == [2, 1, 0]
    assert plan.fillable(reqs, 0).tolist() == [1, 0, 2]
    assert plan.reserved(reqs, 3).tolist() == [1, 0, 0]
    assert plan.fillable(reqs, 5).tolist() == [3, 1, 2]
    # Never more than the row still has.
    reqs[0, OpeningColumn.NUM_OPENINGS] = 1
    assert plan.reserved(reqs, 0).tolist()[0] == 1
    # Pending starts are for people who haven't joined, so they're never excluded.
    assert plan.excluded(0).tolist() == []


def test_a_world_s_known_events_sanitize_against_its_seed(tmp_path: Path) -> None:
    path = tmp_path / "w.duckdb"
    write_world(generate_world(preset("small", seed=4)), path)
    run = dt.date(2025, 7, 1)
    with duckdb.connect(str(path), read_only=True) as con:
        state, _ = seed_state(con, schema_from_world(con), run)
        events = known_events(con, run, 12)
    employees = state.live_employees[:, EmployeeColumn.EMPLOYEE_ID].astype(np.int64)
    reqs = state.live_openings[:, OpeningColumn.REQ_ID].astype(np.int64)
    plan = sanitize(events, employees.tolist(), reqs.tolist())
    notices = [e for e in plan.events.events if e.kind is T]
    assert notices  # people who gave notice are in the seed until they leave
    assert set(plan.excluded(0).tolist()) == {e.employee_id for e in notices}
    starts = [e for e in plan.events.events if e.kind is EventKind.PENDING_START]
    on_seeded = [e for e in starts if e.req_id in set(reqs.tolist())]
    assert 0 < len(on_seeded) <= len(starts)
    assert sum(len(v) for v in plan.reservations.values()) == len(on_seeded)
    assert plan.fillable(state.live_openings, 0).min() >= 0


def test_a_reorganization_from_outside_the_seed_crosses_the_boundary_in() -> None:
    joining = KnownEvent(
        R, 2, Source.DECLARED, employee_id=99, destination={"cost_center": "CC-01"}
    )
    plan = sanitize(KnownEvents((joining,)), employees=[1], requisitions=[])
    (kept,) = plan.events.events
    assert kept.destination == {"cost_center": "CC-01", "boundary": "in"}
    assert plan.dropped == {}
    assert plan.excluded(0).tolist() == [99]  # held until they arrive
