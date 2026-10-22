"""Tests for collecting known events from the warehouse and from declarations."""

from __future__ import annotations

import dataclasses
import datetime as dt
from collections.abc import Iterator

import duckdb
import pytest

from hcsim.engine.known_events import (
    DeclaredMovement,
    EventKind,
    KnownEvent,
    KnownEventError,
    KnownEvents,
    Source,
    TerminationProgram,
    collect,
    from_declarations,
    from_history,
    from_pending_starts,
    from_programs,
    known_events,
    step_of,
)
from hcsim.synth.config import QuirkTruth, preset
from hcsim.synth.world import generate_world, write_world

RUN = dt.date(2025, 7, 1)


@pytest.fixture(scope="module", params=[False, True], ids=["plain", "aliased"])
def con(
    request: pytest.FixtureRequest, tmp_path_factory: pytest.TempPathFactory
) -> Iterator[duckdb.DuckDBPyConnection]:
    quirks = QuirkTruth(cost_center_aliases=request.param)
    path = tmp_path_factory.mktemp("world") / "w.duckdb"
    write_world(generate_world(dataclasses.replace(preset("small", seed=4), quirks=quirks)), path)
    with duckdb.connect(str(path), read_only=True) as connection:
        yield connection


def test_steps_count_months_from_the_run_s_first() -> None:
    assert step_of(RUN, dt.date(2025, 7, 31)) == 0
    assert step_of(RUN, "2025-09") == 2
    assert step_of(RUN, dt.date(2025, 6, 30)) == -1


def test_notices_given_before_the_run_become_terminations(
    con: duckdb.DuckDBPyConnection,
) -> None:
    events = from_history(con, RUN)
    assert events
    assert all(e.kind is EventKind.TERMINATION and e.source is Source.HISTORY for e in events)
    assert all(e.step >= 0 for e in events)
    # Each one is a notice reported before the run, for a later exit.
    expected = con.execute(
        "SELECT count(*) FROM workforce_txn WHERE trans_type = 'TERM_PENDING' "
        "AND reporting_date < ? AND event_date >= ?",
        [RUN, RUN],
    ).fetchone()
    assert (len(events),) == expected


def test_pending_starts_fill_their_requisitions(con: duckdb.DuckDBPyConnection) -> None:
    events = from_pending_starts(con, RUN)
    assert events
    for e in events:
        assert e.kind is EventKind.PENDING_START
        assert e.req_id is not None
        assert e.employee_id is not None
        assert set(e.destination) == {"cost_center", "job_code", "location", "company"}
        assert e.destination["cost_center"].startswith("CC-")
        assert e.step >= 0


def test_declared_movements_and_programs_name_their_people() -> None:
    movements = [
        DeclaredMovement(
            EventKind.REORGANIZATION, "2025-09", (7, 8), {"cost_center": "CC-02"}, "merge"
        ),
        DeclaredMovement(EventKind.CONVERSION_OUT, "2025-07", (9,)),
    ]
    events = from_declarations(movements, RUN)
    assert [(e.kind, e.step, e.employee_id) for e in events] == [
        (EventKind.REORGANIZATION, 2, 7),
        (EventKind.REORGANIZATION, 2, 8),
        (EventKind.CONVERSION_OUT, 0, 9),
    ]
    assert events[0].destination == {"cost_center": "CC-02"}
    program = TerminationProgram("reduction", "2025-08", 3, (1, 2, 3, 4, 5))
    assert program.schedule() == [
        (1, "2025-08"),
        (2, "2025-08"),
        (3, "2025-09"),
        (4, "2025-09"),
        (5, "2025-10"),
    ]
    terminations = from_programs([program], RUN)
    assert [e.step for e in terminations] == [1, 1, 2, 2, 3]
    assert {e.reason for e in terminations} == {"program reduction"}


def test_bad_declarations_are_refused_with_every_reason() -> None:
    with pytest.raises(KnownEventError) as caught:
        from_declarations(
            [
                DeclaredMovement(EventKind.PROMOTION, "2025-13", ()),
                DeclaredMovement(EventKind.REORGANIZATION, "2025-08", (1,)),
                DeclaredMovement(EventKind.CONVERSION_OUT, "2025-08", (1,), {"cost_center": "X"}),
            ],
            RUN,
        )
    problems = caught.value.problems
    assert problems[0].startswith("item 1: a declaration can't move people by 'promotion'")
    assert problems[1:3] == [
        "item 1: '2025-13' isn't a month",
        "item 1: a declaration names at least one employee",
    ]
    assert problems[3] == "item 2: a reorganization needs a destination"
    assert problems[4].startswith("item 3: a conversion out leaves the workforce")
    with pytest.raises(KnownEventError) as caught:
        from_programs([TerminationProgram("x", "soon", 0, ())], RUN)
    assert len(caught.value.problems) == 3


def test_events_outside_the_window_are_counted_not_kept() -> None:
    inside = KnownEvent(EventKind.TERMINATION, 0, Source.DECLARED, employee_id=1)
    late = KnownEvent(EventKind.TERMINATION, 12, Source.PROGRAM, employee_id=2)
    early = KnownEvent(EventKind.TERMINATION, -1, Source.PROGRAM, employee_id=3)
    collected = collect(RUN, 12, [(Source.DECLARED, [inside]), (Source.PROGRAM, [late, early])])
    assert collected.events == (inside,)
    assert collected.outside_window == {"program": 2}


def test_known_events_hold_a_fixed_order_and_a_digest() -> None:
    a = KnownEvent(EventKind.TERMINATION, 2, Source.DECLARED, employee_id=5)
    b = KnownEvent(EventKind.PENDING_START, 0, Source.PENDING_START, employee_id=9, req_id=3)
    c = KnownEvent(EventKind.TERMINATION, 0, Source.HISTORY, employee_id=4)
    one, other = KnownEvents((a, b, c)), KnownEvents((c, a, b))
    assert one.events == other.events == (b, c, a)
    assert one.digest() == other.digest()
    assert one.digest().startswith("ke1-")
    assert KnownEvents((a, b)).digest() != one.digest()
    assert one.at(0) == [b, c]
    assert one.at(0, EventKind.TERMINATION) == [c]
    assert one.counts() == {"pending_start": 1, "termination": 2}
    assert len(one) == 3
    assert list(one.frame()["employee_id"]) == [9, 4, 5]


def test_every_source_comes_together(con: duckdb.DuckDBPyConnection) -> None:
    events = known_events(
        con,
        RUN,
        6,
        movements=[DeclaredMovement(EventKind.CONVERSION_OUT, "2026-03", (1,))],
        programs=[TerminationProgram("p", "2025-08", 2, (11, 12))],
    )
    sources = {e.source for e in events.events}
    assert sources == {Source.HISTORY, Source.PENDING_START, Source.PROGRAM}
    assert events.outside_window["declared"] == 1  # March is the run's ninth month
    assert all(0 <= e.step < 6 for e in events.events)
    quiet = known_events(con, RUN, 6, history=False, pending_starts=False)
    assert len(quiet) == 0
