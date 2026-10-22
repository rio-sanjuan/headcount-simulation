"""Tests for resolving known events against the schema, and checking that they fired."""

from __future__ import annotations

import datetime as dt
from pathlib import Path

import duckdb
import numpy as np
import pytest

from hcsim.engine.known_events import (
    EventKind,
    KnownEvent,
    KnownEventError,
    KnownEvents,
    Source,
    known_events,
)
from hcsim.engine.known_plan import KnownEventPlan, sanitize
from hcsim.engine.known_resolution import (
    FiringLog,
    check_fired,
    jobs_from_world,
    resolve_plan,
)
from hcsim.engine.seeding import schema_from_world, seed_state
from hcsim.model.core.layouts import EmployeeColumn, OpeningColumn
from hcsim.model.core.schema import (
    COMPANY,
    COST_CENTER,
    JOB_FAMILY,
    JOB_LEVEL,
    LOCATION,
    TECH_INDICATOR,
    SimulationSchema,
)
from hcsim.synth.config import preset
from hcsim.synth.world import generate_world, write_world

SCHEMA = SimulationSchema.build(
    {
        COST_CENTER: ["CC-01", "CC-02"],
        JOB_FAMILY: ["Data", "Sales"],
        JOB_LEVEL: [1, 2],
        TECH_INDICATOR: ["tech", "other"],
        LOCATION: ["L1", "L2"],
        COMPANY: ["CO-1"],
    }
)
JOBS = {"DAT-2": {"job_family": "Data", "job_level": "2", "tech_indicator": "tech"}}


def _plan(*events: KnownEvent) -> KnownEventPlan:
    ids = [e.employee_id for e in events if e.employee_id is not None]
    return sanitize(KnownEvents(events), employees=ids, requisitions=[7])


def test_destinations_resolve_to_buffer_codes() -> None:
    reorg = KnownEvent(
        EventKind.REORGANIZATION,
        2,
        Source.DECLARED,
        employee_id=1,
        destination={"cost_center": "CC-02", "is_manager": "True"},
    )
    hire = KnownEvent(
        EventKind.PENDING_START,
        1,
        Source.PENDING_START,
        employee_id=900,
        req_id=7,
        destination={"cost_center": "CC-01", "job_code": "DAT-2", "location": "L2"},
    )
    leave = KnownEvent(EventKind.TERMINATION, 3, Source.HISTORY, employee_id=2)
    resolved = resolve_plan(_plan(reorg, hire, leave), SCHEMA, JOBS)
    (moved,) = resolved.at(2)
    assert dict(moved.codes) == {EmployeeColumn.COST_CENTER: 1, EmployeeColumn.IS_MANAGER: 1}
    (started,) = resolved.at(1, EventKind.PENDING_START)
    assert dict(started.codes) == {
        EmployeeColumn.LEVEL: SCHEMA.vocabulary(JOB_LEVEL).code(2),
        EmployeeColumn.LOCATION: 1,
        EmployeeColumn.JOB_FAMILY: 0,
        EmployeeColumn.TECH_INDICATOR: SCHEMA.vocabulary(TECH_INDICATOR).code("tech"),
        EmployeeColumn.COST_CENTER: 0,
    }
    (left,) = resolved.at(3)
    assert dict(left.codes) == {}
    assert resolved.digest().startswith("kr1-")


def test_a_digest_changes_with_any_resolved_code() -> None:
    def plan(cc: str) -> KnownEventPlan:
        return _plan(
            KnownEvent(
                EventKind.REORGANIZATION,
                0,
                Source.DECLARED,
                employee_id=1,
                destination={"cost_center": cc},
            )
        )

    one = resolve_plan(plan("CC-01"), SCHEMA).digest()
    assert one == resolve_plan(plan("CC-01"), SCHEMA).digest()
    assert one != resolve_plan(plan("CC-02"), SCHEMA).digest()


def test_every_destination_problem_is_reported_at_once() -> None:
    bad = KnownEvent(
        EventKind.REORGANIZATION,
        0,
        Source.DECLARED,
        employee_id=1,
        destination={"cost_center": "CC-09", "org_level1": "ORG-A", "is_manager": "maybe"},
    )
    hire = KnownEvent(
        EventKind.PENDING_START,
        0,
        Source.PENDING_START,
        employee_id=900,
        req_id=7,
        destination={"job_code": "NOPE"},
    )
    with pytest.raises(KnownEventError) as caught:
        resolve_plan(_plan(bad, hire), SCHEMA, JOBS)
    assert caught.value.problems == [
        "pending_start of 900 at step 0: no job 'NOPE'",
        "reorganization of 1 at step 0: cost_center has no label 'CC-09'",
        "reorganization of 1 at step 0: is_manager reads true or false, not 'maybe'",
        "reorganization of 1 at step 0: a destination can't set 'org_level1'",
    ]


def test_every_known_event_must_fire() -> None:
    a = KnownEvent(EventKind.TERMINATION, 0, Source.DECLARED, employee_id=1)
    b = KnownEvent(EventKind.CONVERSION_OUT, 2, Source.DECLARED, employee_id=2)
    resolved = resolve_plan(_plan(a, b), SCHEMA)
    log = FiringLog()
    log.record(a)
    assert log.unfired(resolved) == [b]
    with pytest.raises(KnownEventError, match=r"conversion_out of 2 at step 2 \(declared\)"):
        check_fired(resolved, log)
    log.record(b)
    check_fired(resolved, log)


def test_a_world_s_known_events_resolve(tmp_path: Path) -> None:
    path = tmp_path / "w.duckdb"
    write_world(generate_world(preset("small", seed=4)), path)
    run = dt.date(2025, 7, 1)
    with duckdb.connect(str(path), read_only=True) as con:
        schema = schema_from_world(con)
        jobs = jobs_from_world(con)
        state, _ = seed_state(con, schema, run)
        events = known_events(con, run, 12)
    employees = state.live_employees[:, EmployeeColumn.EMPLOYEE_ID].astype(np.int64).tolist()
    reqs = state.live_openings[:, OpeningColumn.REQ_ID].astype(np.int64).tolist()
    resolved = resolve_plan(sanitize(events, employees, reqs), schema, jobs)
    starts = [e for e in resolved.events if e.event.kind is EventKind.PENDING_START]
    assert starts
    assert all(EmployeeColumn.JOB_FAMILY in e.codes for e in starts)
    assert set(jobs[next(iter(jobs))]) == {"job_family", "job_level", "tech_indicator"}


def test_a_boundary_crossing_is_checked() -> None:
    def reorg(person: int, destination: dict[str, str]) -> KnownEvent:
        return KnownEvent(EventKind.REORGANIZATION, 0, Source.DECLARED, person, None, destination)

    out = resolve_plan(_plan(reorg(1, {"boundary": "out"})), SCHEMA)
    assert out.events[0].boundary == "out"
    assert dict(out.events[0].codes) == {}
    bad = [
        reorg(1, {"boundary": "out", "cost_center": "CC-01"}),
        reorg(2, {"boundary": "sideways"}),
        KnownEvent(EventKind.CONVERSION_OUT, 0, Source.DECLARED, 3, None, {"boundary": "out"}),
    ]
    with pytest.raises(KnownEventError) as caught:
        resolve_plan(_plan(*bad), SCHEMA)
    assert caught.value.problems == [
        "conversion_out of 3 at step 0: only a reorganization crosses the run's boundary",
        "reorganization of 1 at step 0: a reorganization out of the run sets nothing else",
        "reorganization of 2 at step 0: a boundary is in or out, not 'sideways'",
    ]
    # Someone from outside the seed crosses in, and needs a whole role.
    joining = sanitize(
        KnownEvents((reorg(77, {"cost_center": "CC-01"}),)), employees=[], requisitions=[]
    )
    with pytest.raises(KnownEventError, match="needs a whole role, not"):
        resolve_plan(joining, SCHEMA)
    thin = SimulationSchema.build({COST_CENTER: ["CC-01"]})
    with pytest.raises(KnownEventError, match="the schema has no location vocabulary"):
        resolve_plan(_plan(reorg(1, {"location": "L1"})), thin)
