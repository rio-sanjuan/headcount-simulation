"""Tests for consuming aligned guidance: the declared movements its alignment counted, checked
against the run's."""

from __future__ import annotations

import dataclasses
from pathlib import Path

import pytest

from hcsim.engine.alignment import declared_movements, movement_refusals
from hcsim.engine.config import ConfigError
from hcsim.engine.known_events import EventKind, KnownEvent, KnownEvents, Source
from hcsim.engine.known_plan import KnownEventPlan
from hcsim.engine.known_resolution import ResolvedEvent, ResolvedPlan
from hcsim.engine.loop import simulate
from hcsim.model.core.guidance import GuidanceArtifactError, GuidanceForecast, movements_digest
from hcsim.model.core.layouts import EmployeeColumn
from tests.engine.test_growth import plan
from tests.engine.test_loop import config, guidance, initial_state

CC = EmployeeColumn.COST_CENTER
TARGET = plan({"CC-01": 30.0, "CC-02": 30.0})


def declared(kind: EventKind, step: int, employee: int | None, **resolved: object) -> ResolvedEvent:
    return ResolvedEvent(KnownEvent(kind, step, Source.DECLARED, employee), **resolved)  # type: ignore[arg-type]


#: Employees 0-29 sit in CC-01 and 30-59 in CC-02; guidance groups only those two.
KNOWN = ResolvedPlan(
    KnownEventPlan(KnownEvents(())),
    (
        declared(EventKind.REORGANIZATION, 1, 0, codes={CC: 1}),  # CC-01 to CC-02
        declared(EventKind.CONVERSION_OUT, 2, 30),
        declared(EventKind.REORGANIZATION, 2, 1, codes={CC: 0}),  # within CC-01: nets to nothing
        declared(EventKind.REORGANIZATION, 3, None, codes={CC: 2}, boundary="in"),  # ungrouped
        ResolvedEvent(KnownEvent(EventKind.TERMINATION, 1, Source.HISTORY, 2)),  # not declared
    ),
)


#: What the run itself fires: the moves in and out of the guided cost centers.
FIRED = ResolvedPlan(
    KnownEventPlan(KnownEvents(tuple(e.event for e in KNOWN.events[:3]))), KNOWN.events[:3]
)


def aligned(movements: str | None = None) -> GuidanceForecast:
    covering = guidance({"attrition": {"CC-01": 1.0, "CC-02": 1.0}}, horizon=6)
    return dataclasses.replace(covering, aligned_to=TARGET.digest(), movements=movements)


def test_a_movement_digest_is_canonical() -> None:
    empty = movements_digest(None)
    assert movements_digest({}) == empty == movements_digest({"A": [0.0, 0.0]})
    assert movements_digest({"A": [1.0], "B": [-1.0]}) == movements_digest(
        {"B": [-1.0], "A": [1.0]}
    )
    assert movements_digest({"A": [1.0]}) != movements_digest({"A": [-1.0]})
    assert movements_digest({"A": [1.0000001]}) == movements_digest({"A": [1.0]})


def test_the_stamp_rides_with_the_artifact(tmp_path: Path) -> None:
    stamped = aligned(movements_digest({"CC-01": [1.0]}))
    stamped.save(tmp_path / "g")
    assert GuidanceForecast.load(tmp_path / "g").movements == stamped.movements
    assert stamped.fingerprint() != aligned(movements_digest(None)).fingerprint()
    with pytest.raises(GuidanceArtifactError, match="can't combine"):
        GuidanceForecast.combine([stamped, aligned(movements_digest(None))])
    with pytest.raises(GuidanceArtifactError, match="movements must be a dm1- digest"):
        aligned("dm1-nothex")
    with pytest.raises(GuidanceArtifactError, match="only an aligned forecast records"):
        dataclasses.replace(
            guidance({"attrition": {"CC-01": 1.0}}), movements=movements_digest(None)
        )


def test_a_runs_declared_movements_are_each_groups_net_change_by_month() -> None:
    run = config(guidance=aligned(), plan=TARGET, alignment_method="demand")
    assert run.aligner is not None
    found = declared_movements(KNOWN, initial_state().live_employees, run.guidance, run.aligner)  # type: ignore[arg-type]
    assert found == {
        "CC-01": [0.0, -1.0, 0.0, 0.0, 0.0, 0.0],
        "CC-02": [0.0, 1.0, -1.0, 0.0, 0.0, 0.0],
    }
    assert declared_movements(None, initial_state().live_employees, run.guidance, run.aligner) == {}  # type: ignore[arg-type]


def test_guidance_aligned_under_other_movements_is_refused() -> None:
    state = initial_state()
    unstamped = config(
        guidance=aligned(), plan=TARGET, alignment_method="demand", num_trajectories=1
    )
    with pytest.raises(ConfigError, match="aligned under declared movements dm1-"):
        simulate(unstamped, state, known=FIRED)
    simulate(unstamped, state)  # declaring none, as the guidance counted
    assert unstamped.guidance is not None
    assert unstamped.aligner is not None
    ours = declared_movements(FIRED, state.live_employees, unstamped.guidance, unstamped.aligner)
    matching = dataclasses.replace(unstamped, guidance=aligned(movements_digest(ours)))
    assert len(simulate(matching, state, known=FIRED).trajectories) == 1
    supply = config(plan=TARGET)
    assert movement_refusals(supply.alignment_method, None, None, FIRED, state.live_employees) == []
