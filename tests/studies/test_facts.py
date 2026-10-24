"""Tests for event facts: sparse long-form rows a run and history both count."""

from __future__ import annotations

import dataclasses
import datetime as dt
import re
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
import pytest

from hcsim.engine.loop import SimulationResult
from hcsim.engine.seeding import schema_from_world
from hcsim.model.core.categories import GuidanceCategory
from hcsim.model.core.schema import COST_CENTER
from hcsim.studies.facts import (
    ACTUAL,
    BY_NAME,
    COUNT,
    EVENT,
    EVENTS,
    FACT_COLUMNS,
    STEP,
    TRAJECTORY,
    Event,
    FactError,
    actual_facts,
    check_events,
    dense,
    empty,
    mean,
    run_facts,
)
from hcsim.studies.harness import StudyRun, run_study
from hcsim.studies.profile import StudyProfile

PAST = dt.date(2025, 7, 1)
MONTHS = 4
#: The events history counts from month-end snapshots, which a replay needn't match.
SNAPSHOTTED = {"requisitions_opened"}


def study(world: Path, **fields: object) -> StudyRun:
    profile = StudyProfile.from_mapping(
        {
            "name": "facts",
            "seed": 7,
            "trajectories": 3,
            "months": MONTHS,
            "run_date": PAST.isoformat(),
            "scope": "hcf",
            "mode": "unguided",
        }
        | fields
    )
    return run_study(profile, world)


@pytest.fixture(scope="module")
def unguided(inputs: tuple[Path, Path]) -> StudyRun:
    return study(inputs[0])


@pytest.fixture(scope="module")
def history(inputs: tuple[Path, Path]) -> pd.DataFrame:
    with duckdb.connect(str(inputs[0]), read_only=True) as con:
        return actual_facts(con, schema_from_world(con), PAST, MONTHS)


def test_the_events_are_one_vocabulary() -> None:
    assert check_events(EVENTS) == []
    assert [e.name for e in EVENTS] == list(BY_NAME)
    bad = [
        Event("a", ("A", "Z"), (GuidanceCategory.ATTRITION,)),
        Event("b", ("A",), (GuidanceCategory.ATTRITION, GuidanceCategory.HEADCOUNT)),
    ]
    assert check_events(bad) == [
        "no run records the flows ['Z']",
        "two events count the flows ['A']",
        "two events count the categories ['attrition']",
        "the categories ['headcount'] are stocks, not events",
    ]


def test_a_runs_facts_are_its_flows_sparse(unguided: StudyRun) -> None:
    result = unguided.simulated
    facts = run_facts(result)
    assert tuple(facts.columns) == FACT_COLUMNS
    assert (facts[COUNT] != 0).all()  # sparse: a zero has no row
    assert not facts.duplicated([TRAJECTORY, STEP, COST_CENTER, EVENT]).any()
    assert facts.equals(facts.sort_values(list(FACT_COLUMNS[:4]), ignore_index=True))
    cost_centers = result.headcount.shape[2]
    arrays = dense(facts, [t.trajectory for t in result.trajectories], MONTHS, cost_centers)
    for event in EVENTS:
        expected = sum(result.flow(f) for f in event.flows)
        assert np.array_equal(arrays[event.name], expected), event.name
    total = sum(int(np.count_nonzero(a)) for a in arrays.values())
    assert len(facts) == total


def test_a_mean_counts_trajectories_without_a_fact_as_zero(unguided: StudyRun) -> None:
    result = unguided.simulated
    facts = run_facts(result)
    means = mean(facts, len(result.trajectories), [EVENT, STEP])
    expected = result.flow("A").sum(axis=2).mean(axis=0)
    got = [means.get(("attrition", s), 0.0) for s in range(MONTHS)]
    assert got == pytest.approx(expected)
    with pytest.raises(FactError, match="at least one trajectory"):
        mean(facts, 0, [EVENT])


def test_historys_facts_are_the_realized_totals(history: pd.DataFrame) -> None:
    assert tuple(history.columns) == FACT_COLUMNS
    assert (history[TRAJECTORY] == ACTUAL).all()
    assert (history[COUNT] != 0).all()
    assert set(history[STEP]) <= set(range(MONTHS))
    assert (history[COST_CENTER] >= 0).all()  # the tiny world's cost centers are all in schema
    totals = history.groupby(EVENT, observed=True)[COUNT].sum()
    assert totals["attrition"] > 0
    assert totals["hires"] > 0


def test_history_can_count_some_events(inputs: tuple[Path, Path], history: pd.DataFrame) -> None:
    with duckdb.connect(str(inputs[0]), read_only=True) as con:
        schema = schema_from_world(con)
        hires = actual_facts(con, schema, PAST, MONTHS, ["hires"])
        with pytest.raises(FactError, match=re.escape("no events ['departures']")):
            actual_facts(con, schema, PAST, MONTHS, ["departures"])
    pd.testing.assert_frame_equal(
        hires.reset_index(drop=True),
        history[history[EVENT] == "hires"].reset_index(drop=True),
        check_categorical=False,
    )


def test_a_replay_counts_what_history_counts(
    inputs: tuple[Path, Path], history: pd.DataFrame
) -> None:
    replayed = study(inputs[0], mode="replay", trajectories=2)
    facts = run_facts(replayed.simulated)
    keys = [EVENT, STEP, COST_CENTER]
    for trajectory, rows in facts.groupby(TRAJECTORY):
        ran = rows.groupby(keys, observed=True)[COUNT].sum()
        happened = history.groupby(keys, observed=True)[COUNT].sum()
        both = pd.concat({"ran": ran, "happened": happened}, axis=1).fillna(0)
        for event in set(BY_NAME) - SNAPSHOTTED:
            cells = both.xs(event, level=EVENT)
            assert (cells["ran"] == cells["happened"]).all(), (trajectory, event)


def test_dense_refuses_facts_it_cant_place(unguided: StudyRun) -> None:
    facts = run_facts(unguided.simulated)
    with pytest.raises(FactError, match="outside those given"):
        dense(facts, [0], MONTHS, 10)
    with pytest.raises(FactError, match="cost centers outside the schema"):
        dense(facts, [0, 1, 2], MONTHS, 1)
    zeros = dense(empty(), [0], MONTHS, 2)
    assert set(zeros) == set(BY_NAME)
    assert all(a.shape == (1, MONTHS, 2) and not a.any() for a in zeros.values())


def test_a_streamed_run_has_no_facts_to_count(unguided: StudyRun) -> None:
    streamed = dataclasses.replace(unguided.simulated, trajectories=(), streamed=("rd1-x",))
    with pytest.raises(FactError, match="streamed run"):
        run_facts(streamed)
    nothing = SimulationResult(unguided.config, ())
    assert run_facts(nothing).empty
