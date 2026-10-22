"""Tests for buffer snapshots, streaming, and result digests."""

from __future__ import annotations

import dataclasses
from typing import Any

import numpy as np
import pandas as pd
import pytest

from hcsim.engine.config import ForecastScope
from hcsim.engine.digest import digest
from hcsim.engine.loop import SimulationResult, TrajectoryResult, simulate, stream
from hcsim.model.core.layouts import EmployeeColumn, OpeningColumn
from hcsim.model.core.schema import COST_CENTER, JOB_LEVEL
from tests.engine.test_loop import CCS, GUIDED, SCHEMA, config, guidance, initial_state

MONTHS = 6
STEPS = (0, 2, MONTHS)


@pytest.fixture(scope="module")
def kept() -> SimulationResult:
    return simulate(config(results_snapshot_steps=STEPS), initial_state(), validate=True)


def per_cc(rows: np.ndarray, column: int, weights: np.ndarray | None = None) -> np.ndarray:
    return np.bincount(rows[:, column].astype(np.int64), weights=weights, minlength=len(CCS))


def test_each_snapshot_agrees_with_the_stocks_at_its_step(kept: SimulationResult) -> None:
    for t in kept.trajectories:
        assert sorted(t.snapshots) == list(STEPS)
        for step, snap in t.snapshots.items():
            employees = per_cc(snap.employees, EmployeeColumn.COST_CENTER)
            assert np.array_equal(employees, t.headcount[step])
            held = snap.openings[:, OpeningColumn.NUM_OPENINGS]
            openings = per_cc(snap.openings, OpeningColumn.COST_CENTER, held)
            assert np.array_equal(openings, t.open_openings[step])


def test_the_first_snapshot_is_the_seeded_state(kept: SimulationResult) -> None:
    state = initial_state()
    for t in kept.trajectories:
        assert np.array_equal(t.snapshots[0].employees, state.live_employees)
        assert np.array_equal(t.snapshots[0].openings, state.live_openings)
        assert not np.shares_memory(t.snapshots[0].employees, t.snapshots[MONTHS].employees)


def test_a_run_keeps_no_snapshots_unless_asked() -> None:
    result = simulate(config(simulation_months=2, num_trajectories=1), initial_state())
    assert result.trajectories[0].snapshots == {}
    with pytest.raises(KeyError, match="kept no snapshot of step 0"):
        result.snapshot(0)


def test_the_snapshot_view_is_a_frame_per_buffer(kept: SimulationResult) -> None:
    employees = kept.snapshot(2)
    assert list(employees.columns) == ["trajectory", *(c.name.lower() for c in EmployeeColumn)]
    assert len(employees) == int(kept.headcount[:, 2].sum())
    openings = kept.snapshot(MONTHS, "opening", trajectory=1)
    assert set(openings["trajectory"]) == {1}
    assert openings["num_openings"].sum() == kept.open_openings[1, MONTHS].sum()
    decoded = kept.snapshot(2, decoded=True)
    assert decoded[COST_CENTER].tolist() == [CCS[int(c)] for c in employees[COST_CENTER]]
    expected = SCHEMA.decode(JOB_LEVEL, employees["level"].astype(int)).tolist()
    assert decoded["level"].tolist() == expected
    assert decoded["employee_id"].equals(employees["employee_id"])  # an id isn't encoded
    with pytest.raises(ValueError, match="no buffer 'contractor'"):
        kept.snapshot(2, "contractor")
    with pytest.raises(KeyError, match="step 1"):
        kept.snapshot(1)


# --- streaming --------------------------------------------------------------------------------


def assert_identical(a: Any, b: Any, path: str = "result") -> None:
    """Every value the same, bit for bit."""
    assert type(a) is type(b), path
    if isinstance(a, np.ndarray):
        assert (a.dtype, a.shape) == (b.dtype, b.shape), path
        assert a.tobytes() == b.tobytes(), path
    elif isinstance(a, pd.DataFrame):
        pd.testing.assert_frame_equal(a, b, check_exact=True)
    elif dataclasses.is_dataclass(a):
        for f in dataclasses.fields(a):
            assert_identical(getattr(a, f.name), getattr(b, f.name), f"{path}.{f.name}")
    elif isinstance(a, dict):
        assert list(a) == list(b), path
        for key in a:
            assert_identical(a[key], b[key], f"{path}[{key!r}]")
    elif isinstance(a, tuple | list):
        assert len(a) == len(b), path
        for k, (x, y) in enumerate(zip(a, b, strict=True)):
            assert_identical(x, y, f"{path}[{k}]")
    else:
        assert a == b, path


SCENARIOS: dict[str, dict[str, Any]] = {
    "unguided": {},
    "guided": {"guidance": guidance(GUIDED)},
    "headcount only": {"awards": None, "forecast_scope": ForecastScope.HEADCOUNT},
}


@pytest.mark.parametrize("name", SCENARIOS)
def test_a_streamed_run_is_bit_identical_to_a_standard_run(name: str) -> None:
    overrides = SCENARIOS[name] | {"results_snapshot_steps": STEPS}
    standard = simulate(config(**overrides), initial_state())
    received: list[TrajectoryResult] = []
    streamed = simulate(
        config(**overrides, results_streaming=True), initial_state(), sink=received.append
    )
    assert streamed.trajectories == ()
    assert len(received) == len(standard.trajectories)
    for held, sent in zip(standard.trajectories, received, strict=True):
        assert_identical(held, sent)
    assert streamed.streamed == standard.digests()
    assert streamed.digests() == standard.digests()
    assert streamed.unregistered == standard.unregistered


def test_a_streamed_run_holds_no_trajectories() -> None:
    result = simulate(
        config(simulation_months=2, results_streaming=True), initial_state(), sink=lambda _: None
    )
    assert len(result.streamed) == 3
    with pytest.raises(ValueError, match="holds no trajectories"):
        _ = result.headcount
    with pytest.raises(ValueError, match="holds no trajectories"):
        result.event_cells()


def test_stream_runs_each_trajectory_as_it_s_asked_for() -> None:
    cfg = config(simulation_months=3)
    standard = simulate(cfg, initial_state())
    runs = stream(cfg, initial_state())
    first = next(runs)
    assert_identical(first, standard.trajectories[0])
    assert [t.trajectory for t in runs] == [1, 2]


def test_a_sink_and_streaming_go_together() -> None:
    with pytest.raises(ValueError, match="needs a sink"):
        simulate(config(results_streaming=True), initial_state())
    with pytest.raises(ValueError, match="results_streaming"):
        simulate(config(), initial_state(), sink=lambda _: None)


# --- digests ----------------------------------------------------------------------------------


def test_a_digest_tells_apart_any_change_in_bits(kept: SimulationResult) -> None:
    t = kept.trajectories[0]
    assert digest(t) == digest(dataclasses.replace(t))
    assert digest(t) != digest(kept.trajectories[1])
    bumped = t.headcount.copy()
    bumped[-1, -1] += 1
    assert digest(t) != digest(dataclasses.replace(t, headcount=bumped))
    zero, negative_zero = np.array([0.0]), np.array([-0.0])
    assert digest(zero) != digest(negative_zero)  # equal values, different bits
    assert digest(np.array([np.nan])) == digest(np.array([np.nan]))
    assert digest(np.array([1], dtype=np.int64)) != digest(np.array([1], dtype=np.int32))
    frame = pd.DataFrame({"a": [1.0, 2.0]})
    assert digest(frame) != digest(frame.rename(columns={"a": "b"}))
    assert digest({"x": 1, "y": 2}) == digest({"y": 2, "x": 1})
    labels = pd.DataFrame({"c": pd.Categorical(["a", "b"])})
    assert digest(labels) != digest(labels.assign(c=pd.Categorical(["a", "b"], ["b", "a"])))
    assert digest(np.array(["a", None], dtype=object)) != digest(
        np.array(["a", "None"], dtype=object)
    )
