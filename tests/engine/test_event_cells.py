"""Tests for event cells, channel counts, the diagnostics, and their views."""

from __future__ import annotations

import dataclasses

import numpy as np
import pandas as pd
import pytest

from hcsim.engine.events import BUFFERS, CELL_COLUMNS, CHANGES, EventCells
from hcsim.engine.loop import DOMAIN_INDEX, FLOW_SYMBOLS, SimulationResult
from hcsim.engine.results import DomainResult, RowChange
from hcsim.model.core.layouts import EMPLOYEE, OPENING, EmployeeColumn, OpeningColumn
from hcsim.model.core.schema import COST_CENTER, JOB_FAMILY, JOB_LEVEL
from tests.engine.test_loop import CCS, GUIDED, SCHEMA, guidance, initial_state, run


def frame_of(cells: EventCells) -> pd.DataFrame:
    return pd.DataFrame(cells.array(), columns=list(CELL_COLUMNS[1:]))


def test_a_capture_counts_each_change_by_the_rows_it_touches() -> None:
    state = initial_state()
    written = state.live_employees[[1]].copy()
    written[:, EmployeeColumn.LEVEL] = 0  # an L2 in CC-01 moves to L1
    written[:, EmployeeColumn.COST_CENTER] = 2  # and to CC-03
    opened = OPENING.empty(2)
    opened[:, OpeningColumn.COST_CENTER] = 1
    opened[:, OpeningColumn.NUM_OPENINGS] = [2, 5]
    result = DomainResult(
        "test",
        employees=RowChange(
            EMPLOYEE, drop=np.array([0, 2, 30]), write_index=np.array([1]), write_rows=written
        ),
        openings=RowChange(OPENING, drop=np.array([1]), append=opened),
    )
    cells = EventCells(SCHEMA)
    cells.capture(4, 9, result, state)
    frame = frame_of(cells)
    assert (frame["step"] == 4).all()
    assert (frame["domain"] == 9).all()
    assert frame[[COST_CENTER, JOB_LEVEL, JOB_FAMILY, "rows", "openings"]].values.tolist() == [
        [0, 0, 0, 2, 0],  # drops: two L1s in CC-01
        [1, 0, 0, 1, 0],  # and one L1 in CC-02
        [0, 1, 0, 1, 0],  # the written row as it was
        [2, 0, 0, 1, 0],  # and as it's written
        [0, 1, 0, 1, 3],  # a dropped requisition holding three openings
        [1, 0, 0, 2, 7],  # two appended requisitions, seven openings between them
    ]
    assert [CHANGES[k] for k in frame["change"]] == [
        "drop",
        "drop",
        "write_from",
        "write_to",
        "drop",
        "append",
    ]
    assert [BUFFERS[b] for b in frame["buffer"]] == ["employee"] * 4 + ["opening"] * 2


def test_an_empty_result_captures_nothing() -> None:
    cells = EventCells(SCHEMA)
    cells.capture(0, 0, DomainResult("test"), initial_state())
    assert cells.array().shape == (0, len(CELL_COLUMNS) - 1)


def test_an_unattributed_code_has_its_own_cell_and_decodes_to_a_missing_label() -> None:
    opened = OPENING.empty(1)
    opened[:, OpeningColumn.JOB_FAMILY] = -1
    cells = EventCells(SCHEMA)
    cells.capture(
        0, 0, DomainResult("x", openings=RowChange(OPENING, append=opened)), initial_state()
    )
    assert frame_of(cells)[JOB_FAMILY].tolist() == [-1]


@pytest.fixture(scope="module")
def guided() -> SimulationResult:
    return run(guidance=guidance(GUIDED))


@pytest.fixture(scope="module")
def unguided() -> SimulationResult:
    return run()


def by_step_and_cc(frame: pd.DataFrame, value: str, months: int) -> np.ndarray:
    """(months, cost centers) sums of `value` from a cell frame."""
    out = np.zeros((months, len(CCS)), dtype=np.int64)
    np.add.at(out, (frame["step"].to_numpy(), frame[COST_CENTER].to_numpy()), frame[value])
    return out


@pytest.mark.parametrize("name", ["guided", "unguided"])
def test_the_cells_reconcile_with_every_stock_s_change(
    name: str, request: pytest.FixtureRequest
) -> None:
    result: SimulationResult = request.getfixturevalue(name)
    months = result.config.simulation_months
    cells = result.event_cells()
    for t in result.trajectories:
        mine = cells[cells["trajectory"] == t.trajectory]
        for buffer, stock, value in (
            ("employee", t.headcount, "rows"),
            ("opening", t.open_openings, "openings"),
        ):
            rows = mine[mine["buffer"] == buffer]
            change = sum(
                sign * by_step_and_cc(rows[rows["change"] == kind], value, months)
                for kind, sign in (("append", 1), ("write_to", 1), ("drop", -1), ("write_from", -1))
            )
            assert np.array_equal(change, np.diff(stock, axis=0))


def test_each_single_flow_domain_s_cells_are_its_flow(guided: SimulationResult) -> None:
    months = guided.config.simulation_months
    cells = guided.event_cells(trajectory=1)
    t = guided.trajectories[1]
    for domain, buffer, change, value, symbol in (
        ("attrition", "employee", "drop", "rows", "A"),
        ("cancellation", "opening", "drop", "openings", "U"),
        ("growth", "opening", "append", "openings", "G"),
        ("backfill", "opening", "append", "openings", "B_att"),
    ):
        mine = cells[
            (cells["domain"] == domain) & (cells["buffer"] == buffer) & (cells["change"] == change)
        ]
        assert np.array_equal(by_step_and_cc(mine, value, months), t.flows[symbol]), domain
    hires = cells[(cells["domain"] == "fills") & (cells["change"] == "append")]
    assert np.array_equal(
        by_step_and_cc(hires, "rows", months), t.flows["F_ext"] + t.flows["F_conv"]
    )


def test_the_cell_frame_is_code_keyed_with_categories_for_the_engine_s_names(
    guided: SimulationResult,
) -> None:
    frame = guided.event_cells()
    assert tuple(frame.columns) == CELL_COLUMNS
    assert list(frame["domain"].cat.categories) == list(DOMAIN_INDEX)
    assert list(frame["buffer"].cat.categories) == list(BUFFERS)
    assert list(frame["change"].cat.categories) == list(CHANGES)
    assert frame[COST_CENTER].dtype == np.int64
    assert set(frame["trajectory"]) == {0, 1, 2}
    assert (frame["rows"] > 0).all()


def test_the_decoded_view_carries_the_schema_s_labels(guided: SimulationResult) -> None:
    codes, labels = guided.event_cells(), guided.event_cells(decoded=True)
    assert labels[COST_CENTER].tolist() == [CCS[c] for c in codes[COST_CENTER]]
    assert labels[JOB_LEVEL].tolist() == SCHEMA.decode(JOB_LEVEL, codes[JOB_LEVEL]).tolist()
    assert labels.drop(columns=[COST_CENTER, JOB_LEVEL, JOB_FAMILY]).equals(
        codes.drop(columns=[COST_CENTER, JOB_LEVEL, JOB_FAMILY])
    )


def test_a_view_by_trajectory_keeps_only_that_trajectory(guided: SimulationResult) -> None:
    whole = guided.event_cells()
    one = guided.event_cells(trajectory=2)
    assert set(one["trajectory"]) == {2}
    expected = whole[whole["trajectory"] == 2].reset_index(drop=True)
    assert one.reset_index(drop=True).equals(expected)
    with pytest.raises(KeyError, match="no trajectory 9"):
        guided.event_cells(trajectory=9)


def test_channel_counts_are_the_flows_in_long_form(guided: SimulationResult) -> None:
    frame = guided.channel_counts()
    assert list(frame.columns) == ["trajectory", "step", "flow", COST_CENTER, "count"]
    assert (frame["count"] != 0).all()
    assert set(frame["flow"]) <= set(FLOW_SYMBOLS)
    for symbol in FLOW_SYMBOLS:
        mine = frame[frame["flow"] == symbol]
        assert int(mine["count"].sum()) == int(guided.flow(symbol).sum()), symbol
    decoded = guided.channel_counts(decoded=True, trajectory=0)
    assert set(decoded[COST_CENTER]) <= set(CCS)
    assert set(decoded["trajectory"]) == {0}


def test_the_diagnostics_and_run_counts_views(guided: SimulationResult) -> None:
    months = guided.config.simulation_months
    degenerate = np.zeros((months, len(CCS)), dtype=np.int64)
    degenerate[2, 1] = 4
    first = dataclasses.replace(
        guided.trajectories[0],
        diagnostics={"degenerate_rows": degenerate},
        bypasses={"promotion": 3, "growth": 0},
        known_fired={"termination": 2},
        unsized_grants={"fills": 5, "growth": 0},
    )
    result = dataclasses.replace(guided, trajectories=(first, *guided.trajectories[1:]))
    frame = result.diagnostics(decoded=True)
    assert frame.values.tolist() == [[0, 2, "degenerate_rows", "CC-02", 4]]
    assert result.diagnostics(trajectory=1).empty
    counts = result.run_counts(trajectory=0)
    assert counts.values.tolist() == [
        [0, "bypassed", "promotion", 3],
        [0, "known_fired", "termination", 2],
        [0, "unsized_grants", "fills", 5],
    ]


def test_a_sized_run_leaves_no_grant_unsized(guided: SimulationResult) -> None:
    for t in guided.trajectories:
        assert set(t.unsized_grants) == set(guided.trajectories[0].unsized_grants)
        assert not any(t.unsized_grants.values())
