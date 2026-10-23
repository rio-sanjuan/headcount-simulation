"""Tests for domain results and the apply step."""

from __future__ import annotations

import dataclasses

import numpy as np
import pandas as pd
import pytest
from hypothesis import given
from hypothesis import strategies as st

from hcsim.engine.apply import apply
from hcsim.engine.config import AwardSettings
from hcsim.engine.results import DomainResult, Grant, RowChange
from hcsim.engine.state import BufferCapacityError, SimulationState
from hcsim.financial.prices import StaticPrice
from hcsim.financial.ranges import RangeTable
from hcsim.financial.routing import RECORDED_COLUMNS, RecordedAwards
from hcsim.financial.sizing import NewHireGapFill
from hcsim.financial.timing import AnnualCutoff, AwardTiming, LagDistribution
from hcsim.financial.tracker import NOT_FORFEITED, HolderLedger
from hcsim.model.core.layouts import EMPLOYEE, OPENING, Buffer, EmployeeColumn, OpeningColumn
from hcsim.model.core.schema import (
    AWARD_TYPE,
    COST_CENTER,
    JOB_FAMILY,
    JOB_LEVEL,
    LOCATION,
    REQ_TYPE,
    SimulationSchema,
)

SCHEMA = SimulationSchema.build(
    {
        COST_CENTER: ["CC-01", "CC-02", "CC-03"],
        JOB_FAMILY: ["F"],
        JOB_LEVEL: [1],
        LOCATION: ["L"],
        REQ_TYPE: ["backfill", "growth"],
        AWARD_TYPE: ["new_hire", "promotion"],
    }
)
N_CC = 3


def employees(cost_centers: list[int], first_id: int | None = 100) -> Buffer:
    """Employee rows in the given cost centers. No first id marks every row as new."""
    rows = EMPLOYEE.empty(len(cost_centers))
    rows[:, EmployeeColumn.EMPLOYEE_ID] = (
        -1 if first_id is None else first_id + np.arange(len(cost_centers))
    )
    rows[:, EmployeeColumn.COST_CENTER] = cost_centers
    return rows


def openings(cost_centers: list[int], counts: list[int], first_id: int | None = 500) -> Buffer:
    rows = OPENING.empty(len(cost_centers))
    rows[:, OpeningColumn.REQ_ID] = (
        -1 if first_id is None else first_id + np.arange(len(cost_centers))
    )
    rows[:, OpeningColumn.COST_CENTER] = cost_centers
    rows[:, OpeningColumn.NUM_OPENINGS] = counts
    return rows


def make_state(capacity: int = 10, ledger: bool = False) -> SimulationState:
    """Four employees (CC-01, CC-01, CC-02, CC-02) and two requisitions (3 and 1 openings)."""
    e = EMPLOYEE.empty(capacity)
    e[:4] = employees([0, 0, 1, 1])
    e[:4, EmployeeColumn.VEST_INDEX] = np.arange(4)
    o = OPENING.empty(capacity)
    o[:2] = openings([0, 1], [3, 1])
    return SimulationState(
        e,
        4,
        o,
        2,
        N_CC,
        next_employee_id=1_000,
        next_requisition_id=2_000,
        next_vest_slot=4,
        ledger=HolderLedger() if ledger else None,
    )


def result(**changes: object) -> DomainResult:
    return DomainResult("test", **changes)  # type: ignore[arg-type]


def ids(state: SimulationState) -> list[int]:
    return state.live_employees[:, EmployeeColumn.EMPLOYEE_ID].astype(int).tolist()


def req_ids(state: SimulationState) -> list[int]:
    return state.live_openings[:, OpeningColumn.REQ_ID].astype(int).tolist()


def awards() -> AwardSettings:
    ranges = pd.DataFrame(
        [
            {
                "job_family": "F",
                "job_level": 1,
                "location": "L",
                "min_base": 100_000,
                "max_base": 120_000,
                "min_target": 150_000,
                "max_target": 170_000,
                "base_adj": 1.0,
            }
        ]
    )
    return AwardSettings(NewHireGapFill(RangeTable.from_frame(ranges, SCHEMA)), StaticPrice(100.0))


# --- results ----------------------------------------------------------------------------------


def test_an_empty_result_changes_nothing() -> None:
    state = make_state()
    before = state.clone()
    applied = apply(state, result(), step=0, schema=SCHEMA)
    assert RowChange(EMPLOYEE).is_empty
    assert np.array_equal(state.employees, before.employees)
    assert np.array_equal(state.openings, before.openings)
    assert (state.n_employees, state.n_openings) == (4, 2)
    assert state.counters_agree()
    assert (applied.forfeited_tranches, applied.recorded_tranches) == (0, 0)


def test_a_row_change_validates_its_shape() -> None:
    with pytest.raises(ValueError, match="one written row per write index"):
        RowChange(EMPLOYEE, write_index=np.array([0, 1]), write_rows=employees([0]))
    with pytest.raises(ValueError, match="both dropped and written"):
        RowChange(
            EMPLOYEE, drop=np.array([1]), write_index=np.array([1]), write_rows=employees([0])
        )
    with pytest.raises(ValueError, match="employee layout"):
        RowChange(EMPLOYEE, append=OPENING.empty(1))
    with pytest.raises(ValueError, match="their own layouts"):
        DomainResult("x", employees=RowChange(OPENING))
    with pytest.raises(ValueError, match="departures"):
        DomainResult("x", departures=OPENING.empty(1))


@pytest.mark.parametrize(
    ("change", "error"),
    [
        (RowChange(EMPLOYEE, drop=np.array([4])), IndexError),
        (RowChange(EMPLOYEE, drop=np.array([-1])), IndexError),
        (RowChange(EMPLOYEE, drop=np.array([1, 1])), ValueError),
        (RowChange(EMPLOYEE, write_index=np.array([7]), write_rows=employees([0])), IndexError),
        # A write that would swap in another employee.
        (
            RowChange(EMPLOYEE, write_index=np.array([0]), write_rows=employees([0], 999)),
            ValueError,
        ),
    ],
)
def test_indices_must_name_distinct_live_rows(change: RowChange, error: type[Exception]) -> None:
    state = make_state()
    with pytest.raises(error):
        apply(state, result(employees=change), step=0, schema=SCHEMA)
    assert state.n_employees == 4


# --- employees --------------------------------------------------------------------------------


def test_dropping_compacts_in_order_and_zeros_the_tail() -> None:
    state = make_state()
    change = RowChange(EMPLOYEE, drop=np.array([2, 0]))
    applied = apply(state, result(employees=change), step=0, schema=SCHEMA)
    assert ids(state) == [101, 103]
    assert state.headcount.tolist() == [1, 1, 0]
    assert not state.employees[2:].any()
    assert applied.dropped_employees[:, EmployeeColumn.EMPLOYEE_ID].tolist() == [102, 100]
    assert state.counters_agree()


def test_a_leaver_forfeits_only_tranches_vesting_after_the_departure_month() -> None:
    state = make_state(ledger=True)
    assert state.ledger is not None
    schedule = np.zeros((4, 72), dtype=np.int64)
    schedule[:, [2, 5]] = 10  # vests at steps 2 and 5, from an award at step -1
    state.ledger.record(np.arange(4), -1, schedule, 0, np.zeros(4, np.int64), np.zeros(4, np.int64))
    applied = apply(state, result(employees=RowChange(EMPLOYEE, drop=np.array([1]))), 3, SCHEMA)
    tranches = state.ledger.tranches()
    forfeited = tranches.forfeit_month != NOT_FORFEITED
    assert applied.forfeited_tranches == 1
    assert tranches.holder[forfeited].tolist() == [1]
    assert tranches.vest_month[forfeited].tolist() == [5]
    assert tranches.forfeit_month[forfeited].tolist() == [3]


def test_a_result_can_drop_without_forfeiting() -> None:
    state = make_state(ledger=True)
    assert state.ledger is not None
    schedule = np.zeros((4, 72), dtype=np.int64)
    schedule[:, 10] = 5
    state.ledger.record(np.arange(4), 0, schedule, 0, np.zeros(4, np.int64), np.zeros(4, np.int64))
    change = RowChange(EMPLOYEE, drop=np.array([0]))
    applied = apply(state, result(employees=change, forfeit=False), 1, SCHEMA)
    assert applied.forfeited_tranches == 0
    assert (state.ledger.tranches().forfeit_month == NOT_FORFEITED).all()


def test_writes_land_on_the_same_employee_after_earlier_drops() -> None:
    state = make_state()
    moved = state.live_employees[[3]].copy()
    moved[:, EmployeeColumn.COST_CENTER] = 2  # a transfer from CC-02 to CC-03
    change = RowChange(EMPLOYEE, drop=np.array([0]), write_index=np.array([3]), write_rows=moved)
    apply(state, result(employees=change), step=0, schema=SCHEMA)
    assert ids(state) == [101, 102, 103]
    assert state.live_employees[2, EmployeeColumn.COST_CENTER] == 2
    assert state.headcount.tolist() == [1, 1, 1]
    assert state.counters_agree()


def test_appends_get_new_ids_and_vest_slots_and_keep_existing_ids() -> None:
    state = make_state()
    hires = employees([2, 0, 2])
    hires[[0, 2], EmployeeColumn.EMPLOYEE_ID] = -1
    hires[1, EmployeeColumn.EMPLOYEE_ID] = 777  # an internal mover keeps their id
    applied = apply(state, result(employees=RowChange(EMPLOYEE, append=hires)), 0, SCHEMA)
    assert ids(state) == [100, 101, 102, 103, 1_000, 777, 1_001]
    assert state.next_employee_id == 1_002
    assert state.live_employees[4:, EmployeeColumn.VEST_INDEX].tolist() == [4, 5, 6]
    assert state.next_vest_slot == 7
    assert state.headcount.tolist() == [3, 2, 2]
    assert applied.appended_employees[:, EmployeeColumn.EMPLOYEE_ID].tolist() == [1_000, 777, 1_001]
    assert (hires[[0, 2], EmployeeColumn.EMPLOYEE_ID] == -1).all()  # the result is never mutated
    assert state.counters_agree()


def test_an_append_past_capacity_raises_before_writing() -> None:
    state = make_state(capacity=5)
    before = state.clone()
    change = RowChange(EMPLOYEE, append=employees([0, 0]))
    reqs = RowChange(OPENING, drop=np.array([0]))
    with pytest.raises(BufferCapacityError, match="employee buffer holds 5 rows"):
        apply(state, result(employees=change, openings=reqs), step=0, schema=SCHEMA)
    assert np.array_equal(state.employees, before.employees)
    assert np.array_equal(state.openings, before.openings)
    assert state.n_openings == 2


def test_drops_free_room_for_appends_in_the_same_result() -> None:
    state = make_state(capacity=5)
    change = RowChange(EMPLOYEE, drop=np.array([0]), append=employees([1, 1], first_id=None))
    apply(state, result(employees=change), step=0, schema=SCHEMA)
    assert state.n_employees == 5
    assert state.counters_agree()


# --- requisitions -----------------------------------------------------------------------------


def test_a_partial_fill_lowers_openings_but_keeps_the_row() -> None:
    state = make_state()
    filled = state.live_openings[[0]].copy()
    filled[:, OpeningColumn.NUM_OPENINGS] = 1
    change = RowChange(OPENING, write_index=np.array([0]), write_rows=filled, drop=np.array([1]))
    applied = apply(state, result(openings=change), step=0, schema=SCHEMA)
    assert req_ids(state) == [500]
    assert state.open_openings.tolist() == [1, 0, 0]
    assert state.open_rows.tolist() == [1, 0, 0]
    assert applied.dropped_openings[:, OpeningColumn.REQ_ID].tolist() == [501]
    assert state.counters_agree()


def test_new_requisitions_get_new_ids() -> None:
    state = make_state()
    new = openings([2, 2], [2, 4], first_id=None)
    apply(state, result(openings=RowChange(OPENING, append=new)), step=0, schema=SCHEMA)
    assert req_ids(state) == [500, 501, 2_000, 2_001]
    assert state.next_requisition_id == 2_002
    assert state.open_openings.tolist() == [3, 1, 6]
    assert state.open_rows.tolist() == [1, 1, 2]
    assert state.counters_agree()


# --- grants and the category ledger -----------------------------------------------------------


def test_grants_size_and_record_the_appended_employees() -> None:
    state = make_state(ledger=True)
    state.price = StaticPrice(100.0)
    hires = employees([0, 2], first_id=None)
    change = RowChange(EMPLOYEE, append=hires)
    grant = (Grant("new_hire"),)
    applied = apply(state, result(employees=change, grants=grant), 2, SCHEMA, awards())
    # Target 160k, base 110k, a 50k gap a year at a price of 100: 500 shares a year for 4 years.
    assert state.live_employees[4:, EmployeeColumn.BASE_SALARY].tolist() == [110_000, 110_000]
    assert applied.appended_employees[:, EmployeeColumn.BASE_SALARY].tolist() == [110_000] * 2
    assert state.ledger is not None
    tranches = state.ledger.tranches()
    assert applied.recorded_tranches == len(tranches) == 2 * (1 + 4 * 3)
    assert set(tranches.holder.tolist()) == {4, 5}
    assert (tranches.award_month == 2).all()
    assert tranches.vest_month.min() == 2 + 12  # the one-year cliff
    assert tranches.shares.sum() == 2 * 4 * 500
    assert set(tranches.cost_center.tolist()) == {0, 2}


def test_a_headcount_only_run_sizes_no_grants() -> None:
    state = make_state()
    change = RowChange(EMPLOYEE, append=employees([0], first_id=None))
    applied = apply(state, result(employees=change, grants=(Grant("new_hire"),)), 0, SCHEMA)
    assert applied.recorded_tranches == 0
    assert state.live_employees[4, EmployeeColumn.BASE_SALARY] == 0


def test_sizing_without_a_price_model_is_refused() -> None:
    state = make_state(ledger=True)
    change = RowChange(EMPLOYEE, append=employees([0], first_id=None))
    with pytest.raises(RuntimeError, match="price model"):
        apply(state, result(employees=change, grants=(Grant("new_hire"),)), 0, SCHEMA, awards())


def test_the_category_ledger_accumulates_across_results() -> None:
    state = make_state()
    apply(state, result(ledger={"hires": np.array([1, 0, 2])}), step=0, schema=SCHEMA)
    used = np.array([0, 1, 1])
    apply(state, result(ledger={"hires": used}), step=0, schema=SCHEMA)
    assert state.category_ledger["hires"].tolist() == [1, 1, 3]
    used[:] = 9  # the ledger holds its own copy
    assert state.category_ledger["hires"].tolist() == [1, 1, 3]


# --- the invariant ----------------------------------------------------------------------------


def _split(draw: st.DrawFn, live: int, room: int) -> tuple[list[int], list[int], int]:
    """Distinct rows to drop and to write, and how many to append within capacity."""
    order = draw(st.permutations(range(live)))
    n_drop = draw(st.integers(0, live))
    n_write = draw(st.integers(0, live - n_drop))
    return order[:n_drop], order[n_drop : n_drop + n_write], draw(st.integers(0, room + n_drop))


def _cost_centers(draw: st.DrawFn, n: int) -> list[int]:
    return draw(st.lists(st.integers(0, N_CC - 1), min_size=n, max_size=n))


@st.composite
def results(draw: st.DrawFn, state: SimulationState) -> DomainResult:
    """Random drops, transfers, and appends; partial fills, closures, and new requisitions."""
    e_drop, e_write, e_append = _split(
        draw, state.n_employees, len(state.employees) - state.n_employees
    )
    moved = state.live_employees[e_write].copy()
    moved[:, EmployeeColumn.COST_CENTER] = _cost_centers(draw, len(e_write))
    o_drop, o_write, o_append = _split(
        draw, state.n_openings, len(state.openings) - state.n_openings
    )
    edited = state.live_openings[o_write].copy()
    edited[:, OpeningColumn.COST_CENTER] = _cost_centers(draw, len(o_write))
    edited[:, OpeningColumn.NUM_OPENINGS] = draw(
        st.lists(st.integers(1, 10), min_size=len(o_write), max_size=len(o_write))
    )
    counts = draw(st.lists(st.integers(1, 10), min_size=o_append, max_size=o_append))
    return DomainResult(
        "random",
        employees=RowChange(
            EMPLOYEE,
            drop=np.array(e_drop, dtype=np.int64),
            write_index=np.array(e_write, dtype=np.int64),
            write_rows=moved,
            append=employees(_cost_centers(draw, e_append), first_id=None),
        ),
        openings=RowChange(
            OPENING,
            drop=np.array(o_drop, dtype=np.int64),
            write_index=np.array(o_write, dtype=np.int64),
            write_rows=edited,
            append=openings(_cost_centers(draw, o_append), counts, first_id=None),
        ),
    )


@given(st.data())
def test_counters_track_the_buffers_through_any_sequence_of_results(data: st.DataObject) -> None:
    state = make_state(capacity=12)
    seen_ids = set(ids(state))
    for step in range(4):
        r = data.draw(results(state))
        before = state.n_employees
        applied = apply(state, r, step, SCHEMA)
        assert state.counters_agree()
        assert state.n_employees == before - len(r.employees.drop) + len(r.employees.append)
        new_ids = applied.appended_employees[:, EmployeeColumn.EMPLOYEE_ID].astype(int).tolist()
        assert seen_ids.isdisjoint(new_ids)
        seen_ids.update(new_ids)
        assert len(set(ids(state))) == state.n_employees
        assert not state.employees[state.n_employees :].any()
        assert not state.openings[state.n_openings :].any()


def test_a_timed_grant_is_awarded_and_vests_from_its_award_month() -> None:
    timed = dataclasses.replace(
        awards(), timing={"new_hire": AwardTiming(LagDistribution.fixed(2))}
    )
    grant = (Grant("new_hire"),)

    def hire(settings: AwardSettings, rng: np.random.Generator | None = None) -> HolderLedger:
        state = make_state(ledger=True)
        state.price = StaticPrice(100.0)
        change = RowChange(EMPLOYEE, append=employees([0, 2], first_id=None))
        apply(state, result(employees=change, grants=grant), 2, SCHEMA, settings, 1, rng)
        assert state.ledger is not None
        return state.ledger

    now = hire(awards()).tranches()
    later = hire(timed, np.random.default_rng(0)).tranches()
    assert set(now.award_month.tolist()) == {2}
    assert set(later.award_month.tolist()) == {4}
    assert (later.vest_month == now.vest_month + 2).all()
    assert later.shares.sum() == now.shares.sum()
    with pytest.raises(RuntimeError, match="are timed, so need a random stream"):
        hire(timed)


def recorded(*awards_: tuple[int, str, int, int, int]) -> RecordedAwards:
    return RecordedAwards.from_frame(pd.DataFrame(awards_, columns=list(RECORDED_COLUMNS)))


def test_a_lagged_recorded_award_is_awarded_in_its_own_month() -> None:
    # The hires get ids 1000 and 1001: the first's award is recorded a month after their start,
    # and the second's three months after it, beyond the lag.
    history = recorded((1000, "new_hire", 3, 12, 40), (1001, "new_hire", 5, 12, 50))
    timing = {"new_hire": AwardTiming(LagDistribution.fixed(2))}
    settings = dataclasses.replace(
        awards(), replayed=history, replay_lag={"new_hire": 2}, timing=timing
    )
    state = make_state(ledger=True)
    state.price = StaticPrice(100.0)
    change = RowChange(EMPLOYEE, append=employees([0, 2], first_id=None))
    rng = np.random.default_rng(0)
    grant = (Grant("new_hire"),)
    applied = apply(state, result(employees=change, grants=grant), 2, SCHEMA, settings, 3, rng)
    assert applied.replayed_grants == 1
    assert state.replayed_awards == {("new_hire", 3, 1000)}
    assert state.ledger is not None
    t = state.ledger.tranches()
    first = t.holder == 4
    # The recorded award vests from the month it was recorded in, whatever the timing draws.
    assert (t.award_month[first].tolist(), t.vest_month[first].tolist()) == ([3], [15])
    assert t.shares[first].tolist() == [40]
    assert t.recorded[first].all()
    # The second hire's is sized, and awarded after the timing's lag.
    assert set(t.award_month[~first].tolist()) == {4}
    assert not t.recorded[~first].any()


def test_a_recipient_granted_twice_in_the_window_takes_each_recorded_award_once() -> None:
    # Employee 100 is promoted at steps 2, 3, and 4, with awards recorded at steps 3 and 4.
    history = recorded((100, "promotion", 3, 6, 10), (100, "promotion", 4, 6, 20))
    settings = dataclasses.replace(awards(), replayed=history, replay_lag={"promotion": 2})
    state = make_state(ledger=True)
    state.price = StaticPrice(100.0)
    unsized = []
    for step in (2, 3, 4):
        rows = state.employees[:1].copy()
        change = RowChange(EMPLOYEE, write_index=np.array([0]), write_rows=rows)
        grant = (Grant("promotion", rows="written"),)
        applied = apply(state, result(employees=change, grants=grant), step, SCHEMA, settings)
        unsized.append(applied.unsized_grants)
    assert unsized == [0, 0, 1]  # the third finds none left, and no sizer for promotions
    assert state.replayed_awards == {("promotion", 3, 100), ("promotion", 4, 100)}
    assert state.ledger is not None
    t = state.ledger.tranches()
    assert list(zip(t.award_month.tolist(), t.shares.tolist(), strict=True)) == [(3, 10), (4, 20)]


def test_an_eligibility_cutoff_leaves_out_recipients_hired_after_it() -> None:
    cutoff = dataclasses.replace(awards(), eligibility={"new_hire": AnnualCutoff(cutoff_month=1)})
    state = make_state(ledger=True)
    state.price = StaticPrice(100.0)
    state.employees[:4, EmployeeColumn.HIRE_STEP] = [-12, -1, 0, 2]
    grant = (Grant("new_hire", rows="written"),)
    change = RowChange(EMPLOYEE, write_index=np.arange(4), write_rows=state.employees[:4].copy())
    # At step 2, in March, the cutoff is January: step 0. The last two were hired at or after it.
    applied = apply(state, result(employees=change, grants=grant), 2, SCHEMA, cutoff, 3)
    assert applied.ineligible_grants == 2
    assert state.ledger is not None
    assert set(state.ledger.tranches().holder.tolist()) == {0, 1}
    assert (state.employees[2:4, EmployeeColumn.BASE_SALARY] == 0).all()


def test_a_holder_whose_cost_center_changes_moves_in_the_ledger() -> None:
    state = make_state(ledger=True)
    rows = state.employees[:2].copy()
    rows[0, EmployeeColumn.COST_CENTER] = 1  # moves from CC-01 to CC-02
    rows[1, EmployeeColumn.PERF_RATING] = 3  # a write that moves nothing
    apply(
        state,
        result(employees=RowChange(EMPLOYEE, write_index=np.arange(2), write_rows=rows)),
        5,
        SCHEMA,
    )
    assert state.ledger is not None
    moves = state.ledger.intervals()
    assert list(zip(moves.holder, moves.valid_from, moves.cost_center, strict=True)) == [(0, 5, 1)]
