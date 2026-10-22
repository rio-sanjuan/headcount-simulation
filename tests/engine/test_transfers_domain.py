"""Tests for the transfers domain: role writes, grants, departures, caps, and shortfalls."""

from __future__ import annotations

import dataclasses
from typing import Any

import numpy as np

from hcsim.engine.domains import StepContext
from hcsim.engine.loop import FLOW_SYMBOLS, HEADCOUNT_IDENTITY, OPENING_IDENTITY, simulate
from hcsim.engine.results import DomainResult, Grant
from hcsim.engine.seeds import SeedManager
from hcsim.engine.transfers import ShortfallPolicy, transfers
from hcsim.model.core.layouts import EmployeeColumn, OpeningColumn
from hcsim.model.ml.entity import ConstantModel
from hcsim.model.ml.group import BackfillRateModel
from tests.engine.test_loop import config, guidance, initial_state, models

# Requisition 501 is row 1: CC-01, level code 1, three openings.


def place(assigned: dict[int, int], **overrides: Any) -> tuple[DomainResult, Any]:
    state = initial_state()
    state.employees[: state.n_employees, EmployeeColumn.PERF_RATING] = 3
    ctx = StepContext(config(**overrides), 0, 0, SeedManager(5))
    ctx.transfers.update(assigned)
    return transfers(state, ctx), state


def test_a_transfer_takes_the_role_and_keeps_its_identity() -> None:
    result, state = place({501: 1})
    (row,) = result.employees.write_index
    (moved,) = result.employees.write_rows
    before = state.live_employees[row]
    assert moved[EmployeeColumn.COST_CENTER] == 0
    assert moved[EmployeeColumn.LEVEL] == 1
    for column in (
        EmployeeColumn.EMPLOYEE_ID,
        EmployeeColumn.HIRE_STEP,
        EmployeeColumn.PERF_RATING,
        EmployeeColumn.VEST_INDEX,
        EmployeeColumn.BASE_SALARY,
    ):
        assert moved[column] == before[column]
    assert before[EmployeeColumn.COST_CENTER] != 0
    assert result.flows["F_tr"].tolist() == [1, 0, 0]
    assert int(result.flows["V_tr"].sum()) == 1
    assert result.openings.write_index.tolist() == [1]
    assert result.openings.write_rows[0, OpeningColumn.NUM_OPENINGS] == 2
    # The old row is the departure the transfer backfill reads, and a transfer grant goes out.
    assert np.array_equal(result.departures, state.live_employees[[row]])
    assert Grant("transfer", rows="written") in result.grants


def test_an_unmatched_opening_stays_open_by_default() -> None:
    result, _ = place({501: 2}, transfers_availability_factor=0.0)
    assert result.employees.is_empty
    assert result.openings.is_empty
    assert int(result.flows["F_tr"].sum()) == 0


def test_an_unmatched_opening_can_spill_to_another_channel() -> None:
    result, _ = place(
        {501: 2},
        transfers_availability_factor=0.0,
        transfers_shortfall_policy=ShortfallPolicy.SPILL,
        models=dataclasses.replace(models(fill=0.5), conversion_fill=ConstantModel(0.0)),
    )
    hires = result.employees.append
    assert len(hires) == 2
    assert (hires[:, EmployeeColumn.COST_CENTER] == 0).all()
    assert result.flows["F_ext"].tolist() == [2, 0, 0]
    assert result.openings.write_rows[0, OpeningColumn.NUM_OPENINGS] == 1


def test_guidance_caps_placements_and_records_matching_shortfalls() -> None:
    capped, _ = place({501: 3}, guidance=guidance({"internal_starts": {"CC-01": 1.0}}))
    assert len(capped.employees.write_index) == 1
    record = capped.quotas["internal_starts"]
    assert (record.quota.tolist(), record.realized.tolist()) == ([1], [1])
    short, _ = place(
        {501: 2},
        guidance=guidance({"internal_starts": {"CC-01": 2.0}}),
        transfers_availability_factor=0.0,
    )
    assert short.quotas["internal_starts"].shortfall.tolist() == [2]


def test_an_assignment_for_a_closed_requisition_is_skipped() -> None:
    result, _ = place({9_999: 1})
    assert result.employees.is_empty
    assert result.quotas == {}


def test_transfers_run_through_the_loop_and_keep_the_identities() -> None:
    built = dataclasses.replace(
        models(fill=0.2, backfill=1.0),
        transfer_fill=ConstantModel(0.3),
        backfill_transfer=BackfillRateModel.constant(1.0, 3),
    )
    result = simulate(config(models=built), initial_state(), validate=True)
    f = {s: result.flow(s) for s in FLOW_SYMBOLS}
    assert int(f["F_tr"].sum()) > 0
    assert int(f["F_tr"].sum()) == int(f["V_tr"].sum())
    assert int(f["B_tr"].sum()) > 0  # every transfer out is backfilled
    h, o = result.headcount, result.open_openings
    heads = sum(v * f[s] for s, v in HEADCOUNT_IDENTITY.items())
    openings = sum(v * f[s] for s, v in OPENING_IDENTITY.items())
    assert np.array_equal(h[:, 1:], h[:, :-1] + heads)
    assert np.array_equal(o[:, 1:], o[:, :-1] + openings)


def test_a_group_guidance_doesn_t_cover_isn_t_capped() -> None:
    result, _ = place({501: 2}, guidance=guidance({"internal_starts": {"CC-02": 0.0}}))
    assert len(result.employees.write_index) == 2
