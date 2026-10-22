"""Tests for the unified fills domain: categorical draws, and guided two-stage assignment."""

from __future__ import annotations

import dataclasses
from typing import Any

import numpy as np

from hcsim.engine.domains import StepContext
from hcsim.engine.fills import fills
from hcsim.engine.loop import FLOW_SYMBOLS, HEADCOUNT_IDENTITY, OPENING_IDENTITY, simulate
from hcsim.engine.seeds import SeedManager
from hcsim.engine.state import SimulationState
from hcsim.model.core.layouts import OpeningColumn
from hcsim.model.ml.entity import ConstantModel
from tests.engine.test_loop import config, guidance, initial_state, models


def channels(external: float, conversion: float, transfer: float) -> Any:
    return dataclasses.replace(
        models(fill=external, attrition=0.0, cancel=0.0),
        conversion_fill=ConstantModel(conversion),
        transfer_fill=ConstantModel(transfer),
    )


def many_openings(n: int = 300) -> SimulationState:
    """`n` one-opening requisitions in CC-01 and CC-02, alternating."""
    state = initial_state()
    state.openings[:] = 0
    reqs = state.openings[:n]
    reqs[:, OpeningColumn.REQ_ID] = 1_000 + np.arange(n)
    reqs[:, OpeningColumn.COST_CENTER] = np.arange(n) % 2
    reqs[:, OpeningColumn.NUM_OPENINGS] = 1
    return SimulationState(state.employees, state.n_employees, state.openings, n, 3)


def run_fills(state: SimulationState, seed: int = 1, **overrides: Any) -> Any:
    ctx = StepContext(config(**overrides), 0, 0, SeedManager(seed))
    return fills(state, ctx), ctx


def test_an_unguided_opening_fills_by_one_channel_at_its_hazards() -> None:
    n = 2_000  # the toy buffer's capacity
    state = many_openings(n)
    result, _ = run_fills(state, models=channels(0.3, 0.2, 0.1))
    external = int(result.flows["F_ext"].sum())
    conversion = int(result.flows["F_conv"].sum())
    transfer = sum(result.transfers.values())
    for count, share in ((external, 0.3), (conversion, 0.2), (transfer, 0.1)):
        assert abs(count / n - share) < 4 * np.sqrt(share * (1 - share) / n)
    assert external + conversion == len(result.employees.append)
    # Each requisition fills at most once: hires and transfers never share a requisition.
    hired = set(result.openings.drop.tolist())
    assert len(hired) == external + conversion
    assert all(count == 1 for count in result.transfers.values())


def test_a_carried_root_realizes_its_quota_exactly() -> None:
    state = many_openings()
    result, _ = run_fills(
        state,
        models=channels(0.3, 0.2, 0.0),
        guidance=guidance({"external_hires": {"CC-01": 7.0}}),
    )
    assert result.quotas["external_hires"].realized.tolist() == [7]
    assert int(result.flows["F_ext"][0]) == 7
    # CC-02 has no series, so its external fills draw on their own; conversion is free everywhere.
    assert int(result.flows["F_ext"][1]) > 0
    assert int(result.flows["F_conv"].sum()) > 0


def test_a_root_s_residual_pool_goes_to_its_free_children() -> None:
    state = many_openings()
    result, _ = run_fills(
        state,
        models=channels(0.3, 0.1, 0.2),
        guidance=guidance({"fills": {"CC-01": 10.0}, "hires": {"CC-01": 6.0}}),
    )
    assert result.quotas["fills"].realized.tolist() == [10]
    assert result.quotas["hires"].realized.tolist() == [6]
    hires_cc01 = int(result.flows["F_ext"][0] + result.flows["F_conv"][0])
    assert hires_cc01 == 6
    transfers_cc01 = sum(c for r, c in result.transfers.items() if (r - 1_000) % 2 == 0)
    assert transfers_cc01 == 4  # the residual pool: fills less hires go to internal starts


def test_a_carried_child_takes_its_exact_share() -> None:
    state = many_openings()
    result, _ = run_fills(
        state,
        models=channels(0.3, 0.1, 0.0),
        guidance=guidance({"hires": {"CC-01": 8.0}, "conversion_hires": {"CC-01": 2.0}}),
    )
    assert int(result.flows["F_conv"][0]) == 2
    assert int(result.flows["F_ext"][0]) == 6


def test_transfer_assignments_wait_for_matching_and_consume_nothing() -> None:
    state = many_openings(40)
    result, _ = run_fills(state, models=channels(0.0, 0.0, 1.0))
    assert sum(result.transfers.values()) == 40
    assert result.openings.is_empty
    assert len(result.employees.append) == 0


def test_conversion_fills_keep_the_identities() -> None:
    result = simulate(
        config(models=dataclasses.replace(models(), conversion_fill=ConstantModel(0.2))),
        initial_state(),
        validate=True,
    )
    f = {s: result.flow(s) for s in FLOW_SYMBOLS}
    assert int(f["F_conv"].sum()) > 0
    h, o = result.headcount, result.open_openings
    heads = sum(v * f[s] for s, v in HEADCOUNT_IDENTITY.items())
    openings = sum(v * f[s] for s, v in OPENING_IDENTITY.items())
    assert np.array_equal(h[:, 1:], h[:, :-1] + heads)
    assert np.array_equal(o[:, 1:], o[:, :-1] + openings)
