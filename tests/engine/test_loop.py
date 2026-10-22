"""Tests for the domain handlers and the step loop, on a toy three-cost-center state."""

from __future__ import annotations

import datetime as dt
from typing import Any

import numpy as np
import pandas as pd
import pytest

from hcsim.engine import loop
from hcsim.engine.apply import apply
from hcsim.engine.config import AwardSettings, DomainModels, ForecastScope, SimulationConfig
from hcsim.engine.domains import (
    ATTRITION,
    Departures,
    StepContext,
    backfill,
    cancellation,
    growth,
)
from hcsim.engine.fills import fills
from hcsim.engine.loop import (
    FLOW_SYMBOLS,
    HEADCOUNT_IDENTITY,
    InvariantError,
    SimulationResult,
    calendar_month,
    simulate,
)
from hcsim.engine.results import DomainResult
from hcsim.engine.seeds import SeedManager
from hcsim.engine.state import SimulationState
from hcsim.financial.prices import AnnualGrowthPrice
from hcsim.financial.ranges import RangeTable
from hcsim.financial.sizing import NewHireGapFill
from hcsim.financial.tracker import HolderLedger
from hcsim.model.core.categories import GuidanceCategory
from hcsim.model.core.guidance import GuidanceForecast
from hcsim.model.core.layouts import EMPLOYEE, OPENING, EmployeeColumn, OpeningColumn
from hcsim.model.core.schema import (
    AWARD_TYPE,
    COST_CENTER,
    JOB_FAMILY,
    JOB_LEVEL,
    LOCATION,
    REQ_TYPE,
    SimulationSchema,
)
from hcsim.model.ml.entity import ConstantModel
from hcsim.model.ml.generative import EmpiricalAttributeGenerator
from hcsim.model.ml.group import BackfillRateModel, PoissonGrowthModel
from hcsim.model.ml.sampling import apportion
from tests.support.stat_check import bernoulli_sum_se, stat_check

RUN_DATE = dt.date(2026, 1, 1)
SNAPSHOT = "2025-12-31"
CCS = ["CC-01", "CC-02", "CC-03"]
SCHEMA = SimulationSchema.build(
    {
        COST_CENTER: CCS,
        JOB_FAMILY: ["F"],
        JOB_LEVEL: [1, 2],
        LOCATION: ["L"],
        REQ_TYPE: ["backfill", "campus", "growth"],
        AWARD_TYPE: ["new_hire"],
    }
)
CAPACITY = 2_000


def initial_state() -> SimulationState:
    """30 employees per cost center, alternating levels, and two requisitions per cost center
    with one and three openings."""
    employees = EMPLOYEE.empty(CAPACITY)
    live = employees[:90]
    live[:, EmployeeColumn.EMPLOYEE_ID] = np.arange(90)
    live[:, EmployeeColumn.COST_CENTER] = np.repeat([0, 1, 2], 30)
    live[:, EmployeeColumn.LEVEL] = np.arange(90) % 2
    live[:, EmployeeColumn.VEST_INDEX] = np.arange(90)
    openings = OPENING.empty(CAPACITY)
    reqs = openings[:6]
    reqs[:, OpeningColumn.REQ_ID] = 500 + np.arange(6)
    reqs[:, OpeningColumn.COST_CENTER] = [0, 0, 1, 1, 2, 2]
    reqs[:, OpeningColumn.LEVEL] = [0, 1, 0, 1, 0, 1]
    reqs[:, OpeningColumn.NUM_OPENINGS] = [1, 3, 1, 3, 1, 3]
    return SimulationState(
        employees,
        90,
        openings,
        6,
        3,
        next_employee_id=1_000,
        next_requisition_id=2_000,
        next_vest_slot=90,
    )


def models(
    attrition: float = 0.05, fill: float = 0.3, cancel: float = 0.05, backfill: float = 0.6
) -> DomainModels:
    attributes = np.array([[1, 0, 0, 0, 0, 0], [0, 0, 0, 0, 1, 0], [1, 0, 0, 0, 0, 0]])
    return DomainModels(
        attrition=ConstantModel(attrition),
        external_fill=ConstantModel(fill),
        cancellation=ConstantModel(cancel),
        backfill=BackfillRateModel.constant(backfill, 3),
        growth=PoissonGrowthModel(3, min_months=1).fit(np.array([[2.0, 1.0, 0.5]])),
        attributes=EmpiricalAttributeGenerator(min_rows=1).fit(np.arange(3), attributes),
    )


def awards() -> AwardSettings:
    ranges = pd.DataFrame(
        [
            {
                "job_family": "F",
                "job_level": level,
                "location": "L",
                "min_base": 100_000 * level,
                "max_base": 120_000 * level,
                "min_target": 150_000 * level,
                "max_target": 170_000 * level,
                "base_adj": 1.0,
            }
            for level in (1, 2)
        ]
    )
    price = AnnualGrowthPrice(100.0, growth_rate=1.1, growth_month=3)
    return AwardSettings(NewHireGapFill(RangeTable.from_frame(ranges, SCHEMA)), price)


def guidance(cells: dict[str, dict[str, float]], horizon: int = 12) -> GuidanceForecast:
    rows = [
        {"forecast_month": t, "category": category, "cost_center": cc, "value": value}
        for category, by_cc in cells.items()
        for cc, value in by_cc.items()
        for t in range(horizon)
    ]
    return GuidanceForecast(pd.DataFrame(rows), COST_CENTER, SNAPSHOT, horizon)


GUIDED = {
    "attrition": {"CC-01": 2.0, "CC-02": 1.5},  # CC-03 runs unguided for attrition
    "external_hires": {
        "CC-01": 1.0,
        "CC-02": 2.0,
        "CC-03": 50.0,
    },  # CC-03's quota outruns its openings
    "requisitions_opened": {"CC-01": 3.0, "CC-02": 2.0},
}


def config(**overrides: Any) -> SimulationConfig:
    args: dict[str, Any] = {
        "run_date": RUN_DATE,
        "schema": SCHEMA,
        "models": models(),
        "simulation_months": 6,
        "num_trajectories": 3,
        "seed": 7,
        "awards": awards(),
    } | overrides
    return SimulationConfig(**args)


def run(**overrides: Any) -> SimulationResult:
    return simulate(config(**overrides), initial_state(), validate=True)


def context(state: SimulationState, cfg: SimulationConfig, step: int = 0) -> StepContext:
    return StepContext(cfg, step, 0, SeedManager(1))


# --- handlers ---------------------------------------------------------------------------------


def test_a_fill_of_every_opening_closes_the_requisition_and_hires_into_its_role() -> None:
    state = initial_state()
    result = fills(state, context(state, config(models=models(fill=1.0))))
    assert result.openings.drop.tolist() == list(range(6))
    hires = result.employees.append
    assert len(hires) == 12
    assert (hires[:, EmployeeColumn.EMPLOYEE_ID] == -1).all()
    assert hires[:, EmployeeColumn.COST_CENTER].tolist() == [0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2]
    assert hires[:, EmployeeColumn.LEVEL].tolist() == [0, 1, 1, 1] * 3
    assert [g.award_type for g in result.grants] == ["new_hire"]


def test_a_guided_partial_fill_keeps_the_requisition_with_fewer_openings() -> None:
    state = initial_state()
    cfg = config(models=models(fill=1.0), guidance=guidance({"external_hires": {"CC-01": 2.0}}))
    result = fills(state, context(state, cfg))
    # CC-01 fills exactly two of its four openings; the unguided cost centers fill everything.
    hires = result.employees.append[:, EmployeeColumn.COST_CENTER]
    assert np.bincount(hires.astype(int)).tolist() == [2, 4, 4]
    (record,) = result.quotas.values()
    assert (record.quota.tolist(), record.eligible.tolist(), record.realized.tolist()) == (
        [2],
        [4],
        [2],
    )
    # Whichever openings filled, CC-01's three-opening requisition stays open with what's left.
    assert 1 in result.openings.write_index.tolist()
    apply(state, result, 0, SCHEMA)
    assert state.open_openings.tolist() == [2, 0, 0]
    assert state.counters_agree()


def test_backfill_copies_the_departed_role_and_respects_the_headroom() -> None:
    state = initial_state()
    cfg = config(
        models=models().__class__(
            **{**models().__dict__, "backfill": BackfillRateModel.constant(1.0, 3)}
        ),
        guidance=guidance({"requisitions_opened": {"CC-01": 2.0}}),
    )
    ctx = context(state, cfg)
    departed = state.live_employees[[0, 1, 2, 3, 30, 31]].copy()
    ctx.departures.append(Departures(ATTRITION, departed, np.zeros(6, dtype=bool)))
    state.category_ledger = {"requisitions_opened": np.array([1])}  # one of two already used
    result = backfill(state, ctx)
    rows = result.openings.append
    # CC-01 has one opening of headroom left; CC-02 is unguided and backfills both departures.
    assert rows[:, OpeningColumn.COST_CENTER].tolist() == [0, 1, 1]
    assert rows[:, OpeningColumn.LEVEL].tolist()[1:] == [0, 1]
    assert (rows[:, OpeningColumn.REQ_TYPE] == SCHEMA.vocabulary(REQ_TYPE).code("backfill")).all()
    assert (rows[:, OpeningColumn.NUM_OPENINGS] == 1).all()
    assert result.ledger["requisitions_opened"].tolist() == [1]


def test_backfill_without_departures_opens_nothing() -> None:
    state = initial_state()
    assert backfill(state, context(state, config())).openings.is_empty


def test_growth_opens_the_residual_where_guided_and_poisson_counts_elsewhere() -> None:
    state = initial_state()
    cfg = config(guidance=guidance({"requisitions_opened": {"CC-01": 3.0, "CC-02": 1.0}}))
    state.category_ledger = {"requisitions_opened": np.array([1, 2])}
    result = growth(state, context(state, cfg))
    rows = result.openings.append
    by_cc = np.bincount(rows[:, OpeningColumn.COST_CENTER].astype(int), minlength=3)
    assert by_cc[:2].tolist() == [2, 0]  # CC-02 already used more than its quota
    assert result.ledger["requisitions_opened"].tolist() == [2, 0]
    record = result.quotas["requisitions_opened"]
    assert record.realized.tolist() == [3, 2]
    # Roles come from the generator: CC-01's one observed profile is level code 1.
    assert (rows[rows[:, OpeningColumn.COST_CENTER] == 0, OpeningColumn.LEVEL] == 1).all()
    assert (rows[:, OpeningColumn.REQ_TYPE] == SCHEMA.vocabulary(REQ_TYPE).code("growth")).all()


def test_cancellation_drops_whole_requisitions() -> None:
    state = initial_state()
    result = cancellation(state, context(state, config(models=models(cancel=1.0))))
    assert result.openings.drop.tolist() == list(range(6))


def test_a_guided_handler_without_guidance_refuses() -> None:
    state = initial_state()
    with pytest.raises(RuntimeError, match="needs guidance"):
        context(state, config()).quota(GuidanceCategory.EXTERNAL_HIRES)


# --- the loop ---------------------------------------------------------------------------------


@pytest.mark.parametrize("cells", [None, GUIDED], ids=["unguided", "guided"])
def test_the_stock_flow_identities_hold_exactly(cells: dict[str, dict[str, float]] | None) -> None:
    result = run(guidance=None if cells is None else guidance(cells))
    h, o = result.headcount, result.open_openings
    f = {symbol: result.flow(symbol) for symbol in FLOW_SYMBOLS}
    change = sum(sign * f[symbol] for symbol, sign in HEADCOUNT_IDENTITY.items())
    assert np.array_equal(h[:, 1:], h[:, :-1] + change)
    assert np.array_equal(o[:, 1:], o[:, :-1] + f["G"] + f["B_att"] - f["U"] - f["F_ext"])
    assert h.shape == (3, 7, 3)
    assert (h[:, 0] == [30, 30, 30]).all()


def test_guided_draws_realize_their_quotas_exactly() -> None:
    artifact = guidance(GUIDED)
    result = run(guidance=artifact, simulation_months=12, num_trajectories=4)
    rows = [row for t in result.trajectories for row in t.quotas]
    assert {(r.domain, r.category) for r in rows} == {
        ("attrition", "attrition"),
        ("fills", "external_hires"),
        ("growth", "requisitions_opened"),
    }
    binding = 0
    for row in rows:
        r = row.record
        assert np.array_equal(r.realized, np.minimum(r.quota, r.eligible))
        binding += int((r.eligible < r.quota).sum())
    assert binding  # CC-03's external hires quota outruns its openings

    # Requisitions opened in a guided cost center equal its quota every month.
    opened = result.flow("B_att") + result.flow("G")
    quota = apportion(
        artifact.get_quota(
            0, next(c for c in artifact.categories if c.value == "requisitions_opened")
        )
    )
    assert (opened[:, :, :2] == quota[:2]).all()
    # Attrition in CC-01 is exactly its quota of two a month.
    assert (result.flow("A")[:, :, 0] == 2).all()


def test_unguided_rates_match_the_models() -> None:
    result = run(
        simulation_months=12,
        num_trajectories=20,
        awards=None,
        forecast_scope=ForecastScope.HEADCOUNT,
    )
    at_risk = int(result.headcount[:, :-1].sum())
    departures = int(result.flow("A").sum())
    stat_check(departures, 0.05 * at_risk, bernoulli_sum_se([0.05] * at_risk), label="attrition")
    backfills = int(result.flow("B_att").sum())
    stat_check(backfills, 0.6 * departures, bernoulli_sum_se([0.6] * departures), label="backfill")


def test_a_run_is_reproducible_and_trajectories_are_independent_of_the_count() -> None:
    first, second = run(guidance=guidance(GUIDED)), run(guidance=guidance(GUIDED))
    assert np.array_equal(first.headcount, second.headcount)
    assert np.array_equal(first.open_openings, second.open_openings)
    for a, b in zip(first.trajectories, second.trajectories, strict=True):
        assert all(np.array_equal(a.flows[s], b.flows[s]) for s in FLOW_SYMBOLS)
        assert a.sbc is not None
        assert b.sbc is not None
        pd.testing.assert_frame_equal(a.sbc, b.sbc)
    fewer = run(guidance=guidance(GUIDED), num_trajectories=2)
    assert np.array_equal(fewer.headcount, first.headcount[:2])
    other = run(guidance=guidance(GUIDED), seed=8)
    assert not np.array_equal(other.headcount, first.headcount)
    # Trajectories differ from each other.
    assert not np.array_equal(first.headcount[0], first.headcount[1])


def test_the_seeded_state_is_left_untouched() -> None:
    state = initial_state()
    before = state.clone()
    simulate(config(), state)
    assert np.array_equal(state.employees, before.employees)
    assert np.array_equal(state.openings, before.openings)
    assert state.ledger is None
    assert state.price is None


def test_sbc_values_each_grant_at_its_award_month_price_and_ties_out() -> None:
    result = run()
    for t in result.trajectories:
        # Growth applies in March, the third step from January, and never on the first step.
        assert t.prices.tolist() == pytest.approx([100, 100, 110, 110, 110, 110])
        assert t.tranches is not None
        assert t.expense is not None
        assert t.sbc is not None
        hires = int(t.flows["F_ext"].sum())
        assert len(set(t.tranches.holder.tolist())) == hires
        assert t.tranches.holder.min() >= 90  # vest slots of the new hires only
        assert t.expense.max_identity_residual <= 0.01
        assert t.sbc["month"].min() == t.tranches.award_month.min()


def test_a_headcount_only_run_computes_no_sbc() -> None:
    result = run(awards=None, forecast_scope=ForecastScope.HEADCOUNT)
    for t in result.trajectories:
        assert t.tranches is None
        assert t.expense is None
        assert t.sbc is None
        assert len(t.prices) == 0


@pytest.mark.parametrize(("month", "expected"), [(1, [1, 2, 3, 4]), (11, [11, 12, 1, 2])])
def test_calendar_months_wrap_at_year_end(month: int, expected: list[int]) -> None:
    start = dt.date(2026, month, 1)
    assert [calendar_month(start, step) for step in range(4)] == expected


def test_validation_catches_a_handler_that_writes_state(monkeypatch: pytest.MonkeyPatch) -> None:
    def rogue(state: SimulationState, ctx: StepContext) -> DomainResult:
        state.employees[state.n_employees, EmployeeColumn.COST_CENTER] = 1
        state.n_employees += 1  # a write that bypasses the apply step
        return DomainResult("rogue")

    monkeypatch.setattr(loop, "STEP_ORDER", ((ATTRITION, rogue),))
    with pytest.raises(InvariantError, match="after attrition at step 0"):
        simulate(config(), initial_state(), validate=True)


def test_tranches_awarded_before_the_run_are_refused() -> None:
    state = initial_state()
    state.ledger = HolderLedger()
    schedule = np.zeros((1, 72), dtype=np.int64)
    schedule[0, 12] = 10
    state.ledger.record(
        np.array([0]), -1, schedule, 0, np.zeros(1, np.int64), np.zeros(1, np.int64)
    )
    with pytest.raises(ValueError, match="awarded during the run"):
        simulate(config(), state)
