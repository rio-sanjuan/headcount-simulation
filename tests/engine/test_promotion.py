"""Tests for the promotion domain."""

from __future__ import annotations

import dataclasses
from typing import Any

import numpy as np

from hcsim.engine.domains import StepContext
from hcsim.engine.eligibility import EligibilityRule
from hcsim.engine.known_events import EventKind, KnownEvent, KnownEvents, Source
from hcsim.engine.known_plan import sanitize
from hcsim.engine.known_resolution import ResolvedPlan, resolve_plan
from hcsim.engine.loop import simulate
from hcsim.engine.promotion import promotion
from hcsim.engine.seeds import SeedManager
from hcsim.model.core.layouts import EmployeeColumn
from hcsim.model.ml.entity import ConstantModel
from tests.engine.test_loop import SCHEMA, config, guidance, initial_state, models

MARCH = 2  # the run starts in January, and promotions take effect in March and September


def promote(p: float = 1.0) -> Any:
    return dataclasses.replace(models(attrition=0.0), promotion=ConstantModel(p))


def run_step(step: int = MARCH, known: ResolvedPlan | None = None, **overrides: Any) -> Any:
    overrides.setdefault("models", promote())
    ctx = StepContext(config(**overrides), step, 0, SeedManager(4), known=known)
    return promotion(initial_state(), ctx), ctx


def test_everyone_below_the_top_level_moves_up_in_a_cycle_month() -> None:
    result, _ = run_step()
    rows = initial_state().live_employees
    lower = np.flatnonzero(rows[:, EmployeeColumn.LEVEL] == 0)
    assert result.employees.write_index.tolist() == lower.tolist()
    assert (result.employees.write_rows[:, EmployeeColumn.LEVEL] == 1).all()
    assert result.flows["Pi"].tolist() == [15, 15, 15]


def test_no_one_is_sampled_outside_a_cycle_month() -> None:
    result, _ = run_step(step=0)
    assert result.employees.is_empty
    assert run_step(promotion_cycle_months=(1,), step=0)[0].employees.write_index.size == 45


def test_a_cycle_can_be_skipped_at_random() -> None:
    assert run_step(promotion_gate_probability=0.0)[0].employees.is_empty
    sizes = {
        len(
            promotion(
                initial_state(), StepContext(cfg, MARCH, j, SeedManager(4))
            ).employees.write_index
        )
        for j in range(20)
        for cfg in [config(models=promote(), promotion_gate_probability=0.5)]
    }
    assert sizes == {0, 45}


def test_known_promotions_fire_in_any_month_to_their_destination() -> None:
    event = KnownEvent(EventKind.PROMOTION, 0, Source.DECLARED, 4, destination={"job_level": "2"})
    known = resolve_plan(sanitize(KnownEvents((event,)), range(90), []), SCHEMA)
    result, ctx = run_step(step=0, known=known, models=models())
    assert result.employees.write_index.tolist() == [4]
    assert result.employees.write_rows[0, EmployeeColumn.LEVEL] == 1
    assert ctx.fired.unfired(known) == []


def test_eligibility_and_guidance_shape_who_moves() -> None:
    rule = EligibilityRule.of("promotion", cost_center=["CC-02"])
    result, _ = run_step(eligibility=(rule,))
    moved = initial_state().live_employees[result.employees.write_index]
    assert 1 not in set(moved[:, EmployeeColumn.COST_CENTER].astype(int).tolist())
    guided, _ = run_step(models=promote(0.5), guidance=guidance({"promotions": {"CC-01": 4.0}}))
    cost_centers = initial_state().live_employees[guided.employees.write_index][
        :, EmployeeColumn.COST_CENTER
    ]
    assert int((cost_centers == 0).sum()) == 4
    assert guided.quotas["promotions"].realized.tolist() == [4]


def test_promotions_run_through_the_loop() -> None:
    result = simulate(config(models=promote(0.1)), initial_state(), validate=True)
    assert "promotion" not in result.unregistered
    pi = result.flow("Pi")
    assert int(pi[:, MARCH].sum()) > 0
    assert int(pi[:, [0, 1, 3, 4, 5]].sum()) == 0
