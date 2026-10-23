"""Tests for the plan clamp: the effective ceiling bounds what the supply rule aims at, and
backfill counts what the ceiling withholds."""

from __future__ import annotations

import dataclasses
from typing import Any

import numpy as np

from hcsim.engine.ceiling import Ceiling
from hcsim.engine.domains import ATTRITION, BACKFILL_WITHHELD, Departures, StepContext, backfill
from hcsim.engine.seeds import SeedManager
from hcsim.engine.supply import SupplyRule
from hcsim.model.core.layouts import OpeningColumn
from tests.engine.test_ceiling import ALWAYS
from tests.engine.test_loop import RUN_DATE, SCHEMA, config, initial_state, models
from tests.engine.test_supply_rules import dated, grown

H, OPEN = np.array([30, 30, 30]), np.array([4, 4, 4])  # S = 34 in each cost center
#: CC-01's target is 40, rising to 52 in April.
RISING = dated({"CC-01": [40.0, 52.0]}, ["2026-01", "2026-04"])


def room(step: int, rule: SupplyRule, fill_yield: float = 1.0, **caps: int) -> list[float]:
    ceiling = Ceiling.build(SCHEMA, caps, RISING, None, rule, fill_yield)
    return ceiling.room(RUN_DATE, step, H, OPEN).tolist()


def test_the_plan_clamp_bounds_what_each_rule_aims_at() -> None:
    inf = float("inf")
    # Committed strength at today's target of 40.
    assert room(1, SupplyRule.STRENGTH) == [6.0, inf, inf]
    # Expected heads at today's target: 30 and 2 to come at a yield of a half, 8 short.
    assert room(1, SupplyRule.EXPECTED_FILL, 0.5) == [16.0, inf, inf]
    # Committed strength at the next date's 52.
    assert room(1, SupplyRule.SOFT_BRIDGE) == [18.0, inf, inf]
    # The cap bounds committed strength whatever the rule.
    for rule in SupplyRule:
        assert room(1, rule, 0.5, **{"CC-01": 36, "CC-02": 35}) == [2.0, 1.0, inf]
    assert Ceiling.build(SCHEMA, {}, RISING).rule is SupplyRule.STRENGTH


def test_growth_opens_what_its_rule_asks_for_under_the_clamp() -> None:
    assert grown(1, plan=RISING)[0] == 6
    half = {"requisitions_supply_rule": "expected_fill", "requisitions_fill_yield": 0.5}
    assert grown(1, plan=RISING, **half)[0] == 16
    # The soft bridge spreads the 18 short of April's 52 over the 3 months left.
    assert grown(1, plan=RISING, requisitions_supply_rule="soft_bridge")[0] == 6
    steep = dated({"CC-01": [40.0, 58.0]}, ["2026-01", "2026-04"])
    assert grown(1, plan=steep, requisitions_supply_rule="soft_bridge")[0] == 8
    assert (
        grown(1, plan=steep, requisitions_supply_rule="soft_bridge", capacity_caps={"CC-01": 39})[0]
        == 5
    )


def withheld(**overrides: Any) -> tuple[list[int], list[int]]:
    """Five departures that all draw a backfill: two in CC-01, two in CC-02, one in CC-03."""
    state = initial_state()
    run = config(models=dataclasses.replace(models(), backfill=ALWAYS), **overrides)
    ctx = StepContext(run, 0, 0, SeedManager(9))
    rows = state.live_employees[[0, 1, 30, 31, 60]].copy()
    ctx.departures.append(Departures(ATTRITION, rows, np.zeros(5, dtype=bool)))
    result = backfill(state, ctx)
    opened = np.bincount(
        result.openings.append[:, OpeningColumn.COST_CENTER].astype(int), minlength=3
    ).tolist()
    counts = result.diagnostics.get(BACKFILL_WITHHELD)
    return opened, [] if counts is None else counts.tolist()


def test_backfill_counts_what_the_ceiling_withholds() -> None:
    # CC-01's plan of 35 leaves room for one; CC-02's cap of 34 for none.
    plan = dated({"CC-01": [35.0]}, ["2026-01"])
    opened, held = withheld(plan=plan, capacity_caps={"CC-02": 34})
    assert opened == [1, 0, 1]
    assert held == [1, 2, 0]
    assert withheld() == ([2, 2, 1], [])  # nothing binds, so nothing is withheld
    # Under a method where the plan drives no supply, its target withholds nothing.
    assert withheld(plan=plan, alignment_method="none") == ([2, 2, 1], [0, 0, 0])
