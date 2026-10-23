"""Tests for the alignment methods: which valve the plan drives, and each arm's override."""

from __future__ import annotations

import numpy as np
import pytest

from hcsim.engine.alignment import (
    VALVES,
    AlignmentMethod,
    MethodChoice,
    MethodError,
    Valve,
    supplied,
)
from hcsim.engine.config import ConfigError
from hcsim.engine.domains import Regime, StepContext, regimes
from hcsim.engine.seeds import SeedManager
from tests.engine.test_growth import grow, opened, plan
from tests.engine.test_loop import RUN_DATE, config, guidance, initial_state

M = AlignmentMethod


def test_each_method_drives_one_kind_of_valve() -> None:
    assert VALVES[M.NONE] == frozenset()
    assert VALVES[M.SUPPLY] == {Valve.SUPPLY}
    assert VALVES[M.DEMAND] == {Valve.FLOW}
    assert VALVES[M.HYBRID] == {Valve.SUPPLY, Valve.FLOW}
    assert VALVES[M.RATE_CALIBRATION] == {Valve.RATE}
    covered = np.array([True, False, True])
    assert supplied(M.SUPPLY, covered).tolist() == [True, True, True]
    assert supplied(M.HYBRID, covered).tolist() == [False, True, False]
    for method in (M.NONE, M.DEMAND, M.RATE_CALIBRATION):
        assert not supplied(method, covered).any()


#: The plan targets CC-01 and CC-02, and guidance covers CC-01's requisitions opened.
def both(method: str) -> tuple[list[int], list[float]]:
    overrides = {
        "guidance": guidance({"requisitions_opened": {"CC-01": 2.0}}),
        "plan": plan({"CC-01": 40.0, "CC-02": 40.0}),
        "alignment_method": method,
    }
    ctx = StepContext(config(**overrides), 0, 0, SeedManager(1))
    targets = ctx.config.ceiling.targets(RUN_DATE, 0)
    return regimes(initial_state(), ctx).tolist(), np.nan_to_num(targets, nan=-1).tolist()


def test_supply_seeks_the_target_wherever_the_plan_has_one() -> None:
    regime, targets = both("supply")
    assert regime == [Regime.TARGET, Regime.TARGET, Regime.RATE]
    assert targets == [40.0, 40.0, -1]


def test_hybrid_seeks_the_target_only_where_guidance_doesnt_cover_supply() -> None:
    regime, targets = both("hybrid")
    assert regime == [Regime.GUIDED, Regime.TARGET, Regime.RATE]
    assert targets == [-1, 40.0, -1]  # CC-01's plan drives flow, so it's no ceiling there


@pytest.mark.parametrize("method", ["none", "demand", "rate_calibration"])
def test_the_other_methods_leave_supply_to_the_rate_and_guidance(method: str) -> None:
    regime, targets = both(method)
    assert regime == [Regime.GUIDED, Regime.RATE, Regime.RATE]
    assert targets == [-1, -1, -1]


def test_without_supply_the_plan_bounds_nothing() -> None:
    # CC-03 draws 2 at its rate. A target of 30, below its committed 34, stops that under
    # supply, and binds nothing under the other methods.
    free, _ = grow()
    assert opened(free)[2] == 2
    held, _ = grow(plan=plan({"CC-03": 30.0}))
    assert opened(held)[2] == 0
    for method in ("none", "demand", "rate_calibration"):
        unbound, _ = grow(plan=plan({"CC-03": 30.0}), alignment_method=method)
        assert opened(unbound) == opened(free)
    capped, _ = grow(
        plan=plan({"CC-03": 30.0}), capacity_caps={"CC-03": 35}, alignment_method="none"
    )
    assert opened(capped)[2] == 1  # the cap still vetoes


def test_the_method_is_a_setting() -> None:
    assert config().alignment_method is M.SUPPLY
    assert config(alignment_method="hybrid").alignment_method is M.HYBRID
    with pytest.raises(ConfigError, match="alignment_method"):
        config(alignment_method="both")


def test_an_arm_runs_under_its_override_or_the_runs_method() -> None:
    choice = MethodChoice.from_mapping(
        {"method": "supply", "overrides": {"demand-arm": "demand", "flat": "none"}}
    )
    assert choice.resolve() is M.SUPPLY
    assert choice.resolve("demand-arm") is M.DEMAND
    assert choice.resolve("flat") is M.NONE
    assert choice.resolve("champion") is M.SUPPLY
    assert MethodChoice.from_mapping(choice.to_dict()) == choice
    assert MethodChoice.from_mapping({}) == MethodChoice()
    with pytest.raises(MethodError) as caught:
        MethodChoice.from_mapping({"method": "both", "overrides": {"a": "rates"}, "arms": ["a"]})
    assert caught.value.problems == [
        "no setting 'arms'",
        "no method 'both'; the methods are demand, hybrid, none, rate_calibration, supply",
        "arm 'a' overrides with no method 'rates'",
    ]
    with pytest.raises(MethodError, match="overrides map each arm to its method"):
        MethodChoice.from_mapping({"overrides": ["a"]})
