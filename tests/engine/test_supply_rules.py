"""Tests for simulation supply: growth's three routes, and the supply rules on the plan's."""

from __future__ import annotations

import dataclasses
from typing import Any

import numpy as np
import pytest

from hcsim.engine.config import ConfigError
from hcsim.engine.domains import ROUTES, Regime, StepContext, growth
from hcsim.engine.loop import simulate
from hcsim.engine.seeds import SeedManager
from hcsim.engine.supply import Route, expected_fill, soft_bridge, strength
from hcsim.model.core.alignment import AlignmentTarget, Interpolation, marginal
from tests.engine.test_growth import opened, plan
from tests.engine.test_loop import RUN_DATE, config, guidance, initial_state

H, OPEN = np.array([30, 30, 30]), np.array([4, 4, 4])  # S = 34 in each cost center
NAN = np.nan


def test_strength_closes_the_gain_of_the_gap_to_committed_strength() -> None:
    target = np.array([40.0, NAN, 30.0])
    assert strength(target, H, OPEN, 1.0).tolist() == [6, 0, 0]
    assert strength(target, H, OPEN, 0.5).tolist() == [3, 0, 0]


def test_expected_fill_counts_an_opening_as_the_head_it_yields() -> None:
    target = np.array([40.0, NAN, 33.0])
    # Four openings at a yield of a half are two heads to come: 8 short, so 16 openings.
    assert expected_fill(target, H, OPEN, 1.0, 0.5).tolist() == [16, 0, 2]
    assert (
        expected_fill(target, H, OPEN, 1.0, 1.0).tolist() == strength(target, H, OPEN, 1.0).tolist()
    )
    assert expected_fill(target, H, OPEN, 0.5, 0.5).tolist() == [8, 0, 1]


def test_soft_bridge_closes_the_gap_to_the_next_date_over_the_months_left() -> None:
    upcoming = np.array([46.0, 46.0, NAN])
    ahead = np.array([2.0, 0.0, NAN])
    assert soft_bridge(upcoming, ahead, H, OPEN).tolist() == [4, 12, 0]
    assert soft_bridge(np.array([20.0]), np.array([3.0]), H[:1], OPEN[:1]).tolist() == [0]


def dated(values: dict[str, list[float]], dates: list[str], months: int = 6) -> AlignmentTarget:
    m = marginal("cost_center", list(values), dates, [values[g] for g in values])
    return AlignmentTarget((m,), m.dates[0], months, months, Interpolation.HOLD_LAST)


def test_the_next_target_date_and_the_months_until_it() -> None:
    target = dated({"CC-02": [40.0, 52.0]}, ["2026-01", "2026-04"])
    ceiling = config(plan=target).ceiling
    for step, expected in ((0, (40.0, 0.0)), (1, (52.0, 2.0)), (3, (52.0, 0.0)), (5, (52.0, 0.0))):
        values, ahead = ceiling.next_targets(RUN_DATE, step)
        assert (values[1], ahead[1]) == expected
        assert np.isnan(values[[0, 2]]).all()  # no target, so no next one
    assert np.isnan(config().ceiling.next_targets(RUN_DATE, 0)[0]).all()
    later = dated({"CC-02": [40.0]}, ["2026-03"])
    assert np.isnan(config(plan=later).ceiling.next_targets(RUN_DATE, 0)[0]).all()


def grown(step: int, **overrides: Any) -> list[int]:
    ctx = StepContext(config(**overrides), step, 0, SeedManager(8))
    return opened(growth(initial_state(), ctx))


def test_growth_sizes_its_target_seeking_cost_centers_by_the_runs_rule() -> None:
    # CC-02's plan falls from 52 to 40 by April. Strength closes the gap to today's 52 at once;
    # the soft bridge aims at April's 40 and spreads the 6 short over the 3 months left.
    target = dated({"CC-02": [52.0, 40.0]}, ["2026-01", "2026-04"])
    assert grown(1, plan=target)[1] == 18
    assert grown(1, plan=target, requisitions_supply_rule="soft_bridge")[1] == 2
    assert grown(3, plan=target, requisitions_supply_rule="soft_bridge")[1] == 6
    rate = grown(1)
    assert grown(1, plan=target, requisitions_supply_rule="soft_bridge")[2] == rate[2]


def test_a_run_records_each_months_route() -> None:
    run = config(
        guidance=guidance({"requisitions_opened": {"CC-01": 2.0}}), plan=plan({"CC-02": 40.0})
    )
    t = simulate(dataclasses.replace(run, num_trajectories=1), initial_state()).trajectories[0]
    assert t.growth_regimes is not None
    assert t.growth_regimes.shape == (run.simulation_months, 3)
    assert (t.growth_regimes == [Regime.GUIDED, Regime.TARGET, Regime.RATE]).all()
    assert [ROUTES[Regime(r)] for r in t.growth_regimes[0]] == [
        Route.GUIDED,
        Route.SUPPLY,
        Route.RATE,
    ]
    assert [r.value for r in Route] == ["G-A", "G-B", "G-C"]


def test_the_supply_settings_are_checked() -> None:
    assert config().requisitions_supply_rule == "strength"
    with pytest.raises(ConfigError, match="requisitions_supply_rule"):
        config(requisitions_supply_rule="bridge")
    for bad in (0.0, 1.5):
        with pytest.raises(ConfigError, match=r"requisitions_fill_yield must lie in \(0, 1\]"):
            config(requisitions_fill_yield=bad)
