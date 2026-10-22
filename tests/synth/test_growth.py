"""Tests for plan-driven growth requisitions."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from hcsim.synth.config import GrowthTruth, WorldConfig
from hcsim.synth.growth import LatentPlan
from hcsim.synth.world import World
from tests.support.stat_check import stat_check

MONTHS = [pd.Period("2021-01", freq="M") + k for k in range(36)]


def test_the_plan_compounds_from_the_starting_headcount() -> None:
    plan = LatentPlan(GrowthTruth(), np.array([100, 40, 0]), MONTHS, np.random.default_rng(0))
    assert plan.target[0].tolist() == [100, 40, 0]
    for k in range(1, len(MONTHS)):
        expected = plan.target[k - 1] * (1 + plan.annual_growth[k]) ** (1 / 12)
        assert plan.target[k] == pytest.approx(expected)
    # The rate is revised only in January.
    changes = [
        k for k in range(1, 36) if not np.allclose(plan.annual_growth[k], plan.annual_growth[k - 1])
    ]
    assert changes == [12, 24]


def test_openings_are_negative_binomial_around_the_seasonal_gap() -> None:
    truth = GrowthTruth(gap_closing_rate=0.2, dispersion=2.0, seasonal_amplitude=0.3)
    plan = LatentPlan(truth, np.full(20_000, 50), MONTHS, np.random.default_rng(1))
    plan.target[:] = 60.0  # a gap of 10 everywhere, in every month
    rng = np.random.default_rng(2)
    for k, season in ((0, 1.3), (6, 0.7)):  # January peak, July trough
        counts = plan.openings(k, np.full(20_000, 50), rng)
        mean = 10 * 0.2 * season
        variance = mean + mean**2 / 2.0
        stat_check(float(counts.mean()), mean, float(np.sqrt(variance / 20_000)), label=f"mean {k}")
        assert counts.var() == pytest.approx(variance, rel=0.1)


def test_a_cost_center_at_plan_opens_nothing() -> None:
    plan = LatentPlan(GrowthTruth(), np.array([10, 10]), MONTHS, np.random.default_rng(3))
    counts = plan.openings(0, np.array([10, 25]), np.random.default_rng(4))
    assert counts.tolist() == [0, 0]


def test_the_world_records_its_plan_and_follows_it(small_world: World) -> None:
    plan = small_world.truth_tables["latent_plan"]
    assert len(plan) == 250 * (60 + 24)  # the plan runs two years past the world's end
    txn = small_world.tables["workforce_txn"]
    regular = txn[(txn["trans_type"] == "ACTIVE") & (txn["account"] == "ACCT-100")]
    start = regular[regular["event_date"] == regular["event_date"].min()]
    first = plan[pd.to_datetime(plan["month_end"]) == pd.Timestamp("2021-01-31")]
    assert first.set_index("cost_center")["target_headcount"].sum() == len(start)
    end = regular[regular["event_date"] == regular["event_date"].max()]
    target = plan[pd.to_datetime(plan["month_end"]) == pd.Timestamp("2025-12-31")][
        "target_headcount"
    ].sum()
    assert 0.8 * target < len(end) < 1.1 * target  # headcount tracks the plan, a little behind it


def test_growth_openings_follow_the_season(small_world: World) -> None:
    reqs = small_world.tables["req_interval"]
    growth = reqs[reqs["req_type"] == "growth"]
    month = pd.to_datetime(growth["opened_date"]).dt.month
    assert month.isin([11, 12, 1]).sum() > month.isin([5, 6, 7]).sum()


@pytest.mark.parametrize(
    ("truth", "message"),
    [
        (GrowthTruth(gap_closing_rate=0.0), "growth.gap_closing_rate"),
        (GrowthTruth(dispersion=0.0), "growth.dispersion"),
        (GrowthTruth(seasonal_amplitude=1.0), "growth.seasonal_amplitude"),
        (GrowthTruth(seasonal_peak_month=0), "growth.seasonal_peak_month"),
        (GrowthTruth(plan_growth_sigma=-1.0), "growth plan spreads"),
    ],
)
def test_a_bad_growth_truth_is_refused(truth: GrowthTruth, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        WorldConfig(growth=truth)
