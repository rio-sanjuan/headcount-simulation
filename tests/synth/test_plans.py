"""Tests for plan vintages."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from hcsim.synth.config import PlanTruth, WorldConfig
from hcsim.synth.world import World
from tests.support.stat_check import stat_check


@pytest.fixture
def vintages(small_world: World) -> pd.DataFrame:
    return small_world.tables["plan_headcount"].assign(
        month=lambda f: pd.to_datetime(f["month_end"]).dt.to_period("M"),
        published=lambda f: pd.to_datetime(f["published_date"]).dt.to_period("M"),
    )


def test_vintages_are_published_twice_a_year_from_the_second_year(vintages: pd.DataFrame) -> None:
    published = sorted(vintages["vintage"].unique())
    assert published == [f"{y}-{m:02d}" for y in range(2022, 2026) for m in (1, 7)]
    spans = vintages.groupby("vintage")["month"].agg(["min", "max", "nunique"])
    assert (spans["nunique"] == 18).all()
    assert (spans["min"] == spans.index.map(lambda v: pd.Period(v, freq="M"))).all()
    assert spans.loc["2025-07", "max"] == pd.Period("2026-12", freq="M")  # past the world's end
    assert (vintages.groupby("vintage").size() == 250 * 18).all()


def test_regimes_follow_their_shares(vintages: pd.DataFrame) -> None:
    cells = vintages.drop_duplicates(["vintage", "cost_center"])
    for regime, p in zip(("planned", "flat", "decaying"), PlanTruth().regime_shares, strict=True):
        n = len(cells)
        stat_check(
            int((cells["regime"] == regime).sum()),
            n * p,
            float(np.sqrt(n * p * (1 - p))),
            label=regime,
        )


def _headcount_before(world: World) -> pd.Series:
    txn = world.tables["workforce_txn"]
    regular = txn[(txn["trans_type"] == "ACTIVE") & (txn["account"] == "ACCT-100")]
    return regular.groupby(
        [pd.to_datetime(regular["event_date"]).dt.to_period("M"), "cost_center"], observed=True
    ).size()


def test_flat_cells_hold_the_headcount_at_publication(
    small_world: World, vintages: pd.DataFrame
) -> None:
    flat = vintages[vintages["regime"] == "flat"]
    before = _headcount_before(small_world)
    for (vintage, cc), rows in flat.groupby(["vintage", "cost_center"]):
        expected = before.get((pd.Period(str(vintage), freq="M") - 1, str(cc)), 0)
        assert (rows["target_headcount"] == expected).all()


def test_planned_cells_track_the_latent_plan_with_their_vintage_noise(
    small_world: World, vintages: pd.DataFrame
) -> None:
    latent = (
        small_world.tables["truth_latent_plan"]
        .assign(month=lambda f: pd.to_datetime(f["month_end"]).dt.to_period("M"))
        .set_index(["month", "cost_center"])["target_headcount"]
    )
    planned = vintages[(vintages["regime"] == "planned") & (vintages["target_headcount"] > 5)]
    ratio = (
        planned["target_headcount"].to_numpy()
        / latent.loc[list(zip(planned["month"], planned["cost_center"], strict=True))].to_numpy()
    )
    spread = (
        pd.Series(ratio)
        .groupby([planned["vintage"].to_numpy(), planned["cost_center"].to_numpy()])
        .agg(["std", "mean"])
    )
    assert spread["std"].max() < 1e-3  # one noise draw per vintage and cost center
    assert np.log(spread["mean"]).std() == pytest.approx(PlanTruth().vintage_noise_sigma, rel=0.15)


def test_decaying_cells_move_from_the_plan_toward_flat(
    small_world: World, vintages: pd.DataFrame
) -> None:
    """target = base + (latent * noise - base) * exp(-t / decay), with the cell's noise read
    off its first month, where the decay is 1."""
    latent = (
        small_world.tables["truth_latent_plan"]
        .assign(month=lambda f: pd.to_datetime(f["month_end"]).dt.to_period("M"))
        .set_index(["month", "cost_center"])["target_headcount"]
    )
    before = _headcount_before(small_world)
    tau = PlanTruth().decay_months
    decaying = vintages[vintages["regime"] == "decaying"]
    for (vintage, cc), rows in list(decaying.groupby(["vintage", "cost_center"]))[:200]:
        rows = rows.sort_values("month")
        plan = np.array([latent.loc[(m, str(cc))] for m in rows["month"]])
        if plan[0] < 1:
            continue
        base = before.get((pd.Period(str(vintage), freq="M") - 1, str(cc)), 0)
        noise = rows["target_headcount"].iloc[0] / plan[0]
        expected = base + (plan * noise - base) * np.exp(-np.arange(len(rows)) / tau)
        assert rows["target_headcount"].to_numpy() == pytest.approx(expected, abs=1e-2)


def test_successive_vintages_disagree_by_cost_center(vintages: pd.DataFrame) -> None:
    month = pd.Period("2023-03", freq="M")
    both = vintages[(vintages["month"] == month) & vintages["vintage"].isin(["2022-07", "2023-01"])]
    wide = both.pivot(index="cost_center", columns="vintage", values="target_headcount")
    assert (wide["2022-07"] != wide["2023-01"]).mean() > 0.2


@pytest.mark.parametrize(
    ("truth", "message"),
    [
        (PlanTruth(publication_months=(0,)), "plans.publication_months"),
        (PlanTruth(horizon_months=30, future_months=24), "positive horizon within future_months"),
        (PlanTruth(regime_shares=(0.5, 0.5, 0.5)), "plans.regime_shares"),
        (PlanTruth(decay_months=0.0), "positive decay time"),
    ],
)
def test_a_bad_plan_truth_is_refused(truth: PlanTruth, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        WorldConfig(plans=truth)
