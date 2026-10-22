"""Tests for plan vintages: regimes, reallocation, the fan-out grain, negative cells, and
hedges."""

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
    )


@pytest.fixture
def by_cost_center(vintages: pd.DataFrame) -> pd.DataFrame:
    """Real cost centers' targets, families summed."""
    real = vintages[~vintages["is_hedge"]]
    keys = ["vintage", "cost_center", "regime", "month"]
    return real.groupby(keys)["target_headcount"].sum().reset_index()


def _latent(world: World) -> pd.Series:
    return (
        world.tables["truth_latent_plan"]
        .assign(month=lambda f: pd.to_datetime(f["month_end"]).dt.to_period("M"))
        .set_index(["month", "cost_center"])["target_headcount"]
    )


def _headcount_before(world: World) -> pd.Series:
    txn = world.tables["workforce_txn"]
    regular = txn[(txn["trans_type"] == "ACTIVE") & (txn["account"] == "ACCT-100")]
    month = pd.to_datetime(regular["event_date"]).dt.to_period("M")
    return regular.groupby([month, "cost_center"], observed=True).size()


def test_vintages_are_published_twice_a_year_from_the_second_year(vintages: pd.DataFrame) -> None:
    published = sorted(vintages["vintage"].unique())
    assert published == [f"{y}-{m:02d}" for y in range(2022, 2026) for m in (1, 7)]
    spans = vintages.groupby("vintage")["month"].agg(["min", "max", "nunique"])
    assert (spans["nunique"] == 18).all()
    assert (spans["min"] == spans.index.map(lambda v: pd.Period(v, freq="M"))).all()
    assert spans.loc["2025-07", "max"] == pd.Period("2026-12", freq="M")  # past the world's end


def test_regimes_follow_their_shares(by_cost_center: pd.DataFrame) -> None:
    cells = by_cost_center.drop_duplicates(["vintage", "cost_center"])
    for regime, p in zip(("planned", "flat", "decaying"), PlanTruth().regime_shares, strict=True):
        n = len(cells)
        observed = int((cells["regime"] == regime).sum())
        stat_check(observed, n * p, float(np.sqrt(n * p * (1 - p))), label=regime)


def test_flat_cells_hold_their_publication_headcount(
    small_world: World, by_cost_center: pd.DataFrame
) -> None:
    """Flat over the horizon, at the headcount at publication times the cost center's
    reallocation factor."""
    flat = by_cost_center[by_cost_center["regime"] == "flat"]
    spread = flat.groupby(["vintage", "cost_center"])["target_headcount"].agg(["min", "max"])
    assert (spread["max"] - spread["min"]).max() < 1e-3
    before = _headcount_before(small_world)
    keys = [(pd.Period(str(v), freq="M") - 1, str(c)) for v, c in spread.index]
    base = np.array([before.get(k, 0) for k in keys], dtype=float)
    ratio = spread["min"].to_numpy()[base > 5] / base[base > 5]
    assert np.log(ratio).std() == pytest.approx(PlanTruth().reallocation_sigma, rel=0.25)


def test_planned_cells_keep_one_ratio_to_the_latent_plan(
    small_world: World, by_cost_center: pd.DataFrame
) -> None:
    latent = _latent(small_world)
    planned = by_cost_center[
        (by_cost_center["regime"] == "planned") & (by_cost_center["target_headcount"] > 5)
    ]
    keys = list(zip(planned["month"], planned["cost_center"], strict=True))
    ratio = planned["target_headcount"].to_numpy() / np.array([latent.loc[k] for k in keys])
    groups = [planned["vintage"].to_numpy(), planned["cost_center"].to_numpy()]
    assert pd.Series(ratio).groupby(groups).std().max() < 1e-3


def test_decaying_cells_decay_toward_a_fixed_level(
    small_world: World, by_cost_center: pd.DataFrame
) -> None:
    """target = base + (latent * noise - base) * exp(-t / decay). The first month gives the
    noise, and every later month must then imply the same base."""
    latent = _latent(small_world)
    tau = PlanTruth().decay_months
    decaying = by_cost_center[by_cost_center["regime"] == "decaying"]
    for (_, cc), rows in list(decaying.groupby(["vintage", "cost_center"]))[:200]:
        rows = rows.sort_values("month")
        plan = np.array([latent.loc[(m, str(cc))] for m in rows["month"]])
        if plan[0] < 1:
            continue
        target = rows["target_headcount"].to_numpy()
        noise = target[0] / plan[0]
        d = np.exp(-np.arange(1, len(rows)) / tau)
        implied = (target[1:] - plan[1:] * noise * d) / (1 - d)
        assert implied.max() - implied.min() < 0.05


def test_vintages_agree_by_org_and_disagree_by_cost_center(
    small_world: World, by_cost_center: pd.DataFrame
) -> None:
    org = small_world.tables["dim_org"].set_index("cost_center")["org_level1"]
    pair = by_cost_center[by_cost_center["vintage"].isin(["2022-07", "2023-01"])]
    pair = pair[pair["month"].isin(pd.period_range("2023-01", "2023-12", freq="M"))]
    cc = pair.pivot_table(
        index="cost_center", columns="vintage", values="target_headcount", aggfunc="sum"
    )
    top = cc.groupby(cc.index.map(org.to_dict())).sum()
    cc_gap = np.median(np.abs(np.log(cc["2023-01"] / cc["2022-07"])))
    top_gap = np.median(np.abs(np.log(top["2023-01"] / top["2022-07"])))
    assert cc_gap > 3 * top_gap


def test_cells_fan_out_by_family_in_proportion_to_staff(
    small_world: World, vintages: pd.DataFrame
) -> None:
    real = vintages[~vintages["is_hedge"]]
    assert set(real["job_family"]) <= set(small_world.config.families)
    assert ((real["target_headcount"] % 1) != 0).mean() > 0.9  # fractional cells
    first = real[real["month"] == real["vintage"].map(lambda v: pd.Period(v, "M"))]
    txn = small_world.tables["workforce_txn"]
    jobs = small_world.tables["dim_job"].set_index("job_code")["job_family"]
    stock = txn[
        (txn["trans_type"] == "ACTIVE")
        & (txn["account"] == "ACCT-100")
        & (pd.to_datetime(txn["event_date"]) == pd.Timestamp("2021-12-31"))
    ]
    mix = (
        stock.assign(family=stock["job_code"].astype(str).map(jobs))
        .groupby(["cost_center", "family"], observed=True)
        .size()
    )
    cells = first[(first["vintage"] == "2022-01") & (first["target_headcount"] > 0)]
    totals = cells.groupby("cost_center")["target_headcount"].transform("sum")
    share = cells["target_headcount"] / totals
    staff = mix / mix.groupby(level=0, observed=True).transform("sum")
    expected = [
        staff.get((c, f), 0.0)
        for c, f in zip(cells["cost_center"], cells["job_family"], strict=True)
    ]
    close = np.isclose(share.to_numpy(), np.array(expected), atol=1e-3)
    assert close.mean() > 0.9  # the rest sit in cost centers with a negative cell


def test_a_few_cells_are_negative_and_leave_their_cost_center_whole(vintages: pd.DataFrame) -> None:
    real = vintages[~vintages["is_hedge"]]
    negative = real[real["target_headcount"] < 0]
    assert len(negative) > 0
    hit = negative.drop_duplicates(["vintage", "cost_center"])
    share = len(hit) / len(real.drop_duplicates(["vintage", "cost_center"]))
    assert share == pytest.approx(PlanTruth().negative_cell_share, abs=0.015)
    totals = real.groupby(["vintage", "cost_center", "month"])["target_headcount"].sum()
    assert (totals >= -1e-6).all()


def test_each_org_carries_a_hedge_pseudo_cost_center(
    small_world: World, vintages: pd.DataFrame
) -> None:
    hedges = vintages[vintages["is_hedge"]]
    orgs = sorted(small_world.tables["dim_org"]["org_level1"].unique())
    assert sorted(hedges["cost_center"].unique()) == [f"HEDGE-{o}" for o in orgs]
    assert not set(hedges["cost_center"]) & set(small_world.tables["dim_org"]["cost_center"])
    assert hedges["job_family"].isna().all()
    real = vintages[~vintages["is_hedge"]]
    org = small_world.tables["dim_org"].set_index("cost_center")["org_level1"]
    org_total = real.groupby(["vintage", real["cost_center"].map(org)])["target_headcount"].sum()
    hedge_total = hedges.groupby(["vintage", hedges["cost_center"].str.removeprefix("HEDGE-")])[
        "target_headcount"
    ].sum()
    ratio = (hedge_total / org_total.reindex(hedge_total.index)).to_numpy()
    assert ratio.mean() == pytest.approx(PlanTruth().hedge_share, abs=0.015)
    assert (ratio < 0).any() or (ratio > 0).all()


@pytest.mark.parametrize(
    ("truth", "message"),
    [
        (PlanTruth(publication_months=(0,)), "plans.publication_months"),
        (PlanTruth(horizon_months=30, future_months=24), "positive horizon within future_months"),
        (PlanTruth(regime_shares=(0.5, 0.5, 0.5)), "plans.regime_shares"),
        (PlanTruth(decay_months=0.0), "positive decay time"),
        (PlanTruth(reallocation_sigma=-1.0), "plans spreads"),
        (PlanTruth(negative_cell_share=2.0), "plans.negative_cell_share"),
    ],
)
def test_a_bad_plan_truth_is_refused(truth: PlanTruth, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        WorldConfig(plans=truth)
