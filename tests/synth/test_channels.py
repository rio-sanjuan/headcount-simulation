"""Tests for the requisition channels' hazards by age.

The checks use growth requisitions open at the start of a month, from the month-end snapshot
before it, so each one's age is exact. Bulk placeholders are left out: their openings share one
exit draw, so they aren't independent.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from hcsim.synth.config import MovementTruth, RequisitionTruth, WorldConfig, by_age
from hcsim.synth.world import World
from tests.support.stat_check import stat_check


def exposure(world: World) -> pd.DataFrame:
    """Growth requisitions open at each month's start, with their age and openings then, and
    what happened to them that month."""
    reqs = world.tables["req_interval"].assign(
        start=lambda f: pd.to_datetime(f["interval_start"]),
        end=lambda f: pd.to_datetime(f["closed_date"]),
        opened=lambda f: pd.to_datetime(f["opened_date"]).dt.to_period("M"),
    )
    growth = reqs[
        (reqs["req_type"] == "growth") & (reqs.groupby("req_id")["openings"].transform("max") <= 5)
    ]
    rows = []
    for month in pd.period_range("2021-03", "2025-12", freq="M"):
        before = (month - 1).end_time.normalize()
        live = growth[
            (growth["start"] <= before) & (growth["end"].isna() | (growth["end"] > before))
        ]
        events = growth[growth["end"].dt.to_period("M") == month]
        for req, interval in live.set_index("req_id").iterrows():
            mine = events[events["req_id"] == req]
            rows.append(
                {
                    "age": (month - interval["opened"]).n,
                    "openings": int(interval["openings"]),
                    "exited": bool((mine["close_reason"] == "unfilled").any()),
                    "external": int((mine["close_reason"] == "filled_external").sum()),
                    "transfer": int((mine["close_reason"] == "filled_transfer").sum()),
                }
            )
    return pd.DataFrame(rows)


@pytest.fixture(scope="module")
def exposed(small_world: World) -> pd.DataFrame:
    return exposure(small_world)


def test_unfilled_exits_follow_their_hazard_by_age(
    small_world: World, exposed: pd.DataFrame
) -> None:
    p = by_age(small_world.config.requisitions.unfilled_exit_hazard, exposed["age"].to_numpy())
    stat_check(
        int(exposed["exited"].sum()),
        float(p.sum()),
        float(np.sqrt((p * (1 - p)).sum())),
        label="exits",
    )


def test_external_fills_follow_their_hazard_by_age(
    small_world: World, exposed: pd.DataFrame
) -> None:
    truth = small_world.config.requisitions
    ages = exposed["age"].to_numpy()
    stay = 1 - by_age(truth.unfilled_exit_hazard, ages)
    for low, high in ((1, 1), (2, 2), (3, 120)):
        band = (ages >= low) & (ages <= high)
        p = (stay * by_age(truth.external_fill_hazard, ages))[band]
        n = exposed["openings"].to_numpy()[band]
        stat_check(
            int(exposed["external"].to_numpy()[band].sum()),
            float((n * p).sum()),
            float(np.sqrt((n * p * (1 - p)).sum())),
            label=f"external age {low}",
        )


def test_transfers_never_exceed_their_hazard(small_world: World, exposed: pd.DataFrame) -> None:
    """A transfer needs an internal candidate, so it can only fall short of its hazard."""
    truth = small_world.config.requisitions
    ages = exposed["age"].to_numpy()
    p = (1 - by_age(truth.unfilled_exit_hazard, ages)) * by_age(truth.transfer_fill_hazard, ages)
    expected = float((exposed["openings"].to_numpy() * p).sum())
    assert 0 < exposed["transfer"].sum() <= expected + 4 * np.sqrt(expected)


def test_by_age_holds_the_last_entry() -> None:
    assert by_age((0.1, 0.2, 0.3), np.array([0, 1, 2, 3, 40])).tolist() == [0.1, 0.2, 0.3, 0.3, 0.3]


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"requisitions": RequisitionTruth(transfer_fill_hazard=())}, "needs hazards in"),
        ({"requisitions": RequisitionTruth(unfilled_exit_hazard=(1.5,))}, "needs hazards in"),
        (
            {"movements": MovementTruth(conversion_fill_hazard=(0.9, 0.9))},
            "hazards exceed 1",
        ),
    ],
)
def test_bad_channel_hazards_are_refused(overrides: dict[str, object], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        WorldConfig(**overrides)  # type: ignore[arg-type]
