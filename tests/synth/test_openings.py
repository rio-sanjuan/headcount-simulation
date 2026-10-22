"""Tests for the openings a requisition carries."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from hcsim.synth.config import OpeningsTruth, WorldConfig
from hcsim.synth.world import World
from tests.support.stat_check import stat_check


def per_requisition(world: World) -> pd.DataFrame:
    reqs = world.tables["req_interval"].astype({"req_type": str, "close_reason": object})
    filled = reqs["close_reason"].astype(str).str.startswith("filled")
    return (
        reqs.assign(fill=filled)
        .groupby("req_id")
        .agg(
            req_type=("req_type", "first"),
            openings=("openings", "max"),
            fills=("fill", "sum"),
            # The final row, missing values included: `last` would skip an open interval's.
            last_openings=("openings", lambda s: s.iloc[-1]),
            last_reason=("close_reason", lambda s: s.iloc[-1]),
        )
    )


def test_growth_requisitions_draw_their_openings(small_world: World) -> None:
    reqs = per_requisition(small_world)
    growth = reqs[reqs["req_type"] == "growth"]["openings"]
    truth = small_world.config.openings
    for label, mask, p in (
        ("multi", growth.between(*truth.multi_range), truth.multi_share),
        ("bulk", growth.between(*truth.bulk_range), truth.bulk_share),
    ):
        n = len(growth)
        stat_check(int(mask.sum()), n * p, float(np.sqrt(n * p * (1 - p))), label=label)
    assert (reqs.loc[reqs["req_type"] != "growth", "openings"] == 1).all()


def test_every_opening_is_filled_exited_or_still_open(small_world: World) -> None:
    reqs = per_requisition(small_world)
    still_open = reqs["last_reason"].isna()
    unfilled = reqs["last_reason"] == "unfilled"
    left = np.where(still_open | unfilled, reqs["last_openings"], 0)
    assert (reqs["fills"] + left == reqs["openings"]).all()


def test_bulk_requisitions_fill_over_several_months(small_world: World) -> None:
    reqs = small_world.tables["req_interval"]
    bulk = reqs.groupby("req_id").filter(lambda rows: rows["openings"].max() >= 15)
    months = (
        bulk.dropna(subset=["closed_date"])
        .groupby("req_id")["closed_date"]
        .apply(lambda d: d.dt.to_period("M").nunique())
    )
    assert months.max() > 1


@pytest.mark.parametrize(
    ("truth", "message"),
    [
        (OpeningsTruth(multi_share=0.9, bulk_share=0.2), "openings shares"),
        (OpeningsTruth(bulk_range=(1, 10)), "openings.bulk_range"),
    ],
)
def test_a_bad_openings_truth_is_refused(truth: OpeningsTruth, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        WorldConfig(openings=truth)
