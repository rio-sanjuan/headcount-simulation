"""Tests for the seed stream: holder statuses, seed tranches, the opening balance, and the
departure hazard."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from hcsim.financial.seed import (
    COLUMNS,
    DepartureHazard,
    HolderStatus,
    SeedError,
    SeedTranches,
    classify,
)


def seed_frame(**overrides: list[object]) -> pd.DataFrame:
    rows = {
        "award_id": [1, 2, 3],
        "holder": [10, 11, 12],
        "award_type": [0, 1, 1],
        "award_month": [-3, -12, -1],
        "vest_month": [2, 0, 5],
        "shares": [6, 4, 10],
        "value_per_share": [10.0, 20.0, 5.0],
        "level": [0, 1, 2],
        "cost_center": [0, 0, 1],
        "status": ["active", "outside", "future"],
        "start_month": [-1, -1, 3],
        "forfeit_month": [-1, 4, -1],
    }
    rows.update(overrides)
    return pd.DataFrame(rows)


def test_a_holder_is_active_outside_a_future_entrant_or_departed() -> None:
    holders = np.array([1, 2, 3, 4, 5])
    statuses = classify(
        holders, active=np.array([1]), outside=np.array([2]), future=np.array([3, 1])
    )
    assert statuses.tolist() == ["active", "outside", "future", "departed", "departed"]


def test_seed_tranches_are_grouped_by_status() -> None:
    seed = SeedTranches(seed_frame())
    assert len(seed) == 3
    assert seed.counts() == {"active": 1, "outside": 1, "future": 1, "departed": 0}
    assert seed.of(HolderStatus.OUTSIDE, HolderStatus.FUTURE)["holder"].tolist() == [11, 12]
    t = seed.tranches(HolderStatus.ACTIVE)
    assert (t.holder.tolist(), t.values.tolist()) == ([10], [10.0])
    assert t.forfeit_month.tolist() == [-1]


def test_seed_tranches_are_checked() -> None:
    with pytest.raises(SeedError, match=r"need the columns \['status'\]"):
        SeedTranches(seed_frame().drop(columns="status"))
    with pytest.raises(SeedError) as error:
        SeedTranches(
            seed_frame(
                award_month=[0, -1, -1],
                vest_month=[2, -1, 5],
                shares=[6, 0, 10],
                value_per_share=[10.0, np.nan, 5.0],
                status=["active", "gone", "future"],
            )
        )
    message = str(error.value)
    for part in ("negative month", "still vests", "at least one share", "finite", "'gone'"):
        assert part in message
    assert set(COLUMNS) == set(seed_frame().columns)


def test_the_opening_balance_is_what_each_tranche_recognized_before_the_run() -> None:
    # 60 of value over the six months -3 to 2: 10 a month, 9 net of the 10% rate, so 27 before.
    one = SeedTranches(seed_frame().iloc[:1])
    assert one.opening_balance().tolist() == pytest.approx([27.0])
    balances = SeedTranches(seed_frame()).opening_balance()
    # 80 over 13 months, 12 of them before the run; and 50 over 7 months, one before.
    assert balances.tolist() == pytest.approx([27.0, 80 / 13 * 12 * 0.9, 50 / 7 * 0.9])


def test_a_replay_forfeits_in_the_month_history_records() -> None:
    survival = DepartureHazard.replayed(np.array([-1, 2])).survival(2, 4)
    assert survival.tolist() == [[1.0, 1.0, 1.0, 1.0], [1.0, 1.0, 0.0, 0.0]]
    with pytest.raises(SeedError, match="one realized forfeit month per tranche"):
        DepartureHazard.replayed(np.array([1])).survival(2, 4)


def test_a_forecast_survives_a_model_averaged_hazard() -> None:
    survival = DepartureHazard.averaged(0.1).survival(2, 3)
    assert survival[0].tolist() == pytest.approx([0.9, 0.81, 0.729])
    assert np.array_equal(survival[0], survival[1])
    with pytest.raises(SeedError, match=r"lies in \[0, 1\)"):
        DepartureHazard.averaged(1.0)
