"""Known-answer financial cases, computed by hand: an annual grant sized step by step, and one
tranche followed down three paths. `docs/notebooks` walks through the same two cases."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from hcsim.financial.calculator import RatingTable
from hcsim.financial.expense import ExpenseConfig, calculate, prepare
from hcsim.financial.prices import AnnualGrowthPrice
from hcsim.financial.ranges import RangeTable
from hcsim.financial.sizing import AnnualGapFill, SizingContext
from hcsim.financial.tracker import HolderLedger
from hcsim.financial.vesting import quarterly
from hcsim.model.core.layouts import EMPLOYEE, EmployeeColumn
from hcsim.model.core.schema import (
    AWARD_TYPE,
    COST_CENTER,
    JOB_FAMILY,
    JOB_LEVEL,
    LOCATION,
    SimulationSchema,
)

SCHEMA = SimulationSchema.build(
    {
        COST_CENTER: ["CC-01", "CC-02"],
        JOB_FAMILY: ["F"],
        JOB_LEVEL: [1, 2],
        LOCATION: ["L"],
        AWARD_TYPE: ["annual"],
    }
)


# --- the hand-computed annual grant -------------------------------------------------------------


def annual_grant() -> tuple[AnnualGapFill, np.ndarray, AnnualGrowthPrice, SizingContext]:
    """A level-2 employee rated 3, who already vests 150 shares in the coming year and 90 in
    the next, granted an annual award at step 12."""
    ranges = RangeTable.from_frame(
        pd.DataFrame(
            [
                {
                    "job_family": "F",
                    "job_level": level,
                    "location": "L",
                    "min_base": base,
                    "max_base": base + 40_000,
                    "min_target": target,
                    "max_target": target + 40_000,
                    "base_adj": 1.0,
                }
                for level, base, target in ((1, 80_000, 120_000), (2, 100_000, 160_000))
            ]
        ),
        SCHEMA,
    )
    strategy = AnnualGapFill(
        ranges, quarterly("annual_quarterly", 4), RatingTable({1: 0.25, 2: 0.5, 3: 0.75, 4: 1.0})
    )
    rows = EMPLOYEE.empty(1)
    rows[0, EmployeeColumn.EMPLOYEE_ID] = 100001
    rows[0, EmployeeColumn.LEVEL] = 1  # level 2, the vocabulary's second
    rows[0, EmployeeColumn.PERF_RATING] = 3
    ledger = HolderLedger()
    earlier = np.zeros((1, 72), dtype=np.int64)
    earlier[0, [14, 17, 20, 23]] = [30, 40, 40, 40]  # vests in months 15 to 24: 150 shares
    earlier[0, [26, 29, 32, 35]] = [20, 20, 25, 25]  # and in months 27 to 36: 90
    ledger.record(np.array([0]), 0, earlier, 0, np.array([1]), np.array([0]))
    price = AnnualGrowthPrice(100.0, growth_rate=1.25)
    return strategy, rows, price, SizingContext(12, 1, ledger)


def test_the_hand_computed_annual_grant() -> None:
    strategy, rows, price, context = annual_grant()
    sized = strategy.sized(rows, price, context)
    # 2. Rated 3, the employee sits three quarters of the way into the range.
    assert sized.penetration.tolist() == [0.75]
    # 3-4. Target 160,000 + 0.75 x 40,000, base 100,000 + 0.75 x 40,000: 60,000 of equity.
    assert (sized.target.tolist(), sized.base.tolist()) == ([190_000.0], [130_000.0])
    # 5. Each forward year's gap, at its projected price, 100 x 1.25^y, net of existing vests:
    #    60,000 - 150 x 125, 60,000 - 90 x 156.25, then 60,000 twice.
    assert sized.gap.tolist() == [[41_250.0, 45_937.5, 60_000.0, 60_000.0]]
    # 6. Shares: each gap over its year's price, 125, 156.25, 195.3125, and 244.140625.
    assert sized.shares_float[0].tolist() == pytest.approx([330.0, 294.0, 307.2, 245.76])
    # 7. Whole shares, laid out a quarter at a time, the year's last vest taking the remainder.
    assert sized.shares.tolist() == [[330, 294, 307, 246]]
    schedule = strategy.size(rows, price, context).vest_schedule[0]
    vests = {int(m) + 1: int(schedule[m]) for m in np.flatnonzero(schedule)}
    assert vests == {
        3: 82, 6: 82, 9: 82, 12: 84,
        15: 73, 18: 73, 21: 73, 24: 75,
        27: 76, 30: 76, 33: 76, 36: 79,
        39: 61, 42: 61, 45: 61, 48: 63,
    }  # fmt: skip
    assert sum(vests.values()) == 330 + 294 + 307 + 246


# --- the three-path tranche ---------------------------------------------------------------------


def three_paths() -> HolderLedger:
    """One tranche, 120 shares awarded in month 0 to vest in month 5, granted to three holders:
    one who stays, one who leaves in month 3, and one who moves to CC-02 in month 2 and leaves in
    month 4."""
    ledger = HolderLedger()
    schedule = np.zeros((3, 72), dtype=np.int64)
    schedule[:, 4] = 120  # column c vests c + 1 months after the award
    ledger.record(np.array([1, 2, 3]), 0, schedule, 0, np.zeros(3, int), np.zeros(3, int))
    ledger.forfeit(np.array([2]), 3)
    ledger.move(np.array([3]), 2, np.array([1]))
    ledger.forfeit(np.array([3]), 4)
    return ledger


def test_the_three_path_tranche() -> None:
    ledger = three_paths()
    t = ledger.tranches()
    prepared = prepare(t, np.full(3, 10.0), intervals=ledger.intervals())
    result = calculate(prepared, ExpenseConfig(forfeiture_rate=0.10))
    rows = result.rows
    rows["recognized"] = rows[["net", "trueup_attrition", "trueup_rate", "trueup_vesting"]].sum(
        axis=1
    )
    by = rows.groupby("tranche")
    # Each is worth 1,200 over the six months 0 to 5: 200 a month gross, 180 net of the 10%
    # expected to forfeit.
    assert rows.loc[rows["gross"] > 0, "gross"].tolist() == pytest.approx([200.0] * 15)
    assert rows.loc[rows["gross"] > 0, "net"].tolist() == pytest.approx([180.0] * 15)
    # Stays: 6 x 180 = 1,080 accrued, then a vesting true-up of 120 in month 5, the 10% that
    # didn't forfeit after all: 1,200, the vested value.
    stays = by.get_group(0)
    assert stays["trueup_vesting"].tolist() == pytest.approx([0.0] * 5 + [120.0])
    assert stays["recognized"].sum() == pytest.approx(1_200.0)
    # Leaves in month 3, which still accrues: the attrition true-up reverses 4 x 180 = 720.
    leaves = by.get_group(1)
    assert leaves["month"].tolist() == [0, 1, 2, 3]
    assert leaves["trueup_attrition"].tolist() == pytest.approx([0.0, 0.0, 0.0, -720.0])
    assert leaves["recognized"].sum() == pytest.approx(0.0)
    # Moves, then leaves: months 0 and 1 are booked in CC-01, 2 to 4 in CC-02, and the
    # attrition true-up reverses each piece where it was booked, 360 and 540.
    moves = by.get_group(2)
    books = moves.groupby("cost_center")[["net", "trueup_attrition"]].sum()
    assert books.to_dict("index") == {
        0: pytest.approx({"net": 360.0, "trueup_attrition": -360.0}),
        1: pytest.approx({"net": 540.0, "trueup_attrition": -540.0}),
    }
    assert (rows["trueup_rate"] == 0).all()  # a flat rate never changes
    # The identity: what each recognized ties to its vested value.
    assert np.abs(result.residual).max() < 0.01
