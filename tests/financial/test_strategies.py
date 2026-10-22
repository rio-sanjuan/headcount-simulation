"""Tests for the sizing strategies: gap fills net of the ledger, in-year plus forward years, and
retention."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from hcsim.financial.calculator import SizingError
from hcsim.financial.prices import StaticPrice
from hcsim.financial.ranges import RangeTable
from hcsim.financial.sizing import (
    AnnualGapFill,
    InYearPlusForward,
    NewHireGapFill,
    RetentionGrant,
    SizingContext,
)
from hcsim.financial.tracker import HolderLedger
from hcsim.financial.vesting import VestTemplate
from hcsim.model.core.layouts import EMPLOYEE, EmployeeColumn
from hcsim.model.core.schema import COST_CENTER, JOB_FAMILY, JOB_LEVEL, LOCATION, SimulationSchema

SCHEMA = SimulationSchema.build(
    {COST_CENTER: ["CC-01"], JOB_FAMILY: ["Finance"], JOB_LEVEL: [1], LOCATION: ["LOC-1"]}
)
#: Base 100,000 and target 150,000 at the midpoint: a 50,000 gap a year, 500 shares at 100.
RANGES = RangeTable.from_frame(
    pd.DataFrame(
        [
            {
                "job_family": "Finance",
                "job_level": 1,
                "location": "LOC-1",
                "min_base": 80_000,
                "max_base": 120_000,
                "min_target": 120_000,
                "max_target": 180_000,
                "base_adj": 1.0,
            }
        ]
    ),
    SCHEMA,
)
PRICE = StaticPrice(100.0)
#: Two forward years, each vesting at its end.
TWO_YEARS = VestTemplate("annual", (((12, 1.0),), ((12, 1.0),)))


def rows(*holders: int) -> np.ndarray:
    out = EMPLOYEE.empty(len(holders))
    out[:, EmployeeColumn.VEST_INDEX] = holders
    return out


def ledger_with(holder: int, step: int, vests: dict[int, int]) -> HolderLedger:
    """A ledger where `holder` vests `shares` in each month after `step`."""
    ledger = HolderLedger()
    schedule = np.zeros((1, 72), dtype=np.int64)
    for month, shares in vests.items():
        schedule[0, month - 1] = shares
    ledger.record(np.array([holder]), step, schedule, 0, np.array([0]), np.array([0]))
    return ledger


# --- the ledger's forward years -----------------------------------------------------------------


def test_the_ledger_sums_each_holders_live_vests_by_forward_year() -> None:
    ledger = ledger_with(7, 0, {3: 10, 12: 20, 13: 40, 30: 80})
    ledger.record(
        np.array([9]), 0, np.eye(1, 72, 5, dtype=np.int64) * 5, 0, np.array([0]), np.array([0])
    )
    by_year = ledger.vesting_by_year(np.array([9, 7, 4]), 0, 2)
    # Year 1 is months 1-12, year 2 months 13-24; month 30 lies beyond, and holder 4 holds none.
    assert by_year.tolist() == [[5.0, 0.0], [30.0, 40.0], [0.0, 0.0]]
    # From month 12, month 12's vest is past, and the later ones shift a year earlier.
    assert ledger.vesting_by_year(np.array([7]), 12, 2).tolist() == [[40.0, 80.0]]
    ledger.forfeit(np.array([7]), 6)  # forfeits what vests after month 6
    assert ledger.vesting_by_year(np.array([7]), 0, 2).tolist() == [[10.0, 0.0]]


# --- gap fills ----------------------------------------------------------------------------------


def test_an_annual_gap_fill_is_net_of_what_the_recipient_already_vests() -> None:
    strategy = AnnualGapFill(RANGES, TWO_YEARS)
    ledger = ledger_with(3, 0, {6: 100, 18: 600})
    batch = strategy.size(rows(3, 4), PRICE, SizingContext(0, 1, ledger))
    # Holder 3's 100 shares cut year 1 to 400; their 600 cover year 2. Holder 4 holds nothing.
    assert batch.shares_per_year.tolist() == [[400, 0], [500, 500]]
    assert batch.award_type == "annual"
    assert int(batch.vest_schedule[0, 11]) == 400
    without = strategy.size(rows(3), PRICE)
    assert without.shares_per_year.tolist() == [[500, 500]]


def test_a_new_hire_is_sized_over_every_year_of_its_template() -> None:
    batch = NewHireGapFill(RANGES).size(rows(0), PRICE)
    assert batch.shares_per_year.tolist() == [[500, 500, 500, 500]]
    assert batch.award_type == "new_hire"


# --- in-year plus forward years -----------------------------------------------------------------


@pytest.mark.parametrize(("month", "left"), [(1, 3), (3, 1), (4, 12), (5, 11), (12, 4)])
def test_the_in_year_portion_is_the_months_until_the_next_cycle(month: int, left: int) -> None:
    strategy = InYearPlusForward(RANGES, TWO_YEARS, cycle_month=4)
    assert strategy.months_left(month) == left
    batch = strategy.size(rows(0), PRICE, SizingContext(0, month))
    assert batch.shares_per_year.tolist() == [[round(500 * left / 12), 500]]
    assert batch.award_type == "promotion"


def test_a_cycle_month_and_a_calendar_month_run_1_to_12() -> None:
    with pytest.raises(SizingError, match="cycle_month is a calendar month 1-12"):
        InYearPlusForward(RANGES, TWO_YEARS, cycle_month=0)
    with pytest.raises(SizingError, match="a calendar month runs 1-12"):
        SizingContext(0, 13)


# --- retention ----------------------------------------------------------------------------------


def test_a_retention_grant_is_a_share_of_the_target_on_top_of_existing_vests() -> None:
    strategy = RetentionGrant(RANGES, TWO_YEARS, fraction=0.3)
    ledger = ledger_with(3, 0, {6: 100, 18: 600})
    batch = strategy.size(rows(3), PRICE, SizingContext(0, 1, ledger))
    assert batch.shares_per_year.tolist() == [[150, 150]]  # 0.3 x 500, whatever they hold
    assert batch.award_type == "retention"
    with pytest.raises(SizingError, match=r"a retention fraction lies in \(0, 1\]"):
        RetentionGrant(RANGES, TWO_YEARS, fraction=0.0)
