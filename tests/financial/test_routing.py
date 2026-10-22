"""Tests for routing grants between recorded awards and sizing, and for overrides."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from hcsim.financial.calculator import SizingError
from hcsim.financial.prices import StaticPrice
from hcsim.financial.routing import HybridWindow, Overridden, RecordedAwards, per_year, route
from hcsim.financial.sizing import AnnualGapFill, SizingContext
from hcsim.financial.vesting import VestTemplate
from hcsim.model.core.layouts import EMPLOYEE, EmployeeColumn
from tests.financial.test_strategies import RANGES

PRICE = StaticPrice(100.0)
TWO_YEARS = VestTemplate("annual", (((12, 1.0),), ((12, 1.0),)))
STRATEGY = AnnualGapFill(RANGES, TWO_YEARS)  # 500 shares a year, on base 100,000


def rows(*ids: int) -> np.ndarray:
    out = EMPLOYEE.empty(len(ids))
    out[:, EmployeeColumn.EMPLOYEE_ID] = ids
    out[:, EmployeeColumn.BASE_SALARY] = 1.0
    return out


def tranches(*rows_: tuple[int, str, int, int, int]) -> pd.DataFrame:
    return pd.DataFrame(rows_, columns=["employee_id", "award_type", "step", "after", "shares"])


RECORDED = RecordedAwards.from_frame(
    tranches(
        (7, "annual", 2, 3, 10),
        (7, "annual", 2, 15, 20),
        (7, "annual", 2, 15, 5),  # a second award the same month adds up
        (8, "annual", 2, 0, 4),  # vesting in its award month, so the month after
        (7, "promotion", 2, 6, 99),
    )
)


def test_recorded_tranches_become_one_schedule_per_recipient_type_and_month() -> None:
    assert len(RECORDED) == 3
    covered, schedules = RECORDED.lookup(np.array([8, 9, 7]), 2, "annual")
    assert covered.tolist() == [True, False, True]
    assert {int(c) + 1: int(v) for c, v in enumerate(schedules[2]) if v} == {3: 10, 15: 25}
    assert schedules[0, 0] == 4
    assert not schedules[1].any()
    assert per_year(schedules).tolist()[2] == [10, 25, 0, 0, 0, 0]
    assert not RECORDED.lookup(np.array([7]), 3, "annual")[0].any()


def test_recorded_tranches_are_checked() -> None:
    with pytest.raises(SizingError, match="vest more than 72 months"):
        RecordedAwards.from_frame(tranches((7, "annual", 0, 80, 1)))
    with pytest.raises(SizingError, match=r"need the columns \['shares'\]"):
        RecordedAwards.from_frame(tranches((7, "annual", 0, 1, 1)).drop(columns="shares"))
    with pytest.raises(SizingError, match="a recorded schedule is 72 non-negative months"):
        RecordedAwards({("annual", 0): {7: np.zeros(5, dtype=np.int64)}})


def test_a_recorded_recipient_takes_their_award_and_the_rest_are_sized() -> None:
    routed = route(STRATEGY, "annual", rows(7, 9), PRICE, SizingContext(2), RECORDED)
    assert routed is not None
    assert routed.replayed.tolist() == [True, False]
    assert routed.sized.tolist() == [False, True]
    assert routed.batch.shares_per_year.tolist() == [[10, 25, 0, 0, 0, 0], [500, 500, 0, 0, 0, 0]]
    assert routed.batch.base_salary.tolist() == [100_000.0, 100_000.0]  # both imputed


def test_after_the_hybrid_window_every_grant_is_sized() -> None:
    window = HybridWindow(through=1)
    assert window.covers(1)
    assert not window.covers(2)
    routed = route(STRATEGY, "annual", rows(7), PRICE, SizingContext(2), RECORDED, window)
    assert routed is not None
    assert routed.replayed.tolist() == [False]
    assert routed.batch.shares_per_year[0, :2].tolist() == [500, 500]


def test_without_a_strategy_only_recorded_recipients_are_granted() -> None:
    routed = route(None, "annual", rows(7, 9), PRICE, SizingContext(2), RECORDED)
    assert routed is not None
    assert (routed.replayed | routed.sized).tolist() == [True, False]
    assert routed.batch.base_salary.tolist() == [1.0, 1.0]  # kept as they were
    assert not routed.batch.vest_schedule[1].any()
    assert route(None, "annual", rows(9), PRICE, SizingContext(2), RECORDED) is None


def test_an_aggregate_override_scales_and_an_individual_one_sets_shares() -> None:
    overridden = Overridden(STRATEGY, scale=0.5, shares={9: (1, 2)})
    batch = overridden.size(rows(7, 9), PRICE)
    assert batch.shares_per_year.tolist() == [[250, 250], [1, 2]]
    assert overridden.award_type == "annual"
    assert int(batch.vest_schedule[1, 23]) == 2


def test_overrides_are_checked() -> None:
    with pytest.raises(SizingError) as error:
        Overridden(STRATEGY, scale=-1.0, shares={4: (1,), 5: (1, -1)})
    message = str(error.value)
    assert "an aggregate override is a non-negative factor, not -1.0" in message
    assert "employees [4, 5] don't" in message
    three = VestTemplate("three", (((12, 1.0),),) * 3)
    with pytest.raises(SizingError, match="the override's template has 3 years"):
        Overridden(STRATEGY, template=three).size(rows(7), PRICE)
