"""Tests for routing grants between recorded awards and sizing, and for overrides."""

from __future__ import annotations

import dataclasses

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
    covered, schedules, steps = RECORDED.lookup(np.array([8, 9, 7]), 2, "annual")
    assert covered.tolist() == [True, False, True]
    assert steps.tolist() == [2, -1, 2]
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


LAGGED = RecordedAwards.from_frame(
    tranches(
        (7, "new_hire", 3, 12, 40),  # a month after a start at step 2
        (8, "new_hire", 5, 12, 50),  # three months after it
        (9, "new_hire", 2, 12, 1),
        (9, "new_hire", 4, 12, 2),  # a second award of the type in the window
    )
)
HIRES = dataclasses.replace(STRATEGY, award_type="new_hire")


def test_the_window_spans_each_types_lag_through_its_last_step() -> None:
    window = HybridWindow(lag={"new_hire": 2})
    assert window.steps(2, "new_hire") == range(2, 5)
    assert window.steps(2, "annual") == range(2, 3)  # no lag: the dispatch step alone
    assert HybridWindow(3, {"new_hire": 2}).steps(2, "new_hire") == range(2, 4)
    assert not HybridWindow(1, {"new_hire": 2}).steps(2, "new_hire")
    with pytest.raises(SizingError, match=r"which \['new_hire'\] isn't"):
        HybridWindow(lag={"new_hire": -1, "annual": 0})


def test_each_recipient_takes_their_earliest_award_in_the_window_not_yet_granted() -> None:
    ids = np.array([7, 8, 9])
    covered, schedules, steps = LAGGED.lookup(ids, range(2, 5), "new_hire")
    assert covered.tolist() == [True, False, True]
    assert steps.tolist() == [3, -1, 2]
    assert schedules[:, 11].tolist() == [40, 0, 1]
    granted = {("new_hire", 2, 9), ("annual", 3, 7)}  # another type's award takes nothing
    covered, schedules, steps = LAGGED.lookup(ids, range(4, 1, -1), "new_hire", granted)
    assert steps.tolist() == [3, -1, 4]
    assert schedules[:, 11].tolist() == [40, 0, 2]
    assert not LAGGED.lookup(ids, range(5, 5), "new_hire")[0].any()


def test_a_lagged_award_routes_with_its_own_step_and_the_rest_are_sized() -> None:
    window = HybridWindow(lag={"new_hire": 2})
    context = SizingContext(2)
    routed = route(HIRES, "new_hire", rows(7, 8), PRICE, context, LAGGED, window)
    assert routed is not None
    assert routed.replayed.tolist() == [True, False]
    assert routed.award_steps.tolist() == [3, -1]
    assert routed.batch.shares_per_year.tolist() == [[40, 0, 0, 0, 0, 0], [500, 500, 0, 0, 0, 0]]
    granted = {("new_hire", 2, 9), ("new_hire", 4, 9)}
    routed = route(HIRES, "new_hire", rows(9), PRICE, context, LAGGED, window, granted)
    assert routed is not None
    assert routed.sized.tolist() == [True]  # both of their awards are granted already
    exact = route(HIRES, "new_hire", rows(7), PRICE, context, LAGGED)
    assert exact is not None
    assert exact.sized.tolist() == [True]  # without a lag, the award a month on is out of reach


#: Awards an offline stream holds: 8's a month after the dispatch, and 7's beside a primary one.
OFFLINE = RecordedAwards.from_frame(tranches((8, "new_hire", 3, 12, 60), (7, "new_hire", 2, 12, 9)))


def test_an_award_the_offline_stream_holds_routes_its_recipient_there_with_no_shares() -> None:
    window = HybridWindow(lag={"new_hire": 2})
    context = SizingContext(2)
    routed = route(
        HIRES, "new_hire", rows(7, 8, 10), PRICE, context, LAGGED, window, set(), OFFLINE
    )
    assert routed is not None
    # 7 takes the primary plan's award, 8 the stream's, and 10 holds neither, so is sized.
    assert routed.replayed.tolist() == [True, False, False]
    assert routed.offline.tolist() == [False, True, False]
    assert routed.sized.tolist() == [False, False, True]
    assert routed.award_steps.tolist() == [3, 3, -1]
    assert routed.batch.shares_per_year[:, 0].tolist() == [40, 0, 500]
    # Once granted, the stream's award routes no one again, and without a strategy, a
    # recipient the stream covers is still granted.
    granted = {("new_hire", 3, 8)}
    again = route(HIRES, "new_hire", rows(8), PRICE, context, None, window, granted, OFFLINE)
    assert again is not None
    assert again.sized.tolist() == [True]
    alone = route(None, "new_hire", rows(8), PRICE, context, None, window, set(), OFFLINE)
    assert alone is not None
    assert alone.offline.tolist() == [True]
    assert not alone.batch.vest_schedule.any()


def test_a_window_settles_when_it_lies_wholly_within_the_complete_record() -> None:
    window = HybridWindow(lag={"new_hire": 2}, complete_through=5)
    assert window.settled(3, "new_hire")
    assert not window.settled(4, "new_hire")  # it reaches step 6, past the record's end
    assert window.settled(5, "annual")  # no lag: the dispatch step alone
    assert not HybridWindow(4, {"new_hire": 2}, complete_through=5).settled(3, "new_hire")
    assert not HybridWindow(lag={"new_hire": 2}).settled(0, "new_hire")  # no complete record


def test_a_settled_window_with_no_award_grants_nothing_and_one_past_the_record_is_sized() -> None:
    window = HybridWindow(lag={"new_hire": 2}, complete_through=4)
    # 7's award falls a month on and 8's three months on, beyond the lag, so the record says 8
    # was granted none in the window.
    routed = route(HIRES, "new_hire", rows(7, 8), PRICE, SizingContext(2), LAGGED, window)
    assert routed is not None
    assert routed.replayed.tolist() == [True, False]
    assert routed.ungranted.tolist() == [False, True]
    assert not routed.sized.any()
    assert routed.batch.shares_per_year[:, 0].tolist() == [40, 0]
    assert routed.batch.base_salary.tolist() == [100_000.0, 100_000.0]  # both imputed
    # At step 3, the window reaches past the record's end, so a grant that finds none is sized.
    later = route(HIRES, "new_hire", rows(10), PRICE, SizingContext(3), LAGGED, window)
    assert later is not None
    assert (later.sized.tolist(), later.ungranted.tolist()) == ([True], [False])
    # Without a strategy, a recipient the record grants nothing is routed all the same.
    alone = route(None, "annual", rows(9), PRICE, SizingContext(2), RECORDED, window)
    assert alone is not None
    assert alone.ungranted.tolist() == [True]
    assert not alone.batch.vest_schedule.any()


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
