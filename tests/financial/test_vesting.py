"""Tests for the vest templates: quarterly, cliff, and calendar-keyed, in a 72-month buffer."""

from __future__ import annotations

import numpy as np
import pytest

from hcsim.financial.prices import StaticPrice
from hcsim.financial.sizing import AnnualGapFill, SizingContext
from hcsim.financial.vesting import (
    CANONICAL_VEST_MONTHS,
    NEW_HIRE_TEMPLATE,
    CalendarTemplate,
    VestTemplateError,
    cliff,
    quarterly,
)
from tests.financial.test_strategies import RANGES, rows


def vests(schedule: np.ndarray) -> dict[int, int]:
    """Months after the award to the shares vesting then."""
    return {int(c) + 1: int(v) for c, v in enumerate(schedule) if v}


def test_quarterly_vests_a_quarter_of_each_year_every_three_months() -> None:
    schedule = quarterly("q", 2).expand(np.array([[100, 41]]))
    assert schedule.shape == (1, CANONICAL_VEST_MONTHS)
    assert vests(schedule[0]) == {3: 25, 6: 25, 9: 25, 12: 25, 15: 10, 18: 10, 21: 10, 24: 11}


def test_a_cliff_vests_its_year_at_once_then_quarterly_or_nothing() -> None:
    assert cliff("new_hire_cliff", 4) == NEW_HIRE_TEMPLATE
    assert vests(cliff("c", 2).expand(np.array([[40, 40]]))[0]) == {
        12: 40,
        15: 10,
        18: 10,
        21: 10,
        24: 10,
    }
    assert vests(cliff("c", 1, then="none").expand(np.array([[40]]))[0]) == {12: 40}
    with pytest.raises(VestTemplateError, match="covers one forward year"):
        cliff("c", 2, then="none")
    with pytest.raises(VestTemplateError, match="'quarterly' or 'none'"):
        cliff("c", 2, then="monthly")


OFF_CYCLE = CalendarTemplate("off_cycle", (2, 5, 8, 11), years=2)


@pytest.mark.parametrize(
    ("month", "expected"),
    [
        # In January, the next vest dates are February, May, August, and November.
        (1, {1: 25, 4: 25, 7: 25, 10: 25, 13: 25, 16: 25, 19: 25, 22: 25}),
        # In February itself, the first is May; February comes round at month 12.
        (2, {3: 25, 6: 25, 9: 25, 12: 25, 15: 25, 18: 25, 21: 25, 24: 25}),
    ],
)
def test_a_calendar_template_vests_on_fixed_calendar_months(
    month: int, expected: dict[int, int]
) -> None:
    assert vests(OFF_CYCLE.expand(np.array([[100, 100]]), month)[0]) == expected


def test_a_calendar_templates_first_vest_waits() -> None:
    waiting = CalendarTemplate("w", (2, 5, 8, 11), years=1, wait=4)
    # From January, February (month 1) comes too soon: May, August, and November share it.
    assert vests(waiting.expand(np.array([[100]]), 1)[0]) == {4: 33, 7: 33, 10: 34}
    # A year with no vest date after the wait vests at its last month.
    june = CalendarTemplate("j", (6,), years=2, wait=12)
    assert june.dates(1) == [[12], [17]]


def test_every_share_vests_once_whatever_the_month() -> None:
    shares = np.array([[97, 13], [0, 5]])
    for month in range(1, 13):
        schedule = OFF_CYCLE.expand(shares, month)
        assert schedule.sum(axis=1).tolist() == [110, 5]


def test_calendar_templates_are_checked() -> None:
    with pytest.raises(VestTemplateError) as error:
        CalendarTemplate("bad", (0, 13), years=7, wait=0)
    message = str(error.value)
    assert "vest months are calendar months 1-12" in message
    assert "1 to 6 forward years, not 7" in message
    assert "waits 1 to 12 months, not 0" in message
    with pytest.raises(VestTemplateError, match="vest months repeat"):
        CalendarTemplate("bad", (3, 3), years=1)
    with pytest.raises(VestTemplateError, match="a calendar month runs 1-12"):
        OFF_CYCLE.expand(np.array([[1, 1]]), 0)


def test_a_strategy_lays_out_a_calendar_template_from_its_awards_month() -> None:
    strategy = AnnualGapFill(RANGES, OFF_CYCLE)
    january = strategy.size(rows(0), StaticPrice(100.0), SizingContext(0, 1))
    february = strategy.size(rows(0), StaticPrice(100.0), SizingContext(0, 2))
    assert january.shares_per_year.tolist() == february.shares_per_year.tolist() == [[500, 500]]
    assert min(vests(january.vest_schedule[0])) == 1
    assert min(vests(february.vest_schedule[0])) == 3
