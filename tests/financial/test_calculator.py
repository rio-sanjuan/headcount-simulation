"""Tests for the award calculator: its seven steps, worked by hand, and its rating table."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from hcsim.financial.calculator import RatingTable, Recipients, SizingError, size
from hcsim.financial.ranges import RangeTable
from hcsim.financial.vesting import VestTemplate
from hcsim.model.core.schema import (
    COST_CENTER,
    JOB_FAMILY,
    JOB_LEVEL,
    LOCATION,
    SimulationSchema,
)

SCHEMA = SimulationSchema.build(
    {COST_CENTER: ["CC-01"], JOB_FAMILY: ["Finance"], JOB_LEVEL: [1], LOCATION: ["LOC-1"]}
)
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
                "base_adj": 1.03,
            }
        ]
    ),
    SCHEMA,
)
RATINGS = RatingTable({1: 0.25, 2: 0.25, 3: 0.5, 4: 0.75, 5: 0.9}, unrated=0.5)
#: Two forward years, each vesting at its end.
TWO_YEARS = VestTemplate("annual", (((12, 1.0),), ((12, 1.0),)))


def recipients(*ratings: int) -> Recipients:
    n = len(ratings)
    zeros = np.zeros(n, dtype=np.int64)
    return Recipients(zeros, zeros, zeros, np.array(ratings), np.arange(100, 100 + n))


def test_each_step_of_a_hand_computed_grant() -> None:
    # Rated 4, at 0.75 of the range. Target: 120,000 + 0.75 x 60,000 = 165,000. Base: 1.03 x
    # (80,000 + 0.75 x 40,000) = 113,300. Year 1's gap is 51,700, or 517 shares at 100. Year 2
    # already vests 100 shares, worth 12,500 at 125, so its gap is 39,200: 313.6, or 314.
    sized = size(
        recipients(4), RANGES, RATINGS, [100.0, 125.0], TWO_YEARS, np.array([[0.0, 100.0]])
    )
    assert sized.penetration.tolist() == [0.75]
    assert sized.target.tolist() == pytest.approx([165_000.0])
    assert sized.base.tolist() == pytest.approx([113_300.0])
    assert sized.gap[0].tolist() == pytest.approx([51_700.0, 39_200.0])
    assert sized.shares_float[0].tolist() == pytest.approx([517.0, 313.6])
    assert sized.shares.tolist() == [[517, 314]]
    assert {int(c) + 1: int(v) for c, v in enumerate(sized.schedule[0]) if v} == {12: 517, 24: 314}


def test_an_unrated_recipient_takes_the_default_and_the_gap_never_goes_negative() -> None:
    sized = size(
        recipients(0, 1),
        RANGES,
        RATINGS,
        [100.0, 125.0],
        TWO_YEARS,
        np.array([[0.0, 0.0], [1e6, 0.0]]),
    )
    assert sized.penetration.tolist() == [0.5, 0.25]
    assert sized.gap[1, 0] == 0.0  # what they already vest covers the year
    assert sized.shares[1, 0] == 0


def test_the_calculator_is_pure() -> None:
    existing = np.array([[0.0, 100.0]])
    first = size(recipients(4), RANGES, RATINGS, [100.0, 125.0], TWO_YEARS, existing)
    again = size(recipients(4), RANGES, RATINGS, [100.0, 125.0], TWO_YEARS, existing)
    assert existing.tolist() == [[0.0, 100.0]]
    assert np.array_equal(first.schedule, again.schedule)


def test_the_inputs_must_fit_the_template() -> None:
    with pytest.raises(SizingError, match="one projected price per forward year"):
        size(recipients(4), RANGES, RATINGS, [100.0], TWO_YEARS)
    with pytest.raises(SizingError, match="existing vests must be"):
        size(recipients(4), RANGES, RATINGS, [100.0, 125.0], TWO_YEARS, np.zeros((1, 3)))


def test_a_rating_table_reads_each_rank_and_refuses_one_it_lacks() -> None:
    assert RATINGS.penetration(np.array([0, 1, 5])).tolist() == [0.5, 0.25, 0.9]
    with pytest.raises(SizingError, match=r"no penetration for ranks \[6\]"):
        RATINGS.penetration(np.array([6]))
    assert RatingTable.flat(0.4).penetration(np.array([0, 3, 9])).tolist() == [0.4] * 3


@pytest.mark.parametrize(
    ("positions", "unrated", "message"),
    [
        ({0: 0.5}, 0.5, "rating ranks run from 1"),
        ({1: 1.5}, 0.5, r"penetrations lie in \[0, 1\]"),
        ({1: 0.5}, -0.1, r"penetrations lie in \[0, 1\]"),
        ({1: 0.6, 2: 0.4}, 0.5, "a better rating can't sit lower in the range: rank 2"),
    ],
)
def test_a_rating_table_is_checked(
    positions: dict[int, float], unrated: float, message: str
) -> None:
    with pytest.raises(SizingError, match=message):
        RatingTable(positions, unrated)


# --- nothing is sized to a silent zero -----------------------------------------------------------


@pytest.mark.parametrize("bad", [0.0, -5.0, np.inf, np.nan])
def test_a_price_that_isnt_finite_and_positive_refuses_the_batch(bad: float) -> None:
    with pytest.raises(
        SizingError, match="projected prices must be finite and positive: forward year 2"
    ):
        size(recipients(4), RANGES, RATINGS, [100.0, bad], TWO_YEARS)


def test_an_existing_vest_that_isnt_finite_and_non_negative_names_its_recipient() -> None:
    existing = np.array([[0.0, 0.0], [np.nan, 0.0], [0.0, -1.0]])
    with pytest.raises(SizingError, match=r"for employee 101, employee 102$"):
        size(recipients(3, 3, 3), RANGES, RATINGS, [100.0, 125.0], TWO_YEARS, existing)


def test_a_grant_too_large_to_count_names_its_recipient_and_step() -> None:
    # A price this small asks for more shares than can be counted exactly.
    with pytest.raises(SizingError) as error:
        size(recipients(4, 0), RANGES, RATINGS, [1e-12, 125.0], TWO_YEARS)
    message = str(error.value)
    assert message.startswith("2 recipients got a grant that isn't a finite, countable number")
    assert "none is granted zero instead" in message
    assert "employee 100 (at the shares, forward year 1), employee 101 (at the shares" in message


def test_a_broken_range_names_the_step_it_breaks() -> None:
    values = RANGES.values.copy()
    values[0, 0, 0, 2] = np.nan  # a min_target no loader would accept
    with pytest.raises(SizingError, match=r"employee 100 \(at the target\)"):
        size(recipients(4), RangeTable(values), RATINGS, [100.0, 125.0], TWO_YEARS)


def test_a_long_list_of_refused_recipients_is_counted_past_twenty() -> None:
    with pytest.raises(
        SizingError, match=r"employee 119 \(at the shares, forward year 1\), and 5 more$"
    ):
        size(recipients(*[4] * 25), RANGES, RATINGS, [1e-12, 125.0], TWO_YEARS)
