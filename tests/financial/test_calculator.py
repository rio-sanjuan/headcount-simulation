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
