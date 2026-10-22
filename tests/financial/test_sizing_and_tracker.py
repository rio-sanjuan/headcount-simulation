"""Tests for ranges, vest templates, new-hire sizing, and the holder ledger."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from hcsim.financial.calculator import RatingTable
from hcsim.financial.prices import AnnualGrowthPrice
from hcsim.financial.ranges import RangeError, RangeTable
from hcsim.financial.sizing import NewHireGapFill
from hcsim.financial.tracker import NOT_FORFEITED, HolderLedger
from hcsim.financial.vesting import NEW_HIRE_TEMPLATE, VestTemplate, VestTemplateError
from hcsim.model.core.layouts import EMPLOYEE, EmployeeColumn
from hcsim.model.core.schema import (
    COST_CENTER,
    JOB_FAMILY,
    JOB_LEVEL,
    LOCATION,
    SimulationSchema,
)

SCHEMA = SimulationSchema.build(
    {
        COST_CENTER: ["CC-01"],
        JOB_FAMILY: ["Finance", "Sales"],
        JOB_LEVEL: [1, 2],
        LOCATION: ["LOC-1"],
    }
)


def range_frame(**drop: object) -> pd.DataFrame:
    rows = [
        {
            "job_family": f,
            "job_level": lvl,
            "location": "LOC-1",
            "min_base": 80_000,
            "max_base": 120_000,
            "min_target": 120_000,
            "max_target": 180_000,
            "base_adj": 1.0,
        }
        for f in ("Finance", "Sales")
        for lvl in (1, 2)
    ]
    return pd.DataFrame(rows)


# --- ranges -------------------------------------------------------------------------------------


def test_a_full_range_frame_builds_and_looks_up_by_codes() -> None:
    table = RangeTable.from_frame(range_frame(), SCHEMA)
    assert table.values.shape == (2, 2, 1, 5)
    r = table.lookup(np.array([1]), np.array([0]), np.array([0]))
    assert r[0, 0] == 80_000


def test_a_missing_range_is_an_error_naming_the_cell() -> None:
    frame = range_frame().iloc[:-1]  # drop Sales at level 2
    with pytest.raises(
        RangeError, match="1 cells have no range, and nothing is imputed: Sales level 2 at LOC-1"
    ):
        RangeTable.from_frame(frame, SCHEMA)


WIDE = SimulationSchema.build(
    {
        COST_CENTER: ["CC-01"],
        JOB_FAMILY: ["Finance", "Sales"],
        JOB_LEVEL: [1, 2, 9],  # 9 is an executive level, which needs its ranges too
        LOCATION: ["LOC-1", "LOC-2", "LOC-3"],
    }
)


def wide_frame() -> pd.DataFrame:
    base = range_frame().drop(columns=["job_family", "job_level", "location"]).iloc[0]
    cells = pd.MultiIndex.from_product(
        [["Finance", "Sales"], [1, 2, 9], ["LOC-1", "LOC-2", "LOC-3"]],
        names=["job_family", "job_level", "location"],
    )
    return pd.DataFrame([base] * len(cells), index=cells).reset_index()


def test_every_missing_cell_is_named_executive_levels_included() -> None:
    frame = wide_frame()
    keep = ~(
        ((frame["job_level"] == 9) & (frame["job_family"] == "Finance"))
        | ((frame["job_family"] == "Sales") & (frame["location"] == "LOC-2"))
    )
    with pytest.raises(RangeError) as error:
        RangeTable.from_frame(frame[keep], WIDE)
    assert (
        "6 cells have no range, and nothing is imputed: Finance level 9 at LOC-1, LOC-2, LOC-3; "
        "Sales level 1 at LOC-2; Sales level 2 at LOC-2; Sales level 9 at LOC-2"
    ) in str(error.value)
    assert RangeTable.from_frame(wide_frame(), WIDE).values.shape == (2, 3, 3, 5)


def test_every_problem_is_named_at_once() -> None:
    frame = wide_frame()
    frame.loc[0, "min_base"] = 200_000  # Finance level 1 at LOC-1: inverted
    frame.loc[1, "max_target"] = np.nan  # Finance level 1 at LOC-2: not finite
    frame.loc[2, "base_adj"] = 1.1  # a second multiplier
    extra = frame.iloc[[3]].assign(job_family="Legal")  # a family the schema lacks: left out
    frame = pd.concat([frame, frame.iloc[[4]], extra], ignore_index=True)  # a repeated cell
    with pytest.raises(RangeError) as error:
        RangeTable.from_frame(frame.iloc[1:].reset_index(drop=True), WIDE)
    message = str(error.value)
    assert "1 cells are listed more than once: Finance level 2 at LOC-2" in message
    assert "aren't finite and non-negative: Finance level 1 at LOC-2" in message
    assert "base_adj must be one positive multiplier on every row, not [1.0, 1.1]" in message
    assert "1 cells have no range, and nothing is imputed: Finance level 1 at LOC-1" in message
    assert "Legal" not in message
    with pytest.raises(RangeError, match="minimum above their maximum: Finance level 1 at LOC-1"):
        RangeTable.from_frame(frame.iloc[:-2], WIDE)


def test_ranges_need_their_columns() -> None:
    with pytest.raises(RangeError, match=r"need the columns \['base_adj'\]"):
        RangeTable.from_frame(range_frame().drop(columns="base_adj"), SCHEMA)


def test_an_unattributed_recipient_cant_be_sized() -> None:
    table = RangeTable.from_frame(range_frame(), SCHEMA)
    with pytest.raises(RangeError, match="unattributed level"):
        table.lookup(np.array([0]), np.array([-1]), np.array([0]))


# --- vest templates -----------------------------------------------------------------------------


def test_the_new_hire_template_places_a_cliff_then_quarterly_chunks() -> None:
    schedule = NEW_HIRE_TEMPLATE.expand(np.array([[400, 320, 256, 205]]))
    nonzero = {int(c) + 1: int(schedule[0, c]) for c in np.flatnonzero(schedule[0])}
    assert nonzero == {
        12: 400,
        15: 80,
        18: 80,
        21: 80,
        24: 80,
        27: 64,
        30: 64,
        33: 64,
        36: 64,
        39: 51,
        42: 51,
        45: 51,
        48: 52,
    }  # the year's last chunk absorbs rounding
    assert schedule.shape == (1, 72)
    assert int(schedule.sum()) == 400 + 320 + 256 + 205


@pytest.mark.parametrize(
    ("years", "match"),
    [((), "1 to 6"), ((((13, 1.0),),), "1-12"), ((((12, 0.5),),), "sum to 1")],
)
def test_invalid_templates_are_refused(
    years: tuple[tuple[tuple[int, float], ...], ...], match: str
) -> None:
    with pytest.raises(VestTemplateError, match=match):
        VestTemplate("bad", years)


def test_expansion_needs_one_column_per_forward_year() -> None:
    with pytest.raises(VestTemplateError, match=r"\(N, 4\)"):
        NEW_HIRE_TEMPLATE.expand(np.array([[1, 2]]))
    with pytest.raises(VestTemplateError, match="non-negative"):
        NEW_HIRE_TEMPLATE.expand(np.array([[1, -2, 0, 0]]))


# --- new-hire sizing ----------------------------------------------------------------------------


def recipients(n: int = 1) -> np.ndarray:
    rows = EMPLOYEE.empty(n)
    rows[:, EmployeeColumn.JOB_FAMILY] = 0  # Finance
    rows[:, EmployeeColumn.LEVEL] = 1  # level 2
    return rows


def test_a_hand_computed_new_hire_grant() -> None:
    """Base midpoint 100,000, target 150,000 at the midpoint position, so each forward year's
    gap is 50,000. At 125, 156.25, 195.3125, and 244.140625, that's 400, 320, 256, and 205."""
    strategy = NewHireGapFill(RangeTable.from_frame(range_frame(), SCHEMA))
    batch = strategy.size(recipients(), AnnualGrowthPrice(100.0, growth_rate=1.25))
    assert batch.base_salary.tolist() == [100_000.0]
    assert batch.shares_per_year.tolist() == [[400, 320, 256, 205]]
    assert batch.total_shares.tolist() == [1181]
    assert int(batch.vest_schedule.sum()) == 1181


def test_existing_vests_reduce_the_gap_and_the_gap_never_goes_negative() -> None:
    strategy = NewHireGapFill(RangeTable.from_frame(range_frame(), SCHEMA))
    prices = AnnualGrowthPrice(100.0, growth_rate=1.25)
    batch = strategy.size(recipients(), prices, np.array([[100.0, 1_000.0, 0.0, 0.0]]))
    assert batch.shares_per_year.tolist() == [[300, 0, 256, 205]]  # 50,000 - 12,500; then clipped


def test_a_rated_recipient_sits_where_the_rating_table_puts_them() -> None:
    rated = recipients(2)
    rated[1, EmployeeColumn.PERF_RATING] = 5
    table = RatingTable({1: 0.0, 5: 1.0}, unrated=0.5)
    strategy = NewHireGapFill(RangeTable.from_frame(range_frame(), SCHEMA), ratings=table)
    batch = strategy.size(rated, AnnualGrowthPrice(100.0, growth_rate=1.25))
    # At the top of the range: base 120,000 and target 180,000, a 60,000 gap: 480 at 125.
    assert batch.base_salary.tolist() == [100_000.0, 120_000.0]
    assert batch.shares_per_year[:, 0].tolist() == [400, 480]


# --- holder ledger ------------------------------------------------------------------------------


def test_recording_lays_out_one_row_per_vesting_month() -> None:
    ledger = HolderLedger()
    schedule = NEW_HIRE_TEMPLATE.expand(np.array([[400, 320, 256, 205], [4, 0, 0, 0]]))
    count = ledger.record(
        np.array([7, 8]),
        3,
        schedule,
        award_type=0,
        levels=np.array([1, 0]),
        cost_centers=np.array([0, 0]),
    )
    assert count == 13 + 1
    t = ledger.tranches()
    assert len(t) == 14
    assert sorted(t.vest_month[t.holder == 8].tolist()) == [3 + 12]
    assert (t.forfeit_month == NOT_FORFEITED).all()
    assert int(t.shares.sum()) == 1181 + 4


def test_a_departure_forfeits_only_the_tranches_that_vest_after_it() -> None:
    ledger = HolderLedger()
    schedule = NEW_HIRE_TEMPLATE.expand(np.array([[400, 320, 256, 205]]))
    ledger.record(np.array([7]), 0, schedule, 0, np.array([1]), np.array([0]))
    assert ledger.forfeit(np.array([7]), 15) == 13 - 2  # months 12 and 15 already vested
    t = ledger.tranches()
    assert (t.forfeit_month[t.vest_month <= 15] == NOT_FORFEITED).all()
    assert (t.forfeit_month[t.vest_month > 15] == 15).all()
    assert ledger.forfeit(np.array([7]), 20) == 0  # already forfeited rows stay at month 15


def test_the_anchor_can_differ_from_the_award_month() -> None:
    ledger = HolderLedger()
    schedule = NEW_HIRE_TEMPLATE.expand(np.array([[4, 0, 0, 0]]))
    ledger.record(
        np.array([1]),
        award_month=5,
        schedule=schedule,
        award_type=0,
        levels=np.array([0]),
        cost_centers=np.array([0]),
        anchor_month=4,
    )
    t = ledger.tranches()
    assert (t.award_month.tolist(), t.vest_month.tolist()) == ([5], [16])


def test_an_empty_grant_records_nothing() -> None:
    ledger = HolderLedger()
    assert (
        ledger.record(
            np.array([1]), 0, np.zeros((1, 72), dtype=np.int64), 0, np.array([0]), np.array([0])
        )
        == 0
    )
    assert len(ledger) == 0
