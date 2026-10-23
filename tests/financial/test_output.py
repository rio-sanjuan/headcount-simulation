"""Tests for SBC output: parts, bases, and alignment to actuals."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from hcsim.financial.expense import SBC_KEYS, SBC_VALUES, ExpenseConfig, compute_expense
from hcsim.financial.output import PART_KEYS, OutputError, align, sbc_parts
from hcsim.financial.tracker import HolderLedger

FLAT = ExpenseConfig(forfeiture_rate=0.0)


def ledger() -> HolderLedger:
    """A seed tranche, a replayed run grant, and a sized one, each worth 100 over months to 4."""
    book = HolderLedger()
    one = np.ones(1, np.int64)
    book.record_tranches(
        one * 1, one * -1, one * 4, one * 10, one * 0, one * 0, one * 0, np.array([10.0])
    )
    schedule = np.zeros((2, 72), dtype=np.int64)
    schedule[:, 3] = 10  # vests 4 months after an award in month 0
    book.record(
        np.array([2, 3]),
        0,
        schedule,
        1,
        np.zeros(2, int),
        np.zeros(2, int),
        replayed=np.array([True, False]),
    )
    return book


def parts() -> pd.DataFrame:
    t = ledger().tranches()
    values = np.where(np.isnan(t.values), 10.0, t.values)
    return sbc_parts(compute_expense(t, values, FLAT, first_month=0), t)


def test_each_row_is_labelled_by_part_and_basis() -> None:
    frame = parts()
    assert list(frame.columns) == [*PART_KEYS, *SBC_VALUES]
    totals = frame.groupby(["part", "basis"])["recognized"].sum()
    # The seed's 100 accrues over the six months -1 to 4, a sixth of it before the run; the
    # run's two grants recognize all 100 each.
    assert totals.to_dict() == pytest.approx(
        {("seed", "replayed"): 100 - 100 / 6, ("sim", "replayed"): 100.0, ("sim", "sized"): 100.0}
    )


def test_the_offline_and_manual_streams_join_as_their_own_parts() -> None:
    t = ledger().tranches()
    keyed = {k: [0] for k in SBC_KEYS} | {v: [0.0] for v in SBC_VALUES}
    offline = pd.DataFrame(keyed | {"recognized": [7.0]})
    manual = pd.DataFrame(keyed | {"month": [2], "recognized": [-3.0]})
    values = np.where(np.isnan(t.values), 10.0, t.values)
    frame = sbc_parts(compute_expense(t, values, FLAT, first_month=0), t, offline, manual)
    labelled = frame[frame["part"].isin(["offline", "manual"])]
    assert labelled[["part", "basis", "recognized"]].values.tolist() == [
        ["manual", "sized", -3.0],
        ["offline", "replayed", 7.0],
    ]


def test_a_forecast_aligns_to_the_actuals_months_and_keys() -> None:
    forecast = parts()
    actual = forecast[forecast["month"] < 2].copy()
    actual["recognized"] *= 1.1
    actual = pd.concat([actual, actual.iloc[:1].assign(part="offline", recognized=5.0)])
    joined = align(forecast, actual)
    assert set(joined["month"]) == {0, 1}  # only the months the actuals cover
    offline = joined[joined["part"] == "offline"]
    assert offline[["forecast", "actual", "gap"]].values.tolist() == [[0.0, 5.0, -5.0]]
    seed = joined[(joined["part"] == "seed")]
    assert np.allclose(seed["gap"], -0.1 * seed["forecast"])
    with pytest.raises(OutputError, match=r"the actual SBC needs the columns \['part'\]"):
        align(forecast, actual.drop(columns="part"))


def test_chunks_parts_sum_into_one_frame() -> None:
    from hcsim.financial.output import combine_parts, streams

    whole = parts()
    halves = [whole.iloc[::2], whole.iloc[1::2]]
    pd.testing.assert_frame_equal(combine_parts(halves), whole, check_exact=False)
    doubled = combine_parts([whole, whole])
    assert doubled["recognized"].sum() == pytest.approx(2 * whole["recognized"].sum())
    assert combine_parts([]).empty
    assert streams(None, None) == []


def test_only_rows_whose_holder_is_inside_that_month_are_published() -> None:
    from hcsim.financial.output import combine_parts, publish

    book = ledger()
    book.leave(np.array([3]), 2)  # the sized grant's holder leaves the subset from month 2
    t = book.tranches()
    values = np.where(np.isnan(t.values), 10.0, t.values)
    result = compute_expense(t, values, FLAT, first_month=0)
    rows = result.rows
    holder = t.holder[rows["tranche"].to_numpy()]
    shown, held = publish(result, book.inside(holder, rows["month"].to_numpy()))
    assert len(shown.rows) + len(held.rows) == len(rows)
    outside = held.rows[["tranche", "month"]].to_numpy()
    assert set(t.holder[outside[:, 0]]) == {3}
    assert outside[:, 1].min() == 2
    assert np.array_equal(shown.residual, result.residual)
    whole = combine_parts([sbc_parts(shown, t), sbc_parts(held, t)])
    pd.testing.assert_frame_equal(whole, sbc_parts(result, t), check_exact=False)
    with pytest.raises(OutputError, match=r"one membership flag per row: \d+, not \(1,\)"):
        publish(result, np.ones(1, dtype=bool))
