"""Tests for the Phase 3 expense calculator.

The examples are synthetic and computed by hand. One tranche of 600 shares at 50 is worth
30,000. Awarded in month 0 and vesting in month 9, it accrues over 10 months: 3,000 gross a
month, and 2,400 net at a 20% forfeiture rate.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from hcsim.financial.expense import (
    AccountingIdentityError,
    ExpenseConfig,
    check_identity,
    compute_expense,
    sbc_frame,
)
from hcsim.financial.tracker import NOT_FORFEITED, Tranches

RATE = ExpenseConfig(forfeiture_rate=0.20)


def tranches(*rows: tuple[int, int, float, int]) -> Tranches:
    """(award month, vest month, shares, forfeit month) per tranche, one holder each."""
    a, v, s, f = (np.array(col) for col in zip(*rows, strict=True))
    n = len(rows)
    return Tranches(
        holder=np.arange(n),
        award_month=a.astype(np.int64),
        vest_month=v.astype(np.int64),
        shares=s.astype(np.float64),
        forfeit_month=f.astype(np.int64),
        award_type=np.zeros(n, dtype=np.int64),
        level=np.zeros(n, dtype=np.int64),
        cost_center=np.arange(n, dtype=np.int64) % 2,
    )


def per_month(result: pd.DataFrame, column: str) -> dict[int, float]:
    summed = result.groupby("month")[column].sum()
    return {
        int(m): round(float(x), 6)
        for m, x in zip(summed.index.to_numpy(), summed.to_numpy(), strict=True)
        if x
    }


def test_a_tranche_that_vests_recognizes_its_full_value() -> None:
    result = compute_expense(tranches((0, 9, 600, NOT_FORFEITED)), np.array([50.0]), RATE)
    assert per_month(result.rows, "gross") == dict.fromkeys(range(10), 3000.0)
    assert per_month(result.rows, "net") == dict.fromkeys(range(10), 2400.0)
    # The vesting true-up restores the forfeiture discount on shares that did vest.
    assert per_month(result.rows, "trueup_vesting") == {9: 6_000.0}
    assert result.rows[["net", "trueup_vesting"]].to_numpy().sum() == pytest.approx(30_000.0)
    assert result.max_identity_residual < 1e-9


def test_a_forfeited_tranche_reverses_everything_recognized() -> None:
    result = compute_expense(tranches((0, 9, 600, 3)), np.array([50.0]), RATE)
    assert per_month(result.rows, "net") == dict.fromkeys(range(4), 2400.0)  # months 0 to 3
    assert per_month(result.rows, "trueup_attrition") == {3: -9_600.0}
    assert per_month(result.rows, "trueup_vesting") == {}
    recognized = result.rows[["net", "trueup_attrition", "trueup_vesting"]].to_numpy().sum()
    assert recognized == pytest.approx(0.0)  # zero vested shares times 50


def test_a_departure_on_or_after_the_vest_month_forfeits_nothing() -> None:
    result = compute_expense(
        tranches((0, 9, 600, 9), (0, 9, 600, 12)), np.array([50.0, 50.0]), RATE
    )
    assert per_month(result.rows, "trueup_attrition") == {}
    assert per_month(result.rows, "trueup_vesting") == {9: 12_000.0}


def test_the_sbc_frame_sums_by_cost_center_award_type_and_level() -> None:
    t = tranches((0, 9, 600, NOT_FORFEITED), (0, 9, 600, 3), (2, 3, 100, NOT_FORFEITED))
    frame = sbc_frame(compute_expense(t, np.array([50.0, 50.0, 10.0]), RATE), t)
    assert list(frame.columns) == [
        "month",
        "cost_center",
        "award_type",
        "level",
        "gross",
        "net",
        "trueup_attrition",
        "trueup_vesting",
        "trueup",
        "recognized",
    ]
    assert (frame["recognized"] == frame["net"] + frame["trueup"]).all()
    # Tranches 0 and 2 share cost center 0; tranche 1 is in cost center 1 and nets to zero.
    by_cc = frame.groupby("cost_center")["recognized"].sum()
    assert by_cc[0] == pytest.approx(30_000.0 + 1_000.0)
    assert by_cc[1] == pytest.approx(0.0)
    assert frame["month"].max() == 9  # the frame runs to the last vest month


def test_an_empty_ledger_has_no_expense() -> None:
    none = np.zeros(0, dtype=np.int64)
    t = Tranches(
        holder=none,
        award_month=none,
        vest_month=none,
        shares=np.zeros(0),
        forfeit_month=none,
        award_type=none,
        level=none,
        cost_center=none,
    )
    result = compute_expense(t, np.zeros(0), RATE)
    assert result.rows.empty
    assert result.max_identity_residual == 0.0


@pytest.mark.parametrize(
    ("vps", "match"),
    [(np.array([50.0, 1.0]), "one value per share"), (np.array([-1.0]), "non-negative")],
)
def test_bad_values_are_refused(vps: np.ndarray, match: str) -> None:
    with pytest.raises(ValueError, match=match):
        compute_expense(tranches((0, 9, 600, NOT_FORFEITED)), vps, RATE)


def test_a_tranche_that_vests_before_its_award_is_refused() -> None:
    with pytest.raises(ValueError, match="vest before"):
        compute_expense(tranches((5, 4, 1, NOT_FORFEITED)), np.array([1.0]), RATE)


def test_the_forfeiture_rate_must_be_below_one() -> None:
    with pytest.raises(ValueError, match=r"\[0, 1\)"):
        ExpenseConfig(forfeiture_rate=1.0)


def test_an_identity_violation_names_the_worst_tranche() -> None:
    recognized = np.array([100.0, 49.0, 0.004])
    vested = np.array([100.0, 50.0, 0.0])
    with pytest.raises(AccountingIdentityError, match="tranche 1 recognizes 49"):
        check_identity(recognized, vested, 0.01)
    within = check_identity(np.array([100.005]), np.array([100.0]), 0.01)
    assert within.tolist() == pytest.approx([0.005])
