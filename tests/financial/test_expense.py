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
        "trueup_rate",
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


# --- preparation and the stateless calculator ----------------------------------------------------


def test_preparation_gives_windows_values_cohorts_and_survival() -> None:
    from hcsim.financial.expense import prepare

    t = Tranches(
        holder=np.array([0, 1, 2, 3]),
        award_month=np.array([-13, 0, 11, 12]),
        vest_month=np.array([5, 6, 20, 30]),
        shares=np.array([10.0, 10.0, 10.0, 10.0]),
        forfeit_month=np.array([-1, 3, 25, -1]),
        award_type=np.zeros(4, dtype=np.int64),
        level=np.zeros(4, dtype=np.int64),
        cost_center=np.zeros(4, dtype=np.int64),
    )
    held = np.ones((4, 2))
    prepared = prepare(t, np.array([2.0, 3.0, 4.0, 5.0]), pd.Period("2026-01", "M"), held)
    assert prepared.value.tolist() == [20.0, 30.0, 40.0, 50.0]
    assert prepared.forfeited.tolist() == [False, True, False, False]  # 25 is past its vest
    assert prepared.last.tolist() == [5, 3, 20, 30]
    assert prepared.cohort.tolist() == [2024, 2026, 2026, 2027]
    assert prepared.survival is not None
    assert prepare(t, np.ones(4)).cohort.tolist() == [-1, -1, -1, -1]


def test_preparation_refuses_every_problem_at_once() -> None:
    from hcsim.financial.expense import prepare

    t = tranches((5, 2, 1.0, -1))
    with pytest.raises(ValueError, match="one value per share") as error:
        prepare(t, np.array([1.0, 2.0]), survival=np.array([[0.5, 0.9]]))
    message = str(error.value)
    assert "one value per share per tranche" in message
    assert "can't vest before its award month" in message
    assert "survival never rises" in message
    with pytest.raises(ValueError, match=r"survival is a chance in \[0, 1\]"):
        prepare(tranches((0, 2, 1.0, -1)), np.ones(1), survival=np.array([[1.5]]))
    with pytest.raises(ValueError, match="survival is \\(tranches, months\\)"):
        prepare(tranches((0, 2, 1.0, -1)), np.ones(1), survival=np.ones(3))


def test_the_calculator_reads_only_its_table() -> None:
    from hcsim.financial.expense import calculate, prepare

    t = tranches((0, 5, 1.0, -1), (2, 9, 1.0, 4))
    prepared = prepare(t, np.array([10.0, 20.0]))
    first, again = calculate(prepared), calculate(prepared)
    pd.testing.assert_frame_equal(first.rows, again.rows)
    pd.testing.assert_frame_equal(first.rows, compute_expense(t, np.array([10.0, 20.0])).rows)


# --- day weights, survival, cohort rates, and the three true-ups ---------------------------------


def one(award: int, vest: int, value: float, forfeit: int = -1) -> Tranches:
    return tranches((award, vest, value, forfeit))


def column(result: object, name: str) -> list[float]:
    rows = result.rows  # type: ignore[attr-defined]
    return [round(float(x), 6) for x in rows.sort_values(["tranche", "month"])[name]]


def test_gross_follows_calendar_days_given_an_origin() -> None:
    from hcsim.financial.expense import calculate, prepare

    # January, February, and March 2026: 31, 28, and 31 of 90 days, on a value of 90.
    flat = ExpenseConfig(forfeiture_rate=0.0)
    result = calculate(prepare(one(0, 2, 90.0), np.ones(1), pd.Period("2026-01", "M")), flat)
    assert column(result, "gross") == [31.0, 28.0, 31.0]
    assert column(result, "trueup_vesting") == [0.0, 0.0, 0.0]
    # At 10%: 81 net, and a 9 vesting true-up.
    tenth = ExpenseConfig(forfeiture_rate=0.1)
    taxed = calculate(prepare(one(0, 2, 90.0), np.ones(1), pd.Period("2026-01", "M")), tenth)
    assert sum(column(taxed, "net")) == pytest.approx(81.0)
    assert column(taxed, "trueup_vesting")[-1] == pytest.approx(9.0)


def test_uncertain_survival_gives_the_expected_expense() -> None:
    from hcsim.financial.expense import calculate, prepare

    # A value of 100 over two months; the holder has an even chance of leaving in the first.
    # Month 0 accrues 45 net; leaving reverses it with chance 0.5; month 1 accrues 45 with chance
    # 0.5, and the vest trues up 10 with chance 0.5. In all, 50: half the value.
    prepared = prepare(one(0, 1, 100.0), np.ones(1), survival=np.array([[0.5, 0.5]]))
    result = calculate(prepared, ExpenseConfig(forfeiture_rate=0.1))
    assert column(result, "net") == [45.0, 22.5]
    assert column(result, "trueup_attrition") == [-22.5, 0.0]
    assert column(result, "trueup_vesting") == [0.0, 5.0]
    total = result.rows[["net", "trueup_attrition", "trueup_rate", "trueup_vesting"]].sum().sum()
    assert total == pytest.approx(50.0)


def test_a_cohort_rate_is_recomputed_every_month_with_a_rate_true_up() -> None:
    from hcsim.financial.expense import calculate, prepare

    # One cohort: A (100 over months 0-3) forfeits in month 1; B (100 over months 0-2) vests in
    # month 2. The rate starts at the prior 10%, is 100% once A's shares forfeit and nothing has
    # vested, and 50% once B vests.
    t = tranches((0, 3, 100.0, 1), (0, 2, 100.0, -1))
    result = calculate(prepare(t, np.ones(2)), ExpenseConfig(forfeiture="cohort"))
    rows = result.rows.set_index(["tranche", "month"])
    b = rows.loc[1]
    assert b["net"].tolist() == pytest.approx([30.0, 0.0, 100 / 6], abs=1e-6)
    # Month 1 takes back what month 0 accrued at the rate's rise; month 2 gives half of the two
    # months' gross back as the rate falls to 50%.
    assert b["trueup_rate"].tolist() == pytest.approx([0.0, -30.0, 100 / 3], abs=1e-6)
    assert b["trueup_vesting"].tolist() == pytest.approx([0.0, 0.0, 50.0], abs=1e-6)
    a = rows.loc[0]
    assert a[
        ["net", "trueup_attrition", "trueup_rate", "trueup_vesting"]
    ].sum().sum() == pytest.approx(0.0, abs=1e-6)
    assert result.max_identity_residual < 1e-6


def test_cohorts_are_the_calendar_years_awards_fall_in() -> None:
    from hcsim.financial.expense import calculate, prepare

    # Given an origin in December 2025, month 0 is 2025's cohort and month 1 2026's: one's
    # forfeit doesn't move the other's rate.
    cohort = ExpenseConfig(forfeiture="cohort")
    origin = pd.Period("2025-12", "M")
    both = calculate(
        prepare(tranches((0, 3, 100.0, 1), (1, 3, 100.0, -1)), np.ones(2), origin), cohort
    )
    alone = calculate(prepare(tranches((1, 3, 100.0, -1)), np.ones(1), origin), cohort)
    ignore = ["tranche", "cost_center"]
    later = both.rows[both.rows["tranche"] == 1].drop(columns=ignore).reset_index(drop=True)
    pd.testing.assert_frame_equal(later, alone.rows.drop(columns=ignore))
    same_year = calculate(
        prepare(
            tranches((0, 3, 100.0, 1), (1, 3, 100.0, -1)), np.ones(2), pd.Period("2026-01", "M")
        ),
        cohort,
    )
    moved = (
        same_year.rows[same_year.rows["tranche"] == 1].drop(columns=ignore).reset_index(drop=True)
    )
    assert not moved.equals(alone.rows.drop(columns=ignore))


def test_the_identity_holds_in_expectation_for_any_mix() -> None:
    from hcsim.financial.expense import calculate, prepare

    rng = np.random.default_rng(9)
    n = 200
    award = rng.integers(-24, 12, n)
    vest = award + rng.integers(0, 48, n)
    forfeit = np.where(rng.random(n) < 0.3, award + rng.integers(-2, 50, n), -1)
    rows = tuple(zip(award, vest, rng.integers(1, 500, n).astype(float), forfeit, strict=True))
    t = tranches(*rows)
    survival = np.minimum.accumulate(rng.uniform(0.6, 1.0, (n, 36)), axis=1)
    for held in (None, survival):
        for forfeiture in ("flat", "cohort"):
            prepared = prepare(t, rng.uniform(5, 50, n), pd.Period("2025-07", "M"), held)
            config = ExpenseConfig(forfeiture=forfeiture)
            whole = calculate(prepared, config)
            assert whole.max_identity_residual < 1e-6
            split = calculate(prepared, config, first_month=0)
            assert split.opening is not None
            parts = split.rows[["net", "trueup_attrition", "trueup_rate", "trueup_vesting"]]
            assert parts.to_numpy().sum() + split.opening.sum() == pytest.approx(
                whole.rows[parts.columns].to_numpy().sum()
            )


def test_the_forfeiture_setting_is_checked() -> None:
    with pytest.raises(ValueError, match="'flat' or 'cohort'"):
        ExpenseConfig(forfeiture="annual")  # type: ignore[arg-type]


# --- the forfeited-piece path, chunks, toggles, and the identity ---------------------------------


def test_a_forfeit_reverses_each_piece_in_its_own_cost_center() -> None:
    from hcsim.financial.expense import calculate, prepare
    from hcsim.financial.tracker import Intervals

    # 100 over months 0-4, 20 a month. The holder moves from cost center 0 to 1 in month 2 and
    # leaves in month 3, so each cost center recognized 40, and each reverses its own 40.
    t = tranches((0, 4, 100.0, 3))
    t = Tranches(**{**t.__dict__, "cost_center": np.array([0])})
    moves = Intervals(np.array([0]), np.array([2]), np.array([1]))
    flat = ExpenseConfig(forfeiture_rate=0.0)
    result = calculate(prepare(t, np.ones(1), intervals=moves), flat)
    frame = sbc_frame(result, t)
    by_center = frame.groupby("cost_center")[["net", "trueup_attrition", "recognized"]].sum()
    assert by_center.loc[0].to_numpy().tolist() == pytest.approx([40.0, -40.0, 0.0])
    assert by_center.loc[1].to_numpy().tolist() == pytest.approx([40.0, -40.0, 0.0])
    reversals = result.rows[result.rows["trueup_attrition"] != 0]
    assert reversals[["month", "cost_center"]].values.tolist() == [[3, 1], [3, 0]]


def test_chunks_give_bit_identical_results() -> None:
    from hcsim.financial.expense import calculate, prepare
    from hcsim.financial.tracker import Intervals

    rng = np.random.default_rng(4)
    n = 300
    award = rng.integers(-24, 12, n)
    vest = award + rng.integers(0, 48, n)
    forfeit = np.where(rng.random(n) < 0.3, award + rng.integers(0, 50, n), -1)
    t = tranches(*zip(award, vest, rng.integers(1, 500, n).astype(float), forfeit, strict=True))
    survival = np.minimum.accumulate(rng.uniform(0.7, 1.0, (n, 24)), axis=1)
    moves = Intervals(np.arange(0, n, 5), rng.integers(-5, 20, n // 5), rng.integers(0, 4, n // 5))
    order = np.lexsort((moves.valid_from, moves.holder))
    moves = Intervals(moves.holder[order], moves.valid_from[order], moves.cost_center[order])
    prepared = prepare(t, rng.uniform(5, 50, n), pd.Period("2025-07", "M"), survival, moves)
    config = ExpenseConfig(forfeiture="cohort")
    whole = calculate(prepared, config, first_month=0)
    for size in (1, 7, 128):
        chunked = calculate(prepared, config, first_month=0, chunk_size=size)
        pd.testing.assert_frame_equal(chunked.rows, whole.rows, check_exact=True)
        assert np.array_equal(chunked.residual, whole.residual)
        assert chunked.opening is not None
        assert whole.opening is not None
        assert np.array_equal(chunked.opening, whole.opening)
    with pytest.raises(ValueError, match="a chunk holds at least one tranche"):
        calculate(prepared, config, chunk_size=0)
    with pytest.raises(ValueError, match="a chunk holds at least one tranche"):
        ExpenseConfig(chunk_size=0)


def test_a_true_up_left_out_leaves_the_identity_unenforced() -> None:
    from hcsim.financial.expense import calculate, prepare

    t = tranches((0, 9, 100.0, -1), (0, 9, 100.0, 4))
    prepared = prepare(t, np.ones(2))
    full = calculate(prepared, RATE)
    assert full.identity_enforced
    assert full.max_identity_residual < 1e-9
    without = calculate(
        prepared, ExpenseConfig(forfeiture_rate=0.2, trueups=frozenset({"trueup_rate"}))
    )
    assert not without.identity_enforced
    assert (without.rows[["trueup_vesting", "trueup_attrition"]] == 0).all().all()
    # The vesting tranche falls 20% short, and the forfeited one keeps what it accrued.
    assert without.residual.tolist() == pytest.approx([-20.0, 40.0])
    assert without.worst_tranche == 1
    with pytest.raises(ValueError, match="true-ups are"):
        ExpenseConfig(trueups=frozenset({"trueup_bonus"}))


def test_a_defect_in_the_identity_raises() -> None:
    from hcsim.financial.expense import AccountingIdentityError, check_identity

    with pytest.raises(AccountingIdentityError, match="tranche 1 recognizes"):
        check_identity(np.array([10.0, 12.0]), np.array([10.0, 10.0]), 0.01)
