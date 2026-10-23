"""Tests for the RSU tracker's vest buffer: lazy rows, a 120-month tail, and the ledger's live
vests."""

from __future__ import annotations

import numpy as np
import pytest

from hcsim.financial.tracker import (
    EIGHT_D,
    NINE_D,
    NOT_FORFEITED,
    TAIL,
    HolderLedger,
    TrackerError,
    VestBuffer,
)


def schedule(**vests: int) -> np.ndarray:
    """A one-row 72-month schedule from `m<k>=shares` entries."""
    out = np.zeros((1, 72), dtype=np.int64)
    for key, shares in vests.items():
        out[0, int(key[1:]) - 1] = shares
    return out


def test_a_holder_takes_a_row_only_when_first_granted() -> None:
    ledger = HolderLedger()
    ledger.buffer.cover(12)
    assert ledger.buffer.allocated == 0
    ledger.record(np.array([5]), 0, schedule(m3=10), 0, np.array([0]), np.array([0]))
    ledger.record(np.array([5]), 1, schedule(m3=4), 0, np.array([0]), np.array([0]))
    ledger.record(np.array([9]), 1, schedule(m6=7), 0, np.array([0]), np.array([0]))
    assert ledger.buffer.allocated == 2
    held = ledger.buffer.held(np.array([5, 9, 6]))
    assert held.shape == (3, 12 + TAIL)
    assert {int(m): int(v) for m, v in enumerate(held[0]) if v} == {3: 10, 4: 4}
    assert {int(m): int(v) for m, v in enumerate(held[1]) if v} == {7: 7}
    assert not held[2].any()  # never granted: no row


def test_the_buffer_spans_the_run_and_a_120_month_tail() -> None:
    buffer = VestBuffer(months=24)
    assert buffer.width == 24 + TAIL
    buffer.add(np.array([1, 1]), np.array([-2, 24 + TAIL - 1]), np.array([5, 6]))
    assert int(buffer.held(np.array([1])).sum()) == 6  # a vest before the run has happened
    with pytest.raises(TrackerError, match="vest past the vest buffer's 120-month tail"):
        buffer.add(np.array([1]), np.array([24 + TAIL]), np.array([1]))
    buffer.cover(36)
    assert buffer.width == 36 + TAIL
    buffer.add(np.array([1]), np.array([24 + TAIL]), np.array([1]))
    assert int(buffer.held(np.array([1])).sum()) == 7


def test_a_forfeit_clears_the_holders_vests_after_the_month() -> None:
    buffer = VestBuffer(months=12)
    buffer.add(np.array([1, 1, 2]), np.array([3, 9, 9]), np.array([10, 20, 30]))
    buffer.forfeit(np.array([1, 7]), 5)  # 7 holds nothing
    assert buffer.held(np.array([1, 2]))[:, 9].tolist() == [0, 30]
    assert buffer.held(np.array([1]))[0, 3] == 10
    assert buffer.by_year(np.array([1, 2]), 0, 1).tolist() == [[10.0], [30.0]]


def test_the_buffer_agrees_with_the_ledgers_live_tranches() -> None:
    rng = np.random.default_rng(3)
    ledger = HolderLedger()
    ledger.buffer.cover(36)
    for step in range(36):
        holders = rng.choice(40, size=5, replace=False)
        grants = (rng.random((5, 72)) < 0.05) * rng.integers(1, 50, (5, 72))
        ledger.record(holders, step, grants, 0, np.zeros(5, int), np.zeros(5, int))
        if step % 4 == 0:
            ledger.forfeit(rng.choice(40, size=3, replace=False), step)
    t = ledger.tranches()
    live = t.forfeit_month == NOT_FORFEITED
    holders = np.arange(40)
    expected = np.zeros((40, ledger.buffer.width), dtype=np.int64)
    np.add.at(expected, (t.holder[live], t.vest_month[live]), t.shares[live].astype(np.int64))
    assert np.array_equal(ledger.buffer.held(holders), expected)
    for month in (0, 10, 30):
        window = ledger.vesting_by_year(holders, month, 3)
        by_hand = np.stack(
            [
                expected[:, month + 12 * y + 1 : month + 12 * (y + 1) + 1].sum(axis=1)
                for y in range(3)
            ],
            axis=1,
        )
        assert np.array_equal(window, by_hand)


# --- moves and timed grants ----------------------------------------------------------------------


def test_a_move_starts_a_holder_interval_and_the_last_in_a_month_holds() -> None:
    ledger = HolderLedger()
    ledger.move(np.array([1, 2]), 4, np.array([3, 8]))
    ledger.move(np.array([1]), 4, np.array([5]))  # moved again the same month: this one holds
    ledger.move(np.array([1]), 9, np.array([2]))
    moves = ledger.intervals()
    assert list(zip(moves.holder, moves.valid_from, moves.cost_center, strict=True)) == [
        (1, 4, 5),
        (1, 9, 2),
        (2, 4, 8),
    ]
    holders = np.array([1, 1, 1, 1, 1, 2, 2, 6])
    months = np.array([0, 4, 8, 9, 20, 3, 4, 10])
    granted = np.full(8, 7)
    assert ledger.cost_center_at(holders, months, granted).tolist() == [7, 5, 5, 2, 2, 7, 8, 7]


def test_a_timed_grant_waits_and_is_cancelled_if_its_holder_leaves_first() -> None:
    ledger = HolderLedger()
    ledger.buffer.cover(24)
    # Recorded at step 2 for an award in month 6, and another for holder 4 in month 5.
    ledger.record(
        np.array([3, 4]),
        6,
        np.vstack([schedule(m3=10), schedule(m3=10)]),
        0,
        np.zeros(2, int),
        np.zeros(2, int),
    )
    assert ledger.tranches().pending(2).tolist() == [True, True]
    assert int(ledger.vesting_by_year(np.array([3]), 2, 1)[0, 0]) == 10  # already counted
    ledger.forfeit(np.array([3]), 3)  # holder 3 leaves before the award
    t = ledger.tranches()
    assert t.cancelled.tolist() == [True, False]
    assert t.pending(5).tolist() == [False, True]
    assert t.pending(6).tolist() == [False, False]  # awarded
    assert int(ledger.vesting_by_year(np.array([3]), 3, 1)[0, 0]) == 0


def test_a_cancelled_grant_recognizes_nothing() -> None:
    from hcsim.financial.expense import compute_expense

    ledger = HolderLedger()
    ledger.record(np.array([3]), 6, schedule(m3=10), 0, np.zeros(1, int), np.zeros(1, int))
    ledger.forfeit(np.array([3]), 3)
    result = compute_expense(ledger.tranches(), np.array([50.0]))
    rows = result.rows
    assert rows[["gross", "net", "trueup_attrition", "trueup_vesting"]].to_numpy().sum() == 0
    assert result.max_identity_residual == 0


# --- scratch -------------------------------------------------------------------------------------


def test_a_holder_is_inside_until_they_leave_and_again_once_they_join() -> None:
    ledger = HolderLedger()
    ledger.leave(np.array([1]), 4)
    ledger.join(np.array([1]), 9)
    ledger.leave(np.array([2]), 3)
    ledger.join(np.array([2]), 3)  # back the same month: the last change holds
    months = np.array([3, 4, 8, 9, 3, 7])
    holders = np.array([1, 1, 1, 1, 2, 5])
    assert ledger.inside(holders, months).tolist() == [True, False, False, True, True, True]


def test_tranches_move_in_and_out_of_scratch_with_their_holder() -> None:
    ledger = HolderLedger()
    ledger.buffer.cover(24)
    zeros = np.zeros(2, int)
    ledger.record(
        np.array([1, 2]), 0, np.vstack([schedule(m20=10), schedule(m20=10)]), 0, zeros, zeros
    )
    ledger.leave(np.array([1]), 4)  # moves out of the subset, keeping its equity
    ledger.join(np.array([1]), 9)
    assert ledger.scratch(3).tolist() == [False, False]
    assert ledger.scratch(4).tolist() == [True, False]
    assert ledger.scratch(9).tolist() == [False, False]
    assert ledger.tranches().forfeit_month.tolist() == [NOT_FORFEITED, NOT_FORFEITED]


def test_a_known_future_entrant_is_in_scratch_until_they_start() -> None:
    ledger = HolderLedger()
    ledger.buffer.cover(24)
    ledger.record(np.array([7]), 0, schedule(m24=10), 0, np.zeros(1, int), np.zeros(1, int))
    ledger.leave(np.array([7]), 0)  # recorded before they start, in month 5
    ledger.join(np.array([7]), 5)
    assert [bool(ledger.scratch(m)[0]) for m in (0, 4, 5)] == [True, True, False]
    ledger.forfeit(np.array([7]), 8)  # a forfeited tranche is in neither side after it
    assert not ledger.scratch(9)[0]


def test_scratch_is_the_unpublished_part_of_each_tranches_expense() -> None:
    from hcsim.model.data.publication import publish

    ledger = HolderLedger()
    ledger.leave(np.array([1]), 2)
    ledger.join(np.array([1]), 4)
    months = np.arange(6)
    expense = np.full((1, 6), 10.0)
    inside = ledger.inside(np.full(6, 1), months)[None, :]
    split = publish(expense, np.array([0]), inside)
    assert split.published.tolist() == [[10.0, 10.0, 0.0, 0.0, 10.0, 10.0]]
    assert split.scratch.tolist() == [[0.0, 0.0, 10.0, 10.0, 0.0, 0.0]]
    assert np.array_equal(split.total, expense)


# --- views ---------------------------------------------------------------------------------------


def granted() -> HolderLedger:
    """Holder 1, granted in cost center 3 at month 0, vesting at 12; holder 2, granted in cost
    center 4 at month 2, vesting at 14 but leaving at 6."""
    ledger = HolderLedger()
    ledger.buffer.cover(24)
    ledger.record(np.array([1]), 0, schedule(m12=10), 2, np.array([5]), np.array([3]))
    ledger.record(np.array([2]), 2, schedule(m12=20), 1, np.array([6]), np.array([4]))
    ledger.forfeit(np.array([2]), 6)
    return ledger


def test_the_8d_view_is_one_row_per_tranche_with_its_eight_dimensions() -> None:
    view = granted().view8()
    assert tuple(view.columns) == EIGHT_D
    assert view.to_dict("records") == [
        {
            "holder": 1,
            "award_month": 0,
            "vest_month": 12,
            "shares": 10,
            "forfeit_month": NOT_FORFEITED,
            "award_type": 2,
            "level": 5,
            "cost_center": 3,
        },
        {
            "holder": 2,
            "award_month": 2,
            "vest_month": 14,
            "shares": 20,
            "forfeit_month": 6,
            "award_type": 1,
            "level": 6,
            "cost_center": 4,
        },
    ]


def test_the_9d_view_cuts_each_tranche_at_its_holders_moves() -> None:
    ledger = granted()
    ledger.move(np.array([1]), 4, np.array([8]))  # inside holder 1's accrual
    ledger.move(np.array([1]), 20, np.array([9]))  # after its vest: no piece
    ledger.move(np.array([2]), 1, np.array([7]))  # before holder 2's award: it starts there
    ledger.move(np.array([2]), 9, np.array([5]))  # after holder 2 left: no piece
    view = ledger.view9()
    assert tuple(view.columns) == NINE_D
    pieces = view[["holder", "cost_center", "valid_from"]].values.tolist()
    assert pieces == [[1, 3, 0], [1, 8, 4], [2, 7, 2]]
    # Each piece keeps its tranche's other dimensions.
    assert view["shares"].tolist() == [10, 10, 20]
    assert granted().view9()["cost_center"].tolist() == [3, 4]  # no moves: one piece each


def test_tranches_recorded_one_by_one_are_checked() -> None:
    ledger = HolderLedger()
    one = np.ones(1, np.int64)
    none = np.zeros(0, np.int64)
    assert ledger.record_tranches(none, none, none, none, none, none, none, np.zeros(0)) == 0
    with pytest.raises(TrackerError, match="one value of each column per tranche"):
        ledger.record_tranches(one, one, one, one, one, one, one, np.zeros(2))
    with pytest.raises(TrackerError, match="can't vest before its award month"):
        ledger.record_tranches(one, one * 5, one * 2, one, one, one, one, np.ones(1))
    assert ledger.record_tranches(one, one * -2, one * 3, one * 4, one, one, one, np.ones(1)) == 1
    assert ledger.tranches().values.tolist() == [1.0]
