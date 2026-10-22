"""Tests for the RSU tracker's vest buffer: lazy rows, a 120-month tail, and the ledger's live
vests."""

from __future__ import annotations

import numpy as np
import pytest

from hcsim.financial.tracker import NOT_FORFEITED, TAIL, HolderLedger, TrackerError, VestBuffer


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
