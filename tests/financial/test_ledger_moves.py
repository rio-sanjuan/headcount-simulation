"""Tests for a holder ledger with no moves: no intervals, and every holder where they were
granted."""

from __future__ import annotations

import numpy as np

from hcsim.financial.tracker import HolderLedger


def test_a_ledger_without_moves_has_no_intervals_and_keeps_granted_cost_centers() -> None:
    ledger = HolderLedger()
    assert len(ledger.intervals()) == 0
    granted = np.array([3, 4])
    assert ledger.cost_center_at(np.array([1, 2]), np.array([0, 9]), granted).tolist() == [3, 4]
