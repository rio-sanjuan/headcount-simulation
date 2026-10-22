"""Tests for cancellation: a monthly hazard, and weighted Gumbel-top-k against an openings
quota."""

from __future__ import annotations

import numpy as np
from hypothesis import given, settings
from hypothesis import strategies as st

from hcsim.engine.domains import StepContext, cancellation, weighted_top_k
from hcsim.engine.seeds import SeedManager
from hcsim.model.core.layouts import OpeningColumn
from tests.engine.test_loop import config, guidance, initial_state, models

# The toy state's requisitions: two per cost center, with one and three openings.


@given(
    weights=st.lists(st.integers(1, 5), min_size=1, max_size=12),
    quota=st.integers(0, 20),
    seed=st.integers(0, 1_000),
)
@settings(max_examples=60, deadline=None)
def test_a_group_never_passes_its_openings_quota(weights: list[int], quota: int, seed: int) -> None:
    w = np.asarray(weights, dtype=np.int64)
    p = np.full(len(w), 0.5)
    groups = np.zeros(len(w), dtype=np.int64)
    chosen, record = weighted_top_k(p, w, groups, np.array([quota]), np.random.default_rng(seed))
    assert int(w[chosen].sum()) <= quota
    assert record.realized.tolist() == [int(w[chosen].sum())]
    assert record.eligible.tolist() == [int(w.sum())]
    # Nothing left out would still fit: what's taken is as full as the order allows.
    left = quota - int(w[chosen].sum())
    if quota >= int(w.sum()):
        assert chosen.all()
    else:
        assert left < int(w.max())


def test_rows_with_no_chance_or_no_group_are_never_taken() -> None:
    p = np.array([0.0, 0.9, 0.9])
    groups = np.array([0, -1, 0])
    chosen, _ = weighted_top_k(
        p, np.ones(3, dtype=np.int64), groups, np.array([5]), np.random.default_rng(1)
    )
    assert chosen.tolist() == [False, False, True]


def test_an_unguided_requisition_exits_at_its_hazard() -> None:
    ctx = StepContext(config(models=models(cancel=1.0)), 0, 0, SeedManager(1))
    assert cancellation(initial_state(), ctx).openings.drop.size == 6
    ctx = StepContext(config(models=models(cancel=0.0)), 0, 0, SeedManager(1))
    assert cancellation(initial_state(), ctx).openings.is_empty


def test_a_guided_group_closes_up_to_its_openings_quota() -> None:
    # CC-01 holds a one-opening and a three-opening requisition; guidance asks for 3 openings.
    ctx = StepContext(
        config(
            models=models(cancel=0.5),
            guidance=guidance({"requisitions_closed_unfilled": {"CC-01": 3.0}}),
        ),
        0,
        0,
        SeedManager(4),
    )
    result = cancellation(initial_state(), ctx)
    reqs = initial_state().live_openings
    dropped = reqs[result.openings.drop]
    in_cc01 = dropped[dropped[:, OpeningColumn.COST_CENTER] == 0]
    closed = int(in_cc01[:, OpeningColumn.NUM_OPENINGS].sum())
    assert closed <= 3
    record = result.quotas["requisitions_closed_unfilled"]
    assert record.quota.tolist() == [3]
    assert record.realized.tolist() == [closed]
    assert record.eligible.tolist() == [4]
