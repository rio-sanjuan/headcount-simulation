"""Tests for requisition generation's regimes and the effective-ceiling clamp."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from hcsim.engine.config import ConfigError
from hcsim.engine.domains import Regime, StepContext, growth, regimes
from hcsim.engine.seeds import SeedManager
from hcsim.model.core.alignment import AlignmentTarget, Interpolation, marginal
from hcsim.model.core.layouts import OpeningColumn
from tests.engine.test_loop import config, guidance, initial_state

# The toy state: 30 heads and 4 open openings per cost center, so S = 34 everywhere.


def plan(targets: dict[str, float]) -> AlignmentTarget:
    m = marginal("cost_center", list(targets), ["2026-01"], [[v] for v in targets.values()])
    return AlignmentTarget((m,), m.dates[0], 6, 6, Interpolation.HOLD_LAST)


def grow(**overrides: Any) -> tuple[Any, StepContext]:
    ctx = StepContext(config(**overrides), 0, 0, SeedManager(8))
    return growth(initial_state(), ctx), ctx


def opened(result: Any) -> list[int]:
    cc = result.openings.append[:, OpeningColumn.COST_CENTER].astype(int)
    return np.bincount(cc, minlength=3).tolist()


def test_each_cost_center_takes_its_regime() -> None:
    ctx = StepContext(
        config(
            guidance=guidance({"requisitions_opened": {"CC-01": 2.0}}), plan=plan({"CC-02": 40.0})
        ),
        0,
        0,
        SeedManager(1),
    )
    assert regimes(initial_state(), ctx).tolist() == [Regime.GUIDED, Regime.TARGET, Regime.RATE]


def test_target_seeking_closes_the_gap_by_the_supply_gain() -> None:
    result, _ = grow(plan=plan({"CC-02": 40.0}))
    assert opened(result)[1] == 6  # 40 less the 34 committed
    half, _ = grow(plan=plan({"CC-02": 40.0}), requisitions_supply_gain=0.5)
    assert opened(half)[1] == 3
    behind, _ = grow(plan=plan({"CC-02": 20.0}))
    assert opened(behind)[1] == 0  # already past the target: nothing opens


def test_every_regime_is_clamped_at_the_ceiling() -> None:
    free, _ = grow()
    assert opened(free)[2] > 0  # CC-03 draws at its rate
    capped, _ = grow(capacity_caps={"CC-03": 34})
    assert opened(capped)[2] == 0
    target, _ = grow(plan=plan({"CC-02": 40.0}), capacity_caps={"CC-02": 37})
    assert opened(target)[1] == 3


def test_a_guided_count_is_clamped_and_its_ledger_follows() -> None:
    result, _ = grow(
        guidance=guidance({"requisitions_opened": {"CC-01": 5.0}}), capacity_caps={"CC-01": 36}
    )
    assert opened(result)[0] == 2
    assert result.ledger["requisitions_opened"].tolist() == [2]
    assert result.quotas["requisitions_opened"].realized.tolist() == [2]


def test_the_supply_gain_is_a_share() -> None:
    with pytest.raises(ConfigError, match="requisitions_supply_gain must lie in"):
        config(requisitions_supply_gain=1.5)
