"""Simulation supply: how growth's new requisitions are routed and sized per cost center.

**Routing.** Each month, each cost center's growth takes one route:

- **G-A, the plan's supply**, where the alignment method lets the plan drive supply and the plan
  targets the cost center: the supply rule sizes the requisitions that close its gap;
- **G-B, guidance**, where guidance covers its requisitions opened otherwise: the residual of the
  requisitions-opened quota, after backfill's use of it;
- **G-C, the rate**, everywhere else: the growth model's own draw at its historical rate.

Every route's count is then clamped at the room the effective ceiling leaves.

**Supply rules.** On G-A, with `S = H + O` the committed strength and `β` the supply gain:

- `strength` closes `β` of the gap between the target and committed strength, counting every
  open opening as a head to come: `ceil(β (H* - S))`;
- `expected_fill` counts an open opening as only the `y` of a head it's expected to yield, and
  opens enough for the gap to fill at that yield: `ceil(β (H* - H - y O) / y)`;
- `soft_bridge` aims at the target at the next target date, `d` months ahead, and closes the gap
  to it evenly over the months left, with gain `k = 1 / (d + 1)`: `ceil(k (H*(next) - S))`. Past
  the plan's last date, it closes the gap to the last target at once.

No rule opens a negative count: a cost center at or past its target opens nothing.
"""

from __future__ import annotations

from enum import StrEnum

import numpy as np
from numpy.typing import NDArray

Floats = NDArray[np.float64]
Ints = NDArray[np.int64]


class Route(StrEnum):
    """Growth's three routes, as its diagnostics name them."""

    SUPPLY = "G-A"
    GUIDED = "G-B"
    RATE = "G-C"


class SupplyRule(StrEnum):
    STRENGTH = "strength"
    EXPECTED_FILL = "expected_fill"
    SOFT_BRIDGE = "soft_bridge"


def _opened(wanted: Floats) -> Ints:
    out: Ints = np.maximum(0, np.ceil(np.nan_to_num(wanted, nan=0.0))).astype(np.int64)
    return out


def strength(target: Floats, headcount: Ints, openings: Ints, gain: float) -> Ints:
    """`ceil(β (H* - S))`, never negative, and nothing where there's no target."""
    return _opened(gain * (target - (headcount + openings)))


def expected_fill(
    target: Floats, headcount: Ints, openings: Ints, gain: float, fill_yield: float
) -> Ints:
    """`ceil(β (H* - H - y O) / y)`, never negative, and nothing where there's no target."""
    expected = headcount + fill_yield * openings
    return _opened(gain * (target - expected) / fill_yield)


def soft_bridge(next_target: Floats, months_ahead: Floats, headcount: Ints, openings: Ints) -> Ints:
    """`ceil(k (H*(next) - S))` with `k = 1 / (d + 1)`, never negative, and nothing where
    there's no target ahead."""
    return _opened((next_target - (headcount + openings)) / (months_ahead + 1.0))
