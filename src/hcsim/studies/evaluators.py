"""Evaluators: score a run and report metrics with their thresholds.

This is the walking skeleton's one evaluator, `invariants`. Each of its checks holds by
construction, so any failure is a defect, never noise:

- **Quota.** Every guided draw realizes exactly `min(quota, eligible)` in every group.
- **Headcount identity.** `H(t+1) = H(t) + F_ext - A`, exactly, per cost center.
- **Opening identity.** `O(t+1) = O(t) + G + B_att - U - F_ext`, exactly, per cost center.
- **Accounting identity.** Every tranche's recognized expense ties to its vested value.

The other evaluators arrive in M8.6.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Final

import numpy as np

from hcsim.engine.loop import SimulationResult

#: The skeleton's identities, as flow symbol to sign. They grow as M4 adds flows.
HEADCOUNT_IDENTITY: Final[Mapping[str, int]] = {"F_ext": +1, "A": -1}
OPENING_IDENTITY: Final[Mapping[str, int]] = {"G": +1, "B_att": +1, "U": -1, "F_ext": -1}


@dataclass(frozen=True)
class Metric:
    evaluator: str
    name: str
    value: float
    #: The largest passing value. None marks a metric that's reported but not gated.
    threshold: float | None
    detail: str = ""

    @property
    def passed(self) -> bool:
        return self.threshold is None or self.value <= self.threshold


def _identity_gap(stock: np.ndarray, result: SimulationResult, terms: Mapping[str, int]) -> int:
    expected = stock[:, :-1] + sum(sign * result.flow(symbol) for symbol, sign in terms.items())
    return int(np.abs(stock[:, 1:] - expected).max(initial=0))


def invariants(result: SimulationResult) -> list[Metric]:
    name = "invariants"
    rows = [row for t in result.trajectories for row in t.quotas]
    violations = sum(
        int((r.record.realized != np.minimum(r.record.quota, r.record.eligible)).sum())
        for r in rows
    )
    shortfall = sum(int((r.record.quota - r.record.realized).sum()) for r in rows)
    metrics = [
        Metric(name, "quota_violations", violations, 0, f"{len(rows)} guided draws checked"),
        Metric(name, "quota_shortfall", shortfall, None, "quota that eligibility couldn't supply"),
        Metric(
            name,
            "headcount_identity_max_gap",
            _identity_gap(result.headcount, result, HEADCOUNT_IDENTITY),
            0,
        ),
        Metric(
            name,
            "opening_identity_max_gap",
            _identity_gap(result.open_openings, result, OPENING_IDENTITY),
            0,
        ),
    ]
    if result.config.computes_sbc:
        assert result.config.awards is not None
        residual = max(
            (t.expense.max_identity_residual for t in result.trajectories if t.expense),
            default=0.0,
        )
        tranches = sum(len(t.tranches) for t in result.trajectories if t.tranches)
        tolerance = result.config.awards.expense.identity_tolerance
        metrics.append(
            Metric(
                name,
                "accounting_identity_max_residual",
                residual,
                tolerance,
                f"{tranches} tranches checked",
            )
        )
    return metrics
