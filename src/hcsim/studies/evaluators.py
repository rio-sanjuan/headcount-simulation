"""Evaluators: score a run and report metrics with their thresholds.

The walking skeleton has two evaluators. `summary` reports the run's outcomes without gating
them; the metric baselines compare them. Each of the `invariants` checks holds by construction,
so any failure is a defect, never noise:

- **Quota.** Every guided draw realizes exactly `min(quota, eligible)` in every group.
- **Headcount identity.** `H(t+1) = H(t) + F_ext - A - V_conv + R_in - R_out + W_in - W_out`,
  exactly, per cost center.
- **Opening identity.** `O(t+1) = O(t) + G + B_att - U - F_ext`, exactly, per cost center.
- **Accounting identity.** Every tranche's recognized expense ties to its vested value.

The other evaluators arrive in M8.6.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np

from hcsim.engine.loop import HEADCOUNT_IDENTITY, OPENING_IDENTITY, SimulationResult

#: The skeleton's identities, as flow symbol to sign. They grow as M4 adds flows.


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


def summary(result: SimulationResult) -> list[Metric]:
    """The run's outcomes, as means over trajectories. Reported, never gated."""
    name = "summary"
    months = result.config.simulation_months
    headcount = result.headcount.sum(axis=2)  # (trajectories, months + 1)
    end = headcount[:, -1]

    def flow_total(symbol: str) -> float:
        return float(result.flow(symbol).sum(axis=(1, 2)).mean())

    metrics = [
        Metric(name, "headcount_start", float(headcount[0, 0]), None),
        Metric(name, "headcount_end_mean", float(end.mean()), None),
        Metric(name, "headcount_end_sd", float(end.std(ddof=1)) if len(end) > 1 else 0.0, None),
        Metric(
            name,
            "open_openings_end_mean",
            float(result.open_openings.sum(axis=2)[:, -1].mean()),
            None,
        ),
        Metric(name, "attrition_mean", flow_total("A"), None, "A over the run"),
        Metric(name, "external_fills_mean", flow_total("F_ext"), None, "F_ext over the run"),
        Metric(name, "backfills_opened_mean", flow_total("B_att"), None, "B_att over the run"),
        Metric(name, "growth_opened_mean", flow_total("G"), None, "G over the run"),
        Metric(name, "unfilled_exits_mean", flow_total("U"), None, "U over the run"),
    ]
    if result.config.computes_sbc:
        recognized = [
            float(t.sbc.loc[t.sbc["month"] < months, "recognized"].sum())
            for t in result.trajectories
            if t.sbc is not None
        ]
        tranches = [len(t.tranches) for t in result.trajectories if t.tranches is not None]
        metrics += [
            Metric(
                name,
                "sbc_recognized_mean",
                float(np.mean(recognized)),
                None,
                "recognized inside the run's months",
            ),
            Metric(name, "tranches_mean", float(np.mean(tranches)), None),
        ]
    return metrics
