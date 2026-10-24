"""Evaluators: score a run and report metrics with their thresholds.

The walking skeleton has two evaluators. `summary` reports the run's outcomes without gating
them; the metric baselines compare them. Each of the `invariants` checks holds by construction,
so any failure is a defect, never noise:

- **Quota.** Every guided draw honors its quota in every group: it realizes exactly
  `min(quota, eligible)`, less what no whole requisition could fit, unless known events alone
  passed it (`QuotaRecord.honored`).
- **Headcount identity.** `H(t+1) = H(t) + F_ext - A - V_conv + R_in - R_out + W_in - W_out`,
  exactly, per cost center.
- **Opening identity.** `O(t+1) = O(t) + G + B_att - U - F_ext`, exactly, per cost center.
- **Accounting identity.** Every tranche's recognized expense ties to its vested value.

**Plan alignment** (`plan_alignment`) scores how close a run lands on its plan at the plan's
evaluation month `t*`, on the headcount at that month's end, at cost-center grain and never the
total, since plans can agree on totals while disagreeing on most cost centers. Per target, as the
notation defines them: `bias = |mean_j H - H*|`, `spread = sd_j H`, and `D = max_j |H - H*|`.
The regimes say how each target was driven: the valve the alignment method gave it (supply, flow,
rate, or none, and `undriven` for one no valve reaches), and the route its growth took most
months up to `t*` (G-A, G-B, or G-C); the metrics break the bias down by valve, and
`plan_alignment_table` holds a row per target.

The other evaluators arrive in M8.6.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Final

import numpy as np
import pandas as pd

from hcsim.engine.alignment import AlignmentMethod
from hcsim.engine.ceiling import Ceiling
from hcsim.engine.loop import HEADCOUNT_IDENTITY, OPENING_IDENTITY, SimulationResult
from hcsim.model.core.schema import COST_CENTER
from hcsim.studies.inputs import FittedModels

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
    violations = sum(int((~r.record.honored).sum()) for r in rows)
    shortfall = sum(int(r.record.shortfall.sum()) for r in rows)
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
            float(t.sbc.loc[t.sbc["month"].between(0, months - 1), "recognized"].sum())
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


# --- plan alignment ---------------------------------------------------------------------------

#: The valves a target can be driven by, and `undriven` for one no valve reaches.
VALVE_LABELS: Final = ("supply", "flow", "rate", "undriven", "none")
#: Growth's regime codes (`engine.domains.Regime`) as the routes they are.
ROUTE_LABELS: Final = {0: "G-C", 1: "G-A", 2: "G-B"}
#: The plan-alignment table's columns, one row per target.
PLAN_COLUMNS: Final = (
    "cost_center",
    "target",
    "mean",
    "bias",
    "spread",
    "deviation",
    "valve",
    "route",
)


def plan_alignment_table(result: SimulationResult) -> pd.DataFrame | None:
    """One row per cost center the plan targets at its evaluation month `t*`, scored on the
    headcount at that month's end: the target, the mean over trajectories, `bias`, `spread`,
    and `D` (`deviation`) as the notation defines them, the valve that drives the target, and the
    route its growth took most months up to `t*`. None for a run with no plan, one that ends
    before `t*`, or a streamed run, which holds no trajectories."""
    config = result.config
    if config.plan is None or not result.trajectories:
        return None
    s = (config.plan.evaluation_month - pd.Period(config.run_date, freq="M")).n
    if not 0 <= s < config.simulation_months:
        return None
    whole = Ceiling.build(config.schema, {}, config.plan)
    target = whole.targets(config.run_date, s)
    heads = result.headcount[:, s + 1, :].astype(np.float64)  # the end of month t*
    labels = config.schema.vocabulary(COST_CENTER).labels
    alignment = config.alignment
    if alignment.method is AlignmentMethod.NONE:
        valve_of = dict.fromkeys(labels, "none")
    else:
        valve_of = {
            **dict.fromkeys(alignment.undriven, "undriven"),
            **dict.fromkeys(alignment.supply, "supply"),
            **dict.fromkeys(alignment.flow, "flow"),
            **dict.fromkeys(alignment.rate, "rate"),
        }
    regimes = [t.growth_regimes for t in result.trajectories if t.growth_regimes is not None]
    rows = []
    for c in np.flatnonzero(~np.isnan(target)):
        h = heads[:, c]
        seen = np.concatenate([r[: s + 1, c] for r in regimes]) if regimes else np.zeros(0)
        seen = seen[seen >= 0]
        route = ROUTE_LABELS[int(np.bincount(seen.astype(np.int64)).argmax())] if len(seen) else ""
        rows.append(
            {
                "cost_center": labels[c],
                "target": float(target[c]),
                "mean": float(h.mean()),
                "bias": float(abs(h.mean() - target[c])),
                "spread": float(h.std(ddof=1)) if len(h) > 1 else 0.0,
                "deviation": float(np.abs(h - target[c]).max()),
                "valve": valve_of.get(labels[c], "undriven"),
                "route": route,
            }
        )
    return pd.DataFrame(rows, columns=PLAN_COLUMNS)


def plan_alignment(result: SimulationResult) -> list[Metric]:
    """How close the run lands on its plan at `t*`, at cost-center grain, never the total:
    bias, spread, and `D` across the targeted cost centers, and the mean bias of the targets
    each valve drives. Reported, never gated; nothing for a run the table can't score."""
    table = plan_alignment_table(result)
    if table is None:
        return []
    name = "plan_alignment"
    config = result.config
    assert config.plan is not None
    s = (config.plan.evaluation_month - pd.Period(config.run_date, freq="M")).n
    scored = len(table)
    metrics = [
        Metric(name, "evaluation_step", float(s), None, f"t* = {config.plan.evaluation_month}"),
        Metric(name, "targets_scored", float(scored), None, "cost centers the plan targets at t*"),
    ]
    if not scored:
        return metrics
    metrics += [
        Metric(name, "bias_mean", float(table["bias"].mean()), None, "|mean H - H*| per target"),
        Metric(name, "bias_max", float(table["bias"].max()), None),
        Metric(name, "spread_mean", float(table["spread"].mean()), None, "sd of H per target"),
        Metric(name, "deviation_max", float(table["deviation"].max()), None, "D: max |H - H*|"),
    ]
    for valve in _valves_in_order(table["valve"]):
        mine = table[table["valve"] == valve]
        metrics.append(
            Metric(
                name,
                f"bias_mean_{valve}",
                float(mine["bias"].mean()),
                None,
                f"{len(mine)} targets driven by {valve}",
            )
        )
    return metrics


def _valves_in_order(column: pd.Series) -> list[str]:
    present = set(column)
    return [v for v in VALVE_LABELS if v in present]


#: Every evaluator a profile can name that scores a simulation's results.
EVALUATORS: Final = MappingProxyType(
    {"invariants": invariants, "summary": summary, "plan_alignment": plan_alignment}
)


def fitted_rates(fitted: FittedModels) -> list[Metric]:
    """The pooled rates the models were fitted at, over the year before the run: an offline
    evaluator, which needs no simulation. Reported, never gated."""
    return [
        Metric("fitted_rates", f"{name}_rate", float(value), None, "pooled over the fit window")
        for name, value in sorted(fitted.rates.items())
    ]


#: Every evaluator that scores a study's inputs offline, with no simulation.
OFFLINE_EVALUATORS: Final = MappingProxyType({"fitted_rates": fitted_rates})
