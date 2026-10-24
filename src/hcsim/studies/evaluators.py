"""Evaluators: score a run and report metrics with their thresholds.

An evaluator reads a `Scoring`, an arm's results, the study's fitted models, and the window's
actuals, and returns metrics. Each declares what it needs of a study (`Need`): a plan, SBC, two
or more trajectories, or actuals, which only a wholly closed window has, and history's are the
whole world's, so a subset study has none. A study without what an evaluator needs skips it,
with the reason. Every metric but the invariants is reported, never gated; gates arrive with
M8.7.

`summary` reports the run's outcomes; the metric baselines compare them. Each of the
`invariants` checks holds by construction, so any failure is a defect, never noise:

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

**Against history**, over a closed window:

- `fidelity`: each event's WAPE and total bias against history's event facts, at cost-center
  and month grain (see `facts`), and the month-end headcount's;
- `sbc`: the WAPE of the monthly SBC and of each month and cost center against the reference
  ledger, and the bias of the total and of each part;
- `award_grants`: the grants of each award type the run awards beside those history recorded,
  and the grants no sizer sized or a replay granted nothing;
- `award_sizing`: the mean shares of each award type's sized grants beside history's.

**Of the run alone**: `domain`, each domain's rate as the run realized it beside the rate its
model was fitted at; and `dispersion`, how widely the trajectories spread.

**Offline**, with no simulation: `fitted_rates`, the rates the models were fitted at; and
`offline_domain`, each rate refitted over the study's own closed window beside the one fitted
over the year before it.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from enum import StrEnum
from types import MappingProxyType
from typing import Any, Final

import numpy as np
import pandas as pd

from hcsim.engine.alignment import AlignmentMethod
from hcsim.engine.ceiling import Ceiling
from hcsim.engine.loop import HEADCOUNT_IDENTITY, OPENING_IDENTITY, SimulationResult
from hcsim.financial.expense import SBC_KEYS
from hcsim.financial.output import align
from hcsim.financial.routing import RecordedAwards
from hcsim.model.core.schema import AWARD_TYPE, COST_CENTER
from hcsim.studies.actuals import Actuals
from hcsim.studies.facts import ACTUAL, dense, run_facts
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


# --- the scoring context and the registry ----------------------------------------------------


@dataclass(frozen=True)
class Scoring:
    """What an evaluator reads: an arm's results, none on the offline backend, the models the
    study fitted, and the window's actuals, where it has them."""

    result: SimulationResult | None
    fitted: FittedModels
    actuals: Actuals | None = None

    @property
    def run(self) -> SimulationResult:
        if self.result is None:
            raise ValueError("an offline arm has no simulated results to score")
        return self.result

    def actual(self, part: str) -> Any:
        if self.actuals is None:
            raise ValueError("the study has no actuals to score against")
        return self.actuals.need(part)


class Need(StrEnum):
    """What an evaluator needs of a study, and is skipped without."""

    PLAN = "plan"
    SBC = "sbc"
    SPREAD = "spread"
    ACTUALS = "actuals"


#: Why an evaluator is skipped without each need.
SKIP_REASONS: Final = MappingProxyType(
    {
        Need.PLAN: "the study has no plan to score against",
        Need.SBC: "the study computes no SBC",
        Need.SPREAD: "a spread across trajectories needs at least two of them",
        Need.ACTUALS: "the window isn't wholly closed, so there are no actuals to score against",
    }
)
#: Why an evaluator that scores against actuals is skipped for a study on a subset.
WHOLE_WORLD: Final = "the study runs a subset, and history's actuals are the whole world's"


@dataclass(frozen=True)
class Evaluator:
    """A scorer: its name, what it scores, the parts of the window's actuals it reads, the input
    kinds those need, and what it needs of a study."""

    name: str
    score: Callable[[Scoring], list[Metric]]
    actuals: tuple[str, ...] = ()
    kinds: tuple[str, ...] = ()
    needs: frozenset[Need] = frozenset()

    def skipped(
        self, *, plan: bool, sbc: bool, trajectories: int, closed: bool, subset: bool
    ) -> str | None:
        """Why a study with these properties skips this evaluator, or None if it runs it."""
        missing = {
            Need.PLAN: not plan,
            Need.SBC: not sbc,
            Need.SPREAD: trajectories < 2,
            Need.ACTUALS: not closed,
        }
        for need in Need:
            if need in self.needs and missing[need]:
                return SKIP_REASONS[need]
        if Need.ACTUALS in self.needs and subset:
            return WHOLE_WORLD
        return None


def _of(score: Callable[[SimulationResult], list[Metric]]) -> Callable[[Scoring], list[Metric]]:
    return lambda s: score(s.run)


# --- fidelity ---------------------------------------------------------------------------------


def _wape(forecast: np.ndarray, actual: np.ndarray) -> float | None:
    total = float(np.abs(actual).sum())
    return float(np.abs(forecast - actual).sum()) / total if total else None


def _bias(forecast: np.ndarray, actual: np.ndarray) -> float | None:
    total = float(actual.sum())
    return (float(forecast.sum()) - total) / total if total else None


def _mean_events(result: SimulationResult) -> dict[str, np.ndarray]:
    """Each event's mean over trajectories, (months, cost centers)."""
    trajectories = [t.trajectory for t in result.trajectories]
    shape = result.headcount.shape
    arrays = dense(run_facts(result), trajectories, shape[1] - 1, shape[2])
    return {name: a.mean(axis=0) for name, a in arrays.items()}


def fidelity(s: Scoring) -> list[Metric]:
    """How closely the run's mean reproduces history over a closed window, at cost-center and
    month grain: each event's WAPE and total bias against history's event facts, and the
    month-end headcount's. Reported, never gated."""
    name = "fidelity"
    result = s.run
    months = result.config.simulation_months
    cost_centers = result.headcount.shape[2]
    history = dense(s.actual("facts"), [ACTUAL], months, cost_centers)
    metrics = []
    for event, mean in _mean_events(result).items():
        actual = history[event][0]
        wape, bias = _wape(mean, actual), _bias(mean, actual)
        if wape is not None and bias is not None:
            detail = f"{int(actual.sum())} in history"
            metrics += [
                Metric(name, f"{event}_wape", wape, None, detail),
                Metric(name, f"{event}_bias", bias, None, "mean total less history's, over it"),
            ]
    heads = result.headcount[:, 1:, :].mean(axis=0)
    actual_heads = s.actual("headcount")
    wape = _wape(heads, actual_heads)
    if wape is not None:
        metrics += [
            Metric(name, "headcount_wape", wape, None, "month-end headcount per cost center"),
            Metric(
                name,
                "headcount_end_bias",
                _bias(heads[-1], actual_heads[-1]) or 0.0,
                None,
                f"{int(actual_heads[-1].sum())} at the window's end",
            ),
        ]
    return metrics


# --- domain and offline_domain ----------------------------------------------------------------

#: The domains whose rates the study's models are fitted at, in order.
DOMAINS: Final = ("attrition", "external_fill", "cancellation", "backfill", "growth")


def run_rates(result: SimulationResult) -> dict[str, float]:
    """Each domain's rate as the run realized it, by the estimator the models were fitted with:
    departures over the headcount at risk; fills and unfilled exits over the openings exposed,
    open at a month's start or opened in it; backfill opened over vacancies; and growth
    requisitions per cost center and month. A rate with nothing at risk is left out."""
    heads = float(result.headcount[:, :-1, :].sum())

    def total(*flows: str) -> float:
        return float(sum(result.flow(f).sum() for f in flows))

    exposed = float(result.open_openings[:, :-1, :].sum()) + total("G", "G_camp", "B_att", "B_tr")
    cells = float(np.prod(result.flow("G").shape))
    pairs = {
        "attrition": (total("A"), heads),
        "external_fill": (total("F_ext"), exposed),
        "cancellation": (total("U"), exposed),
        "backfill": (total("B_att", "B_tr"), total("A", "V_tr")),
        "growth": (total("G"), cells),
    }
    return {d: n / at for d, (n, at) in pairs.items() if at}


def _rate_metrics(
    name: str, rates: Mapping[str, float], fitted: Mapping[str, float]
) -> list[Metric]:
    metrics = []
    for domain in DOMAINS:
        if domain not in rates:
            continue
        rate, base = rates[domain], fitted.get(domain, 0.0)
        detail = f"fitted at {base:.6g}"
        metrics.append(Metric(name, f"{domain}_rate", rate, None, detail))
        if base:
            metrics.append(Metric(name, f"{domain}_rate_ratio", rate / base, None, detail))
    return metrics


def domain(s: Scoring) -> list[Metric]:
    """Each domain's rate as the run realized it, beside the rate its model was fitted at: in an
    unguided run they agree up to sampling, and a guided one shows how far guidance moves each
    domain from its model. Reported, never gated."""
    return _rate_metrics("domain", run_rates(s.run), s.fitted.rates)


def offline_domain(s: Scoring) -> list[Metric]:
    """Each domain's rate out of time, with no simulation: the same estimator refitted over the
    study's own closed window, beside the rate fitted over the year before it. Reported, never
    gated."""
    refit: FittedModels = s.actual("refit")
    return _rate_metrics("offline_domain", refit.rates, s.fitted.rates)


# --- dispersion -------------------------------------------------------------------------------


def _cv(values: np.ndarray) -> float | None:
    mean = float(values.mean())
    return float(values.std(ddof=1)) / mean if mean else None


def dispersion(s: Scoring) -> list[Metric]:
    """How widely the run's trajectories spread: the coefficient of variation of each event's
    total over the run and of the final headcount, the final headcount's 90% interval relative
    to its mean, and the median, over cost centers, of each one's final headcount's coefficient
    of variation. Reported, never gated."""
    name = "dispersion"
    result = s.run
    if len(result.trajectories) < 2:
        return []  # a screen's pass may run fewer trajectories than the study
    metrics = []
    trajectories = [t.trajectory for t in result.trajectories]
    shape = result.headcount.shape
    for event, arrays in dense(run_facts(result), trajectories, shape[1] - 1, shape[2]).items():
        cv = _cv(arrays.sum(axis=(1, 2)).astype(np.float64))
        if cv is not None:
            metrics.append(Metric(name, f"{event}_cv", cv, None, "sd over mean of the run's total"))
    end = result.headcount[:, -1, :].astype(np.float64)
    total = end.sum(axis=1)
    cv = _cv(total)
    if cv is not None:
        low, high = np.quantile(total, [0.05, 0.95])
        metrics += [
            Metric(name, "headcount_end_cv", cv, None, f"over {len(total)} trajectories"),
            Metric(
                name,
                "headcount_end_interval",
                float(high - low) / float(total.mean()),
                None,
                "the 90% interval's width over the mean",
            ),
        ]
    held = end.mean(axis=0) > 0
    if held.any():
        per = end[:, held].std(axis=0, ddof=1) / end[:, held].mean(axis=0)
        detail = f"{int(held.sum())} cost centers holding headcount"
        metrics.append(Metric(name, "headcount_end_cv_median", float(np.median(per)), None, detail))
    return metrics


# --- sbc --------------------------------------------------------------------------------------


def sbc(s: Scoring) -> list[Metric]:
    """How closely the run's mean SBC reproduces the reference ledger over a closed window: the
    WAPE of the monthly total and of each month and cost center, the total's bias, and each
    part's. Reported, never gated."""
    name = "sbc"
    result = s.run
    actual: pd.DataFrame = s.actual("sbc")
    frames = [align(t.sbc, actual) for t in result.trajectories if t.sbc is not None]
    if not frames or actual.empty:
        return []
    n = len(frames)
    keys = ["part", *SBC_KEYS]
    joined = pd.concat(frames, ignore_index=True).groupby(keys)[["forecast", "actual"]].sum()
    joined["forecast"] /= n
    joined["actual"] /= n
    monthly = joined.groupby(level="month")[["forecast", "actual"]].sum()
    cells = joined.groupby(level=["month", "cost_center"])[["forecast", "actual"]].sum()
    metrics = []
    for metric, frame, detail in (
        ("monthly_wape", monthly, "the monthly total"),
        ("cost_center_wape", cells, "each month and cost center"),
    ):
        wape = _wape(frame["forecast"].to_numpy(), frame["actual"].to_numpy())
        if wape is not None:
            metrics.append(Metric(name, metric, wape, None, detail))
    total = _bias(monthly["forecast"].to_numpy(), monthly["actual"].to_numpy())
    if total is not None:
        recognized = f"{float(monthly['actual'].sum()):.6g} recognized in history"
        metrics.append(Metric(name, "total_bias", total, None, recognized))
    parts = joined.groupby(level="part")[["forecast", "actual"]].sum()
    for part, row in parts.iterrows():
        if row["actual"]:
            gap = float(row["forecast"] / row["actual"] - 1)
            metrics.append(Metric(name, f"{part}_bias", gap, None, f"the {part} part"))
    return metrics


# --- award_grants and award_sizing ------------------------------------------------------------


def _run_grants(result: SimulationResult) -> pd.DataFrame:
    """One row per grant the run awarded inside its months, per trajectory: its holder, award
    month, award type label, shares, and whether a recorded award routed it."""
    labels = result.config.schema.vocabulary(AWARD_TYPE).labels
    months = result.config.simulation_months
    rows = []
    for t in result.trajectories:
        tranches = t.tranches
        if tranches is None or not len(tranches):
            continue
        awarded = (tranches.award_month >= 0) & (tranches.award_month < months)
        awarded &= ~tranches.cancelled
        frame = pd.DataFrame(
            {
                "holder": tranches.holder[awarded],
                "award_month": tranches.award_month[awarded],
                "award_type": tranches.award_type[awarded],
                "shares": tranches.shares[awarded],
                "replayed": tranches.recorded[awarded],
            }
        )
        grants = frame.groupby(["holder", "award_month", "award_type"], as_index=False).agg(
            shares=("shares", "sum"), replayed=("replayed", "all")
        )
        grants["trajectory"] = t.trajectory
        rows.append(grants)
    if not rows:
        return pd.DataFrame(columns=["trajectory", "award_type", "shares", "replayed"])
    out = pd.concat(rows, ignore_index=True)
    out["award_type"] = [labels[c] for c in out["award_type"]]
    return out


def _recorded_grants(awards: RecordedAwards, months: int) -> pd.DataFrame:
    """One row per recorded award inside the window: its type and shares."""
    rows = [
        (award_type, float(schedule.sum()))
        for (award_type, step), by_id in awards.schedules.items()
        if 0 <= step < months
        for schedule in by_id.values()
    ]
    return pd.DataFrame(rows, columns=["award_type", "shares"])


def award_grants(s: Scoring) -> list[Metric]:
    """How many grants of each award type the run awards inside its months, on average, beside
    how many history recorded over the closed window, and the grants a run emitted that no
    sizer sized or a replay granted nothing. Reported, never gated."""
    name = "award_grants"
    result = s.run
    months = result.config.simulation_months
    n = len(result.trajectories)
    ran = _run_grants(result).groupby("award_type").size() / n
    recorded = _recorded_grants(s.actual("awards"), months).groupby("award_type").size()
    metrics = []
    for award_type in sorted(set(ran.index) | set(recorded.index)):
        mean, count = float(ran.get(award_type, 0.0)), int(recorded.get(award_type, 0))
        detail = f"{count} recorded"
        metrics.append(Metric(name, f"{award_type}_grants", mean, None, detail))
        if count:
            metrics.append(Metric(name, f"{award_type}_grants_ratio", mean / count, None, detail))
    for counter in ("unsized_grants", "ungranted_grants"):
        total = sum(sum(getattr(t, counter).values()) for t in result.trajectories)
        metrics.append(Metric(name, counter, total / n, None, "per trajectory"))
    return metrics


def award_sizing(s: Scoring) -> list[Metric]:
    """How the run sizes each award type: the mean shares of its sized grants, beside the mean
    of history's recorded awards of that type over the closed window. A replayed grant is
    history's own, so only sized grants are scored. Reported, never gated."""
    name = "award_sizing"
    result = s.run
    grants = _run_grants(result)
    sized = grants[~grants["replayed"].astype(bool)].groupby("award_type")["shares"]
    months = result.config.simulation_months
    recorded = _recorded_grants(s.actual("awards"), months).groupby("award_type")["shares"]
    ran, history = sized.mean(), recorded.mean()
    counts, recorded_counts = sized.size(), recorded.size()
    metrics = []
    for award_type in sorted(ran.index):
        mean = float(ran[award_type])
        detail = (
            f"{int(counts[award_type])} sized grants over {len(result.trajectories)} trajectories"
        )
        metrics.append(Metric(name, f"{award_type}_shares_mean", mean, None, detail))
        if award_type in history.index and history[award_type]:
            base = float(history[award_type])
            detail = f"{int(recorded_counts[award_type])} recorded, {base:.6g} shares on average"
            metrics.append(Metric(name, f"{award_type}_shares_ratio", mean / base, None, detail))
    return metrics


# --- the registries ---------------------------------------------------------------------------

_E = Evaluator
#: Every evaluator a profile can name that scores a simulation's results.
EVALUATORS: Final = MappingProxyType(
    {
        e.name: e
        for e in (
            _E("invariants", _of(invariants)),
            _E("summary", _of(summary)),
            _E(
                "fidelity",
                fidelity,
                ("facts", "headcount"),
                ("realized_totals",),
                frozenset({Need.ACTUALS}),
            ),
            _E("domain", domain),
            _E("dispersion", dispersion, needs=frozenset({Need.SPREAD})),
            _E("plan_alignment", _of(plan_alignment), needs=frozenset({Need.PLAN})),
            _E("sbc", sbc, ("sbc",), ("actual_sbc",), frozenset({Need.SBC, Need.ACTUALS})),
            _E(
                "award_grants",
                award_grants,
                ("awards",),
                ("replay_history",),
                frozenset({Need.SBC, Need.ACTUALS}),
            ),
            _E(
                "award_sizing",
                award_sizing,
                ("awards",),
                ("replay_history",),
                frozenset({Need.SBC, Need.ACTUALS}),
            ),
        )
    }
)


def fitted_rates(fitted: FittedModels) -> list[Metric]:
    """The pooled rates the models were fitted at, over the year before the run: an offline
    evaluator, which needs no simulation. Reported, never gated."""
    return [
        Metric("fitted_rates", f"{name}_rate", float(value), None, "pooled over the fit window")
        for name, value in sorted(fitted.rates.items())
    ]


#: Every evaluator that scores a study's inputs offline, with no simulation.
OFFLINE_EVALUATORS: Final = MappingProxyType(
    {
        e.name: e
        for e in (
            _E("fitted_rates", lambda s: fitted_rates(s.fitted)),
            _E(
                "offline_domain",
                offline_domain,
                ("refit",),
                ("fitted_models",),
                frozenset({Need.ACTUALS}),
            ),
        )
    }
)
