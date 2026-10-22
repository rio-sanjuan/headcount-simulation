"""The guidance study ladder: six arms for each of three series, fitted through a cutoff and
forecast over the year after it.

A ladder study asks what a richer guidance configuration buys over a simpler one, for the
series whose quotas move a run most: attrition, hires, and internal starts. Each series has
six arms, ranked by simplicity from a null arm, the benchmark every other arm is measured
against, to a combination of the rest:

| rank | arm | models |
|---|---|---|
| 0 | `seasonal_naive` (null) | `native.SeasonalNaive` |
| 1 | `benchmarks` | `native.SeasonalNaive`, `native.HistoricalAverage`, `native.Drift` |
| 2 | `exponential_smoothing` | `statsforecast.AutoETS` |
| 3 | `log_count_arima` | `native.LogCountARIMA` |
| 4 | the series' own | a GLM, or Croston's method: see below |
| 5 | `ensemble` | every model above, its top 3 by backtest error |

The fourth arm fits the series: departures are close to Poisson (`native.PoissonGLM`), hires
come in bursts a negative binomial absorbs (`native.NegativeBinomialGLM`), and internal starts
are intermittent (`statsforecast.CrostonSBA`). An arm with more than one
model weights them by inverse backtest error.

Each arm is an ordinary guidance configuration (`config_for`): the series and the headcount
anchor, which marks every loaded month, at the study's grouping; the snapshot fixed at the
cutoff; and no reconciliation, so each arm's forecast is its models' own. `run_ladder` fits
every arm and sets its forecast beside the actuals the following months realized, by the
guidance measures, on the grid of the arm's groups. A group an arm didn't forecast, as when its
history was all zeros, forecasts zero.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, Final

import duckdb
import pandas as pd

from hcsim.guidance.config import GuidanceConfig
from hcsim.guidance.ingest import CATEGORY, GROUP, MONTH, VALUE, ingest
from hcsim.guidance.models import MODELS
from hcsim.guidance.pipeline import run
from hcsim.model.core.categories import ANCHOR, GuidanceCategory

_C = GuidanceCategory
#: The series a ladder study compares arms on.
SERIES: Final = (_C.ATTRITION, _C.HIRES, _C.INTERNAL_STARTS)
#: The months each arm forecasts after the cutoff.
HORIZON: Final = 12
ARMS_PER_SERIES: Final = 6
#: Rolling-origin folds that fit inside a few years of history.
EVALUATION: Final = {"n_splits": 3, "test_months": 6, "step_months": 3, "min_train_months": 24}
FORECAST: Final = "forecast"
ACTUAL: Final = "actual"


class LadderError(ValueError):
    """Arms, or a study, that don't hold."""


@dataclass(frozen=True)
class Arm:
    """One guidance configuration for one series, ranked by simplicity."""

    name: str
    series: GuidanceCategory
    simplicity: int
    models: tuple[str, ...]
    null: bool = False
    top_k: int = 3

    @property
    def key(self) -> str:
        return f"{self.series.value}/{self.name}"


_BENCHMARKS: Final = ("native.SeasonalNaive", "native.HistoricalAverage", "native.Drift")
_OWN: Final = {
    _C.ATTRITION: ("poisson_glm", "native.PoissonGLM"),
    _C.HIRES: ("negative_binomial_glm", "native.NegativeBinomialGLM"),
    _C.INTERNAL_STARTS: ("croston_sba", "statsforecast.CrostonSBA"),
}


def _series_arms(series: GuidanceCategory) -> tuple[Arm, ...]:
    own, model = _OWN[series]
    ladder = [
        ("seasonal_naive", ("native.SeasonalNaive",)),
        ("benchmarks", _BENCHMARKS),
        ("exponential_smoothing", ("statsforecast.AutoETS",)),
        ("log_count_arima", ("native.LogCountARIMA",)),
        (own, (model,)),
    ]
    every = tuple(dict.fromkeys(m for _, models in ladder for m in models))
    ladder.append(("ensemble", every))
    return tuple(
        Arm(name, series, rank, models, null=rank == 0)
        for rank, (name, models) in enumerate(ladder)
    )


#: Every series' arms, in simplicity order.
ARMS: Final = tuple(arm for series in SERIES for arm in _series_arms(series))


def problems(arms: Sequence[Arm]) -> list[str]:
    """Everything wrong with a set of arms, all at once."""
    found = []
    keys = [a.key for a in arms]
    repeated = sorted({k for k in keys if keys.count(k) > 1})
    if repeated:
        found.append(f"arm names repeat within a series: {repeated}")
    for series in dict.fromkeys(a.series for a in arms):
        mine = [a for a in arms if a.series is series]
        nulls = [a for a in mine if a.null]
        if len(nulls) != 1:
            found.append(f"{series.value} needs exactly one null arm, not {len(nulls)}")
        elif nulls[0].simplicity != min(a.simplicity for a in mine):
            found.append(f"{series.value}'s null arm, {nulls[0].name}, isn't its simplest")
        ranks = [a.simplicity for a in mine]
        if len(set(ranks)) != len(ranks):
            found.append(f"{series.value}'s arms share simplicity ranks: {sorted(ranks)}")
    for arm in arms:
        found += [f"{arm.key}: {p}" for p in MODELS.problems(arm.models)]
        if not arm.models:
            found.append(f"{arm.key}: an arm needs at least one model")
    return found


def config_for(
    arm: Arm, cutoff: pd.Period, grouping: str = "org_level1", horizon: int = HORIZON
) -> GuidanceConfig:
    """The guidance configuration an arm fits through `cutoff`."""
    return GuidanceConfig.from_mapping(
        {
            "horizon": horizon,
            "grouping": grouping,
            "categories": [arm.series.value, ANCHOR.value],
            "models": list(arm.models),
            "combination": {"method": "inverse_error", "top_k": min(arm.top_k, len(arm.models))},
            "evaluation": EVALUATION,
            "validation": {"snapshot": str(cutoff)},
            "reconciliation": {"method": None},
        }
    )


@dataclass(frozen=True)
class ArmRun:
    """One arm's forecast beside the actuals: a row per group and month after the cutoff."""

    arm: Arm
    frame: pd.DataFrame
    #: The (group, category) series the arm didn't forecast, which forecast zero.
    unforecast: int = 0
    report: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class LadderRun:
    """Every arm's run, for one cutoff and grouping."""

    cutoff: pd.Period
    grouping: str
    horizon: int
    runs: tuple[ArmRun, ...]

    def frame(self) -> pd.DataFrame:
        """Long-form: series, arm, simplicity, group, month, forecast, and actual."""
        parts = [
            r.frame.assign(series=r.arm.series.value, arm=r.arm.name, simplicity=r.arm.simplicity)
            for r in self.runs
        ]
        columns = ["series", "arm", "simplicity", GROUP, MONTH, FORECAST, ACTUAL]
        return pd.concat(parts, ignore_index=True)[columns]


def _actuals(
    con: duckdb.DuckDBPyConnection,
    grouping: str,
    series: Sequence[GuidanceCategory],
    cutoff: pd.Period,
    horizon: int,
) -> pd.DataFrame:
    frame = ingest(con, grouping, list(series)).frame
    last = frame[MONTH].max()
    if cutoff + horizon > last:
        raise LadderError(
            f"a cutoff of {cutoff} needs actuals through {cutoff + horizon}, but the data ends "
            f"at {last}"
        )
    return frame[(frame[MONTH] > cutoff) & (frame[MONTH] <= cutoff + horizon)]


def run_ladder(
    con: duckdb.DuckDBPyConnection,
    cutoff: pd.Period | str,
    grouping: str = "org_level1",
    horizon: int = HORIZON,
    arms: Sequence[Arm] = ARMS,
) -> LadderRun:
    """Fit every arm through `cutoff` and set its forecast beside the actuals of the `horizon`
    months after it."""
    found = problems(arms)
    if found:
        raise LadderError("; ".join(found))
    cutoff = pd.Period(cutoff, "M")
    series = list(dict.fromkeys(a.series for a in arms))
    actuals = _actuals(con, grouping, series, cutoff, horizon)
    months = pd.period_range(cutoff + 1, periods=horizon, freq="M")
    runs = []
    for arm in arms:
        result = run(con, config_for(arm, cutoff, grouping, horizon))
        artifact = result.artifact.frame
        mine = artifact[artifact[CATEGORY] == arm.series.value]
        index = pd.MultiIndex.from_arrays(
            [mine[grouping].astype(str), [cutoff + 1 + int(k) for k in mine["forecast_month"]]],
            names=[GROUP, MONTH],
        )
        forecast = pd.Series(mine[VALUE].to_numpy(), index=index)
        actual = actuals[actuals[CATEGORY] == arm.series.value].set_index([GROUP, MONTH])[VALUE]
        groups = sorted(set(result.series.groups) | set(actual.index.get_level_values(GROUP)))
        grid = pd.MultiIndex.from_product([groups, months], names=[GROUP, MONTH])
        frame = pd.DataFrame(
            {
                FORECAST: forecast.reindex(grid, fill_value=0.0),
                ACTUAL: actual.reindex(grid, fill_value=0.0),
            }
        ).reset_index()
        skipped = [
            s for s in result.report.get("series_skipped", []) if s["category"] == arm.series
        ]
        selection = {
            k: v["weights"]
            for k, v in result.report.get("selection", {}).items()
            if k.endswith(f"/{arm.series.value}")
        }
        runs.append(ArmRun(arm, frame, len(skipped), {"weights": selection}))
    return LadderRun(cutoff, grouping, horizon, tuple(runs))
