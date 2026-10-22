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
are intermittent (`statsforecast.CrostonSBA`). An arm with more than one model weights them by
inverse backtest error.

Each arm is an ordinary guidance configuration (`config_for`): the series and the headcount
anchor, which marks every loaded month, at the study's grouping; the snapshot fixed at the
cutoff; and no reconciliation, so each arm's forecast is its models' own. `run_ladder` fits
every arm and sets its forecast beside the actuals the following months realized, by the
guidance measures, on the grid of the arm's groups. A group an arm didn't forecast, as when its
history was all zeros, forecasts zero.

**Scoring** (`score`). Each arm is scored by its WAPE over the year after the cutoff: the summed
absolute error over every group and month, over the summed actuals. Selection climbs each
series' ladder from the null arm: the champion passes to the next richer arm only if that arm's
WAPE is at least `margin` (10%) below the champion's, so a richer arm has to earn its place.
Twelve months are few, and neighbouring months' errors are alike, so the uncertainty comes from
a moving-block bootstrap over the months: each draw strings together blocks of `block`
consecutive months, from random starts, into a year, and recomputes the WAPE. The intervals are
the draws' central `level` share, for each arm's WAPE and, drawn on the same months, for each
challenger's gain over the champion it faced.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, Final

import duckdb
import numpy as np
import pandas as pd
from numpy.typing import NDArray

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
    cutoff: pd.Period | None,
    horizon: int,
) -> tuple[pd.Period, pd.DataFrame]:
    """The cutoff, by default the year before the data ends, and the actuals after it."""
    frame = ingest(con, grouping, list(series)).frame
    last = frame[MONTH].max()
    cutoff = last - horizon if cutoff is None else cutoff
    if cutoff + horizon > last:
        raise LadderError(
            f"a cutoff of {cutoff} needs actuals through {cutoff + horizon}, but the data ends "
            f"at {last}"
        )
    return cutoff, frame[(frame[MONTH] > cutoff) & (frame[MONTH] <= cutoff + horizon)]


def run_ladder(
    con: duckdb.DuckDBPyConnection,
    cutoff: pd.Period | str | None = None,
    grouping: str = "org_level1",
    horizon: int = HORIZON,
    arms: Sequence[Arm] = ARMS,
) -> LadderRun:
    """Fit every arm through `cutoff`, by default `horizon` months before the data ends, and
    set its forecast beside the actuals of the `horizon` months after it."""
    found = problems(arms)
    if found:
        raise LadderError("; ".join(found))
    series = list(dict.fromkeys(a.series for a in arms))
    given = None if cutoff is None else pd.Period(cutoff, "M")
    cutoff, actuals = _actuals(con, grouping, series, given, horizon)
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


# --- scoring -------------------------------------------------------------------------------------

#: How far below the champion's WAPE a richer arm's must be to replace it.
MARGIN: Final = 0.10
#: The moving-block bootstrap's block length, in months, its draws, and its interval's level.
BLOCK: Final = 3
DRAWS: Final = 1000
LEVEL: Final = 0.9


def wape(forecast: NDArray[np.float64], actual: NDArray[np.float64]) -> float:
    """The weighted absolute percentage error: summed absolute error over summed actuals."""
    total = float(np.sum(actual))
    if total <= 0:
        raise LadderError("WAPE needs actuals that sum to more than zero")
    return float(np.sum(np.abs(forecast - actual)) / total)


def block_indices(months: int, block: int, draws: int, seed: int = 0) -> NDArray[np.int64]:
    """(draws, months) month indices for a moving-block bootstrap: each row strings together
    blocks of `block` consecutive months, from uniformly drawn starts, cut to `months`."""
    if not 1 <= block <= months:
        raise LadderError(f"a block must be 1 to {months} months, not {block}")
    blocks = -(-months // block)
    starts = np.random.default_rng(seed).integers(0, months - block + 1, size=(draws, blocks))
    rows = (starts[:, :, None] + np.arange(block)).reshape(draws, blocks * block)
    return rows[:, :months]


def _monthly(run: ArmRun) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Each month's summed absolute error and summed actuals, over the arm's groups."""
    frame = run.frame.assign(error=(run.frame[FORECAST] - run.frame[ACTUAL]).abs())
    by_month = frame.groupby(MONTH)[["error", ACTUAL]].sum().sort_index()
    return by_month["error"].to_numpy(float), by_month[ACTUAL].to_numpy(float)


def _drawn(
    errors: NDArray[np.float64], actuals: NDArray[np.float64], rows: NDArray[np.int64]
) -> NDArray[np.float64]:
    totals = actuals[rows].sum(axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(totals > 0, errors[rows].sum(axis=1) / totals, np.nan)


@dataclass(frozen=True)
class ArmScore:
    """One arm's WAPE and its interval, and how it fared against the champion it faced."""

    arm: Arm
    wape: float
    low: float
    high: float
    #: The champion it challenged, its relative gain over it (1 - its WAPE over theirs), and
    #: that gain's interval; none for the null arm.
    challenged: str | None = None
    gain: float | None = None
    gain_low: float | None = None
    gain_high: float | None = None
    #: Whether it cleared the margin and became the champion.
    cleared: bool = False


@dataclass(frozen=True)
class Scorecard:
    """Every arm's score, and each series' selected arm."""

    scores: tuple[ArmScore, ...]
    selected: dict[GuidanceCategory, str]
    margin: float
    block: int
    draws: int
    level: float

    def frame(self) -> pd.DataFrame:
        rows = [
            {
                "series": s.arm.series.value,
                "arm": s.arm.name,
                "simplicity": s.arm.simplicity,
                "wape": s.wape,
                "low": s.low,
                "high": s.high,
                "challenged": s.challenged,
                "gain": s.gain,
                "gain_low": s.gain_low,
                "gain_high": s.gain_high,
                "cleared": s.cleared,
                "selected": self.selected[s.arm.series] == s.arm.name,
            }
            for s in self.scores
        ]
        return pd.DataFrame(rows)


def score(
    ladder: LadderRun,
    margin: float = MARGIN,
    block: int = BLOCK,
    draws: int = DRAWS,
    level: float = LEVEL,
    seed: int = 0,
) -> Scorecard:
    """Score every arm by its WAPE over the run's months, select each series' arm by climbing
    its ladder with `margin`, and give moving-block bootstrap intervals."""
    if not 0 <= margin < 1:
        raise LadderError(f"a margin must be in [0, 1), not {margin}")
    if not 0 < level < 1:
        raise LadderError(f"an interval's level must be in (0, 1), not {level}")
    rows = block_indices(ladder.horizon, block, draws, seed)
    tails = (100 * (1 - level) / 2, 100 * (1 + level) / 2)
    scores: list[ArmScore] = []
    selected: dict[GuidanceCategory, str] = {}
    for series in dict.fromkeys(r.arm.series for r in ladder.runs):
        runs = sorted(
            (r for r in ladder.runs if r.arm.series is series), key=lambda r: r.arm.simplicity
        )
        monthly = {r.arm.name: _monthly(r) for r in runs}
        drawn = {name: _drawn(e, a, rows) for name, (e, a) in monthly.items()}
        champion = None
        for r in runs:
            name = r.arm.name
            point = wape(*(r.frame[c].to_numpy(float) for c in (FORECAST, ACTUAL)))
            low, high = (float(v) for v in np.nanpercentile(drawn[name], tails))
            if champion is None:
                scores.append(ArmScore(r.arm, point, low, high, cleared=True))
                champion = (name, point)
                continue
            held, best = champion
            gain = 1 - point / best if best > 0 else 0.0
            with np.errstate(divide="ignore", invalid="ignore"):
                gains = 1 - drawn[name] / drawn[held]
            gain_low, gain_high = (float(v) for v in np.nanpercentile(gains, tails))
            cleared = point <= (1 - margin) * best
            scores.append(
                ArmScore(r.arm, point, low, high, held, gain, gain_low, gain_high, cleared)
            )
            if cleared:
                champion = (name, point)
        assert champion is not None
        selected[series] = champion[0]
    return Scorecard(tuple(scores), selected, margin, block, draws, level)


def markdown(ladder: LadderRun, card: Scorecard) -> str:
    """The study's report: per series, each arm's WAPE and interval, and the selection."""
    parts = [
        "# Guidance ladder",
        f"Fitted through {ladder.cutoff}, scored on the {ladder.horizon} months after it at "
        f"{ladder.grouping} grain. A richer arm replaces the champion only with a WAPE "
        f"{card.margin:.0%} below its own. Intervals are {card.level:.0%} moving-block "
        f"bootstrap intervals over the months, in blocks of {card.block}, from {card.draws} "
        "draws.",
    ]
    for series in dict.fromkeys(s.arm.series for s in card.scores):
        chosen = card.selected[series]
        lines = [
            f"## {series.value}: {chosen}",
            "",
            "| rank | arm | WAPE | interval | challenged | gain | gain interval | outcome |",
            "|---|---|---|---|---|---|---|---|",
        ]
        for s in (s for s in card.scores if s.arm.series is series):
            gain = "—" if s.gain is None else f"{s.gain:+.1%}"
            band = "—" if s.gain_low is None else f"{s.gain_low:+.1%} to {s.gain_high:+.1%}"
            outcome = (
                "**selected**"
                if s.arm.name == chosen
                else "null"
                if s.arm.null
                else "cleared"
                if s.cleared
                else "held"
            )
            lines.append(
                f"| {s.arm.simplicity} | {s.arm.name} | {s.wape:.3f} | {s.low:.3f}–{s.high:.3f} "
                f"| {s.challenged or '—'} | {gain} | {band} | {outcome} |"
            )
        parts.append("\n".join(lines))
    return "\n\n".join(parts) + "\n"
