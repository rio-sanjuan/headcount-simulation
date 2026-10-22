"""Ground-truth recovery: each fit compared with what the synthetic world really is.

Only a synthetic world allows this: its truth manifest and truth tables hold every parameter
behind the data. `recover` reads them beside a training run and compares, on the run's
held-out side:

- **attrition**: the champion's probability for each held-out person-month with the true
  hazard of the month ahead: the base hazard for the level, tenure, and rating, times the
  season, the cost center's effect, and any shock, capped at 1;
- **performance**: the Markov arm's transitions and entry distribution with the configured
  ones, beside the held-out frequencies;
- **requisitions**: for single-opening, non-campus requisitions, the external-fill and
  unfilled-exit champions' mean probability by age with the true monthly chance: an unfilled
  exit at its hazard, and an external fill at its hazard among those that don't exit;
- **backfill**: the champion's probability for each held-out departure with the true one,
  from the base probability, the level's shift, and the cost center's effect on the logit
  scale.

Each comparison is a `Recovery`: rows of (quantity, truth, fitted, observed) for the sidecar,
a figure, and a note on what the models can't recover by design, since the world's functional
forms aren't the models'. Rows the truth doesn't cover, such as ratings on a revised scale, are
left out and counted.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Final, cast

import duckdb
import numpy as np
import pandas as pd
from numpy.typing import NDArray

from hcsim.model.core.schema import SimulationSchema
from hcsim.model.data.extracts import extract
from hcsim.model.data.population import PopulationSpec, guard
from hcsim.model.interface import EntityModel
from hcsim.model.ml.markov import MarkovModel
from hcsim.model.ml.training import DomainResult, TrainingRun
from hcsim.model.reports.harness import Section, number, table
from hcsim.model.reports.sidecar import SidecarError
from hcsim.model.reports.svg import Series, chart

Floats = NDArray[np.float64]
ROW_COLUMNS: Final = ("quantity", "truth", "fitted", "observed")
#: The fewest held-out departures a cost center needs to be plotted on its own.
MIN_DEPARTURES: Final = 10
_UNRATED: Final = "unrated"


@dataclass(frozen=True)
class Recovery:
    name: str
    title: str
    rows: pd.DataFrame
    note: str
    figures: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if tuple(self.rows.columns) != ROW_COLUMNS:
            raise SidecarError(f"recovery rows need the columns {list(ROW_COLUMNS)}")

    def section(self) -> Section:
        links = "\n\n".join(f"![{name}](figures/{name}.svg)" for name in self.figures)
        body = "\n\n".join(
            [
                self.note.strip(),
                links,
                table(
                    list(ROW_COLUMNS),
                    [list(r) for r in self.rows.itertuples(index=False, name=None)],
                ),
            ]
        )
        return Section(f"Recovery: {self.title}", body, self.figures)

    def records(self) -> list[dict[str, Any]]:
        return [
            {
                "quantity": str(q),
                "truth": float(t),
                "fitted": float(f),
                "observed": None if o is None or pd.isna(o) else float(o),
            }
            for q, t, f, o in self.rows.itertuples(index=False, name=None)
        ]


@dataclass(frozen=True)
class _Truth:
    manifest: Mapping[str, Any]
    tables: Mapping[str, pd.DataFrame]

    @classmethod
    def read(cls, con: duckdb.DuckDBPyConnection) -> _Truth:
        (present,) = con.execute(
            "SELECT count(*) FROM information_schema.tables "
            "WHERE table_schema = 'truth' AND table_name = 'manifest'"
        ).fetchone() or (0,)
        if not present:
            raise SidecarError("recovery needs a synthetic world's truth, which this lacks")
        (text,) = con.execute("SELECT manifest FROM truth.manifest").fetchone() or ("{}",)
        names = [
            "attrition_hazard",
            "attrition_season",
            "attrition_shock",
            "cost_center_effect",
            "requisition_hazard",
        ]
        return cls(
            json.loads(text), {n: con.execute(f"SELECT * FROM truth.{n}").df() for n in names}
        )

    @property
    def config(self) -> Mapping[str, Any]:
        result: Mapping[str, Any] = self.manifest["config"]
        return result

    def effect(self, process: str) -> dict[str, float]:
        frame = self.tables["cost_center_effect"]
        chosen = frame[frame["process"] == process]
        return dict(zip(chosen["cost_center"], chosen["effect"].astype(float), strict=True))


def _rows(records: Sequence[tuple[str, float, float, float | None]]) -> pd.DataFrame:
    return pd.DataFrame(records, columns=list(ROW_COLUMNS))


def _held_out(
    con: duckdb.DuckDBPyConnection, run: TrainingRun, domain: str, population: PopulationSpec
) -> pd.DataFrame:
    full = extract(con, domain, population, run.split.start, run.split.end)
    return run.split.cut(full)[1].frame


def _scores(result: DomainResult, frame: pd.DataFrame, schema: SimulationSchema) -> Floats:
    if result.design is None:
        raise SidecarError(f"{result.domain} has no design to score rows with")
    model = cast(EntityModel, result.champion.model)
    return np.asarray(model.compute_probability(result.design.matrix(frame, schema)), np.float64)


def _deciles(truth: Floats, *others: Floats) -> list[Floats]:
    """Each array's mean within deciles of the truth."""
    order = np.argsort(truth, kind="stable")
    chunks = np.array_split(order, min(10, len(order)))
    return [np.array([a[c].mean() for c in chunks]) for a in (truth, *others)]


def attrition(
    con: duckdb.DuckDBPyConnection,
    run: TrainingRun,
    schema: SimulationSchema,
    population: PopulationSpec,
    truth: _Truth,
) -> Recovery:
    result = run.domains["attrition"]
    frame = _held_out(con, run, "attrition", population)
    fitted = _scores(result, frame, schema)
    scale = list(truth.config["ratings"]["scale"])
    rank = frame["rating_rank"].to_numpy(np.int64)
    original = frame["rating_scale"].eq("original").to_numpy()
    labels = np.array(
        [
            _UNRATED if r == 0 else scale[r - 1] if o and r <= len(scale) else None
            for r, o in zip(rank, original, strict=True)
        ],
        dtype=object,
    )
    base = truth.tables["attrition_hazard"]
    ceiling = int(base["tenure_month"].max())
    keys = pd.DataFrame(
        {
            "job_level": frame["job_level"].to_numpy(np.int64),
            "tenure_month": np.minimum(frame["tenure_months"].to_numpy(np.int64) + 1, ceiling),
            "rating": labels,
        }
    )
    hazard = keys.merge(base, on=["job_level", "tenure_month", "rating"], how="left")
    season = dict(
        zip(
            truth.tables["attrition_season"]["month_of_year"],
            truth.tables["attrition_season"]["factor"],
            strict=True,
        )
    )
    shocks = truth.tables["attrition_shock"]
    shock = {
        (pd.Timestamp(m), c): float(f)
        for m, c, f in shocks[["month_end", "cost_center", "factor"]].itertuples(index=False)
    }
    following = pd.to_datetime(frame["month_end"]) + pd.offsets.MonthEnd(1)
    effect = truth.effect("attrition")
    multiplier = frame["cost_center"].map(effect).to_numpy(np.float64)
    true = np.minimum(
        1.0,
        hazard["monthly_hazard"].to_numpy(np.float64)
        * frame["next_month"].map(season).to_numpy(np.float64)
        * multiplier
        * np.array(
            [shock.get((m, c), 1.0) for m, c in zip(following, frame["cost_center"], strict=True)]
        ),
    )
    covered = np.isfinite(true)
    observed = frame["label"].to_numpy(np.float64)
    true, fitted, observed = true[covered], fitted[covered], observed[covered]
    levels = frame["job_level"].to_numpy(np.int64)[covered]
    records: list[tuple[str, float, float, float | None]] = [
        (
            f"monthly hazard, level {k}",
            true[levels == k].mean(),
            fitted[levels == k].mean(),
            observed[levels == k].mean(),
        )
        for k in np.unique(levels).tolist()
    ]
    records.append(("monthly hazard, everyone", true.mean(), fitted.mean(), observed.mean()))
    correlation = float(np.corrcoef(true, fitted)[0, 1]) if true.std() > 0 else float("nan")
    t, f, o = _deciles(true, fitted, observed)
    figure = chart(
        [Series("champion", list(t), list(f)), Series("observed", list(t), list(o))],
        title="Attrition: fitted against true hazard, by decile of the truth",
        x_label="true monthly hazard",
        y_label="mean probability",
        diagonal=True,
    )
    note = (
        f"The champion, **{result.selection.champion}**, scored {int(covered.sum())} held-out "
        f"person-months ({int((~covered).sum())} left out, outside the truth's ratings). The "
        f"per-row correlation of its probability with the true hazard is {number(correlation)}."
        " The truth multiplies a level, tenure, and rating surface by the season, the cost "
        "center's effect, and shocks. The models see the level, tenure, rating, month ahead, "
        "and cost center, but no shock, so they can recover the shocks' average and not their "
        "timing. Observed rates are each decile's share who left."
    )
    return Recovery("attrition", "attrition hazard", _rows(records), note, {"attrition": figure})


def performance(
    con: duckdb.DuckDBPyConnection,
    run: TrainingRun,
    population: PopulationSpec,
    truth: _Truth,
) -> Recovery:
    result = run.domains["performance"]
    arm = next(a for a in result.arms.values() if a.arm.algorithm == "markov")
    model = cast(MarkovModel, arm.model)
    ratings = truth.config["ratings"]
    scale = list(ratings["scale"])
    k = len(scale)
    true_t = np.asarray(ratings["transitions"], dtype=np.float64)
    true_e = np.asarray(ratings["entry_distribution"], dtype=np.float64)
    fitted_t, fitted_e = model.transitions[:k, :k], model.entry[:k]
    frame = _held_out(con, run, "performance", population)
    both = frame[frame["rating_scale"].eq("original")]
    pairs = both[both["previous_scale"].eq("original")]
    counts = np.zeros((k, k))
    np.add.at(
        counts,
        (
            pairs["previous_rank"].to_numpy(np.int64) - 1,
            pairs["rating_rank"].to_numpy(np.int64) - 1,
        ),
        1.0,
    )
    with np.errstate(invalid="ignore", divide="ignore"):
        seen_t = counts / counts.sum(axis=1, keepdims=True)
    first = both[both["previous_rank"] == 0]["rating_rank"].to_numpy(np.int64) - 1
    seen_e = np.bincount(first, minlength=k) / len(first) if len(first) else np.full(k, np.nan)
    records: list[tuple[str, float, float, float | None]] = [
        (f"{scale[i]} → {scale[j]}", true_t[i, j], fitted_t[i, j], seen_t[i, j])
        for i in range(k)
        for j in range(k)
    ]
    records += [(f"entry {scale[j]}", true_e[j], fitted_e[j], seen_e[j]) for j in range(k)]
    figure = chart(
        [
            Series("transitions", list(true_t.ravel()), list(fitted_t.ravel())),
            Series("entry", list(true_e), list(fitted_e)),
        ],
        title="Performance: fitted against configured probabilities",
        x_label="configured probability",
        y_label="fitted probability",
        diagonal=True,
    )
    error = float(np.abs(fitted_t - true_t).max())
    note = (
        f"The Markov arm, **{arm.arm.name}**, against the configured rating transitions and "
        f"entry distribution. The largest transition error is {number(error)}. Smoothing "
        "pulls thin rows toward the overall distribution, and a person's first rating in the "
        "window counts as an entry. Observed values are the held-out cycle's frequencies."
    )
    return Recovery(
        "performance", "rating transitions", _rows(records), note, {"performance": figure}
    )


def requisitions(
    con: duckdb.DuckDBPyConnection,
    run: TrainingRun,
    schema: SimulationSchema,
    population: PopulationSpec,
    truth: _Truth,
) -> Recovery:
    hazards = truth.tables["requisition_hazard"]

    def schedule(channel: str) -> Floats:
        rows = hazards[hazards["channel"] == channel].sort_values("age_month")
        return rows["monthly_hazard"].to_numpy(np.float64)

    exit_h, external_h = schedule("unfilled"), schedule("external")
    domain = "cancellation" if "cancellation" in run.domains else "external_fill"
    frame = _held_out(con, run, domain, population)
    single = frame[frame["req_type"].ne("campus") & frame["openings"].eq(1)].reset_index(drop=True)
    last = len(exit_h) - 1
    age = np.minimum(single["age_months"].to_numpy(np.int64), last)
    targets = {
        "cancellation": ("unfilled exit", exit_h[age], single["closed_unfilled"]),
        "external_fill": (
            "external fill",
            (1 - exit_h[age]) * external_h[np.minimum(age, len(external_h) - 1)],
            single["fill_channel"].eq("external"),
        ),
    }
    records: list[tuple[str, float, float, float | None]] = []
    figures = {}
    for name, (what, true, label) in targets.items():
        if name not in run.domains:
            continue
        fitted = _scores(run.domains[name], single, schema)
        observed = label.to_numpy(np.float64)
        ages = list(range(last + 1))
        means = [
            (true[age == a].mean(), fitted[age == a].mean(), observed[age == a].mean())
            for a in ages
            if (age == a).any()
        ]
        shown = [a for a in ages if (age == a).any()]
        records += [
            (f"{what}, age {a}{'+' if a == last else ''}", t, f, o)
            for a, (t, f, o) in zip(shown, means, strict=True)
        ]
        figures[name.replace("_", "-")] = chart(
            [
                Series("truth", shown, [m[0] for m in means], "line"),
                Series("champion", shown, [m[1] for m in means]),
                Series("observed", shown, [m[2] for m in means]),
            ],
            title=f"Requisitions: monthly chance of an {what}, by age",
            x_label=f"age in months (the last is {last} and older)",
            y_label="monthly probability",
        )
    note = (
        f"Held-out requisition-months of single-opening, non-campus requisitions "
        f"({len(single)} of {len(frame)} rows). In the world, an open requisition first exits "
        "unfilled at its hazard; each opening left then fills externally, by transfer, or by "
        "conversion, at each channel's hazard. So the chance of an external fill is the "
        "external hazard among those that don't exit. Multi-opening and campus requisitions "
        "follow other rules and are left out."
    )
    return Recovery("requisitions", "requisition hazards", _rows(records), note, figures)


def backfill(
    con: duckdb.DuckDBPyConnection,
    run: TrainingRun,
    schema: SimulationSchema,
    population: PopulationSpec,
    truth: _Truth,
) -> Recovery:
    result = run.domains["backfill"]
    frame = _held_out(con, run, "backfill", population)
    fitted = _scores(result, frame, schema)
    config = truth.config["backfill"]
    base = float(config["base_probability"])
    shift = {int(k): float(v) for k, v in config["level_shift"].items()}
    logit = (
        np.log(base / (1 - base))
        + frame["cost_center"].map(truth.effect("backfill")).to_numpy(np.float64)
        + frame["job_level"].map(shift).to_numpy(np.float64)
    )
    true = 1 / (1 + np.exp(-logit))
    observed = frame["label"].to_numpy(np.float64)
    levels = frame["job_level"].to_numpy(np.int64)
    records: list[tuple[str, float, float, float | None]] = [
        (
            f"backfill probability, level {k}",
            true[levels == k].mean(),
            fitted[levels == k].mean(),
            observed[levels == k].mean(),
        )
        for k in np.unique(levels).tolist()
    ]
    records.append(("backfill probability, everyone", true.mean(), fitted.mean(), observed.mean()))
    by_center = (
        pd.DataFrame({"cc": frame["cost_center"], "true": true, "fitted": fitted})
        .groupby("cc")
        .agg(true=("true", "mean"), fitted=("fitted", "mean"), n=("true", "size"))
    )
    plotted = by_center[by_center["n"] >= MIN_DEPARTURES]
    figure = chart(
        [Series("cost centers", list(plotted["true"]), list(plotted["fitted"]))],
        title="Backfill: fitted against true probability, by cost center",
        x_label="true probability",
        y_label="fitted probability",
        diagonal=True,
    )
    note = (
        f"The champion, **{result.selection.champion}**, on {len(frame)} held-out departures, "
        f"with the {len(plotted)} cost centers holding at least {MIN_DEPARTURES} of them "
        "plotted. The truth shifts a base probability on the logit scale by the level and by "
        "the cost center's effect. The backfill arms read only the cost center, so they can't "
        "recover the level's shift, which the rows by level show."
    )
    return Recovery("backfill", "backfill probability", _rows(records), note, {"backfill": figure})


def recover(
    con: duckdb.DuckDBPyConnection,
    run: TrainingRun,
    schema: SimulationSchema,
    population: PopulationSpec,
) -> list[Recovery]:
    """Every recovery the run's domains allow, from the world's truth."""
    guard(population, run.population, "training run")
    truth = _Truth.read(con)
    found = []
    if "attrition" in run.domains:
        found.append(attrition(con, run, schema, population, truth))
    performance_arms = run.domains.get("performance")
    if performance_arms is not None and any(
        a.arm.algorithm == "markov" for a in performance_arms.arms.values()
    ):
        found.append(performance(con, run, population, truth))
    if {"external_fill", "cancellation"} & set(run.domains):
        found.append(requisitions(con, run, schema, population, truth))
    if "backfill" in run.domains:
        found.append(backfill(con, run, schema, population, truth))
    return found
