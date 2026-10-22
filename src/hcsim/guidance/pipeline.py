"""The guidance pipeline, lite: ingest, forecast, reconcile, and write the artifact.

The pipeline runs as ordered steps over a shared context (see `framework`):

1. **ingest** reads the warehouse into monthly series per group and category, composing an
   inner fill-tree node from its leaves;
2. **validate** settles the series: the snapshot, the feed's gaps, and the all-zero series (see
   `validation`), and gives the calendar covariates for the history and the horizon;
3. **select** scores each series' suite by rolling-origin evaluation and weighs the best (see
   `selection`), for every series long enough;
4. **generate** refits the selected models, reweights them, and forecasts (see `generation`);
5. **cohere** makes each carried fill-tree parent its children's sum, by the configured
   method (see `coherence`), as it does again after reconciling and, bottom up, when writing;
6. **reconcile**, when the configuration asks for it, reconciles each group's flows to its
   headcount anchor, over the identity's terms: the deepest carried cover of the fill tree and
   the other flows the run forecasts. Moves out of a group have no category, so the identity
   is partial wherever people move between groups, and the report names every missing term;
7. **align**, when the configuration names a method, moves each group's anchor onto its plan
   targets and re-derives the flows (see `alignment`), after reconciliation, so the aligned
   values are final, and a coherence pass follows it;
8. **disaggregate**, when the configuration names a strategy, splits each group's forecasts
   onto the leaf grain inside it (see `disaggregation`);
9. **format** writes the artifact, stamped with the alignment target's digest (`aligned_to`)
   when the run aligned to one. Negative cells are clamped to zero there, and the report
   counts them.

The context starts with two slots: `source`, where the warehouse tables are read from (see
`sources`), and `config`.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

import duckdb
import numpy as np
import pandas as pd

from hcsim.guidance import ingest as ingestion
from hcsim.guidance import validation
from hcsim.guidance.alignment import ALIGNERS, AlignmentProblem
from hcsim.guidance.coherence import PASSES, cohere
from hcsim.guidance.coherence import Method as CoherenceMethod
from hcsim.guidance.config import GuidanceConfig
from hcsim.guidance.covariates import covariates
from hcsim.guidance.disaggregation import basis, shares, split
from hcsim.guidance.framework import Context, Pipeline
from hcsim.guidance.generation import generate
from hcsim.guidance.models import ForecastError, capabilities
from hcsim.guidance.reconcile import RECONCILERS, Problem
from hcsim.guidance.report import config_digest
from hcsim.guidance.selection import Selection, select
from hcsim.guidance.sources import Source, SourceError, as_source
from hcsim.model.core.alignment import AlignmentTarget
from hcsim.model.core.categories import ANCHOR, GuidanceCategory
from hcsim.model.core.fill_tree import identity_terms
from hcsim.model.core.guidance import GuidanceForecast

#: The identity's one flow no category forecasts: moves out of a group to another, the
#: counterpart of internal starts, which leaves every group's identity partial.
TRANSFERS_OUT: Final = "transfers_out"

Forecasts = dict[tuple[str, GuidanceCategory], np.ndarray]


@dataclass(frozen=True)
class PipelineResult:
    artifact: GuidanceForecast
    report: dict[str, Any]
    #: The settled history the forecast started from.
    series: ingestion.Series


# --- steps ---------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Ingest:
    name: str = "ingest"
    needs: tuple[str, ...] = ("source", "config")
    gives: tuple[str, ...] = ("series", "grain")

    def validate(self, ctx: Context) -> list[str]:
        config = ctx["config"]
        try:
            source = as_source(ctx["source"])
            with source.connect() as con:
                return ingestion.problems(con, config.grouping, config.categories)
        except SourceError as error:
            return [str(error)]

    def execute(self, ctx: Context) -> Mapping[str, Any]:
        config = ctx["config"]
        source = as_source(ctx["source"])
        with source.connect() as con:
            series = ingestion.ingest(con, config.grouping, config.categories)
        ctx.note(
            self.name,
            source=source.description,
            snapshot=str(series.snapshot),
            grouping=series.grouping,
            categories=[c.value for c in series.categories],
            ungrouped=series.ungrouped,
            counted=[c.value for c in ingestion.counted(config.categories)],
        )
        return {"series": series, "grain": config.grouping}


@dataclass(frozen=True)
class Validate:
    """Settle the series: the snapshot, the gaps, and the all-zero series, with the calendar
    covariates for the history and the horizon."""

    name: str = "validate"
    needs: tuple[str, ...] = ("series", "config")
    gives: tuple[str, ...] = ("series", "covariates")

    def validate(self, ctx: Context) -> list[str]:
        rules = ctx["config"].validation
        return validation.problems(ctx["series"], rules.snapshot, rules.gaps)

    def execute(self, ctx: Context) -> Mapping[str, Any]:
        config = ctx["config"]
        rules = config.validation
        settled = validation.validate(ctx["series"], rules.snapshot, rules.drop_all_zero)
        ctx.note(
            self.name,
            snapshot=str(settled.series.snapshot),
            gaps=[str(m) for m in settled.gaps],
            dropped=[{"group": g, "category": c} for g, c in settled.dropped],
        )
        return {
            "series": settled.series,
            "covariates": covariates(settled.series, config.horizon),
        }


def _exposure(series: Any, group: str, category: GuidanceCategory) -> Any:
    """A flow's exposure: its group's anchor history, when the group has one."""
    if category is not ANCHOR and (group, ANCHOR) in series.keys:
        return series.values(group, ANCHOR).to_numpy()
    return None


@dataclass(frozen=True)
class Select:
    """Score each series' suite and weigh its candidates."""

    name: str = "select"
    needs: tuple[str, ...] = ("series", "covariates", "config")
    gives: tuple[str, ...] = ("selections",)

    def validate(self, ctx: Context) -> list[str]:
        horizon = ctx["config"].horizon
        return [] if horizon >= 1 else [f"the horizon must be at least 1 month, got {horizon}"]

    def execute(self, ctx: Context) -> Mapping[str, Any]:
        series, config = ctx["series"], ctx["config"]
        selections: dict[tuple[str, GuidanceCategory], Selection] = {}
        skipped: list[dict[str, str]] = []
        for group, category in series.keys:
            values = series.values(group, category).to_numpy()
            exposure = _exposure(series, group, category)
            try:
                selections[group, category] = select(
                    values, config.selection, exposure, ctx["covariates"]
                )
            except ForecastError as error:
                skipped.append({"group": group, "category": category.value, "reason": str(error)})
        suite = {name: capabilities(name).as_record() for name in config.selection.suite}
        ctx.note(self.name, series_skipped=skipped, capabilities=suite)
        return {"selections": selections}


@dataclass(frozen=True)
class Generate:
    """Refit each series' selected models, reweight them, and forecast, each group's anchor
    first, so a flow's share models have its forecast as their exposure."""

    name: str = "generate"
    needs: tuple[str, ...] = ("series", "selections", "covariates", "config")
    gives: tuple[str, ...] = ("forecasts", "paths", "residuals")

    def validate(self, ctx: Context) -> list[str]:
        return []

    def execute(self, ctx: Context) -> Mapping[str, Any]:
        series, config = ctx["series"], ctx["config"]
        forecasts: Forecasts = {}
        paths: Forecasts = {}
        residuals: Forecasts = {}
        point_only: list[dict[str, str]] = []
        report: dict[str, dict[str, Any]] = {}
        skipped: list[dict[str, str]] = []
        selections = ctx["selections"]
        for stream, ((group, category), selection) in enumerate(selections.items()):
            values = series.values(group, category).to_numpy()
            exposure = _exposure(series, group, category)
            future = None if exposure is None else forecasts.get((group, ANCHOR))
            try:
                result = generate(
                    selection,
                    values,
                    config.horizon,
                    config.selection,
                    exposure,
                    future,
                    ctx["covariates"],
                    stream,  # each series' own paths, whatever else the run forecasts
                )
            except ForecastError as error:
                skipped.append({"group": group, "category": category.value, "reason": str(error)})
                continue
            forecasts[group, category] = result.values
            if result.residuals is not None:
                residuals[group, category] = result.residuals
            if result.paths is not None:
                paths[group, category] = result.paths
            elif config.selection.uncertainty is not None:
                point_only.append(
                    {"group": group, "category": category.value, "reason": result.point_only}
                )
            report[f"{group}/{category.value}"] = {
                "scores": result.scores,
                "weights": result.weights,
                "method": result.method,
                "folds": {m: [f.as_record() for f in fs] for m, fs in result.folds.items()},
                "parameters": result.parameters,
            }
        ctx.note(self.name, series_skipped=skipped, selection=report, point_only=point_only)
        return {"forecasts": forecasts, "paths": paths, "residuals": residuals}


@dataclass(frozen=True)
class Cohere:
    """A fill-tree coherence pass (see `coherence`): the configured method after generation,
    and bottom up after any stage that adjusts the identity's terms."""

    stage: str = "generated"
    needs: tuple[str, ...] = ("forecasts", "config")
    gives: tuple[str, ...] = ("forecasts",)

    @property
    def name(self) -> str:
        return f"cohere_{self.stage}"

    def validate(self, ctx: Context) -> list[str]:
        return [] if self.stage in PASSES else [f"no coherence pass {self.stage!r}"]

    def execute(self, ctx: Context) -> Mapping[str, Any]:
        method: CoherenceMethod = (
            ctx["config"].coherence.method if self.stage == "generated" else "bottom_up"
        )
        forecasts, closed = cohere(ctx["forecasts"], method)
        ctx.note(self.name, method=method, closed=closed)
        return {"forecasts": forecasts}


@dataclass(frozen=True)
class Reconcile:
    """Reconcile each group's flows to its anchor, rewriting the forecasts."""

    name: str = "reconcile"
    needs: tuple[str, ...] = ("series", "forecasts", "residuals", "config")
    gives: tuple[str, ...] = ("forecasts",)

    def validate(self, ctx: Context) -> list[str]:
        tolerance = ctx["config"].tolerance
        return [] if tolerance >= 0 else [f"the tolerance must be non-negative, got {tolerance}"]

    def execute(self, ctx: Context) -> Mapping[str, Any]:
        series, config = ctx["series"], ctx["config"]
        forecasts: Forecasts = dict(ctx["forecasts"])
        # The identity's terms: the deepest cover of the fill tree the run forecasts, and the
        # other flows it carries. A flow it doesn't carry can't be adjusted, and the gap it
        # leaves falls on the others.
        carried = {category for _, category in forecasts}
        terms_of, missing = identity_terms(carried)
        signs = {c.value: s for c, s in terms_of.items()}
        rules = config.reconciliation
        errors: Forecasts = ctx["residuals"]
        residuals = {}
        methods: dict[str, int] = {}
        for group in series.groups:
            terms = [GuidanceCategory(c) for c in signs]
            if (group, ANCHOR) not in forecasts or any((group, c) not in forecasts for c in terms):
                continue
            keys = [ANCHOR, *terms]
            problem = Problem(
                forecasts[group, ANCHOR],
                float(series.values(group, ANCHOR).iloc[-1]),
                {c.value: forecasts[group, c] for c in terms},
                signs,
                config.tolerance,
                {c.value: errors[group, c] for c in keys if (group, c) in errors},
                rules.sign_aware,
                rules.anchor_fixed,
                rules.shrinkage,
            )
            reconciled = RECONCILERS.build(rules.method)(problem)
            for c in terms:
                forecasts[group, c] = reconciled.flows[c.value]
            if reconciled.anchor is not None:
                forecasts[group, ANCHOR] = reconciled.anchor
            residuals[group] = reconciled.residual
            methods[reconciled.method] = methods.get(reconciled.method, 0) + 1
        largest = max((float(np.abs(r).max()) for r in residuals.values()), default=0.0)
        ctx.note(
            self.name,
            max_abs_residual=largest,
            methods=methods,
            terms=signs,
            missing_terms=[*(c.value for c in missing), TRANSFERS_OUT],
        )
        return {"forecasts": forecasts}


@dataclass(frozen=True)
class Align:
    """Move each group's anchor onto its plan targets, re-deriving the flows (see
    `alignment`), after reconciliation, so the aligned values are final."""

    name: str = "align"
    needs: tuple[str, ...] = ("series", "forecasts", "config", "target", "movements")
    gives: tuple[str, ...] = ("forecasts",)

    def validate(self, ctx: Context) -> list[str]:
        target, config = ctx["target"], ctx["config"]
        if target is None:
            return ["alignment needs a target, and the run has none"]
        if not isinstance(target, AlignmentTarget):
            return [f"the target must be an AlignmentTarget, not {type(target).__name__}"]
        groupings = [m.grouping for m in target.marginals]
        if config.grouping not in groupings:
            return [
                f"the target has no marginal for the run's grouping {config.grouping!r}; it "
                f"has {groupings}"
            ]
        return []

    def execute(self, ctx: Context) -> Mapping[str, Any]:
        series, config, target = ctx["series"], ctx["config"], ctx["target"]
        rules = config.alignment
        forecasts: Forecasts = dict(ctx["forecasts"])
        terms_of, _ = identity_terms({category for _, category in forecasts})
        signs = {c.value: s for c, s in terms_of.items()}
        marginal = target.marginal(config.grouping)
        months = [series.snapshot + 1 + k for k in range(config.horizon)]
        movements = ctx["movements"] or {}
        shortfalls: dict[str, float] = {}
        untargeted, aligned_months = [], 0
        for group in series.groups:
            keys = [ANCHOR, *terms_of]
            if any((group, c) not in forecasts for c in keys):
                continue
            if group not in marginal.groups:
                untargeted.append(group)
                continue
            g = marginal.groups.index(group)
            goal, band = _targets(target, marginal, config.grouping, months, g, rules.all_months)
            problem = AlignmentProblem(
                forecasts[group, ANCHOR],
                float(series.values(group, ANCHOR).iloc[-1]),
                {c.value: forecasts[group, c] for c in terms_of},
                signs,
                goal,
                band,
                frozenset(c.value for c in rules.fixed),
                movements.get(group),
                {c.value: w for c, w in rules.weights.items()},
                rules.rate_aware,
                rules.shortfall_penalty,
            )
            result = ALIGNERS.build(rules.method)(problem)
            forecasts[group, ANCHOR] = result.anchor
            for c in terms_of:
                forecasts[group, c] = result.flows[c.value]
            aligned_months += int((~np.isnan(goal)).sum())
            shortfalls[group] = float(np.abs(result.shortfall).max(initial=0.0))
        ctx.note(
            self.name,
            method=rules.method,
            target=target.digest(),
            # Alignment runs after reconciliation, so its values are the artifact's.
            precedence="aligned after reconciliation" if config.reconcile else "aligned",
            aligned_months=aligned_months,
            untargeted_groups=untargeted,
            max_shortfall=max(shortfalls.values(), default=0.0),
            shortfalls={g: v for g, v in shortfalls.items() if v > 0},
        )
        return {"forecasts": forecasts}


def _targets(
    target: AlignmentTarget,
    marginal: Any,
    grouping: str,
    months: list[Any],
    group: int,
    all_months: bool,
) -> tuple[Any, Any]:
    """Per forecast month, the group's target and band, NaN where the month isn't aligned:
    outside the target's window, past its evaluation month, or, unless `all_months`, off its
    target dates."""
    first, last = target.window
    goal, band = np.full(len(months), np.nan), np.zeros(len(months))
    for k, month in enumerate(months):
        if not first <= month <= last or not target.is_planned(month):
            continue
        if all_months:
            goal[k] = target.target(grouping, month)[group]
            continue
        width = marginal.band_at(month)[group]
        if not np.isnan(width):
            goal[k], band[k] = target.target(grouping, month)[group], width
    return goal, band


@dataclass(frozen=True)
class Disaggregate:
    """Split each group's forecasts onto the leaf grain inside it (see `disaggregation`)."""

    name: str = "disaggregate"
    needs: tuple[str, ...] = ("source", "series", "forecasts", "paths", "grain", "config")
    gives: tuple[str, ...] = ("forecasts", "paths", "grain")

    def validate(self, ctx: Context) -> list[str]:
        config = ctx["config"]
        try:
            with as_source(ctx["source"]).connect() as con:
                return ingestion.problems(con, config.disaggregation.leaf, config.categories)
        except SourceError as error:
            return [str(error)]

    def execute(self, ctx: Context) -> Mapping[str, Any]:
        config, series = ctx["config"], ctx["series"]
        rules = config.disaggregation
        forecasts: Forecasts = ctx["forecasts"]
        groups = sorted({group for group, _ in forecasts})
        categories = sorted({c for _, c in forecasts}, key=list(GuidanceCategory).index)
        with as_source(ctx["source"]).connect() as con:
            found = basis(
                con,
                ctx["grain"],
                rules.leaf,
                groups,
                categories,
                series.snapshot,
                rules.window_months,
                rules.chunk_size,
            )
        weights = shares(found, list(forecasts), rules.strategy, series.snapshot)
        leaf_forecasts, leaf_paths, uncovered = split(forecasts, ctx["paths"], weights)
        ctx.note(
            self.name,
            strategy=rules.strategy,
            leaf=rules.leaf,
            leaves=len({leaf for leaf, _ in leaf_forecasts}),
            chunks=found.chunks,
            fallbacks=weights.fallbacks,
            uncovered=[{"group": g, "category": c.value} for g, c in uncovered],
        )
        return {"forecasts": leaf_forecasts, "paths": leaf_paths, "grain": rules.leaf}


@dataclass(frozen=True)
class Format:
    """Write the artifact: the point forecasts, and with sample paths on, the paths and the
    quantiles read from them. Negative values are clamped to zero, as writers clamp.

    Quantiles go in as columns, which every row must carry, so they're written only when
    every series has paths; otherwise the paths go in alone, and the report says why the
    quantiles didn't. Paths and quantiles are the models' own draws, before reconciliation
    adjusts the point forecasts."""

    name: str = "format"
    needs: tuple[str, ...] = ("series", "forecasts", "paths", "grain", "config")
    gives: tuple[str, ...] = ("artifact",)

    def validate(self, ctx: Context) -> list[str]:
        return []

    def execute(self, ctx: Context) -> Mapping[str, Any]:
        series, config, grain = ctx["series"], ctx["config"], ctx["grain"]
        raw: Forecasts = ctx["forecasts"]
        negative_cells = int(sum(int((v < 0).sum()) for v in raw.values()))
        # Writers clamp, and the final coherence pass rebuilds each parent from its clamped
        # children, which leaves nothing negative.
        clamped = {key: np.maximum(v, 0.0) for key, v in raw.items()}
        forecasts, closed = cohere(clamped, "bottom_up")
        drawn: Forecasts = ctx["paths"]
        levels = config.uncertainty.quantiles
        with_quantiles = bool(drawn) and set(drawn) == set(forecasts)
        quantiles = {
            key: np.quantile(np.maximum(drawn[key], 0.0), np.asarray(levels) / 100, axis=0)
            for key in forecasts
            if with_quantiles
        }
        rows = []
        for (group, category), values in forecasts.items():
            for step, value in enumerate(values):
                row = {
                    "forecast_month": step,
                    "category": category.value,
                    "value": float(value),
                    grain: group,
                }
                if with_quantiles:
                    row |= {
                        f"q{q}": float(v)
                        for q, v in zip(levels, quantiles[group, category][:, step], strict=True)
                    }
                rows.append(row)
        columns = ["forecast_month", "category", "value", grain]
        if with_quantiles:
            columns += [f"q{q}" for q in levels]
        frame = pd.DataFrame(rows, columns=columns)
        keys = [key for key in forecasts if key in drawn]
        clamped_paths = int(sum(int((drawn[k] < 0).sum()) for k in keys))
        paths = np.stack([np.maximum(drawn[k], 0.0) for k in keys]) if keys else None
        artifact = GuidanceForecast(
            frame.astype({"forecast_month": np.int64}),
            grain,
            series.snapshot.end_time.date().isoformat(),
            config.horizon,
            config.population,
            sample_paths=paths,
            path_index=tuple((g, c.value) for g, c in keys),
            aligned_to=ctx.report.get("align", {}).get("target"),
        )
        ctx.note(
            self.name,
            clamped_negative_cells=negative_cells,
            coherence_closed=closed,
            clamped_path_values=clamped_paths,
            series_with_paths=len(keys),
            quantiles=list(levels) if with_quantiles else [],
        )
        return {"artifact": artifact}


def pipeline(config: GuidanceConfig) -> Pipeline:
    """The lite pipeline's steps, reconciling when `config` asks to."""
    middle = (Reconcile(), Cohere("reconciled")) if config.reconcile else ()
    align = (Align(), Cohere("aligned")) if config.alignment.method is not None else ()
    split_ = (Disaggregate(),) if config.disaggregation.strategy is not None else ()
    return Pipeline(
        (
            Ingest(),
            Validate(),
            Select(),
            Generate(),
            Cohere("generated"),
            *middle,
            *align,
            *split_,
            Format(),
        )
    )


def run(
    source: Source | duckdb.DuckDBPyConnection | Path,
    config: GuidanceConfig | None = None,
    target: AlignmentTarget | None = None,
    movements: Mapping[str, Any] | None = None,
) -> PipelineResult:
    """Run the pipeline on `source`: a `Source`, a DuckDB connection, or a path to a DuckDB
    file or a directory of table files. `target` holds the plan targets alignment moves the
    forecast onto, and `movements` each group's declared change, `ε_H`, per forecast month."""
    config = config or GuidanceConfig()
    seed = {"source": source, "config": config, "target": target, "movements": movements}
    ctx = pipeline(config).run(Context(seed))
    artifact: GuidanceForecast = ctx["artifact"]
    sections = ctx.report
    report = {
        "fingerprint": artifact.fingerprint(),
        "configuration": config_digest(config),
        "source": sections["ingest"]["source"],
        "snapshot": sections["validate"]["snapshot"],
        "grouping": config.grouping,
        "series_forecast": len(ctx["forecasts"]),
        "series_skipped": [
            *sections["select"]["series_skipped"],
            *sections["generate"]["series_skipped"],
        ],
        "clamped_negative_cells": sections["format"]["clamped_negative_cells"],
        "alignment": sections.get("align", {"method": None}),
        "disaggregation": sections.get("disaggregate", {"strategy": None}),
        "grain": ctx["grain"],
        "coherence": {
            "method": config.coherence.method,
            "closed": {
                **{
                    stage: sections[f"cohere_{stage}"]["closed"]
                    for stage in PASSES
                    if f"cohere_{stage}" in sections
                },
                "final": sections["format"]["coherence_closed"],
            },
        },
        "uncertainty": {
            "enabled": config.uncertainty.enabled,
            "series_with_paths": sections["format"]["series_with_paths"],
            "quantiles": sections["format"]["quantiles"],
            "clamped_path_values": sections["format"]["clamped_path_values"],
            "point_only": sections["generate"]["point_only"],
        },
        "validation": {
            "gaps": sections["validate"]["gaps"],
            "dropped": sections["validate"]["dropped"],
        },
        "reconciliation": {
            "identity": "partial",
            "terms": sections.get("reconcile", {}).get("terms", {}),
            "methods": sections.get("reconcile", {}).get("methods", {}),
            "missing_terms": sections.get("reconcile", {}).get("missing_terms", [TRANSFERS_OUT]),
            "max_abs_residual": sections.get("reconcile", {}).get("max_abs_residual", 0.0),
        },
        "selection": sections["generate"]["selection"],
        "capabilities": sections["select"]["capabilities"],
        "steps": [
            {"step": r.step, "gave": list(r.gave), "seconds": round(r.seconds, 6)}
            for r in ctx.records
        ],
    }
    return PipelineResult(artifact, report, ctx["series"])
