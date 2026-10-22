"""The guidance pipeline, lite: ingest, forecast, reconcile, and write the artifact.

The pipeline runs as ordered steps over a shared context (see `framework`):

1. **ingest** reads the warehouse into monthly series per group and category, composing an
   inner fill-tree node from its leaves;
2. **validate** settles the series: the snapshot, the feed's gaps, and the all-zero series (see
   `validation`), and gives the calendar covariates for the history and the horizon;
3. **forecast** gives every series long enough a combined benchmark forecast;
4. **reconcile**, when the configuration asks for it, reconciles each group's flows to its
   headcount anchor. The lite identity carries hires and attrition only, so it's partial
   wherever people move between groups, and the report says so;
5. **format** writes the artifact. Negative cells are clamped to zero there, and the report
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
from hcsim.guidance.config import GuidanceConfig
from hcsim.guidance.covariates import covariates
from hcsim.guidance.framework import Context, Pipeline
from hcsim.guidance.models import ForecastError
from hcsim.guidance.reconcile import RECONCILERS
from hcsim.guidance.selection import forecast
from hcsim.guidance.sources import Source, SourceError, as_source
from hcsim.model.core.categories import ANCHOR, GuidanceCategory, coherency_terms
from hcsim.model.core.guidance import GuidanceForecast

#: Flows the full identity carries that the lite categories can't forecast.
MISSING_TERMS: Final = ("internal_starts", "transfers_out", "conversion_out")

Forecasts = dict[tuple[str, GuidanceCategory], np.ndarray]


@dataclass(frozen=True)
class PipelineResult:
    artifact: GuidanceForecast
    report: dict[str, Any]


# --- steps ---------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Ingest:
    name: str = "ingest"
    needs: tuple[str, ...] = ("source", "config")
    gives: tuple[str, ...] = ("series",)

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
            counted=[c.value for c in ingestion.counted(config.categories)],
        )
        return {"series": series}


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


@dataclass(frozen=True)
class Forecast:
    name: str = "forecast"
    needs: tuple[str, ...] = ("series", "config")
    gives: tuple[str, ...] = ("forecasts",)

    def validate(self, ctx: Context) -> list[str]:
        horizon = ctx["config"].horizon
        return [] if horizon >= 1 else [f"the horizon must be at least 1 month, got {horizon}"]

    def execute(self, ctx: Context) -> Mapping[str, Any]:
        series, config = ctx["series"], ctx["config"]
        forecasts: Forecasts = {}
        scores: dict[str, dict[str, Any]] = {}
        skipped: list[dict[str, str]] = []
        for group, category in series.keys:
            values = series.values(group, category).to_numpy()
            try:
                result = forecast(values, config.horizon, config.selection)
            except ForecastError as error:
                skipped.append({"group": group, "category": category.value, "reason": str(error)})
                continue
            forecasts[group, category] = result.values
            scores[f"{group}/{category.value}"] = {
                "scores": result.scores,
                "weights": result.weights,
            }
        ctx.note(self.name, series_skipped=skipped, selection=scores)
        return {"forecasts": forecasts}


@dataclass(frozen=True)
class Reconcile:
    """Reconcile each group's flows to its anchor, rewriting the forecasts."""

    name: str = "reconcile"
    needs: tuple[str, ...] = ("series", "forecasts", "config")
    gives: tuple[str, ...] = ("forecasts",)

    def validate(self, ctx: Context) -> list[str]:
        tolerance = ctx["config"].tolerance
        return [] if tolerance >= 0 else [f"the tolerance must be non-negative, got {tolerance}"]

    def execute(self, ctx: Context) -> Mapping[str, Any]:
        series, config = ctx["series"], ctx["config"]
        forecasts: Forecasts = dict(ctx["forecasts"])
        # The identity's terms this run forecasts. A term with no series can't be adjusted, and
        # the gap it leaves falls on the others.
        forecast = {category for _, category in forecasts}
        signs = {c.value: s for c, s in coherency_terms().items() if c in forecast}
        residuals = {}
        for group in series.groups:
            terms = [GuidanceCategory(c) for c in signs]
            if (group, ANCHOR) not in forecasts or any((group, c) not in forecasts for c in terms):
                continue
            last_actual = float(series.values(group, ANCHOR).iloc[-1])
            reconciled = RECONCILERS.build(config.reconciliation.method)(
                forecasts[group, ANCHOR],
                last_actual,
                {c.value: forecasts[group, c] for c in terms},
                signs,
                config.tolerance,
            )
            for c in terms:
                forecasts[group, c] = reconciled.flows[c.value]
            residuals[group] = reconciled.residual
        largest = max((float(np.abs(r).max()) for r in residuals.values()), default=0.0)
        ctx.note(self.name, max_abs_residual=largest)
        return {"forecasts": forecasts}


@dataclass(frozen=True)
class Format:
    name: str = "format"
    needs: tuple[str, ...] = ("series", "forecasts", "config")
    gives: tuple[str, ...] = ("artifact",)

    def validate(self, ctx: Context) -> list[str]:
        return []

    def execute(self, ctx: Context) -> Mapping[str, Any]:
        series, config = ctx["series"], ctx["config"]
        rows = [
            {
                "forecast_month": step,
                "category": category.value,
                "value": float(value),
                config.grouping: group,
            }
            for (group, category), values in ctx["forecasts"].items()
            for step, value in enumerate(values)
        ]
        columns = ["forecast_month", "category", "value", config.grouping]
        frame = pd.DataFrame(rows, columns=columns)
        negative = frame["value"] < 0
        frame.loc[negative, "value"] = 0.0  # writers clamp; the loader refuses negatives
        artifact = GuidanceForecast(
            frame.astype({"forecast_month": np.int64}),
            config.grouping,
            series.snapshot.end_time.date().isoformat(),
            config.horizon,
            config.population,
        )
        ctx.note(self.name, clamped_negative_cells=int(negative.sum()))
        return {"artifact": artifact}


def pipeline(config: GuidanceConfig) -> Pipeline:
    """The lite pipeline's steps, reconciling when `config` asks to."""
    middle = (Reconcile(),) if config.reconcile else ()
    return Pipeline((Ingest(), Validate(), Forecast(), *middle, Format()))


def run(
    source: Source | duckdb.DuckDBPyConnection | Path, config: GuidanceConfig | None = None
) -> PipelineResult:
    """Run the pipeline on `source`: a `Source`, a DuckDB connection, or a path to a DuckDB
    file or a directory of table files."""
    config = config or GuidanceConfig()
    ctx = pipeline(config).run(Context({"source": source, "config": config}))
    artifact: GuidanceForecast = ctx["artifact"]
    sections = ctx.report
    report = {
        "fingerprint": artifact.fingerprint(),
        "snapshot": sections["validate"]["snapshot"],
        "grouping": config.grouping,
        "series_forecast": len(ctx["forecasts"]),
        "series_skipped": sections["forecast"]["series_skipped"],
        "clamped_negative_cells": sections["format"]["clamped_negative_cells"],
        "validation": {
            "gaps": sections["validate"]["gaps"],
            "dropped": sections["validate"]["dropped"],
        },
        "reconciliation": {
            "identity": "partial",
            "missing_terms": list(MISSING_TERMS),
            "max_abs_residual": sections.get("reconcile", {}).get("max_abs_residual", 0.0),
        },
        "selection": sections["forecast"]["selection"],
        "steps": [
            {"step": r.step, "gave": list(r.gave), "seconds": round(r.seconds, 6)}
            for r in ctx.records
        ],
    }
    return PipelineResult(artifact, report)
