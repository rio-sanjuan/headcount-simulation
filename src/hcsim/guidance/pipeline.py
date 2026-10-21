"""The guidance pipeline, lite: ingest, forecast, reconcile, and write the artifact.

Every series long enough gets a combined benchmark forecast. Per group, the flows are then
reconciled to the headcount anchor. The lite identity carries hires and attrition only, so it's
partial wherever people move between groups, and the report says so. Negative cells are clamped
to zero when the artifact is written, and the report counts them.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Final

import duckdb
import numpy as np
import pandas as pd

from hcsim.guidance.ingest import Series, ingest
from hcsim.guidance.models import ForecastError
from hcsim.guidance.reconcile import reconcile_greedy
from hcsim.guidance.selection import SelectionConfig, forecast
from hcsim.model.core.categories import ANCHOR, GuidanceCategory, coherency_terms
from hcsim.model.core.guidance import GuidanceForecast

#: Flows the full identity carries that the lite categories can't forecast.
MISSING_TERMS: Final = ("internal_starts", "transfers_out", "conversion_out")


@dataclass(frozen=True)
class PipelineConfig:
    horizon: int = 12
    grouping: str = "cost_center"
    selection: SelectionConfig = field(default_factory=SelectionConfig)
    reconcile: bool = True
    tolerance: float = 1.0
    population: str | None = None


@dataclass(frozen=True)
class PipelineResult:
    artifact: GuidanceForecast
    report: dict[str, Any]


def run(con: duckdb.DuckDBPyConnection, config: PipelineConfig | None = None) -> PipelineResult:
    config = config or PipelineConfig()
    series = ingest(con, config.grouping)
    forecasts: dict[tuple[str, GuidanceCategory], np.ndarray] = {}
    scores: dict[str, dict[str, Any]] = {}
    skipped: list[dict[str, str]] = []
    for group in series.groups:
        for category in GuidanceCategory:
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

    residuals = _reconcile(series, forecasts, config) if config.reconcile else {}
    frame, clamped = _format(forecasts, config)
    artifact = GuidanceForecast(
        frame,
        config.grouping,
        series.snapshot.end_time.date().isoformat(),
        config.horizon,
        config.population,
    )
    report = {
        "fingerprint": artifact.fingerprint(),
        "snapshot": str(series.snapshot),
        "grouping": config.grouping,
        "series_forecast": len(forecasts),
        "series_skipped": skipped,
        "clamped_negative_cells": clamped,
        "reconciliation": {
            "identity": "partial",
            "missing_terms": list(MISSING_TERMS),
            "max_abs_residual": max(
                (float(np.abs(r).max()) for r in residuals.values()), default=0.0
            ),
        },
        "selection": scores,
    }
    return PipelineResult(artifact, report)


def _reconcile(
    series: Series,
    forecasts: dict[tuple[str, GuidanceCategory], np.ndarray],
    config: PipelineConfig,
) -> dict[str, np.ndarray]:
    signs = {c.value: s for c, s in coherency_terms().items()}
    residuals = {}
    for group in series.groups:
        terms = [GuidanceCategory(c) for c in signs]
        if (group, ANCHOR) not in forecasts or any((group, c) not in forecasts for c in terms):
            continue
        last_actual = float(series.values(group, ANCHOR).iloc[-1])
        reconciled = reconcile_greedy(
            forecasts[group, ANCHOR],
            last_actual,
            {c.value: forecasts[group, c] for c in terms},
            signs,
            config.tolerance,
        )
        for c in terms:
            forecasts[group, c] = reconciled.flows[c.value]
        residuals[group] = reconciled.residual
    return residuals


def _format(
    forecasts: dict[tuple[str, GuidanceCategory], np.ndarray], config: PipelineConfig
) -> tuple[pd.DataFrame, int]:
    rows = [
        {
            "forecast_month": step,
            "category": category.value,
            "value": float(value),
            config.grouping: group,
        }
        for (group, category), values in forecasts.items()
        for step, value in enumerate(values)
    ]
    frame = pd.DataFrame(rows, columns=["forecast_month", "category", "value", config.grouping])
    negative = frame["value"] < 0
    frame.loc[negative, "value"] = 0.0  # writers clamp; the loader refuses negatives
    return frame.astype({"forecast_month": np.int64}), int(negative.sum())
