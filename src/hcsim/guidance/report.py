"""The guidance run report: what a run forecast, how, and what it adjusted, with plots.

`write_report(result, directory)` writes `report.md` and `figures/<category>.svg`:

- **Stamps**: the artifact's fingerprint, snapshot, grain, horizon, categories, and the step
  offsets its months are written at (step 0 is the first month after the snapshot), its
  population and `aligned_to` stamps, the configuration's digest, the source, and the hcsim
  version;
- **Series**: how many were forecast, and each one skipped, with why;
- **Clamped cells**: the negative cells written as zero, by category and group;
- **Selection**: per category, the combination methods that weighed the series and each
  model's mean weight;
- **Reconciliation, coherence, alignment, uncertainty, disaggregation**: what each step
  adjusted, from the run report;
- **Figures**: per category, the total over groups: the last two years of history, the
  forecast, and its 10th and 90th quantiles where the artifact carries them.

Nothing in the report depends on when it was written; timings stay in the run's JSON report.
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from importlib.metadata import version
from pathlib import Path
from typing import TYPE_CHECKING, Any, Final

import pandas as pd

from hcsim.model.core.guidance import CATEGORY, FORECAST_MONTH, VALUE
from hcsim.model.reports.harness import table
from hcsim.model.reports.svg import Series as Line
from hcsim.model.reports.svg import chart

if TYPE_CHECKING:
    from hcsim.guidance.pipeline import PipelineResult

REPORT_FILE: Final = "report.md"
FIGURES: Final = "figures"
#: The months of history a figure shows before the forecast.
HISTORY_MONTHS: Final = 24


def config_digest(config: Any) -> str:
    """`gc1-` and 16 hex characters of a SHA-256 over the configuration, canonically."""
    text = config.model_dump_json(exclude_none=False)
    return "gc1-" + hashlib.sha256(text.encode()).hexdigest()[:16]


def steps(snapshot: pd.Period, horizon: int) -> list[tuple[int, str]]:
    """Each step offset and the calendar month it stands for."""
    return [(k, str(snapshot + 1 + k)) for k in range(horizon)]


def _stamps(result: PipelineResult) -> list[list[object]]:
    artifact, report = result.artifact, result.report
    offsets = steps(result.series.snapshot, artifact.horizon)
    return [
        ["fingerprint", artifact.fingerprint()],
        ["snapshot", artifact.snapshot_date],
        ["grain", artifact.grouping_column],
        ["horizon", artifact.horizon],
        ["categories", ", ".join(sorted(c.value for c in artifact.categories))],
        ["step offsets", f"0 = {offsets[0][1]} … {offsets[-1][0]} = {offsets[-1][1]}"],
        ["population", artifact.population or "—"],
        ["aligned_to", artifact.aligned_to or "—"],
        ["configuration", report.get("configuration", "—")],
        ["source", report.get("source", "—")],
        ["hcsim", version("hcsim")],
    ]


def _selection(report: Mapping[str, Any]) -> str:
    rows = []
    by_category: dict[str, list[Mapping[str, Any]]] = {}
    for key, entry in sorted(report.get("selection", {}).items()):
        by_category.setdefault(key.rsplit("/", 1)[1], []).append(entry)
    for category, entries in sorted(by_category.items()):
        methods = pd.Series([e["method"] for e in entries]).value_counts()
        weights = pd.DataFrame([e["weights"] for e in entries]).fillna(0.0).mean()
        leading = ", ".join(f"{m} {w:.2f}" for m, w in weights.sort_values(ascending=False).items())
        used = ", ".join(f"{m} x{n}" for m, n in methods.items())
        rows.append([category, len(entries), used, leading])
    return table(["category", "series", "methods", "mean weight"], rows)


def _figure(result: PipelineResult, category: str) -> str | None:
    frame = result.artifact.frame
    rows = frame[frame[CATEGORY] == category]
    if rows.empty:
        return None
    forecast = rows.groupby(FORECAST_MONTH)[VALUE].sum().sort_index()
    history = result.series.frame
    past = history[history["category"] == category].groupby("month")["value"].sum().sort_index()
    past = past.iloc[-HISTORY_MONTHS:]
    start = -len(past)
    lines = []
    if len(past):
        lines.append(Line("history", list(range(start, 0)), past.tolist(), "line"))
    lines.append(Line("forecast", list(forecast.index), forecast.tolist(), "line"))
    for q in (10, 90):
        column = f"q{q}"
        if column in rows.columns:
            band = rows.groupby(FORECAST_MONTH)[column].sum().sort_index()
            lines.append(Line(f"q{q}", list(band.index), band.tolist(), "line"))
    return chart(
        lines,
        title=f"{category}: history and forecast",
        x_label="months from the first forecast month",
        y_label=category,
    )


def _section(title: str, entries: Mapping[str, Any]) -> str:
    rows = [[key, _plain(value)] for key, value in entries.items() if key != "seconds"]
    return f"## {title}\n\n" + (table(["field", "value"], rows) if rows else "Nothing to report.")


def _plain(value: object) -> str:
    if isinstance(value, float):
        return f"{value:.6g}"
    if isinstance(value, Mapping):
        return ", ".join(f"{k}: {_plain(v)}" for k, v in value.items()) or "—"
    if isinstance(value, list):
        return ", ".join(_plain(v) for v in value) or "—"
    return str(value)


def markdown(result: PipelineResult) -> tuple[str, dict[str, str]]:
    """The report's Markdown, and its figures as SVG, by name."""
    report = result.report
    figures = {}
    for category in sorted(c.value for c in result.artifact.categories):
        svg = _figure(result, category)
        if svg is not None:
            figures[category] = svg
    skipped = report.get("series_skipped", [])
    parts = [
        "# Guidance run",
        "## Stamps",
        table(["stamp", "value"], _stamps(result)),
        "## Series",
        f"{report.get('series_forecast', 0)} series forecast, {len(skipped)} skipped.",
    ]
    if skipped:
        parts.append(
            table(
                ["group", "category", "reason"],
                [[s["group"], s["category"], s["reason"]] for s in skipped],
            )
        )
    clamped = report.get("clamped_cells", {})
    parts += [
        "## Clamped cells",
        f"{report.get('clamped_negative_cells', 0)} negative cells clamped to zero.",
    ]
    if clamped:
        rows = [
            [c, g, n]
            for c, by_group in sorted(clamped.items())
            for g, n in sorted(by_group.items())
        ]
        parts.append(table(["category", "group", "cells"], rows))
    parts += ["## Selection", _selection(report)]
    for title in ("reconciliation", "coherence", "alignment", "uncertainty", "disaggregation"):
        entries = report.get(title)
        if isinstance(entries, Mapping):
            parts.append(_section(title.capitalize(), entries))
    if figures:
        parts.append("## Figures")
        parts += [f"![{name}]({FIGURES}/{name}.svg)" for name in sorted(figures)]
    return "\n\n".join(parts) + "\n", figures


def write_report(result: PipelineResult, directory: Path) -> Path:
    """Write the report and its figures into `directory`. Returns the report's path."""
    text, figures = markdown(result)
    directory.mkdir(parents=True, exist_ok=True)
    if figures:
        (directory / FIGURES).mkdir(exist_ok=True)
    for name, svg in sorted(figures.items()):
        (directory / FIGURES / f"{name}.svg").write_text(svg)
    path = directory / REPORT_FILE
    path.write_text(text)
    return path
