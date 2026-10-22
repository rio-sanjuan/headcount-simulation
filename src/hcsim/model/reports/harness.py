"""The report harness: a Markdown report with its provenance first, figures beside it, and a
validated metrics sidecar.

A `Report` is a title, a `Provenance`, and sections in order. A section is Markdown, with
optional figures, as SVG text, that it links by name. `Report.write(directory, sidecar)` writes:

- `report.md`: the title, the provenance as a table, then every section;
- `figures/<name>.svg`: each section's figures;
- `metrics.json`: the sidecar, validated, whose provenance must be the report's.

Nothing in a report depends on when it was written, so the same run writes the same files.

`training_report(run, provenance)` is the report of a training run:

- a selection summary, one row per domain;
- per domain, every arm's metrics, the decisions that chose the champion, and a classifier
  champion's lift table over the held-out rows;
- any ground-truth recoveries, each with its figures, whose rows go into the sidecar.
"""

from __future__ import annotations

import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Final

import pandas as pd

from hcsim.model.ml.metrics import FAMILY_METRICS
from hcsim.model.ml.training import TrainingRun
from hcsim.model.reports.sidecar import MetricsSidecar, Provenance, SidecarError

if TYPE_CHECKING:
    from hcsim.model.reports.recovery import Recovery

REPORT_FILE: Final = "report.md"
FIGURES: Final = "figures"
_NAME: Final = re.compile(r"[a-z0-9][a-z0-9_-]*")


def number(value: object) -> str:
    """A table cell: integers as they are, other numbers to four significant figures, and a
    missing value as an em dash."""
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "—"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return f"{value:.4g}"
    return str(value)


def table(headers: Sequence[str], rows: Sequence[Sequence[object]]) -> str:
    """A Markdown table. A pipe inside a cell is escaped."""

    def cell(value: object) -> str:
        return number(value).replace("|", "\\|")

    lines = ["| " + " | ".join(headers) + " |", "|" + "---|" * len(headers)]
    lines += ["| " + " | ".join(cell(v) for v in row) + " |" for row in rows]
    return "\n".join(lines)


def frame_table(frame: pd.DataFrame) -> str:
    return table(
        [str(c) for c in frame.columns],
        [[v.item() if hasattr(v, "item") else v for v in row] for row in frame.to_numpy()],
    )


@dataclass(frozen=True)
class Section:
    title: str
    body: str
    #: Figures as SVG text, by name. The body links them as `figures/<name>.svg`.
    figures: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        bad = sorted(n for n in self.figures if not _NAME.fullmatch(n))
        if bad:
            raise SidecarError(f"figure names must be lowercase words, not {bad}")


@dataclass
class Report:
    title: str
    provenance: Provenance
    sections: list[Section] = field(default_factory=list)

    def add(self, section: Section) -> Report:
        names = {n for s in self.sections for n in s.figures}
        repeated = sorted(names & set(section.figures))
        if repeated:
            raise SidecarError(f"figures already in the report: {repeated}")
        self.sections.append(section)
        return self

    def markdown(self) -> str:
        fields = self.provenance.model_dump(mode="json")
        models = fields.pop("models")
        rows = [[k, v] for k, v in fields.items()]
        rows += [[f"model: {d}", v] for d, v in sorted(models.items())]
        parts = [f"# {self.title}", "## Provenance", table(["field", "value"], rows)]
        for section in self.sections:
            parts += [f"## {section.title}", section.body.strip()]
        return "\n\n".join(parts) + "\n"

    def write(self, directory: Path, sidecar: MetricsSidecar) -> Path:
        """Write the report, its figures, and its sidecar. Returns the report's path."""
        if sidecar.provenance != self.provenance:
            raise SidecarError("the sidecar's provenance isn't the report's")
        directory.mkdir(parents=True, exist_ok=True)
        figures = {n: svg for s in self.sections for n, svg in s.figures.items()}
        if figures:
            (directory / FIGURES).mkdir(exist_ok=True)
        for name, svg in sorted(figures.items()):
            (directory / FIGURES / f"{name}.svg").write_text(svg)
        sidecar.write(directory)
        path = directory / REPORT_FILE
        path.write_text(self.markdown())
        return path


def _selection(run: TrainingRun) -> Section:
    rows = []
    for name, result in run.domains.items():
        metric = FAMILY_METRICS[result.family][0]
        null = next(iter(result.arms.values()))
        rows.append(
            [
                name,
                result.family,
                metric,
                result.train_rows,
                result.test_rows,
                null.arm.name,
                null.metrics.get(metric),
                result.selection.champion,
                result.champion.metrics.get(metric),
            ]
        )
    headers = [
        "domain",
        "family",
        "primary metric",
        "train rows",
        "held-out rows",
        "null arm",
        "null score",
        "champion",
        "champion score",
    ]
    body = (
        "Every arm is scored on the same held-out rows. A richer arm replaces the champion "
        f"only when its mean per-row gain exceeds {number(run.z)} standard errors of the "
        "paired gains. Scores are each arm's primary metric on the held-out rows: lower is "
        "better for log loss and WAPE, higher for log-likelihood.\n\n" + table(headers, rows)
    )
    return Section("Selection", body)


def _domain(name: str, run: TrainingRun) -> Section:
    result = run.domains[name]
    metrics = FAMILY_METRICS[result.family]
    shown = [m for m in metrics if any(m in a.metrics for a in result.arms.values())]
    arms = table(
        ["arm", "algorithm", "version", "primary", *shown],
        [
            [a.arm.name, a.arm.algorithm, a.version, a.primary, *(a.metrics.get(m) for m in shown)]
            for a in result.arms.values()
        ],
    )
    decisions = table(
        ["arm", "against", "mean gain", "standard error", "cleared"],
        [
            [d.arm, d.against, d.mean_gain, d.standard_error, d.cleared]
            for d in result.selection.decisions
        ],
    )
    parts = [
        f"Champion: **{result.selection.champion}**, on {result.train_rows} training rows and "
        f"{result.test_rows} held-out rows.",
        "### Arms",
        arms,
        "### Decisions",
        decisions if result.selection.decisions else "The domain has only its null arm.",
    ]
    if result.lift is not None:
        parts += [
            "### Lift of the champion",
            "Held-out rows in bands of equal count, highest scores first.",
            frame_table(result.lift),
        ]
    return Section(f"Domain: {name}", "\n\n".join(parts))


def training_report(
    run: TrainingRun, provenance: Provenance, recoveries: Sequence[Recovery] = ()
) -> tuple[Report, MetricsSidecar]:
    """The report of a training run, with any recoveries after the domains, and its
    sidecar."""
    report = Report("Model training report", provenance)
    report.add(_selection(run))
    for name in run.domains:
        report.add(_domain(name, run))
    for recovery in recoveries:
        report.add(recovery.section())
    rows = {r.name: r.records() for r in recoveries}
    return report, MetricsSidecar.of(run, provenance, rows)
