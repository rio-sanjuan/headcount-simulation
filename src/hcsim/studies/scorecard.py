"""The scorecard: one row per pass, arm, evaluator, and metric, each carrying its provenance.

**The schema is versioned** (`SCORECARD_VERSION`, `SCHEMA`): every column, in order, and its
type. A row says which pass the metric came from (`A`, the screen, or `B`), the arm and its
role (`study` for a profile with no arms, else `null`, `compared`, or `reference`), the metric
with its threshold and whether it passed, how far the arm lies from the null arm on the same
metric in pass B (`vs_null`), and its rank, where the study ranks arms by that metric. Every
row carries the provenance of the run behind it (`PROVENANCE_COLUMNS`), so any one row traces
to the exact world, profile, specification, inputs, and code.

**Rankings** order the null and compared arms of pass B by each of the study's ranking metrics
(`StudyProfile.ranked_by`), first first, ties broken by the arm's order in the study; a
reference arm stays out of every ranking. An arm that doesn't report a ranking's metric isn't
ranked by it.

The CSV is deterministic for a given profile and inputs, so it can be compared with a
committed baseline. Anything that varies run to run, such as timing, goes in the **run record**
instead, whose keys are fixed (`RUN_RECORD_KEYS`) and checked as it's written.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
from collections.abc import Hashable, Mapping
from pathlib import Path
from types import MappingProxyType
from typing import Any, Final

import numpy as np
import pandas as pd

from hcsim.engine.loop import SimulationResult
from hcsim.studies.harness import ITSELF, ArmRun, StudyRun
from hcsim.studies.profile import Arm

SCORECARD_VERSION: Final = 2
SCORECARD_FILE: Final = "scorecard.csv"
RUN_RECORD_FILE: Final = "run.json"
#: The columns that say where a metric sits: its pass, its arm, and the arm's role.
PLACE_COLUMNS: Final = ["pass", "arm", "role"]
METRIC_COLUMNS: Final = ["evaluator", "metric", "value", "threshold", "passed", "detail"]
#: The columns that compare an arm with the others.
COMPARISON_COLUMNS: Final = ["vs_null", "rank"]
PROVENANCE_COLUMNS: Final = [
    "source",
    "world_generator",
    "world_preset",
    "world_seed",
    "profile",
    "spec",
    "mode",
    "scope",
    "backend",
    "seed",
    "run_date",
    "schema_fingerprint",
    "guidance_fingerprint",
    "hcsim_version",
]
COLUMNS: Final = [
    "scorecard_version",
    *PLACE_COLUMNS,
    *METRIC_COLUMNS,
    *COMPARISON_COLUMNS,
    *PROVENANCE_COLUMNS,
]
#: Each column's type, as pandas reads it back.
SCHEMA: Final[Mapping[str, str]] = MappingProxyType(
    {
        "scorecard_version": "int64",
        **dict.fromkeys(PLACE_COLUMNS, "string"),
        "evaluator": "string",
        "metric": "string",
        "value": "float64",
        "threshold": "float64",
        "passed": "bool",
        "detail": "string",
        "vs_null": "float64",
        "rank": "Int64",
        **dict.fromkeys(PROVENANCE_COLUMNS, "string"),
    }
)
#: The roles an arm plays.
ROLES: Final = ("study", "null", "compared", "reference")
#: Every key of the run record, in the order it's documented.
RUN_RECORD_KEYS: Final = (
    "scorecard_version",
    "profile",
    "spec",
    "provenance",
    "verdict",
    "passed",
    "closure",
    "arms",
    "skips",
    "deferred",
    "gates",
    "rankings",
    "results_digest",
    "seed",
    "fitted_rates",
    "fit_window",
    "timing",
    "wall_clock_seconds",
)
FLOAT_FORMAT: Final = "%.10g"


class ScorecardError(ValueError):
    """A scorecard or run record that doesn't fit its schema."""


def role(arm: Arm) -> str:
    if arm is ITSELF:
        return "study"
    return "null" if arm.null else "reference" if arm.reference else "compared"


def arm_runs(run: StudyRun) -> tuple[ArmRun, ...]:
    """Every arm's run the scorecard rows: pass A's, then pass B's, or the study itself."""
    itself = ArmRun(ITSELF, run.config, run.result, run.metrics)
    return (*run.screened, *(run.arms or (itself,)))


def rankings(run: StudyRun) -> dict[str, list[dict[str, Any]]]:
    """For each of the study's ranking metrics, the null and compared arms of pass B that report
    it, first first, with their values and ranks."""
    ranked = [a for a in run.arms if a.pass_name == "B" and not a.arm.reference]
    out: dict[str, list[dict[str, Any]]] = {}
    for ranking in run.profile.ranked_by():
        evaluator, name = ranking.metric.split(".", 1)
        values = [
            (a.arm.name, m.value)
            for a in ranked
            for m in a.metrics
            if (m.evaluator, m.name) == (evaluator, name)
        ]
        order = sorted(range(len(values)), key=lambda k: (ranking.key(values[k][1]), k))
        out[ranking.metric] = [
            {"arm": values[k][0], "value": values[k][1], "rank": r}
            for r, k in enumerate(order, start=1)
        ]
    return out


def scorecard(run: StudyRun) -> pd.DataFrame:
    """The study's scorecard, one row per pass, arm, evaluator, and metric."""
    null = next((a for a in run.arms if a.arm.null and a.pass_name == "B"), None)
    baseline = {(m.evaluator, m.name): m.value for m in null.metrics} if null else {}
    ranks = {(metric, r["arm"]): r["rank"] for metric, rows in rankings(run).items() for r in rows}
    provenance = {key: run.provenance.get(key, "") for key in PROVENANCE_COLUMNS}
    rows = []
    for a in arm_runs(run):
        compared = a.pass_name == "B" and null is not None and not a.arm.null
        for m in a.metrics:
            key = (m.evaluator, m.name)
            vs_null = m.value - baseline[key] if compared and key in baseline else np.nan
            rows.append(
                {
                    "scorecard_version": SCORECARD_VERSION,
                    "pass": a.pass_name,
                    "arm": a.arm.name,
                    "role": role(a.arm),
                    "evaluator": m.evaluator,
                    "metric": m.name,
                    "value": float(m.value),
                    "threshold": m.threshold,
                    "passed": m.passed,
                    "detail": m.detail,
                    "vs_null": vs_null,
                    "rank": ranks.get((f"{m.evaluator}.{m.name}", a.arm.name))
                    if a.pass_name == "B"
                    else None,
                }
                | provenance
            )
    frame = pd.DataFrame(rows, columns=COLUMNS)
    return frame.astype({"rank": "Int64", "threshold": "float64", "vs_null": "float64"})


def check(frame: pd.DataFrame) -> list[str]:
    """Every way a scorecard doesn't fit the schema: its columns, their order, the version,
    each role, and every row's provenance."""
    if list(frame.columns) != COLUMNS:
        return [f"a version {SCORECARD_VERSION} scorecard has the columns {COLUMNS}"]
    problems = []
    versions = sorted(set(frame["scorecard_version"].astype(int)))
    if versions and versions != [SCORECARD_VERSION]:
        problems.append(f"scorecard versions {versions}; this is {SCORECARD_VERSION}")
    strangers = sorted(set(frame["role"]) - set(ROLES))
    if strangers:
        problems.append(f"no roles {strangers}")
    for column in ("source", "profile", "spec", "mode", "scope", "seed", "schema_fingerprint"):
        if (frame[column].astype(str).str.len() == 0).any() or frame[column].isna().any():
            problems.append(f"rows without their provenance's {column}")
    return problems


def load(path: Path) -> pd.DataFrame:
    """A scorecard as written, its columns typed by the schema. Only an empty cell is missing,
    so a role or an arm named like a missing value, as `null` is, reads as itself."""
    strings: dict[Hashable, Any] = {c: t for c, t in SCHEMA.items() if t == "string"}
    frame = pd.read_csv(path, dtype=strings, keep_default_na=False, na_values=[""])
    problems = check(frame)
    if problems:
        raise ScorecardError("; ".join(problems))
    return frame.astype(dict(SCHEMA))


def results_digest(result: SimulationResult) -> str:
    """`rd1-` and 16 hex characters of a SHA-256 over every trajectory's raw results: stocks,
    flows, quota records, prices, and SBC, published and scratch. Equal digests mean
    bit-identical results."""
    digest = hashlib.sha256()
    for t in result.trajectories:
        arrays = [t.headcount, t.open_openings, t.prices, *(t.flows[k] for k in sorted(t.flows))]
        for q in t.quotas:
            digest.update(f"{q.step}|{q.domain}|{q.category}".encode())
            arrays += [q.record.quota, q.record.eligible, q.record.realized]
        for sbc in (t.sbc, t.sbc_scratch):
            if sbc is None or sbc.empty:
                continue
            arrays.append(sbc.select_dtypes("number").to_numpy(dtype=np.float64))
            for label in sbc.select_dtypes(exclude="number").columns:
                digest.update("|".join(sbc[label].astype(str)).encode())
        for array in arrays:
            digest.update(np.ascontiguousarray(array).tobytes())
    return "rd1-" + digest.hexdigest()[:16]


def run_record(run: StudyRun, seconds: float) -> dict[str, Any]:
    """Everything about the run that isn't a metric, under `RUN_RECORD_KEYS`."""
    seed = run.seed
    meta = run.metadata()
    record = {
        "scorecard_version": SCORECARD_VERSION,
        "profile": run.profile.model_dump(mode="json"),
        "spec": meta["spec"],
        "provenance": run.provenance,
        "verdict": meta["verdict"],
        "passed": run.passed,
        "closure": meta["closure"],
        "arms": [
            {"arm": a.arm.name, "role": role(a.arm), "pass": a.pass_name, "passed": a.passed}
            for a in arm_runs(run)
        ],
        "skips": meta["skips"],
        "deferred": meta["deferred"],
        "gates": [dataclasses.asdict(g) for g in run.gates],
        "rankings": rankings(run),
        "results_digest": results_digest(run.result) if run.result is not None else "",
        "seed": {
            "snapshot_date": seed.snapshot_date.isoformat(),
            "employees": seed.employees_seeded,
            "employees_pruned": seed.employees_pruned,
            "openings": seed.openings_seeded,
            "openings_pruned": seed.openings_pruned,
            "openings_clipped": seed.openings_clipped,
        },
        "fitted_rates": run.fitted.rates,
        "fit_window": [d.isoformat() for d in run.fitted.window],
        "timing": meta["timing"],
        "wall_clock_seconds": round(seconds, 3),
    }
    problems = check_record(record)
    if problems:
        raise ScorecardError("; ".join(problems))
    return record


def check_record(record: Mapping[str, Any]) -> list[str]:
    """Every way a run record's keys differ from `RUN_RECORD_KEYS`."""
    problems = []
    missing = [k for k in RUN_RECORD_KEYS if k not in record]
    extra = sorted(set(record) - set(RUN_RECORD_KEYS))
    if missing:
        problems.append(f"the run record lacks {missing}")
    if extra:
        problems.append(f"the run record has no keys {extra}")
    if record.get("scorecard_version") != SCORECARD_VERSION:
        problems.append(f"a run record is version {SCORECARD_VERSION}")
    return problems


def write(run: StudyRun, directory: Path, seconds: float) -> Path:
    """Write the scorecard and the run record, returning the scorecard's path."""
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / SCORECARD_FILE
    card = scorecard(run)
    problems = check(card)
    if problems:
        raise ScorecardError("; ".join(problems))
    card.to_csv(path, index=False, float_format=FLOAT_FORMAT)
    record = json.dumps(run_record(run, seconds), indent=2, sort_keys=True, default=str)
    (directory / RUN_RECORD_FILE).write_text(record + "\n")
    return path
