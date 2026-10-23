"""The scorecard: one row per metric, each carrying the provenance of the run behind it.

The CSV is deterministic for a given profile and inputs, so it can be compared with a
committed baseline. Anything that varies run to run, such as timing, goes in the run record
instead. The versioned schema and rankings arrive in M8.8.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Final

import numpy as np
import pandas as pd

from hcsim.engine.loop import SimulationResult
from hcsim.studies.harness import StudyRun

SCORECARD_VERSION: Final = 1
SCORECARD_FILE: Final = "scorecard.csv"
RUN_RECORD_FILE: Final = "run.json"
METRIC_COLUMNS: Final = ["evaluator", "metric", "value", "threshold", "passed", "detail"]
PROVENANCE_COLUMNS: Final = [
    "source",
    "world_generator",
    "world_preset",
    "world_seed",
    "profile",
    "mode",
    "scope",
    "seed",
    "run_date",
    "schema_fingerprint",
    "guidance_fingerprint",
    "hcsim_version",
]
COLUMNS: Final = ["scorecard_version", *METRIC_COLUMNS, *PROVENANCE_COLUMNS]
FLOAT_FORMAT: Final = "%.10g"


def scorecard(run: StudyRun) -> pd.DataFrame:
    rows = [
        {
            "scorecard_version": SCORECARD_VERSION,
            "evaluator": m.evaluator,
            "metric": m.name,
            "value": float(m.value),
            "threshold": m.threshold,
            "passed": m.passed,
            "detail": m.detail,
        }
        | {key: run.provenance[key] for key in PROVENANCE_COLUMNS}
        for m in run.metrics
    ]
    return pd.DataFrame(rows, columns=COLUMNS)


def results_digest(result: SimulationResult) -> str:
    """`rd1-` and 16 hex characters of a SHA-256 over every trajectory's raw results: stocks,
    flows, quota records, prices, and SBC. Equal digests mean bit-identical results."""
    digest = hashlib.sha256()
    for t in result.trajectories:
        arrays = [t.headcount, t.open_openings, t.prices, *(t.flows[k] for k in sorted(t.flows))]
        for q in t.quotas:
            digest.update(f"{q.step}|{q.domain}|{q.category}".encode())
            arrays += [q.record.quota, q.record.eligible, q.record.realized]
        if t.sbc is not None:
            arrays.append(t.sbc.select_dtypes("number").to_numpy(dtype=np.float64))
            for label in t.sbc.select_dtypes(exclude="number").columns:
                digest.update("|".join(t.sbc[label].astype(str)).encode())
        for array in arrays:
            digest.update(np.ascontiguousarray(array).tobytes())
    return "rd1-" + digest.hexdigest()[:16]


def run_record(run: StudyRun, seconds: float) -> dict[str, Any]:
    """Everything about the run that isn't a metric."""
    seed = run.seed
    return {
        "scorecard_version": SCORECARD_VERSION,
        "profile": run.profile.model_dump(mode="json"),
        "provenance": run.provenance,
        "passed": run.passed,
        "results_digest": results_digest(run.result),
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
        "wall_clock_seconds": round(seconds, 3),
    }


def write(run: StudyRun, directory: Path, seconds: float) -> Path:
    """Write the scorecard and the run record, returning the scorecard's path."""
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / SCORECARD_FILE
    scorecard(run).to_csv(path, index=False, float_format=FLOAT_FORMAT)
    record = json.dumps(run_record(run, seconds), indent=2, sort_keys=True)
    (directory / RUN_RECORD_FILE).write_text(record + "\n")
    return path
