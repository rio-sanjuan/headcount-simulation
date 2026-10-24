"""Checkpoints: a study's accuracy, recorded, so a later run can be checked against it.

A checkpoint (`save`) keeps, from a run's written output, the pass-B metrics of the evaluators
it names (by default `sbc`, for the SBC accuracy replay), the run's results digest, and the
provenance that fixes the study: its world, profile, specification, mode, scope, seed, run
date, and schema.

`check` compares a later run's output with a checkpoint:

- a run of another study is refused: its provenance must match;
- accuracy may improve but not regress. Every metric is an error, so it regresses when its
  magnitude grows by more than the tolerance, and each regression is reported;
- a metric the checkpoint holds and the run lacks is a regression too;
- equal results digests mean the run's results are bit-identical to the checkpoint's.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

from hcsim.studies import scorecard

CHECKPOINT_VERSION: Final = 1
#: The provenance a checkpoint and a later run must share.
FIXED: Final = (
    "world_generator",
    "world_preset",
    "world_seed",
    "profile",
    "spec",
    "mode",
    "scope",
    "seed",
    "run_date",
    "schema_fingerprint",
)
#: How far an error's magnitude may grow before it's a regression.
TOLERANCE: Final = 0.005


class CheckpointError(ValueError):
    """A checkpoint that can't be saved or checked against."""


def _read(run: Path) -> tuple[Any, dict[str, Any]]:
    try:
        card = scorecard.load(run / scorecard.SCORECARD_FILE)
        record: dict[str, Any] = json.loads((run / scorecard.RUN_RECORD_FILE).read_text())
    except (OSError, ValueError) as error:
        raise CheckpointError(f"can't read the run at {run}: {error}") from None
    return card, record


def _metrics(card: Any, evaluators: Sequence[str]) -> dict[str, float]:
    rows = card[(card["pass"] == "B") & card["evaluator"].isin(list(evaluators))]
    return {
        f"{a}/{e}.{m}": float(v)
        for a, e, m, v in zip(
            rows["arm"], rows["evaluator"], rows["metric"], rows["value"], strict=True
        )
    }


def save(run: Path, out: Path, evaluators: Sequence[str] = ("sbc",)) -> dict[str, Any]:
    """Record the run written under `run` as a checkpoint at `out`."""
    card, record = _read(run)
    metrics = _metrics(card, evaluators)
    if not metrics:
        raise CheckpointError(f"the run at {run} reports no metrics of {list(evaluators)}")
    checkpoint = {
        "checkpoint_version": CHECKPOINT_VERSION,
        "evaluators": list(evaluators),
        "provenance": {k: str(record["provenance"].get(k, "")) for k in FIXED},
        "results_digest": record["results_digest"],
        "metrics": metrics,
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(checkpoint, indent=2, sort_keys=True) + "\n")
    return checkpoint


@dataclass(frozen=True)
class Check:
    """A run checked against a checkpoint: every regression, and whether its results are
    bit-identical."""

    regressions: tuple[str, ...]
    identical: bool

    @property
    def passed(self) -> bool:
        return not self.regressions


def check(run: Path, against: Path, tolerance: float = TOLERANCE) -> Check:
    """Check the run written under `run` against the checkpoint at `against`."""
    try:
        checkpoint = json.loads(against.read_text())
    except (OSError, ValueError) as error:
        raise CheckpointError(f"can't read the checkpoint at {against}: {error}") from None
    if checkpoint.get("checkpoint_version") != CHECKPOINT_VERSION:
        raise CheckpointError(f"a checkpoint is version {CHECKPOINT_VERSION}")
    card, record = _read(run)
    fixed = {k: str(record["provenance"].get(k, "")) for k in FIXED}
    moved = sorted(k for k in FIXED if fixed[k] != checkpoint["provenance"][k])
    if moved:
        raise CheckpointError(f"the run is another study: its {', '.join(moved)} differ")
    now = _metrics(card, checkpoint["evaluators"])
    regressions = []
    for name, then in sorted(checkpoint["metrics"].items()):
        if name not in now:
            regressions.append(f"{name}: missing from the run")
        elif abs(now[name]) > abs(then) + tolerance:
            regressions.append(f"{name}: {now[name]:.6g}, from {then:.6g}")
    identical = record["results_digest"] == checkpoint["results_digest"]
    return Check(tuple(regressions), identical)
