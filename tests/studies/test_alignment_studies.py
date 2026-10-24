"""A study under each alignment method, on a synthetic world, scored by `plan_alignment`."""

from __future__ import annotations

import datetime as dt
from pathlib import Path
from typing import Any

import duckdb
import pytest

from hcsim.engine.seeding import schema_from_world, seed_state
from hcsim.guidance.config import GuidanceConfig
from hcsim.guidance.pipeline import run as fit_guidance
from hcsim.model.core.alignment_builders import synthetic
from hcsim.model.core.schema import COST_CENTER
from hcsim.studies.harness import run_study
from hcsim.studies.profile import StudyProfile

RUN = dt.date(2026, 1, 1)  # the month after the tiny world's last snapshot


@pytest.fixture(scope="module")
def aligned(
    inputs: tuple[Path, Path], tmp_path_factory: pytest.TempPathFactory
) -> tuple[Path, Path]:
    """A plan 5% above the seed by June, saved, and guidance aligned to it."""
    world, _ = inputs
    root = tmp_path_factory.mktemp("aligned")
    with duckdb.connect(str(world), read_only=True) as con:
        schema = schema_from_world(con)
        state, _ = seed_state(con, schema, RUN)
        labels = schema.vocabulary(COST_CENTER).labels
        heads = {labels[c]: float(n) for c, n in enumerate(state.headcount) if n > 0}
        plan = synthetic(
            "relative",
            COST_CENTER,
            heads,
            start="2026-01",
            months=6,
            dates=["2026-06"],
            factors=[1.05],
        )
        plan.save(root / "plan")
        config = GuidanceConfig.from_mapping({"horizon": 6, "alignment": {"method": "greedy"}})
        fit_guidance(con, config, plan).artifact.save(root / "guidance")
    return root / "plan", root / "guidance"


def study(method: str, plan: Path, **fields: Any) -> StudyProfile:
    return StudyProfile.from_mapping(
        {
            "name": f"alignment-{method}",
            "seed": 7,
            "trajectories": 2,
            "months": 6,
            "scope": "hcf",
            "mode": "unguided",
            "plan": {"path": str(plan)},
            "alignment": {"method": method},
            "evaluators": ["invariants", "plan_alignment"],
        }
        | fields
    )


def test_a_study_under_each_alignment_method_scores_against_its_plan(
    inputs: tuple[Path, Path], aligned: tuple[Path, Path]
) -> None:
    world, _ = inputs
    plan, guidance = aligned
    runs = {
        method: run_study(study(method, plan), world)
        for method in ("none", "supply", "rate_calibration")
    }
    for method in ("demand", "hybrid"):
        runs[method] = run_study(study(method, plan, mode="guided_forecast"), world, guidance)
    for method, run in runs.items():
        assert run.passed, method
        metrics = {m.name: m.value for m in run.metrics if m.evaluator == "plan_alignment"}
        assert metrics["evaluation_step"] == 5.0, method
        assert metrics["targets_scored"] > 0, method
        assert run.config.alignment.method == method
    valves = {
        method: {m.name for m in run.metrics if m.name.startswith("bias_mean_")}
        for method, run in runs.items()
    }
    assert valves["none"] == {"bias_mean_none"}
    assert valves["supply"] == {"bias_mean_supply"}
    assert valves["rate_calibration"] == {"bias_mean_rate"}
    assert valves["demand"] <= {"bias_mean_flow", "bias_mean_undriven"}
    assert runs["rate_calibration"].result.calibration is not None
