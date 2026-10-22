"""Performance smoke test: time each phase of the tiny run and record it.

Shared runners are noisy, so this records rather than gates. Its one timing assertion is a
generous ceiling that catches only a pathological regression. Benchmarks with stored baselines
and regression flags arrive in M9.1.

The record goes to `$HCSIM_PERF_OUT/tiny.json` when that's set, and to the job summary when
running in GitHub Actions.
"""

from __future__ import annotations

import json
import os
import platform
import time
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

import duckdb
import numpy as np
import pytest

from hcsim.guidance.config import GuidanceConfig
from hcsim.guidance.pipeline import run as fit_guidance
from hcsim.studies.harness import run_study
from hcsim.studies.profile import StudyProfile
from hcsim.synth.config import preset
from hcsim.synth.world import generate_world, write_world

pytestmark = pytest.mark.e2e

CEILING_SECONDS = 120.0


def test_the_tiny_run_records_its_wall_clock(tmp_path: Path) -> None:
    seconds: dict[str, float] = {}

    @contextmanager
    def phase(name: str) -> Iterator[None]:
        start = time.perf_counter()
        yield
        seconds[name] = round(time.perf_counter() - start, 3)

    world = tmp_path / "world.duckdb"
    with phase("world"):
        write_world(generate_world(preset("tiny", seed=1)), world)
    with phase("guidance"), duckdb.connect(str(world), read_only=True) as con:
        fit_guidance(con, GuidanceConfig()).artifact.save(tmp_path / "guidance")
    profile = StudyProfile.from_mapping(
        {"name": "perf", "seed": 2026, "trajectories": 20, "months": 12}
    )
    with phase("study"):
        run = run_study(profile, world, tmp_path / "guidance")
    assert run.passed

    total = round(sum(seconds.values()), 3)
    record = {
        "preset": "tiny",
        "employees": run.seed.employees_seeded,
        "trajectories": profile.trajectories,
        "months": profile.months,
        "seconds": seconds,
        "total_seconds": total,
        "python": platform.python_version(),
        "numpy": np.__version__,
        "platform": platform.platform(),
    }
    if out := os.environ.get("HCSIM_PERF_OUT"):
        Path(out).mkdir(parents=True, exist_ok=True)
        (Path(out) / "tiny.json").write_text(json.dumps(record, indent=2) + "\n")
    if summary := os.environ.get("GITHUB_STEP_SUMMARY"):
        rows = "\n".join(f"| {name} | {value:.2f} |" for name, value in seconds.items())
        with Path(summary).open("a") as handle:
            handle.write(
                f"### Tiny run wall clock ({record['platform']})\n\n"
                f"{profile.trajectories} trajectories x {profile.months} months, "
                f"{record['employees']} employees\n\n"
                f"| phase | seconds |\n|---|---|\n{rows}\n"
                f"| **total** | **{total:.2f}** |\n\n"
            )
    print(json.dumps(record))
    assert total < CEILING_SECONDS
