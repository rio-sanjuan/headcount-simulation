"""`hcsim study`: run a study and write its scorecard.

Exit codes: 0 every gated metric passed, 2 a usage, profile, or input error, 3 a gated metric
failed.
"""

from __future__ import annotations

import argparse
import sys
import time
from collections.abc import Sequence
from pathlib import Path

from hcsim.engine.config import ConfigError
from hcsim.engine.seeding import SeedingError
from hcsim.model.core.guidance import GuidanceArtifactError
from hcsim.studies import scorecard
from hcsim.studies.harness import StudyError, run_study
from hcsim.studies.inputs import InputError
from hcsim.studies.profile import ProfileError, StudyProfile

#: Inputs that don't fit the study: reported, never a traceback.
INPUT_ERRORS = (
    ProfileError,
    StudyError,
    GuidanceArtifactError,
    ConfigError,
    SeedingError,
    InputError,
    FileNotFoundError,
)

EXIT_FAILED = 3


def main(argv: Sequence[str]) -> int:
    parser = argparse.ArgumentParser(prog="hcsim study", description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser("run", help="run a study and write its scorecard")
    run.add_argument("--profile", type=Path, required=True, help="the study profile (YAML)")
    run.add_argument("--world", type=Path, required=True, help="the world's DuckDB file")
    run.add_argument("--guidance", type=Path, help="the guidance artifact, for guided modes")
    run.add_argument("--out", type=Path, required=True, help="the directory to write")
    args = parser.parse_args(list(argv))

    started = time.perf_counter()
    try:
        profile = StudyProfile.load(args.profile)
        study = run_study(profile, args.world, args.guidance)
    except INPUT_ERRORS as error:
        print(f"hcsim study: {error}", file=sys.stderr)
        return 2
    path = scorecard.write(study, args.out, time.perf_counter() - started)

    width = max(len(m.name) for m in study.metrics)
    for m in study.metrics:
        status = "info" if m.threshold is None else ("pass" if m.passed else "FAIL")
        print(f"  {status:4}  {m.name:{width}}  {m.value:.6g}")
    verdict = "passed" if study.passed else "failed"
    print(f"{profile.name}: {verdict}; wrote {path}")
    return 0 if study.passed else EXIT_FAILED
