"""`hcsim study`: run a study and write its scorecard.

`hcsim study run --profile P --world W [--guidance G] --out D` runs the study the profile
declares and writes its scorecard and run record to `D`. With:

- `--preflight-only`, it checks everything the study needs, reports what it would run, and
  writes nothing;
- `--offline`, it runs the study on the offline backend, its evaluators scoring its inputs with
  no simulation;
- `--schema-path`, it encodes with the schema saved there, which must cover every label the
  world holds, rather than the world's own;
- `--cache-dir`, it keeps the extracts it reads under that directory, so another study that
  reads the same rows reads them from there.

Exit codes:

- 0: the study passed every gate;
- 2: a usage, profile, or input error, or a problem preflight found;
- 3: the study stopped: a stop gate failed, or a metric failed its threshold;
- 4: the study is provisional: a provisional gate failed, or its inputs include a fixture;
- 5: nothing can run yet: the study's mode reads its window's actuals, and the window isn't
  wholly closed.
"""

from __future__ import annotations

import argparse
import sys
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Final

from hcsim.engine.config import ConfigError
from hcsim.engine.seeding import SeedingError
from hcsim.model.core.guidance import GuidanceArtifactError
from hcsim.studies import catalog, scorecard
from hcsim.studies import checkpoint as checkpoints
from hcsim.studies.gates import Verdict
from hcsim.studies.harness import Preflight, StudyError, StudyRun, preflight, run_study
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

EXIT_PASSED: Final = 0
EXIT_ERROR: Final = 2
EXIT_STOPPED: Final = 3
EXIT_PROVISIONAL: Final = 4
EXIT_WAITING: Final = 5
#: Kept for callers that named the failed-run code before gates had their own.
EXIT_FAILED: Final = EXIT_STOPPED


def exit_code(run: StudyRun) -> int:
    """A finished study's exit code, by its verdict and its metrics' thresholds."""
    if run.verdict is Verdict.STOPPED or not run.passed:
        return EXIT_STOPPED
    if run.verdict is Verdict.PROVISIONAL:
        return EXIT_PROVISIONAL
    return EXIT_PASSED


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="hcsim study", description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser("run", help="run a study and write its scorecard")
    run.add_argument("--profile", type=Path, required=True, help="the study profile (YAML)")
    run.add_argument("--world", type=Path, required=True, help="the world's DuckDB file")
    run.add_argument("--guidance", type=Path, help="the guidance artifact, for guided modes")
    run.add_argument("--out", type=Path, help="the directory to write; not with --preflight-only")
    run.add_argument(
        "--preflight-only", action="store_true", help="check the study, run nothing, write nothing"
    )
    run.add_argument("--offline", action="store_true", help="run on the offline backend")
    run.add_argument("--schema-path", type=Path, help="encode with the schema saved here")
    run.add_argument("--cache-dir", type=Path, help="keep the study's extracts here")

    catalog = commands.add_parser("catalog", help="list, write, or run the study catalog")
    actions = catalog.add_subparsers(dest="action", required=True)
    actions.add_parser("list", help="list the catalog's studies")
    write = actions.add_parser("write", help="write the catalog's studies for a world")
    run_all = actions.add_parser("run", help="run every written study and check its exit code")
    for action in (write, run_all):
        action.add_argument("--world", type=Path, required=True, help="the world's DuckDB file")
    write.add_argument("--out", type=Path, required=True, help="the directory to write")
    write.add_argument("--study", action="append", default=[], help="write only this study")
    run_all.add_argument("--catalog", type=Path, required=True, help="where it was written")
    run_all.add_argument("--out", type=Path, required=True, help="where the runs are written")

    checkpoint = commands.add_parser("checkpoint", help="record or check a run's accuracy")
    steps = checkpoint.add_subparsers(dest="action", required=True)
    save = steps.add_parser("save", help="record a run's accuracy as a checkpoint")
    against = steps.add_parser("check", help="check a run against a checkpoint")
    for step in (save, against):
        step.add_argument("--run", type=Path, required=True, help="a study run's output")
    save.add_argument("--out", type=Path, required=True, help="the checkpoint file to write")
    save.add_argument("--evaluator", action="append", default=[], help="an evaluator to keep")
    against.add_argument("--against", type=Path, required=True, help="the checkpoint")
    against.add_argument(
        "--tolerance", type=float, default=checkpoints.TOLERANCE, help="how far an error may grow"
    )
    return parser


def _report(profile: StudyProfile, checked: Preflight) -> int:
    if not checked.ok:
        for problem in checked.problems:
            print(f"hcsim study: {problem}", file=sys.stderr)
        return EXIT_WAITING if checked.waiting else EXIT_ERROR
    assert checked.inputs is not None
    assert checked.closure is not None
    print(f"{profile.name}: preflight passed")
    print(f"  window     {checked.inputs.run_date} for {profile.months} months, ", end="")
    print(checked.closure.value)
    print(f"  arms       {', '.join(checked.configs)}")
    print(f"  evaluators {', '.join(checked.evaluators) or 'none'}")
    print(f"  inputs     {', '.join(checked.inputs.kinds)}")
    for skip in checked.skips:
        print(f"  skips      {skip.name}: {skip.reason}")
    for waiting in checked.deferred:
        print(f"  defers     {waiting.arm}: {waiting.reason}")
    return EXIT_PASSED


def main(argv: Sequence[str]) -> int:
    parser = _parser()
    args = parser.parse_args(list(argv))
    if args.command == "catalog":
        return _catalog(args)
    if args.command == "checkpoint":
        return _checkpoint(args)
    if not args.preflight_only and args.out is None:
        parser.error("--out is required unless --preflight-only")

    started = time.perf_counter()
    try:
        profile = StudyProfile.load(
            args.profile, **({"backend": "offline"} if args.offline else {})
        )
        options = {"schema_path": args.schema_path, "cache_dir": args.cache_dir}
        if args.preflight_only:
            return _report(profile, preflight(profile, args.world, args.guidance, **options))
        study = run_study(profile, args.world, args.guidance, **options)
    except StudyError as error:
        print(f"hcsim study: {error}", file=sys.stderr)
        return EXIT_WAITING if error.waiting else EXIT_ERROR
    except INPUT_ERRORS as error:
        print(f"hcsim study: {error}", file=sys.stderr)
        return EXIT_ERROR
    path = scorecard.write(study, args.out, time.perf_counter() - started)

    width = max((len(m.name) for m in study.metrics), default=0)
    for m in study.metrics:
        status = "info" if m.threshold is None else ("pass" if m.passed else "FAIL")
        print(f"  {status:4}  {m.name:{width}}  {m.value:.6g}")
    for g in study.gates:
        if g.failed:
            print(f"  gate  {g.gate} failed on {g.arm}: {g.metric} = {g.value:.6g}, {g.bound}")
    for skip in study.skips:
        print(f"  skip  {skip.what} {skip.name}: {skip.reason}")
    code = exit_code(study)
    verdict = (
        "failed" if code == EXIT_STOPPED and study.verdict is Verdict.PASSED else study.verdict
    )
    print(f"{profile.name}: {verdict}; wrote {path}")
    return code


def _catalog(args: argparse.Namespace) -> int:
    if args.action == "list":
        width = max(len(e.name) for e in catalog.ENTRIES)
        for e in catalog.ENTRIES:
            print(f"  {e.name:{width}}  exits {e.expect}  {e.doc}")
        return EXIT_PASSED
    try:
        if args.action == "write":
            written = catalog.write(args.world, args.out, args.study)
            print(f"wrote {len(written)} studies and {catalog.MANIFEST} to {args.out}")
            return EXIT_PASSED
        written = catalog.load(args.catalog)
    except (OSError, ValueError) as error:
        print(f"hcsim study: {error}", file=sys.stderr)
        return EXIT_ERROR
    outcomes = catalog.run(written, args.world, args.out, lambda a: main(a[1:]))
    for o in outcomes:
        status = "ok" if o.matched else "MISMATCH"
        print(f"  {status:8}  {o.name}: exited {o.got}, documented {o.expect}")
    return EXIT_PASSED if all(o.matched for o in outcomes) else EXIT_STOPPED


def _checkpoint(args: argparse.Namespace) -> int:
    try:
        if args.action == "save":
            saved = checkpoints.save(args.run, args.out, args.evaluator or ("sbc",))
            print(f"saved {len(saved['metrics'])} metrics to {args.out}")
            return EXIT_PASSED
        result = checkpoints.check(args.run, args.against, args.tolerance)
    except checkpoints.CheckpointError as error:
        print(f"hcsim study: {error}", file=sys.stderr)
        return EXIT_ERROR
    for regression in result.regressions:
        print(f"  regressed  {regression}")
    same = "bit-identical results" if result.identical else "results differ"
    print(f"{'passed' if result.passed else 'regressed'}: {same}")
    return EXIT_PASSED if result.passed else EXIT_STOPPED
