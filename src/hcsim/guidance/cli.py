"""`hcsim guidance`: fit the guidance forecast on a synthetic world and write the artifact, or
write a window's realized totals as one."""

from __future__ import annotations

import argparse
import datetime as dt
import json
from collections.abc import Sequence
from pathlib import Path

import duckdb
import pandas as pd

from hcsim.guidance.config import GuidanceConfig, GuidanceConfigError
from hcsim.guidance.divergence import against_measures
from hcsim.guidance.framework import PipelineError
from hcsim.guidance.ingest import GROUPINGS, IngestionError
from hcsim.guidance.ladder import BLOCK, DRAWS, MARGIN, LadderError, markdown, run_ladder, score
from hcsim.guidance.pipeline import run
from hcsim.guidance.realized import RealizedError, realized
from hcsim.guidance.report import write_report
from hcsim.model.core.alignment import AlignmentError, AlignmentTarget


def main(argv: Sequence[str]) -> int:
    parser = argparse.ArgumentParser(prog="hcsim guidance", description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    fit = commands.add_parser("fit", help="fit the forecast and write the artifact")
    fit.add_argument("--world", type=Path, required=True, help="the world's DuckDB file")
    fit.add_argument("--out", type=Path, required=True, help="the artifact directory to write")
    fit.add_argument("--config", type=Path, help="a guidance configuration (YAML)")
    fit.add_argument("--target", type=Path, help="the alignment target's directory, to align to")
    fit.add_argument("--horizon", type=int, help="overrides the configuration's horizon")
    fit.add_argument(
        "--grouping", help=f"overrides the configuration's grouping: one of {', '.join(GROUPINGS)}"
    )
    actual = commands.add_parser(
        "realized", help="write a window's realized totals as an artifact, for guided_actuals"
    )
    actual.add_argument("--world", type=Path, required=True, help="the world's DuckDB file")
    actual.add_argument("--out", type=Path, required=True, help="the artifact directory to write")
    actual.add_argument("--start", required=True, help="the window's first month, YYYY-MM")
    actual.add_argument("--months", type=int, required=True, help="the window's length")
    actual.add_argument("--grouping", default="cost_center", help=f"one of {', '.join(GROUPINGS)}")
    actual.add_argument(
        "--allow-snapshot-gaps",
        action="store_true",
        help="count closures across a missing month-end in the month after",
    )
    actual.add_argument(
        "--allow-divergence",
        action="store_true",
        help="write the artifact even where it diverges from the guidance measures beyond "
        "what's declared",
    )
    study = commands.add_parser(
        "ladder", help="fit the ladder's arms through a cutoff, and score them on the year after"
    )
    study.add_argument("--world", type=Path, required=True, help="the world's DuckDB file")
    study.add_argument("--out", type=Path, required=True, help="the directory to write")
    study.add_argument(
        "--cutoff", help="the last month the arms see, YYYY-MM; by default a year before the end"
    )
    study.add_argument("--grouping", default="org_level1", help=f"one of {', '.join(GROUPINGS)}")
    study.add_argument(
        "--margin", type=float, default=MARGIN, help="the share a richer arm's WAPE must fall by"
    )
    study.add_argument("--block", type=int, default=BLOCK, help="the bootstrap's block, in months")
    study.add_argument("--draws", type=int, default=DRAWS, help="the bootstrap's draws")
    study.add_argument("--seed", type=int, default=0, help="the bootstrap's seed")
    args = parser.parse_args(list(argv))

    if not args.world.is_file():
        parser.error(f"no world at {args.world}")
    if args.command == "realized":
        return _realized(parser, args)
    if args.command == "ladder":
        return _ladder(parser, args)
    try:
        base = GuidanceConfig.load(args.config) if args.config else GuidanceConfig()
        overrides = {"horizon": args.horizon, "grouping": args.grouping}
        given = {k: v for k, v in overrides.items() if v is not None}
        config = GuidanceConfig.from_mapping(base.model_dump() | given)
    except GuidanceConfigError as error:
        parser.error(str(error))
    target = None
    if args.target is not None:
        try:
            target = AlignmentTarget.load(args.target)
        except (OSError, AlignmentError) as error:
            parser.error(f"can't read the alignment target at {args.target}: {error}")
    with duckdb.connect(str(args.world), read_only=True) as con:
        try:
            result = run(con, config, target)
        except (IngestionError, PipelineError) as error:
            parser.error(str(error))
    result.artifact.save(args.out)
    (args.out / "report.json").write_text(json.dumps(result.report, indent=2, default=str))
    write_report(result, args.out)
    report = result.report
    print(
        f"wrote {report['fingerprint']} to {args.out}: {report['series_forecast']} series, "
        f"{len(report['series_skipped'])} skipped, {report['clamped_negative_cells']} clamped"
    )
    return 0


def _realized(parser: argparse.ArgumentParser, args: argparse.Namespace) -> int:
    try:
        start = pd.Period(dt.datetime.strptime(args.start, "%Y-%m"), "M")
    except ValueError:
        parser.error(f"--start must be a month, YYYY-MM, not {args.start!r}")
    with duckdb.connect(str(args.world), read_only=True) as con:
        try:
            result = realized(
                con, start, args.months, args.grouping, allow_gaps=args.allow_snapshot_gaps
            )
        except RealizedError as error:
            parser.error(str(error))
        try:
            comparison = against_measures(con, result)
        except IngestionError as error:
            parser.error(f"can't compare with the guidance measures: {error}")
    if not comparison.within and not args.allow_divergence:
        parser.error(
            "the realized totals diverge from the guidance measures beyond what's declared: "
            + "; ".join(comparison.problems())
        )
    report = {**result.report, "divergence": comparison.report()}
    result.artifact.save(args.out)
    (args.out / "report.json").write_text(json.dumps(report, indent=2, default=str))
    print(
        f"wrote {result.artifact.fingerprint()} to {args.out}: {report['window']}, "
        f"{report['groups']} groups, "
        + ("within" if comparison.within else "outside")
        + " the declared divergences"
    )
    return 0


def _ladder(parser: argparse.ArgumentParser, args: argparse.Namespace) -> int:
    cutoff = None
    if args.cutoff is not None:
        try:
            cutoff = pd.Period(dt.datetime.strptime(args.cutoff, "%Y-%m"), "M")
        except ValueError:
            parser.error(f"--cutoff must be a month, YYYY-MM, not {args.cutoff!r}")
    with duckdb.connect(str(args.world), read_only=True) as con:
        try:
            ladder = run_ladder(con, cutoff, args.grouping)
            card = score(ladder, args.margin, args.block, args.draws, seed=args.seed)
        except (LadderError, IngestionError, PipelineError) as error:
            parser.error(str(error))
    args.out.mkdir(parents=True, exist_ok=True)
    ladder.frame().astype({"month": str}).to_csv(args.out / "forecasts.csv", index=False)
    card.frame().to_csv(args.out / "scores.csv", index=False)
    (args.out / "ladder.md").write_text(markdown(ladder, card))
    selected = {s.value: arm for s, arm in card.selected.items()}
    report = {"cutoff": str(ladder.cutoff), "grouping": ladder.grouping, "selected": selected}
    (args.out / "report.json").write_text(json.dumps(report, indent=2))
    print(
        f"scored {len(ladder.runs)} arms through {ladder.cutoff}: "
        + ", ".join(f"{s} {a}" for s, a in selected.items())
    )
    return 0
