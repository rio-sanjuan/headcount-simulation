"""`hcsim guidance`: fit the guidance forecast on a synthetic world and write the artifact."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

import duckdb

from hcsim.guidance.config import GuidanceConfig, GuidanceConfigError
from hcsim.guidance.framework import PipelineError
from hcsim.guidance.ingest import GROUPINGS, IngestionError
from hcsim.guidance.pipeline import run
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
    args = parser.parse_args(list(argv))

    if not args.world.is_file():
        parser.error(f"no world at {args.world}")
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
