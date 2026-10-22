"""`hcsim guidance`: fit the guidance forecast on a synthetic world and write the artifact."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

import duckdb

from hcsim.guidance.framework import PipelineError
from hcsim.guidance.ingest import GROUPINGS, IngestionError
from hcsim.guidance.pipeline import PipelineConfig, run


def main(argv: Sequence[str]) -> int:
    parser = argparse.ArgumentParser(prog="hcsim guidance", description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    fit = commands.add_parser("fit", help="fit the forecast and write the artifact")
    fit.add_argument("--world", type=Path, required=True, help="the world's DuckDB file")
    fit.add_argument("--out", type=Path, required=True, help="the artifact directory to write")
    fit.add_argument("--horizon", type=int, default=12)
    fit.add_argument("--grouping", default="cost_center", help=f"one of {', '.join(GROUPINGS)}")
    args = parser.parse_args(list(argv))

    if not args.world.is_file():
        parser.error(f"no world at {args.world}")
    with duckdb.connect(str(args.world), read_only=True) as con:
        try:
            result = run(con, PipelineConfig(horizon=args.horizon, grouping=args.grouping))
        except (IngestionError, PipelineError) as error:
            parser.error(str(error))
    result.artifact.save(args.out)
    (args.out / "report.json").write_text(json.dumps(result.report, indent=2, default=str))
    report = result.report
    print(
        f"wrote {report['fingerprint']} to {args.out}: {report['series_forecast']} series, "
        f"{len(report['series_skipped'])} skipped, {report['clamped_negative_cells']} clamped"
    )
    return 0
