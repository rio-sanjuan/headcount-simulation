"""`hcsim bench`: time the synthetic world's presets."""

from __future__ import annotations

import argparse
import dataclasses
import json
from collections.abc import Sequence
from pathlib import Path

from hcsim.bench.world import BenchError, run
from hcsim.synth.config import PRESETS


def main(argv: Sequence[str]) -> int:
    parser = argparse.ArgumentParser(prog="hcsim bench", description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    world = commands.add_parser("world", help="generate, write, and check each preset, timed")
    world.add_argument(
        "--preset", action="append", choices=sorted(PRESETS), help="repeat for several"
    )
    world.add_argument("--seed", type=int, default=0)
    world.add_argument("--max-rss-gib", type=float, help="stop a run whose memory passes this")
    world.add_argument("--out", type=Path, help="write the runs as JSON")
    args = parser.parse_args(list(argv))

    try:
        runs = run(args.preset or ["tiny"], args.seed, args.max_rss_gib)
    except BenchError as error:
        print(f"hcsim bench: {error}")
        return 1
    print(
        f"{'preset':8} {'employees':>9} {'generate':>9} {'write':>7} {'validate':>8} "
        f"{'peak MiB':>9} {'rows':>11} checks"
    )
    for r in runs:
        print(
            f"{r.preset:8} {r.employees:>9,} {r.generate_seconds:>8.1f}s {r.write_seconds:>6.1f}s "
            f"{r.validate_seconds:>7.1f}s {r.peak_rss_mib:>9,.0f} {r.rows:>11,} "
            f"{'ok' if r.checks_passed else 'FAIL'}"
        )
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps([dataclasses.asdict(r) for r in runs], indent=2) + "\n")
    return 0 if all(r.checks_passed for r in runs) else 1
