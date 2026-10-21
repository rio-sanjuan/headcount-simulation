"""`hcsim synth`: generate a synthetic world and write it to DuckDB."""

from __future__ import annotations

import argparse
import time
from collections.abc import Sequence
from pathlib import Path

from hcsim.synth.config import PRESETS, preset
from hcsim.synth.world import generate_world, write_world


def main(argv: Sequence[str]) -> int:
    parser = argparse.ArgumentParser(prog="hcsim synth", description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    generate = commands.add_parser("generate", help="generate a world and write it to DuckDB")
    generate.add_argument("--preset", default="tiny", choices=sorted(PRESETS))
    generate.add_argument("--seed", type=int, default=0)
    generate.add_argument("--out", type=Path, required=True, help="the DuckDB file to write")
    args = parser.parse_args(list(argv))

    started = time.perf_counter()
    world = generate_world(preset(args.preset, args.seed))
    write_world(world, args.out)
    elapsed = time.perf_counter() - started
    print(f"wrote the {args.preset!r} world (seed {args.seed}) to {args.out} in {elapsed:.1f}s")
    for name, frame in sorted(world.tables.items()):
        print(f"  {name:16} {len(frame):>8} rows")
    return 0
