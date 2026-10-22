"""`hcsim synth`: generate a synthetic world and write it to DuckDB, then check or profile it."""

from __future__ import annotations

import argparse
import time
from collections.abc import Sequence
from pathlib import Path

import duckdb

from hcsim.synth.config import PRESETS, preset
from hcsim.synth.profile import profile, render
from hcsim.synth.validate import validate
from hcsim.synth.world import generate_world, write_world


def main(argv: Sequence[str]) -> int:
    parser = argparse.ArgumentParser(prog="hcsim synth", description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    generate = commands.add_parser("generate", help="generate a world and write it to DuckDB")
    generate.add_argument("--preset", default="tiny", choices=sorted(PRESETS))
    generate.add_argument("--seed", type=int, default=0)
    generate.add_argument("--out", type=Path, required=True, help="the DuckDB file to write")
    check = commands.add_parser("validate", help="check a written world's coherence")
    check.add_argument("--db", type=Path, required=True, help="the DuckDB file to check")
    report = commands.add_parser("profile", help="write a Markdown profile of a written world")
    report.add_argument("--db", type=Path, required=True, help="the DuckDB file to profile")
    report.add_argument("--out", type=Path, help="the Markdown file to write; else stdout")
    args = parser.parse_args(list(argv))
    if args.command == "validate":
        return _validate(args.db)
    if args.command == "profile":
        return _profile(args.db, args.out)

    started = time.perf_counter()
    world = generate_world(preset(args.preset, args.seed))
    write_world(world, args.out)
    elapsed = time.perf_counter() - started
    print(f"wrote the {args.preset!r} world (seed {args.seed}) to {args.out} in {elapsed:.1f}s")
    for name, frame in sorted(world.tables.items()):
        print(f"  {name:26} {len(frame):>8} rows")
    for name, frame in sorted(world.truth_tables.items()):
        print(f"  {'truth.' + name:26} {len(frame):>8} rows")
    return 0


def _profile(path: Path, out: Path | None) -> int:
    if not path.is_file():
        print(f"no world at {path}")
        return 2
    with duckdb.connect(str(path), read_only=True) as con:
        text = render(profile(con), title=f"World profile: {path.name}")
    if out is None:
        print(text)
    else:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text + "\n")
        print(f"wrote the profile of {path} to {out}")
    return 0


def _validate(path: Path) -> int:
    if not path.is_file():
        print(f"no world at {path}")
        return 2
    with duckdb.connect(str(path), read_only=True) as con:
        results = validate(con)
    for result in results:
        status = "ok" if result.passed else "FAIL"
        count = "" if result.passed else f" ({result.violations} rows)"
        print(f"  {status:4} {result.name:20} {result.description}{count}")
        for example in result.examples:
            print(f"         {example}")
    failed = sum(not r.passed for r in results)
    print(f"{len(results) - failed} of {len(results)} checks passed")
    return 1 if failed else 0
