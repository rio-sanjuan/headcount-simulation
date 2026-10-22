"""`hcsim synth`: generate a synthetic world and write it to DuckDB, then check, profile,
export, or import it."""

from __future__ import annotations

import argparse
import time
from collections.abc import Sequence
from pathlib import Path

import duckdb

from hcsim.synth.config import PRESETS, preset
from hcsim.synth.export import COMPRESSIONS, ExportError, export_parquet, import_parquet
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
    export = commands.add_parser("export", help="write a world's tables as Parquet files")
    export.add_argument("--db", type=Path, required=True, help="the DuckDB file to export")
    export.add_argument("--out", type=Path, required=True, help="a new or empty directory")
    export.add_argument("--compression", default="zstd", choices=COMPRESSIONS)
    load = commands.add_parser("import", help="build a DuckDB world from a Parquet export")
    load.add_argument("--parquet", type=Path, required=True, help="the exported directory")
    load.add_argument("--out", type=Path, required=True, help="the DuckDB file to write")
    args = parser.parse_args(list(argv))
    if args.command == "export":
        return _export(args.db, args.out, args.compression)
    if args.command == "import":
        return _import(args.parquet, args.out)
    if args.command == "validate":
        return _validate(args.db)
    if args.command == "profile":
        return _profile(args.db, args.out)

    started = time.perf_counter()
    world = generate_world(preset(args.preset, args.seed))
    write_world(world, args.out)
    elapsed = time.perf_counter() - started
    print(f"wrote the {args.preset!r} world (seed {args.seed}) to {args.out} in {elapsed:.1f}s")
    with duckdb.connect(str(args.out), read_only=True) as con:
        listed = con.execute(
            "SELECT table_schema, table_name FROM information_schema.tables "
            "WHERE table_type = 'BASE TABLE' ORDER BY table_schema, table_name"
        ).fetchall()
        for schema, name in listed:
            (rows,) = con.execute(f'SELECT count(*) FROM {schema}."{name}"').fetchone() or (0,)
            label = name if schema == "main" else f"{schema}.{name}"
            print(f"  {label:26} {rows:>8} rows")
    return 0


def _export(path: Path, out: Path, compression: str) -> int:
    if not path.is_file():
        print(f"no world at {path}")
        return 2
    try:
        with duckdb.connect(str(path), read_only=True) as con:
            files = export_parquet(con, out, compression)
    except ExportError as error:
        print(f"hcsim synth: {error}")
        return 2
    print(f"wrote {len(files)} files to {out}")
    return 0


def _import(directory: Path, out: Path) -> int:
    try:
        import_parquet(directory, out)
    except ExportError as error:
        print(f"hcsim synth: {error}")
        return 2
    print(f"built {out} from {directory}")
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
