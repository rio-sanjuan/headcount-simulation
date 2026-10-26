"""`hcsim app`: serve the explorer on this machine, and prepare its answers ahead.

`hcsim app serve --world W` serves the explorer's API for the world at `W`, and, with
`--static`, the frontend built into that directory (`make explorer-build` builds it into
apps/explorer/dist). It binds to 127.0.0.1 unless `--host` says otherwise. It needs the `app`
extra: `uv sync --extra app`.

`hcsim app prepare --world W` runs every chapter's presets on the world, keeps their answers,
and exits, so the explorer's first reader never waits. Both commands keep answers in
`--cache DIR`, or in `~/.cache/hcsim/explorer`; a development version keeps them only in a
directory given with `--cache` (see `hcsim.app.cache`).
"""

from __future__ import annotations

import argparse
import tempfile
import time
from collections.abc import Sequence
from pathlib import Path

DEFAULT_STATIC = Path("apps") / "explorer" / "dist"


def main(argv: Sequence[str]) -> int:
    parser = argparse.ArgumentParser(prog="hcsim app", description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    serve = commands.add_parser("serve", help="serve the explorer")
    serve.add_argument("--world", type=Path, required=True, help="the world's DuckDB file")
    serve.add_argument("--static", type=Path, help="the built frontend, if not the default")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8765)
    serve.add_argument("--cache", type=Path, help="where presets' answers are kept")
    prepare = commands.add_parser("prepare", help="run and keep every chapter's presets")
    prepare.add_argument("--world", type=Path, required=True, help="the world's DuckDB file")
    prepare.add_argument("--cache", type=Path, help="where presets' answers are kept")
    args = parser.parse_args(list(argv))
    if args.command == "prepare":
        return _prepare(args.world, args.cache)
    try:
        import uvicorn

        from hcsim.app.demos import DemoError
        from hcsim.app.server import create_app
    except ImportError as error:
        print(
            f"hcsim app: {error.name} isn't installed; install the app extra: uv sync --extra app"
        )
        return 2
    static = args.static
    if static is None and (DEFAULT_STATIC / "index.html").is_file():
        static = DEFAULT_STATIC
    try:
        app = create_app(args.world, static, cache=args.cache)
    except DemoError as error:
        print(f"hcsim app: {error}")
        return 2
    where = f"http://{args.host}:{args.port}"
    print(
        f"serving the explorer at {where}"
        + ("" if static else f" (the API only: {where}/api/docs)")
    )
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")
    return 0


def _prepare(world: Path, cache: Path | None) -> int:
    """Runs every chapter's presets on `world` and keeps their answers in `cache`."""
    from hcsim import __version__
    from hcsim.app.cache import Kept, is_development
    from hcsim.app.demos import DEMOS, PRESETS, DemoError, Explorer

    if cache is None and is_development(__version__):
        print(
            f"hcsim app: {__version__} is a development version, which keeps answers only in a "
            "directory given with --cache"
        )
        return 2
    if not world.is_file():
        print(f"hcsim app: no world at {world}")
        return 2
    kept = Kept.where(cache, world, __version__)
    with tempfile.TemporaryDirectory(prefix="hcsim-explorer-") as scratch:
        try:
            explorer = Explorer(world, Path(scratch), kept)
            for demo in DEMOS:
                start = time.perf_counter()
                for params in PRESETS[demo]:
                    explorer.answer(demo, **params)
                took = time.perf_counter() - start
                print(f"{demo}: {len(PRESETS[demo])} presets kept, in {took:.1f} s")
        except DemoError as error:
            print(f"hcsim app: {error}")
            return 2
    print(f"kept in {kept.root}")
    return 0
