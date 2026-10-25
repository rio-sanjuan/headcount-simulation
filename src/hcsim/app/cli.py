"""`hcsim app`: serve the explorer on this machine.

`hcsim app serve --world W` serves the explorer's API for the world at `W`, and, with
`--static`, the frontend built into that directory (`make explorer-build` builds it into
apps/explorer/dist). It binds to 127.0.0.1 unless `--host` says otherwise. It needs the `app`
extra: `uv sync --extra app`.
"""

from __future__ import annotations

import argparse
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
    args = parser.parse_args(list(argv))
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
        app = create_app(args.world, static)
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
