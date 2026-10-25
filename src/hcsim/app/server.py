"""The explorer's HTTP API, and the built frontend beside it.

`create_app` serves each demo (see `demos`) as JSON under `/api`, its parameters bounded so no
request asks for more than a laptop gives quickly. With a directory of the frontend's built
files, it serves them at `/` too. It's meant for the local machine: `hcsim app serve` binds to
the loopback address unless told otherwise.
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Annotated, Any

from fastapi import FastAPI, HTTPException, Query
from fastapi.staticfiles import StaticFiles

from hcsim.app.demos import DemoError, Explorer, sampling


def create_app(world: Path, static: Path | None = None, scratch: Path | None = None) -> FastAPI:
    """The explorer's server for `world`, serving the frontend built into `static`, if given.
    The valves demo's plans and guidance go to `scratch`, a fresh temporary directory unless
    given."""
    if scratch is None:
        scratch = Path(tempfile.mkdtemp(prefix="hcsim-explorer-"))
    explorer = Explorer(world, scratch)
    app = FastAPI(title="hcsim explorer", docs_url="/api/docs", openapi_url="/api/openapi.json")

    def answer(make: Any) -> dict[str, Any]:
        try:
            result: dict[str, Any] = make()
        except DemoError as error:
            raise HTTPException(status_code=422, detail=str(error)) from None
        return result

    @app.get("/api/health")
    def health() -> dict[str, str]:
        return {"world": world.name, "run_date": explorer.run_date.isoformat()}

    @app.get("/api/sampling")
    def sampling_demo(
        rows: Annotated[int, Query(ge=2, le=2_000)] = 200,
        quota: Annotated[int, Query(ge=1, le=2_000)] = 20,
        trajectories: Annotated[int, Query(ge=2, le=2_000)] = 500,
        spread: Annotated[float, Query(ge=0.0, le=3.0)] = 1.0,
        seed: Annotated[int, Query(ge=0)] = 0,
    ) -> dict[str, Any]:
        return answer(lambda: sampling(rows, quota, trajectories, spread, seed))

    @app.get("/api/valves")
    def valves_demo(
        growth: Annotated[float, Query(ge=-0.2, le=0.3)] = 0.05,
        trajectories: Annotated[int, Query(ge=1, le=8)] = 4,
    ) -> dict[str, Any]:
        return answer(lambda: explorer.valves(round(growth, 3), trajectories))

    @app.get("/api/lifecycle")
    def lifecycle_demo(
        months: Annotated[int, Query(ge=1, le=36)] = 12,
        trajectories: Annotated[int, Query(ge=1, le=8)] = 4,
    ) -> dict[str, Any]:
        return answer(lambda: explorer.lifecycle(months, trajectories))

    @app.get("/api/sbc")
    def sbc_demo(
        months: Annotated[int, Query(ge=1, le=36)] = 12,
        trajectories: Annotated[int, Query(ge=1, le=4)] = 2,
    ) -> dict[str, Any]:
        return answer(lambda: explorer.sbc(months, trajectories))

    if static is not None:
        if not (static / "index.html").is_file():
            raise DemoError(f"no built frontend at {static}: run `make explorer-build`")
        app.mount("/", StaticFiles(directory=static, html=True), name="frontend")
    return app
