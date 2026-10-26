"""The explorer's HTTP API, and the built frontend beside it.

`create_app` serves each demo (see `demos`) as JSON under `/api`, its parameters bounded so no
request asks for more than a laptop gives quickly. With a directory of the frontend's built
files, it serves them at `/` too. It's meant for the local machine: `hcsim app serve` binds to
the loopback address unless told otherwise.

Each preset's answer is kept on disk (`cache`), so a later start answers it at once, and
`/api/health` says about how long each chapter's first preset will take where it isn't kept.
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Annotated, Any

from fastapi import FastAPI, HTTPException, Query
from fastapi.staticfiles import StaticFiles

from hcsim import __version__
from hcsim.app.cache import Kept
from hcsim.app.demos import DEMOS, TRAJECTORIES, DemoError, Explorer

#: The most trajectories a request for a run on the world may ask for.
MOST_TRAJECTORIES = 20


def create_app(
    world: Path, static: Path | None = None, scratch: Path | None = None, cache: Path | None = None
) -> FastAPI:
    """The explorer's server for `world`, serving the frontend built into `static`, if given.
    The valves demo's plans and guidance go to `scratch`, a fresh temporary directory unless
    given. Presets' answers are kept in `cache`, or where `Kept.where` keeps them by default."""
    if not world.is_file():
        raise DemoError(f"no world at {world}")
    if scratch is None:
        scratch = Path(tempfile.mkdtemp(prefix="hcsim-explorer-"))
    explorer = Explorer(world, scratch, Kept.where(cache, world, __version__))
    app = FastAPI(title="hcsim explorer", docs_url="/api/docs", openapi_url="/api/openapi.json")

    def answer(demo: str, **params: Any) -> dict[str, Any]:
        try:
            return explorer.answer(demo, **params)
        except DemoError as error:
            raise HTTPException(status_code=422, detail=str(error)) from None

    @app.get("/api/health")
    def health() -> dict[str, Any]:
        """The run behind every chart: the world, its size, the month its runs start, and the
        engine's version; and about how many seconds each chapter's first preset takes, none
        where it's kept, and null where there's no estimate."""
        return {
            "world": world.name,
            "run_date": explorer.run_date.isoformat(),
            "employees": explorer.employees,
            "cost_centers": explorer.cost_centers,
            "version": __version__,
            "chapters": {demo: {"seconds": explorer.estimate(demo)} for demo in DEMOS},
        }

    @app.get("/api/sampling")
    def sampling_demo(
        rows: Annotated[int, Query(ge=2, le=2_000)] = 200,
        quota: Annotated[int, Query(ge=1, le=2_000)] = 20,
        trajectories: Annotated[int, Query(ge=2, le=2_000)] = 500,
        spread: Annotated[float, Query(ge=0.0, le=3.0)] = 1.0,
        seed: Annotated[int, Query(ge=0)] = 0,
    ) -> dict[str, Any]:
        return answer(
            "sampling", rows=rows, quota=quota, trajectories=trajectories, spread=spread, seed=seed
        )

    @app.get("/api/valves")
    def valves_demo(
        growth: Annotated[float, Query(ge=-0.2, le=0.3)] = 0.05,
        trajectories: Annotated[int, Query(ge=1, le=MOST_TRAJECTORIES)] = TRAJECTORIES,
    ) -> dict[str, Any]:
        return answer("valves", growth=round(growth, 3), trajectories=trajectories)

    @app.get("/api/lifecycle")
    def lifecycle_demo(
        months: Annotated[int, Query(ge=1, le=36)] = 12,
        trajectories: Annotated[int, Query(ge=1, le=MOST_TRAJECTORIES)] = TRAJECTORIES,
    ) -> dict[str, Any]:
        return answer("lifecycle", months=months, trajectories=trajectories)

    @app.get("/api/sbc")
    def sbc_demo(
        months: Annotated[int, Query(ge=1, le=36)] = 12,
        trajectories: Annotated[int, Query(ge=1, le=MOST_TRAJECTORIES)] = TRAJECTORIES,
    ) -> dict[str, Any]:
        return answer("sbc", months=months, trajectories=trajectories)

    if static is not None:
        if not (static / "index.html").is_file():
            raise DemoError(f"no built frontend at {static}: run `make explorer-build`")
        app.mount("/", StaticFiles(directory=static, html=True), name="frontend")
    return app
