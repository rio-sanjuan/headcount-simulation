"""Tests for the explorer's demos, its API, and `hcsim app`."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest
from fastapi.testclient import TestClient

from hcsim.app import cli
from hcsim.app.demos import (
    FILLED,
    OPENED,
    SBC_PARTS,
    VALVES,
    DemoError,
    Explorer,
    _spread,
    sampling,
)
from hcsim.app.server import create_app
from hcsim.synth.config import preset
from hcsim.synth.world import generate_world, write_world


@pytest.fixture(scope="module")
def world(tmp_path_factory: pytest.TempPathFactory) -> Path:
    path = tmp_path_factory.mktemp("world") / "tiny.duckdb"
    write_world(generate_world(preset("tiny", seed=5)), path)
    return path


@pytest.fixture(scope="module")
def explorer(world: Path, tmp_path_factory: pytest.TempPathFactory) -> Explorer:
    return Explorer(world, tmp_path_factory.mktemp("scratch"))


@pytest.fixture(scope="module")
def client(world: Path, tmp_path_factory: pytest.TempPathFactory) -> TestClient:
    static = tmp_path_factory.mktemp("dist")
    (static / "index.html").write_text("<!doctype html><title>explorer</title>")
    return TestClient(create_app(world, static, tmp_path_factory.mktemp("scratch")))


# --- the demos --------------------------------------------------------------------------------


def test_quota_sampling_realizes_the_quota_and_independent_draws_only_expect_it() -> None:
    s = sampling(rows=100, quota=10, trajectories=400, spread=1.0, seed=3)
    assert s["expected"] == pytest.approx(10)
    assert (s["quota"]["mean"], s["quota"]["sd"]) == (10, 0)
    assert s["independent"]["sd"] > 1
    assert abs(s["independent"]["mean"] - 10) < 4 * s["independent"]["sd"] / np.sqrt(400)
    for name in ("quota", "independent"):
        assert sum(s[name]["histogram"]) == 400
        assert len(s[name]["rates"]) == 100
    assert len(s["bins"]) == len(s["quota"]["histogram"])
    assert s["probabilities"] == sorted(s["probabilities"], reverse=True)
    # Both draw the likelier rows more often.
    rates = np.array(s["quota"]["rates"])
    assert rates[:10].mean() > rates[-10:].mean()
    even = sampling(rows=10, quota=5, trajectories=10, spread=0.0)
    assert even["probabilities"] == [0.5] * 10
    with pytest.raises(DemoError, match="between 1 and the rows"):
        sampling(rows=5, quota=6)


def assert_spread(spread: dict[str, list[float]], months: int) -> None:
    """A spread is a mean and three percentiles, each a value per month, and its band, from the
    10th to the 90th percentile, holds its median."""
    assert set(spread) == {"mean", "p10", "p50", "p90"}
    assert all(len(v) == months for v in spread.values())
    low, median, high = (np.array(spread[k]) for k in ("p10", "p50", "p90"))
    assert (low <= median).all()
    assert (median <= high).all()


def test_a_spread_is_the_mean_and_percentiles_of_each_month() -> None:
    values = np.array([[0.0, 10.0], [1.0, 20.0], [2.0, 30.0], [3.0, 40.0], [4.0, 50.0]])
    s = _spread(values)
    assert s == {
        "mean": [2.0, 30.0],
        "p10": [0.4, 14.0],
        "p50": [2.0, 30.0],
        "p90": [3.6, 46.0],
    }


def test_each_valve_drives_the_same_plan(explorer: Explorer) -> None:
    v = explorer.valves(growth=0.05, trajectories=4)
    assert set(v["runs"]) == set(VALVES)
    assert len(v["plan"]) == v["months"] == 6
    assert v["plan"] == sorted(v["plan"])  # a plan that rises
    for method, run in v["runs"].items():
        assert_spread(run["headcount"], 7)
        assert run["valve"] == VALVES[method]
        assert run["bias_mean"] >= 0
    starts = {run["headcount"][k][0] for run in v["runs"].values() for k in run["headcount"]}
    assert len(starts) == 1  # every run, and every trajectory, starts from the same seed
    # Trajectories part as the months go by, so the band opens.
    band = [run["headcount"]["p90"][-1] - run["headcount"]["p10"][-1] for run in v["runs"].values()]
    assert max(band) > 0
    assert explorer.valves(growth=0.05, trajectories=4) == v  # kept


def test_demand_holds_a_flat_plan_from_its_first_month(explorer: Explorer) -> None:
    # Guidance aligned to a flat plan once asked for 53 exits in the first month and as many
    # hires. Every exit happened, but only the hires the open openings allowed did, and demand
    # lost 33 heads at once.
    v = explorer.valves(growth=0.0, trajectories=2)
    demand = v["runs"]["demand"]["headcount"]["mean"]
    assert abs(demand[1] - v["plan"][0]) <= 3
    assert abs(demand[-1] - v["plan"][-1]) <= 8


def test_the_lifecycle_balances_the_open_openings(explorer: Explorer) -> None:
    life = explorer.lifecycle(months=6, trajectories=3)
    assert life["identity_residual"] == 0
    assert_spread(life["open"], 7)
    assert set(life["opened"]) == set(OPENED)
    assert set(life["filled"]) == set(FILLED)
    change = np.diff(life["open"]["mean"])  # the flows are means, so they explain the mean
    inflow = sum(np.array(v) for v in life["opened"].values())
    outflow = sum(np.array(v) for v in life["filled"].values()) + np.array(life["unfilled"])
    assert change == pytest.approx(inflow - outflow, abs=1e-3)  # each mean rounded


def test_the_sbc_layer_splits_expense_into_parts_that_add_up_to_it(explorer: Explorer) -> None:
    out = explorer.sbc(months=6, trajectories=2)
    assert_spread(out["total"], 6)
    assert set(out["parts"]) == set(SBC_PARTS)
    assert all(len(v) == 6 for v in out["parts"].values())
    # The parts are means, so they add up to the total's mean, month by month.
    parts = sum(np.array(v) for v in out["parts"].values())
    assert parts == pytest.approx(out["total"]["mean"], abs=1e-3)
    assert min(out["parts"]["net"]) > 0
    assert min(out["parts"]["attrition"]) <= 0  # a leaver's unvested tranches reverse expense
    # So do the grants it comes from: those made before the run, and by it.
    assert {"seed", "sim"} <= set(out["origins"])
    origins = sum(np.array(v) for v in out["origins"].values())
    assert origins == pytest.approx(out["total"]["mean"], abs=1e-3)
    assert {"grants", "example_grant", "recognized"}.isdisjoint(out)


def test_an_explorer_needs_its_world(tmp_path: Path) -> None:
    with pytest.raises(DemoError, match="no world"):
        Explorer(tmp_path / "missing.duckdb", tmp_path)


# --- the API ----------------------------------------------------------------------------------


def test_the_api_serves_each_demo(client: TestClient) -> None:
    assert client.get("/api/health").json()["world"] == "tiny.duckdb"
    s = client.get("/api/sampling", params={"rows": 50, "quota": 5, "trajectories": 20})
    assert s.status_code == 200
    assert s.json()["quota"]["mean"] == 5
    assert (
        client.get("/api/lifecycle", params={"months": 3, "trajectories": 1}).json()[
            "identity_residual"
        ]
        == 0
    )
    v = client.get("/api/valves", params={"growth": 0.02, "trajectories": 1}).json()
    assert set(v["runs"]) == set(VALVES)
    sbc = client.get("/api/sbc", params={"months": 3, "trajectories": 1}).json()
    assert set(sbc["parts"]) == set(SBC_PARTS)


def test_the_api_bounds_and_explains_bad_requests(client: TestClient) -> None:
    assert client.get("/api/sampling", params={"rows": 10**6}).status_code == 422
    assert client.get("/api/lifecycle", params={"months": 0}).status_code == 422
    for demo in ("valves", "lifecycle", "sbc"):
        assert client.get(f"/api/{demo}", params={"trajectories": 21}).status_code == 422
    bad = client.get("/api/sampling", params={"rows": 5, "quota": 6})
    assert bad.status_code == 422
    assert "between 1 and the rows" in bad.json()["detail"]


def test_the_server_serves_the_built_frontend(client: TestClient, world: Path) -> None:
    page = client.get("/")
    assert page.status_code == 200
    assert "<title>explorer</title>" in page.text
    assert client.get("/api/docs").status_code == 200
    with pytest.raises(DemoError, match="no built frontend"):
        create_app(world, world.parent)


# --- the command ------------------------------------------------------------------------------


def test_serve_runs_the_server_on_the_loopback_address(
    world: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    import uvicorn

    served: dict[str, Any] = {}
    monkeypatch.setattr(uvicorn, "run", lambda app, **kw: served.update(kw, app=app))
    monkeypatch.chdir(world.parent)  # no built frontend here
    assert cli.main(["serve", "--world", str(world)]) == 0
    assert (served["host"], served["port"]) == ("127.0.0.1", 8765)
    assert "the API only" in capsys.readouterr().out
    assert cli.main(["serve", "--world", str(world.parent / "missing.duckdb")]) == 2
    assert "no world" in capsys.readouterr().out
    dist = world.parent / "apps" / "explorer" / "dist"
    dist.mkdir(parents=True)
    (dist / "index.html").write_text("<!doctype html>")
    assert cli.main(["serve", "--world", str(world), "--port", "9000"]) == 0
    assert served["port"] == 9000
    assert "the API only" not in capsys.readouterr().out
