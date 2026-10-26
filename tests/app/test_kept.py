"""Tests for the explorer's kept answers: the cache and its keys, the locks, and preparing."""

from __future__ import annotations

import inspect
import json
import threading
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

import hcsim
from hcsim.app import cache as cache_module
from hcsim.app import cli
from hcsim.app.cache import Kept, default_dir, is_development
from hcsim.app.demos import DEMOS, OVERHEAD, PRESETS, DemoError, Explorer, first, sampling
from hcsim.app.server import create_app
from hcsim.synth.config import preset
from hcsim.synth.world import generate_world, write_world

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def world(tmp_path_factory: pytest.TempPathFactory) -> Path:
    path = tmp_path_factory.mktemp("world") / "tiny.duckdb"
    write_world(generate_world(preset("tiny", seed=5)), path)
    return path


# --- the cache --------------------------------------------------------------------------------


def test_an_answer_is_kept_under_everything_that_made_it(tmp_path: Path, world: Path) -> None:
    kept = Kept(tmp_path / "kept", world, "1.0.0")
    params = {"months": 6, "trajectories": 10}
    assert kept.get("sbc", params) is None
    kept.put("sbc", params, {"total": [1.0]})
    assert kept.get("sbc", params) == {"total": [1.0]}
    assert kept.get("sbc", {"months": 6.0, "trajectories": 10.0}) == {"total": [1.0]}  # 6 is 6.0
    assert kept.get("lifecycle", params) is None  # another demo
    assert kept.get("sbc", {"months": 12, "trajectories": 10}) is None  # other parameters
    assert Kept(tmp_path / "kept", world, "1.0.1").get("sbc", params) is None  # a new version
    other = tmp_path / "other.duckdb"
    other.write_bytes(world.read_bytes() + b"\0")  # the same name's no help: its contents differ
    assert Kept(tmp_path / "kept", other, "1.0.0").get("sbc", params) is None
    assert not list((tmp_path / "kept").rglob("*.partial"))  # written whole, then moved


def test_an_unreadable_answer_is_a_miss(tmp_path: Path, world: Path) -> None:
    kept = Kept(tmp_path, world, "1.0.0")
    kept.put("sbc", {"months": 6}, {"total": [1.0]})
    (path,) = (tmp_path / "sbc").glob("*.json")
    path.write_text("{not json")
    assert kept.get("sbc", {"months": 6}) is None


def test_answers_are_kept_by_default_except_for_a_development_version(
    tmp_path: Path, world: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "xdg"))
    assert default_dir() == tmp_path / "xdg" / "hcsim" / "explorer"
    assert is_development("0.1.0.dev0")
    assert not is_development("0.1.0")
    assert Kept.where(None, world, "0.1.0").root == default_dir()
    assert Kept.where(None, world, "0.1.0.dev0").root is None
    assert Kept.where(tmp_path / "asked", world, "0.1.0.dev0").root == tmp_path / "asked"
    nowhere = Kept.where(None, world, "0.1.0.dev0")
    nowhere.put("sbc", {"months": 6}, {"total": []})
    assert nowhere.get("sbc", {"months": 6}) is None


def test_the_world_is_hashed_only_where_answers_are_kept(
    world: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def refuse(path: Path) -> str:
        raise AssertionError("hashed a world with nowhere to keep its answers")

    monkeypatch.setattr(cache_module, "file_hash", refuse)
    assert Kept(None, world, "1.0.0").world == ""


# --- the presets ------------------------------------------------------------------------------


def test_the_server_keeps_the_presets_the_page_offers() -> None:
    page = json.loads((ROOT / "apps" / "explorer" / "src" / "presets.json").read_text())
    defaults = {
        "sampling": inspect.signature(sampling).parameters,
        **{demo: inspect.signature(getattr(Explorer, demo)).parameters for demo in DEMOS[1:]},
    }
    assert list(page) == list(DEMOS)
    for demo in DEMOS:
        full = [
            {
                name: query.get(name, p.default)
                for name, p in defaults[demo].items()
                if name != "self"
            }
            for query in page[demo]
        ]
        assert full == [dict(p) for p in PRESETS[demo]], demo
    assert first("valves") == {"growth": 0.05, "trajectories": 10}


# --- the explorer's answers --------------------------------------------------------------------


def test_a_second_start_reads_the_presets_it_kept(tmp_path: Path, world: Path) -> None:
    kept = Kept(tmp_path / "kept", world, "1.0.0")
    one = Explorer(world, tmp_path / "scratch", kept)
    params = {"months": 6, "trajectories": 2}
    made = one.answer("lifecycle", **dict(PRESETS["lifecycle"][0]))
    assert made["seconds"] > 0
    one.answer("lifecycle", **params)  # Adjust's: kept in memory only
    assert len(list((tmp_path / "kept" / "lifecycle").glob("*.json"))) == 1

    two = Explorer(world, tmp_path / "scratch", Kept(tmp_path / "kept", world, "1.0.0"))
    two._make = lambda demo: pytest.fail("ran a preset that was kept")  # type: ignore[method-assign]
    assert two.answer("lifecycle", **dict(PRESETS["lifecycle"][0])) == made
    assert two.has("lifecycle", dict(PRESETS["lifecycle"][0]))
    assert not two.has("lifecycle", params)


def test_each_demo_takes_its_own_lock(tmp_path: Path, world: Path) -> None:
    explorer = Explorer(world, tmp_path)
    done = threading.Event()

    def draw() -> None:
        explorer.answer("sampling", rows=20, quota=2)
        done.set()

    with explorer._locks["valves"]:  # a slow chapter, still running
        threading.Thread(target=draw).start()
        assert done.wait(timeout=30), "sampling waited on the valves"
    with pytest.raises(DemoError, match="the demos are"):
        explorer.answer("payroll")


def test_the_engine_estimates_each_chapters_first_preset(tmp_path: Path, world: Path) -> None:
    explorer = Explorer(world, tmp_path)
    baseline = tmp_path / "baseline.json"
    fit = {"prepare": [0.5, 0.0], "simulate": [0.0, 0.001]}
    baseline.write_text(json.dumps({"capacity": {"fits": {"hcf": fit, "hcf+sbc": fit}}}))
    work = explorer.employees * 12 * 10
    lifecycle = (0.5 + 0.001 * work) * OVERHEAD["lifecycle"]
    assert explorer.estimate("lifecycle", baseline) == round(lifecycle, 1)
    four = 4 * (0.5 + 0.001 * explorer.employees * 6 * 10)  # a run for each valve
    assert explorer.estimate("valves", baseline) == round(four * OVERHEAD["valves"], 1)
    assert explorer.estimate("sampling", baseline) == 0
    assert explorer.estimate("sbc", tmp_path / "missing.json") is None
    explorer.answer("lifecycle", **first("lifecycle"))
    assert explorer.estimate("lifecycle", baseline) == 0  # at hand


def test_the_committed_baseline_estimates_every_chapter(tmp_path: Path, world: Path) -> None:
    explorer = Explorer(world, tmp_path)
    for demo in DEMOS:
        seconds = explorer.estimate(demo, ROOT / "benchmarks" / "baseline.json")
        assert seconds is not None
        assert 0 <= seconds < 60, demo


# --- the server and the commands ---------------------------------------------------------------


def test_a_failed_demo_leaves_the_others_answering(tmp_path: Path, world: Path) -> None:
    client = TestClient(create_app(world, scratch=tmp_path, cache=tmp_path / "kept"))
    assert client.get("/api/sampling", params={"rows": 5, "quota": 6}).status_code == 422
    assert client.get("/api/sampling", params={"rows": 50, "quota": 5}).status_code == 200
    health = client.get("/api/health").json()
    assert set(health["chapters"]) == set(DEMOS)
    assert health["chapters"]["sampling"] == {"seconds": 0}


def test_prepare_keeps_every_preset_for_serve_to_read(
    tmp_path: Path, world: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    kept = tmp_path / "kept"
    assert cli.main(["prepare", "--world", str(world), "--cache", str(kept)]) == 0
    out = capsys.readouterr().out
    assert "valves: 3 presets kept" in out
    assert str(kept) in out
    assert len(list(kept.rglob("*.json"))) == sum(len(p) for p in PRESETS.values())
    app = create_app(world, scratch=tmp_path / "scratch", cache=kept)
    client = TestClient(app)
    health = client.get("/api/health").json()
    assert all(c["seconds"] == 0 for c in health["chapters"].values())  # nothing left to run

    served: dict[str, Any] = {}
    import uvicorn

    monkeypatch.setattr(uvicorn, "run", lambda app, **kw: served.update(kw, app=app))
    assert cli.main(["serve", "--world", str(world), "--cache", str(kept)]) == 0
    assert served["app"] is not None


def test_prepare_refuses_a_development_version_without_a_cache(
    world: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(hcsim, "__version__", "0.1.0.dev0")
    assert cli.main(["prepare", "--world", str(world)]) == 2
    assert "development version" in capsys.readouterr().out
    monkeypatch.setattr(hcsim, "__version__", "0.1.0")
    assert cli.main(["prepare", "--world", str(world.parent / "missing.duckdb")]) == 2
    assert "no world" in capsys.readouterr().out
