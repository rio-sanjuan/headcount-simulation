"""Tests for `hcsim study run`'s options and exit codes."""

from __future__ import annotations

import dataclasses
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import duckdb
import pandas as pd
import pytest
import yaml

from hcsim.cli import main
from hcsim.engine.seeding import schema_from_world
from hcsim.model.core.schema import SimulationSchema
from hcsim.studies import cli as study_cli
from hcsim.studies.gates import Verdict
from hcsim.studies.harness import StudyRun, run_study, schema_problems
from hcsim.studies.profile import StudyProfile

#: A closed window in the tiny world, which ends with 2025.
CLOSED = {"run_date": "2025-07-01", "months": 4}
BASE = {"name": "cli", "seed": 4, "trajectories": 2, "mode": "unguided", "scope": "hcf"}


def run(tmp_path: Path, world: Path, profile: Mapping[str, Any], *extra: str) -> int:
    tmp_path.mkdir(parents=True, exist_ok=True)
    path = tmp_path / "profile.yaml"
    path.write_text(yaml.safe_dump(BASE | CLOSED | dict(profile)))
    args = ["study", "run", "--profile", str(path), "--world", str(world)]
    if "--preflight-only" not in extra:
        args += ["--out", str(tmp_path / "out")]
    return main([*args, *extra])


def card(tmp_path: Path) -> pd.DataFrame:
    return pd.read_csv(tmp_path / "out" / "scorecard.csv", keep_default_na=False)


def record(tmp_path: Path) -> dict[str, object]:
    data: dict[str, object] = json.loads((tmp_path / "out" / "run.json").read_text())
    return data


# --- exit codes -------------------------------------------------------------------------------


def test_each_exit_code_says_how_the_study_ended(
    tmp_path: Path, inputs: tuple[Path, Path], capsys: pytest.CaptureFixture[str]
) -> None:
    world = inputs[0]
    fidelity = {"evaluators": ["invariants", "fidelity"]}
    assert run(tmp_path / "passed", world, fidelity) == 0
    flagged = {"name": "f", "metric": "fidelity.headcount_wape", "max": 0}
    assert run(tmp_path / "provisional", world, fidelity | {"gates": [flagged]}) == 4
    assert "gate  f failed on study" in capsys.readouterr().out
    assert card(tmp_path / "provisional")["arm"].eq("study").all()  # still written
    stopping = flagged | {"action": "stop"}
    assert run(tmp_path / "stopped", world, fidelity | {"gates": [stopping]}) == 3
    assert "cli: stopped" in capsys.readouterr().out
    assert record(tmp_path / "stopped")["verdict"] == "stopped"


def test_a_study_that_waits_for_its_window_exits_5(
    tmp_path: Path, inputs: tuple[Path, Path], capsys: pytest.CaptureFixture[str]
) -> None:
    open_window = {"mode": "replay", "run_date": None}
    assert run(tmp_path, inputs[0], open_window) == 5
    assert "nothing can run yet" in capsys.readouterr().err
    assert not (tmp_path / "out").exists()
    assert run(tmp_path, inputs[0], open_window, "--preflight-only") == 5
    # Waiting and wrong is wrong.
    assert run(tmp_path, inputs[0], open_window | {"subset": {"cost_center": ["CC-01"]}}) == 2


def test_the_exit_code_follows_the_verdict(tmp_path: Path, inputs: tuple[Path, Path]) -> None:
    study = run_study(StudyProfile.from_mapping(BASE | CLOSED), inputs[0])
    assert study_cli.exit_code(study) == 0
    assert study_cli.exit_code(dataclasses.replace(study, verdict=Verdict.PROVISIONAL)) == 4
    assert study_cli.exit_code(dataclasses.replace(study, verdict=Verdict.STOPPED)) == 3
    failed = dataclasses.replace(study.metrics[0], value=5.0, threshold=0.0)
    assert study_cli.exit_code(dataclasses.replace(study, metrics=(failed,))) == 3


def test_out_is_needed_to_run(tmp_path: Path, inputs: tuple[Path, Path]) -> None:
    path = tmp_path / "p.yaml"
    path.write_text(yaml.safe_dump(BASE))
    with pytest.raises(SystemExit) as raised:
        main(["study", "run", "--profile", str(path), "--world", str(inputs[0])])
    assert raised.value.code == 2


# --- --preflight-only -------------------------------------------------------------------------


def test_preflight_only_checks_and_writes_nothing(
    tmp_path: Path, inputs: tuple[Path, Path], capsys: pytest.CaptureFixture[str]
) -> None:
    arms = [
        {"name": "base", "null": True},
        {"name": "x", "overrides": {"promotion.gate_probability": 0.5}},
        {"name": "history", "reference": True, "mode": "replay"},
    ]
    profile = {"arms": arms, "run_date": "2025-11-01", "evaluators": ["invariants", "fidelity"]}
    assert run(tmp_path, inputs[0], profile, "--preflight-only") == 0
    out = capsys.readouterr().out
    assert "cli: preflight passed" in out
    assert "partially_closed" in out
    assert "arms       base, x" in out
    assert "skips      fidelity:" in out
    assert "defers     history:" in out
    assert not (tmp_path / "out").exists()
    assert run(tmp_path, inputs[0], {"subset": {"nope": ["x"]}}, "--preflight-only") == 2
    assert "subset" in capsys.readouterr().err


# --- --offline --------------------------------------------------------------------------------


def test_offline_runs_on_the_offline_backend(
    tmp_path: Path, inputs: tuple[Path, Path], capsys: pytest.CaptureFixture[str]
) -> None:
    offline = {"evaluators": ["fitted_rates", "offline_domain"]}
    assert run(tmp_path, inputs[0], offline, "--offline") == 0
    rows = card(tmp_path)
    assert set(rows["evaluator"]) == {"fitted_rates", "offline_domain"}
    assert set(rows["backend"]) == {"offline"}
    assert record(tmp_path)["results_digest"] == ""
    assert run(tmp_path / "simulation", inputs[0], {}, "--offline") == 2
    assert "offline backend runs offline evaluators" in capsys.readouterr().err


# --- --schema-path ----------------------------------------------------------------------------


def test_a_study_can_encode_with_a_saved_schema(
    tmp_path: Path, inputs: tuple[Path, Path], capsys: pytest.CaptureFixture[str]
) -> None:
    with duckdb.connect(str(inputs[0]), read_only=True) as con:
        world = schema_from_world(con)
    wider = world.to_dict()
    wider["vocabularies"]["location"] = sorted([*wider["vocabularies"]["location"], "ZZ-new"])
    saved = tmp_path / "schema.json"
    SimulationSchema.from_dict(wider).save(saved)
    assert run(tmp_path, inputs[0], {}, "--schema-path", str(saved)) == 0
    assert set(card(tmp_path)["schema_fingerprint"]) == {SimulationSchema.load(saved).fingerprint()}

    narrower = world.to_dict()
    narrower["vocabularies"]["location"] = narrower["vocabularies"]["location"][1:]
    del narrower["vocabularies"]["company"]
    problems = schema_problems(SimulationSchema.from_dict(narrower), world)
    assert problems == [
        "the schema has no vocabulary 'company'",
        f"the schema's location lacks 1 of the world's: {world.vocabulary('location').labels[0]}",
    ]
    short = tmp_path / "short.json"
    SimulationSchema.from_dict(narrower).save(short)
    assert run(tmp_path / "short", inputs[0], {}, "--schema-path", str(short)) == 2
    assert "lacks 1 of the world's" in capsys.readouterr().err
    assert run(tmp_path / "none", inputs[0], {}, "--schema-path", str(tmp_path / "x.json")) == 2
    assert "can't load the schema" in capsys.readouterr().err


# --- --cache-dir ------------------------------------------------------------------------------


def test_a_cache_dir_keeps_the_extracts_a_study_reads(
    tmp_path: Path, inputs: tuple[Path, Path]
) -> None:
    profile = {"extracts": ["departure_months", "promotion_months"]}
    cache = tmp_path / "cache"
    assert run(tmp_path / "first", inputs[0], profile, "--cache-dir", str(cache)) == 0
    first = record(tmp_path / "first")["extracts"]
    assert isinstance(first, dict)
    assert set(first) == {"departure_months", "promotion_months"}
    assert all(e["hits"] == 0 and e["pieces"] >= 1 and e["rows"] > 0 for e in first.values())
    files = sorted(p.name for p in cache.iterdir())
    assert len(files) == sum(e["pieces"] for e in first.values())
    assert run(tmp_path / "second", inputs[0], profile, "--cache-dir", str(cache)) == 0
    second = record(tmp_path / "second")["extracts"]
    assert isinstance(second, dict)
    assert all(e["hits"] == e["pieces"] for e in second.values())
    assert {k: e["rows"] for k, e in second.items()} == {k: e["rows"] for k, e in first.items()}
    # Without a cache dir, nothing is kept.
    assert run(tmp_path / "third", inputs[0], profile) == 0
    third = record(tmp_path / "third")["extracts"]
    assert isinstance(third, dict)
    assert all(e["hits"] == 0 for e in third.values())
    assert sorted(p.name for p in cache.iterdir()) == files


def test_a_profile_reads_only_extract_kinds() -> None:
    with pytest.raises(ValueError, match=r"no extracts \['seed_state'\]"):
        StudyProfile.from_mapping(BASE | {"extracts": ["seed_state"]})


def test_failed_runs_report_the_cli_names(
    tmp_path: Path, inputs: tuple[Path, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    def broken(*args: object, **options: object) -> StudyRun:
        study = run_study(*args, **options)  # type: ignore[arg-type]
        return dataclasses.replace(study, verdict=Verdict.STOPPED)

    monkeypatch.setattr(study_cli, "run_study", broken)
    assert run(tmp_path, inputs[0], {}) == study_cli.EXIT_STOPPED == study_cli.EXIT_FAILED
