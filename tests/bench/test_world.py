"""Tests for the preset timing harness."""

from __future__ import annotations

import dataclasses
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from hcsim.bench import cli
from hcsim.bench import world as bench
from hcsim.bench.world import BenchError, PresetRun, run, run_one
from hcsim.synth.config import preset
from hcsim.synth.world import digest, generate_world


def test_one_run_times_each_step_and_checks_the_world(tmp_path: Path) -> None:
    result = run_one("tiny", 3, tmp_path)
    assert result.checks_passed
    assert (result.preset, result.seed, result.employees) == ("tiny", 3, 200)
    assert min(result.generate_seconds, result.write_seconds, result.validate_seconds) >= 0
    assert result.total_seconds > 0
    assert result.peak_rss_mib > 0
    assert result.rows > 0
    assert result.digest == digest(generate_world(preset("tiny", 3)))


def test_the_child_entry_point_prints_its_run(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert bench.main(["tiny", "3", str(tmp_path)]) == 0
    printed = json.loads(capsys.readouterr().out)
    assert PresetRun(**printed).checks_passed


@pytest.mark.parametrize(("platform", "unit"), [("darwin", 2**20), ("linux", 2**10)])
def test_peak_memory_reads_each_platform_s_unit(
    monkeypatch: pytest.MonkeyPatch, platform: str, unit: int
) -> None:
    monkeypatch.setattr(sys, "platform", platform)
    monkeypatch.setattr(bench.resource, "getrusage", lambda _: SimpleNamespace(ru_maxrss=unit))
    assert bench.peak_rss_mib() == 1.0


@pytest.mark.slow
def test_each_preset_runs_in_its_own_process() -> None:
    (result,) = run(["tiny"], seed=3)
    assert result.digest == digest(generate_world(preset("tiny", 3)))


@pytest.mark.slow
def test_a_run_past_its_memory_limit_is_stopped() -> None:
    with pytest.raises(BenchError, match=r"memory limit of 0\.05 GiB"):
        run(["small"], max_rss_gib=0.05)


@pytest.mark.slow
def test_a_failed_run_is_reported() -> None:
    with pytest.raises(BenchError, match="'enormous' preset failed"):
        run(["enormous"])


@pytest.mark.slow
def test_the_small_preset_generates_within_a_minute(tmp_path: Path) -> None:
    result = run_one("small", 0, tmp_path)
    assert result.generate_seconds < 60
    assert result.checks_passed


FAKE = PresetRun("tiny", 0, 200, 0.3, 0.1, 0.05, 180.0, 96_000, True, "abc")


def test_the_cli_prints_and_writes_runs(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    calls = []

    def fake(presets: list[str], seed: int, limit: float | None) -> list[PresetRun]:
        calls.append((presets, seed, limit))
        return [FAKE]

    monkeypatch.setattr(cli, "run", fake)
    out = tmp_path / "bench" / "runs.json"
    args = ["world", "--preset", "tiny", "--seed", "2", "--max-rss-gib", "8", "--out", str(out)]
    assert cli.main(args) == 0
    assert calls == [(["tiny"], 2, 8.0)]
    assert "tiny" in capsys.readouterr().out
    assert json.loads(out.read_text()) == [dataclasses.asdict(FAKE)]
    assert cli.main(["world"]) == 0  # tiny by default
    assert calls[-1][0] == ["tiny"]

    monkeypatch.setattr(cli, "run", lambda *_: [dataclasses.replace(FAKE, checks_passed=False)])
    assert cli.main(["world"]) == 1
    assert "FAIL" in capsys.readouterr().out

    def broken(*_: object) -> list[PresetRun]:
        raise BenchError("the 'tiny' preset failed")

    monkeypatch.setattr(cli, "run", broken)
    assert cli.main(["world"]) == 1
    assert "hcsim bench: the 'tiny' preset failed" in capsys.readouterr().out
