"""Tests for the hcsim command."""

from __future__ import annotations

import subprocess
from pathlib import Path

import duckdb
import pytest

from hcsim.cli import main


def test_no_arguments_prints_usage_and_exits_2(capsys: pytest.CaptureFixture[str]) -> None:
    assert main([]) == 2
    assert "usage: hcsim" in capsys.readouterr().out


def test_help_exits_0(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["--help"]) == 0
    assert "synth" in capsys.readouterr().out


def test_an_unknown_command_exits_2(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["teleport"]) == 2
    assert "unknown command 'teleport'" in capsys.readouterr().err


def test_synth_generate_writes_a_world(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    out = tmp_path / "world.duckdb"
    assert main(["synth", "generate", "--preset", "tiny", "--seed", "3", "--out", str(out)]) == 0
    printed = capsys.readouterr().out
    assert "workforce_txn" in printed
    assert "truth.latent_plan" in printed
    with duckdb.connect(str(out), read_only=True) as con:
        tables = {row[0] for row in con.execute("SHOW TABLES").fetchall()}
        truth = {
            row[0]
            for row in con.execute(
                "SELECT table_name FROM information_schema.tables WHERE table_schema = 'truth'"
            ).fetchall()
        }
    assert {"workforce_txn", "dim_org"} <= tables
    assert {"manifest", "latent_plan"} <= truth


def test_synth_refuses_an_unknown_preset(tmp_path: Path) -> None:
    with pytest.raises(SystemExit) as stop:
        main(["synth", "generate", "--preset", "enormous", "--out", str(tmp_path / "x.duckdb")])
    assert stop.value.code == 2


def test_the_installed_entry_point_runs(tmp_path: Path) -> None:
    out = tmp_path / "w.duckdb"
    result = subprocess.run(
        ["uv", "run", "--quiet", "hcsim", "synth", "generate", "--seed", "1", "--out", str(out)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert out.exists()
