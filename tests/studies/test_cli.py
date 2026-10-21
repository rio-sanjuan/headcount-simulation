"""Tests for the scorecard and the `hcsim study run` command."""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import pandas as pd
import pytest

from hcsim.cli import main
from hcsim.studies import cli as study_cli
from hcsim.studies.evaluators import Metric
from hcsim.studies.harness import StudyRun, run_study
from hcsim.studies.scorecard import COLUMNS, PROVENANCE_COLUMNS, SCORECARD_VERSION

PROFILE = "name: skeleton\nseed: 4\ntrajectories: 2\nmonths: 3\n"


def write_profile(tmp_path: Path, text: str = PROFILE) -> Path:
    path = tmp_path / "profile.yaml"
    path.write_text(text)
    return path


def study(tmp_path: Path, inputs: tuple[Path, Path], *extra: str) -> list[str]:
    world, guidance = inputs
    return [
        "study",
        "run",
        "--profile",
        str(write_profile(tmp_path)),
        "--world",
        str(world),
        "--guidance",
        str(guidance),
        "--out",
        str(tmp_path / "out"),
        *extra,
    ]


def test_a_study_run_writes_its_scorecard_and_record(
    tmp_path: Path, inputs: tuple[Path, Path], capsys: pytest.CaptureFixture[str]
) -> None:
    assert main(study(tmp_path, inputs)) == 0
    out = capsys.readouterr().out
    assert "skeleton: passed" in out
    assert "pass  quota_violations" in out
    assert "info  quota_shortfall" in out

    card = pd.read_csv(tmp_path / "out" / "scorecard.csv")
    assert list(card.columns) == COLUMNS
    assert (card["scorecard_version"] == SCORECARD_VERSION).all()
    assert set(card.loc[card["evaluator"] == "invariants", "metric"]) == {
        "quota_violations",
        "quota_shortfall",
        "headcount_identity_max_gap",
        "opening_identity_max_gap",
        "accounting_identity_max_residual",
    }
    summary = card[card["evaluator"] == "summary"].set_index("metric")["value"]
    assert {"headcount_end_mean", "sbc_recognized_mean", "tranches_mean"} <= set(summary.index)
    assert card.loc[card["evaluator"] == "summary", "threshold"].isna().all()
    assert card["passed"].all()
    assert card[PROVENANCE_COLUMNS].notna().all().all()
    assert (card["guidance_fingerprint"].str.startswith("gf1-")).all()

    record = json.loads((tmp_path / "out" / "run.json").read_text())
    assert record["passed"] is True
    assert record["results_digest"].startswith("rd1-")
    assert record["profile"]["seed"] == 4
    assert record["seed"]["snapshot_date"] == "2025-12-31"
    assert record["wall_clock_seconds"] > 0
    assert set(record["fitted_rates"]) >= {"attrition", "growth"}


def test_the_scorecard_is_identical_across_runs(tmp_path: Path, inputs: tuple[Path, Path]) -> None:
    first, second = tmp_path / "a", tmp_path / "b"
    first.mkdir()
    second.mkdir()
    assert main(study(first, inputs)) == 0
    assert main(study(second, inputs)) == 0
    a = (first / "out" / "scorecard.csv").read_bytes()
    assert a == (second / "out" / "scorecard.csv").read_bytes()


@pytest.mark.parametrize(
    ("profile", "guided", "message"),
    [
        ("name: x\n", True, "seed: Field required"),
        (PROFILE, False, "needs a guidance artifact"),
        (PROFILE + "run_date: 2025-06-01\n", True, "guidance was fitted through 2025-12-31"),
    ],
)
def test_input_errors_exit_2_without_a_traceback(
    tmp_path: Path,
    inputs: tuple[Path, Path],
    capsys: pytest.CaptureFixture[str],
    profile: str,
    guided: bool,
    message: str,
) -> None:
    world, guidance = inputs
    args = [
        "study",
        "run",
        "--profile",
        str(write_profile(tmp_path, profile)),
        "--world",
        str(world),
        "--out",
        str(tmp_path / "out"),
    ]
    if guided:
        args += ["--guidance", str(guidance)]
    assert main(args) == 2
    assert message in capsys.readouterr().err
    assert not (tmp_path / "out").exists()


def test_a_failed_metric_exits_3(
    tmp_path: Path,
    inputs: tuple[Path, Path],
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    def failing(*args: object) -> StudyRun:
        run = run_study(*args)  # type: ignore[arg-type]
        broken = Metric("invariants", "quota_violations", 2, 0)
        return dataclasses.replace(run, metrics=(broken, *run.metrics[1:]))

    monkeypatch.setattr(study_cli, "run_study", failing)
    assert main(study(tmp_path, inputs)) == 3
    assert "FAIL  quota_violations" in capsys.readouterr().out
    card = pd.read_csv(tmp_path / "out" / "scorecard.csv")
    assert not card.loc[card["metric"] == "quota_violations", "passed"].item()


def test_the_command_is_listed(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["--help"]) == 0
    assert "study" in capsys.readouterr().out
