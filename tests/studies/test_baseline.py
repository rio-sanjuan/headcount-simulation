"""Tests for the metric baselines and the summary evaluator."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from hcsim.studies import baseline
from hcsim.studies.harness import StudyRun, run_study
from hcsim.studies.profile import StudyProfile
from hcsim.studies.scorecard import results_digest, scorecard


@pytest.fixture(scope="module")
def run(inputs: tuple[Path, Path]) -> StudyRun:
    profile = StudyProfile.from_mapping({"name": "b", "seed": 3, "trajectories": 2, "months": 3})
    return run_study(profile, *inputs)


@pytest.fixture
def card(run: StudyRun) -> pd.DataFrame:
    return scorecard(run)


def saved(card: pd.DataFrame, tmp_path: Path) -> pd.DataFrame:
    baseline.write(card, tmp_path / "b.csv")
    return baseline.load(tmp_path / "b.csv")


def test_a_scorecard_matches_its_own_baseline(card: pd.DataFrame, tmp_path: Path) -> None:
    assert baseline.compare(card, saved(card, tmp_path)) == []
    written = pd.read_csv(tmp_path / "b.csv")
    assert list(written.columns) == baseline.COLUMNS
    assert "hcsim_version" not in written.columns
    assert "guidance_fingerprint" not in written.columns


def test_last_bit_noise_is_tolerated_but_a_real_move_is_not(
    card: pd.DataFrame, tmp_path: Path
) -> None:
    reference = saved(card, tmp_path)
    metric = card["metric"] == "sbc_recognized_mean"
    noisy = card.copy()
    noisy.loc[metric, "value"] *= 1 + 1e-12
    assert baseline.compare(noisy, reference) == []
    moved = card.copy()
    moved.loc[metric, "value"] += 1.0
    (line,) = baseline.compare(moved, reference)
    assert line.startswith("B/study/summary/sbc_recognized_mean: value is ")


def test_added_dropped_and_relabelled_rows_are_differences(
    card: pd.DataFrame, tmp_path: Path
) -> None:
    reference = saved(card, tmp_path)
    dropped = card[card["metric"] != "tranches_mean"]
    assert baseline.compare(dropped, reference) == [
        "B/study/summary/tranches_mean: missing from the run"
    ]
    added = pd.concat([card, card.iloc[[0]].assign(metric="new_metric")], ignore_index=True)
    assert baseline.compare(added, reference) == [
        "B/study/invariants/new_metric: not in the baseline"
    ]
    reseeded = card.assign(seed="99")
    assert len(baseline.compare(reseeded, reference)) == len(card)
    flipped = card.copy()
    flipped.loc[flipped["metric"] == "quota_violations", "passed"] = False
    assert baseline.compare(flipped, reference) == [
        "B/study/invariants/quota_violations: passed is False, baseline True"
    ]


def test_the_results_digest_tracks_every_bit(run: StudyRun) -> None:
    digest = results_digest(run.simulated)
    assert digest.startswith("rd1-")
    assert results_digest(run.simulated) == digest
    first = run.simulated.trajectories[0]
    first.headcount[1, 0] += 1  # the run is module-scoped: restore it afterwards
    try:
        assert results_digest(run.simulated) != digest
    finally:
        first.headcount[1, 0] -= 1


def test_the_summary_reports_outcomes_without_gating(run: StudyRun) -> None:
    summary = {m.name: m for m in run.metrics if m.evaluator == "summary"}
    assert all(m.threshold is None for m in summary.values())
    headcount = run.simulated.headcount.sum(axis=2)
    assert summary["headcount_start"].value == headcount[0, 0]
    assert summary["headcount_end_mean"].value == pytest.approx(headcount[:, -1].mean())
    assert summary["attrition_mean"].value == pytest.approx(
        run.simulated.flow("A").sum(axis=(1, 2)).mean()
    )
    assert summary["tranches_mean"].value > 0


def test_a_single_trajectory_reports_no_spread(inputs: tuple[Path, Path]) -> None:
    profile = StudyProfile.from_mapping(
        {
            "name": "one",
            "seed": 3,
            "trajectories": 1,
            "months": 2,
            "mode": "unguided",
            "scope": "hcf",
        }
    )
    metrics = {m.name: m.value for m in run_study(profile, inputs[0]).metrics}
    assert metrics["headcount_end_sd"] == 0.0
    assert "sbc_recognized_mean" not in metrics
