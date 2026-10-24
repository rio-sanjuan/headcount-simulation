"""Tests for the study catalog, the SBC accuracy checkpoint, and the independent-sampling
baseline."""

from __future__ import annotations

import json
import re
from pathlib import Path

import pandas as pd
import pytest

from hcsim.cli import main
from hcsim.studies import catalog
from hcsim.studies import checkpoint as checkpoints
from hcsim.studies.catalog import BY_NAME, ENTRIES, MANIFEST, Guidance, Window
from hcsim.studies.harness import run_study
from hcsim.studies.profile import StudyProfile
from hcsim.studies.scorecard import SCORECARD_FILE, check


@pytest.fixture(scope="module")
def written(inputs: tuple[Path, Path], tmp_path_factory: pytest.TempPathFactory) -> Path:
    out = tmp_path_factory.mktemp("catalog")
    catalog.write(inputs[0], out)
    return out


@pytest.fixture(scope="module")
def replayed(
    inputs: tuple[Path, Path], written: Path, tmp_path_factory: pytest.TempPathFactory
) -> Path:
    """The SBC accuracy replay, run through the command."""
    (study,) = [w for w in catalog.load(written) if w.name == "sbc_accuracy_replay"]
    out = tmp_path_factory.mktemp("replay")
    assert main(study.args(inputs[0], out)) == study.expect
    return out / study.name


# --- the catalog ------------------------------------------------------------------------------


def test_the_catalog_holds_every_study_the_plan_names() -> None:
    names = [e.name for e in ENTRIES]
    assert len(names) == len(set(names)) == len(BY_NAME)
    alignment = [n for n in names if n.startswith("alignment_")]
    assert len(alignment) == 7
    assert {
        "fidelity_ceiling",
        "sbc_accuracy_replay",
        "variation",
        "dispersion",
        "regime_comparison",
        "backtest",
        "domain_sweeps",
        "subsets",
        "award_sizing",
        "why_quota_sampling",
    } <= set(names)
    assert all(e.doc and e.expect in (0, 2, 3, 4, 5) for e in ENTRIES)


def test_writing_sets_each_window_from_the_world(written: Path) -> None:
    manifest = json.loads((written / MANIFEST).read_text())
    assert [r["name"] for r in manifest] == [e.name for e in ENTRIES]
    for row in manifest:
        entry = BY_NAME[row["name"]]
        profile = StudyProfile.load(Path(row["profile"]))
        assert profile.name == entry.name
        # The tiny world's last snapshot is 2025-12-31.
        expected = "2025-07-01" if entry.window is Window.CLOSED else "2026-01-01"
        assert str(profile.run_date) == expected, entry.name
        assert (row["guidance"] is None) == (entry.guidance is Guidance.NONE)
        if row["guidance"]:
            assert Path(row["guidance"]).exists()
    demand = StudyProfile.load(written / "profiles" / "alignment_demand.yaml")
    assert demand.plan is not None
    assert demand.plan.path == str(written / "inputs" / "plan")
    supply = StudyProfile.load(written / "profiles" / "alignment_supply.yaml")
    assert supply.plan is not None
    assert supply.plan.synthetic is not None
    assert len(supply.plan.synthetic["dates"]) == catalog.MONTHS  # a target every month
    subsets = StudyProfile.load(written / "profiles" / "subsets.yaml")
    assert subsets.subset == {"cost_center": ["CC-01", "CC-02"]}


def test_a_study_can_be_written_alone(inputs: tuple[Path, Path], tmp_path: Path) -> None:
    (only,) = catalog.write(inputs[0], tmp_path, ["backtest"])
    assert only.guidance is None
    assert only.args(inputs[0], tmp_path / "runs")[:3] == ["study", "run", "--profile"]
    assert not (tmp_path / "inputs" / "guidance").exists()
    with pytest.raises(ValueError, match=re.escape("no catalog studies ['nope']")):
        catalog.write(inputs[0], tmp_path, ["nope"])


def test_the_catalog_commands(
    inputs: tuple[Path, Path], tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert main(["study", "catalog", "list"]) == 0
    assert "why_quota_sampling" in capsys.readouterr().out
    out = tmp_path / "catalog"
    args = ["--world", str(inputs[0])]
    assert main(["study", "catalog", "write", *args, "--out", str(out), "--study", "subsets"]) == 0
    runs = ["study", "catalog", "run", *args, "--catalog", str(out), "--out", str(tmp_path / "r")]
    assert main(runs) == 0
    assert "ok        subsets: exited 0, documented 0" in capsys.readouterr().out
    manifest = json.loads((out / MANIFEST).read_text())
    manifest[0]["expect"] = 4
    (out / MANIFEST).write_text(json.dumps(manifest))
    assert main(runs) == 3
    assert "MISMATCH  subsets: exited 0, documented 4" in capsys.readouterr().out
    missing = ["study", "catalog", "run", *args, "--catalog", str(tmp_path / "x"), "--out", "y"]
    assert main(missing) == 2


# --- the checkpoint ---------------------------------------------------------------------------


def test_a_checkpoint_records_the_replays_accuracy(replayed: Path, tmp_path: Path) -> None:
    saved = checkpoints.save(replayed, tmp_path / "cp.json")
    assert set(saved["metrics"]) >= {"study/sbc.monthly_wape", "study/sbc.total_bias"}
    assert saved["results_digest"].startswith("rd1-")
    same = checkpoints.check(replayed, tmp_path / "cp.json")
    assert same.passed
    assert same.identical


def _regressed(replayed: Path, tmp_path: Path, by: float) -> Path:
    """The replay's output with its monthly WAPE worse by `by`."""
    out = tmp_path / "worse"
    out.mkdir()
    card = pd.read_csv(replayed / SCORECARD_FILE, keep_default_na=False, na_values=[""])
    card.loc[card["metric"] == "monthly_wape", "value"] += by
    card.to_csv(out / SCORECARD_FILE, index=False)
    record = json.loads((replayed / "run.json").read_text()) | {"results_digest": "rd1-other"}
    (out / "run.json").write_text(json.dumps(record))
    return out


def test_a_regression_fails_the_check(
    replayed: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    cp = tmp_path / "cp.json"
    assert main(["study", "checkpoint", "save", "--run", str(replayed), "--out", str(cp)]) == 0
    worse = _regressed(replayed, tmp_path, 0.02)
    result = checkpoints.check(worse, cp)
    assert not result.passed
    assert not result.identical
    (line,) = result.regressions
    assert line.startswith("study/sbc.monthly_wape: ")
    assert checkpoints.check(worse, cp, tolerance=0.05).passed
    check_args = ["study", "checkpoint", "check", "--run", str(worse), "--against", str(cp)]
    assert main(check_args) == 3
    assert "regressed  study/sbc.monthly_wape" in capsys.readouterr().out
    assert main([*check_args[:4], str(replayed), *check_args[5:]]) == 0
    assert "bit-identical results" in capsys.readouterr().out


def test_a_checkpoint_is_one_studys(
    replayed: Path, tmp_path: Path, written: Path, inputs: tuple[Path, Path]
) -> None:
    cp = tmp_path / "cp.json"
    checkpoints.save(replayed, cp)
    (other,) = [w for w in catalog.load(written) if w.name == "fidelity_ceiling"]
    assert main(other.args(inputs[0], tmp_path)) == 0
    another = "another study: its profile, scope, spec differ"
    with pytest.raises(checkpoints.CheckpointError, match=another):
        checkpoints.check(tmp_path / "fidelity_ceiling", cp)
    with pytest.raises(checkpoints.CheckpointError, match="reports no metrics of"):
        checkpoints.save(tmp_path / "fidelity_ceiling", tmp_path / "none.json")
    with pytest.raises(checkpoints.CheckpointError, match="can't read the run"):
        checkpoints.save(tmp_path / "missing", cp)
    with pytest.raises(checkpoints.CheckpointError, match="can't read the checkpoint"):
        checkpoints.check(replayed, tmp_path / "missing.json")
    (tmp_path / "old.json").write_text(json.dumps({"checkpoint_version": 0}))
    with pytest.raises(checkpoints.CheckpointError, match="version 1"):
        checkpoints.check(replayed, tmp_path / "old.json")
    gone = json.loads(cp.read_text())
    gone["metrics"]["study/sbc.vanished"] = 0.0
    (tmp_path / "gone.json").write_text(json.dumps(gone))
    assert checkpoints.check(replayed, tmp_path / "gone.json").regressions == (
        "study/sbc.vanished: missing from the run",
    )
    bad = ["study", "checkpoint", "check", "--run", str(replayed), "--against", "x.json"]
    assert main(bad) == 2


# --- the independent-sampling baseline --------------------------------------------------------


def test_the_baseline_draws_attrition_independently(inputs: tuple[Path, Path]) -> None:
    arms = [
        {"name": "independent", "null": True, "overrides": {"sampling.method": "independent"}},
        {"name": "quota"},
    ]
    profile = StudyProfile.from_mapping(
        {
            "name": "baseline",
            "seed": 3,
            "trajectories": 4,
            "months": 6,
            "run_date": "2025-07-01",
            "mode": "guided_actuals",
            "scope": "hcf",
            "evaluators": ["invariants", "fidelity", "dispersion"],
            "arms": arms,
        }
    )
    run = run_study(profile, inputs[0])
    independent, quota = run.arms
    for t in independent.simulated.trajectories:
        assert not [q for q in t.quotas if q.category == "attrition"]  # no quota was drawn
    assert all(
        any(q.category == "attrition" for q in t.quotas) for t in quota.simulated.trajectories
    )
    scores = {a.arm.name: {m.name: m.value for m in a.metrics} for a in run.arms}
    assert scores["quota"]["attrition_wape"] == 0
    assert scores["independent"]["attrition_wape"] > 0
    assert scores["quota"]["attrition_cv"] == 0 < scores["independent"]["attrition_cv"]


# --- the milestone's acceptance ----------------------------------------------------------------


@pytest.mark.acceptance
def test_every_catalog_study_runs_with_its_documented_exit_code(
    inputs: tuple[Path, Path], written: Path, tmp_path: Path
) -> None:
    outcomes = catalog.run(catalog.load(written), inputs[0], tmp_path, lambda a: main(a))
    assert [(o.name, o.got) for o in outcomes if not o.matched] == []
    assert len(outcomes) == len(ENTRIES)
    for o in outcomes:
        card = pd.read_csv(
            tmp_path / o.name / SCORECARD_FILE, keep_default_na=False, na_values=[""]
        )
        assert len(card), o.name
        assert check(card) == [], o.name  # every row carries its provenance
        assert card["world_preset"].astype(str).eq("tiny").all(), o.name
