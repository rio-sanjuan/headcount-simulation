"""Tests for greedy reconciliation and the lite guidance pipeline."""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path

import duckdb
import numpy as np
import pytest

from hcsim.cli import main as hcsim
from hcsim.guidance.pipeline import PipelineConfig, PipelineResult, run
from hcsim.guidance.reconcile import reconcile_greedy
from hcsim.model.core.categories import GuidanceCategory
from hcsim.model.core.guidance import GuidanceForecast
from hcsim.synth.config import preset
from hcsim.synth.world import generate_world, write_world

SIGNS = {"hires": +1, "attrition": -1}


def flows(hires: list[float], attrition: list[float]) -> dict[str, np.ndarray]:
    return {"hires": np.array(hires), "attrition": np.array(attrition)}


# --- reconciliation -----------------------------------------------------------------------------


def test_coherent_months_are_left_alone() -> None:
    result = reconcile_greedy(np.array([102.0]), 100.0, flows([3.0], [1.0]), SIGNS)
    assert result.flows["hires"].tolist() == [3.0]
    assert result.flows["attrition"].tolist() == [1.0]


def test_the_gap_goes_to_the_smallest_flow() -> None:
    # Headcount rises 4, flows say 2: attrition (1) is smallest, so it falls... to 0, clamped,
    # and the remaining 1 goes to hires.
    result = reconcile_greedy(np.array([104.0]), 100.0, flows([3.0], [1.0]), SIGNS, tolerance=0.0)
    assert result.flows["attrition"].tolist() == [0.0]
    assert result.flows["hires"].tolist() == [4.0]
    assert result.residual.tolist() == [0.0]


def test_a_falling_anchor_raises_outflows_or_cuts_inflows() -> None:
    result = reconcile_greedy(np.array([97.0]), 100.0, flows([1.0], [2.0]), SIGNS, tolerance=0.0)
    # Flows say -1, headcount falls 3: hires (1) is smallest and falls to 0, then attrition +1.
    assert result.flows["hires"].tolist() == [0.0]
    assert result.flows["attrition"].tolist() == [3.0]


def test_after_reconciliation_the_identity_holds_every_month() -> None:
    rng = np.random.default_rng(2)
    anchor = 100 + np.cumsum(rng.normal(0, 3, 12))
    result = reconcile_greedy(
        anchor, 100.0, flows(list(rng.uniform(0, 4, 12)), list(rng.uniform(0, 4, 12))), SIGNS, 0.0
    )
    change = np.diff(np.r_[100.0, anchor])
    net = result.flows["hires"] - result.flows["attrition"]
    assert np.allclose(change, net)
    assert all((v >= 0).all() for v in result.flows.values())


# --- the pipeline -------------------------------------------------------------------------------


@pytest.fixture(scope="module")
def world_path(tmp_path_factory: pytest.TempPathFactory) -> Path:
    path = tmp_path_factory.mktemp("world") / "world.duckdb"
    write_world(generate_world(preset("tiny", seed=8)), path)
    return path


@pytest.fixture(scope="module")
def result(world_path: Path) -> Iterator[PipelineResult]:
    with duckdb.connect(str(world_path), read_only=True) as con:
        yield run(con, PipelineConfig(horizon=12))


def test_every_group_gets_every_category_for_the_horizon(result: PipelineResult) -> None:
    artifact = result.artifact
    assert artifact.groups == tuple(f"CC-{i:02d}" for i in range(1, 11))
    assert artifact.categories == frozenset(GuidanceCategory)
    assert artifact.horizon == 12
    assert artifact.snapshot_date == "2025-12-31"
    assert len(artifact.frame) == 10 * 4 * 12
    assert result.report["series_skipped"] == []


def test_reconciled_flows_satisfy_the_identity_against_the_anchor(
    result: PipelineResult, world_path: Path
) -> None:
    artifact = result.artifact
    with duckdb.connect(str(world_path), read_only=True) as con:
        last = dict(
            con.execute(
                "SELECT cost_center, count(*) FROM workforce_txn "
                "WHERE trans_type='ACTIVE' AND event_date = DATE '2025-12-31' GROUP BY ALL"
            ).fetchall()
        )
    for group in artifact.groups:
        index = artifact.groups.index(group)
        anchor = np.array(
            [artifact.get_quota(t, GuidanceCategory.HEADCOUNT)[index] for t in range(12)]
        )
        hires = np.array([artifact.get_quota(t, GuidanceCategory.HIRES)[index] for t in range(12)])
        exits = np.array(
            [artifact.get_quota(t, GuidanceCategory.ATTRITION)[index] for t in range(12)]
        )
        change = np.diff(np.r_[float(last[group]), anchor])
        assert np.all(np.abs(change - (hires - exits)) <= 1.0 + 1e-9), group


def test_the_report_records_the_partial_identity_and_clamping(result: PipelineResult) -> None:
    report = result.report
    assert report["fingerprint"] == result.artifact.fingerprint()
    assert report["reconciliation"]["identity"] == "partial"
    assert "internal_starts" in report["reconciliation"]["missing_terms"]
    assert report["clamped_negative_cells"] >= 0
    assert len(report["selection"]) == 40


def test_the_artifact_saves_and_loads(result: PipelineResult, tmp_path: Path) -> None:
    result.artifact.save(tmp_path / "g")
    assert GuidanceForecast.load(tmp_path / "g").fingerprint() == result.artifact.fingerprint()


def test_a_coarser_grouping_and_no_reconciliation(world_path: Path) -> None:
    with duckdb.connect(str(world_path), read_only=True) as con:
        topline = run(con, PipelineConfig(horizon=6, grouping="TOPLINE", reconcile=False))
    assert topline.artifact.groups == ("ALL",)
    assert topline.report["reconciliation"]["max_abs_residual"] == 0.0


# --- the command --------------------------------------------------------------------------------


def test_guidance_fit_writes_the_artifact_and_report(
    world_path: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    out = tmp_path / "artifact"
    assert hcsim(["guidance", "fit", "--world", str(world_path), "--out", str(out)]) == 0
    assert "wrote gf1-" in capsys.readouterr().out
    assert GuidanceForecast.load(out).horizon == 12
    assert json.loads((out / "report.json").read_text())["grouping"] == "cost_center"


def test_guidance_fit_refuses_a_missing_world(tmp_path: Path) -> None:
    with pytest.raises(SystemExit) as stop:
        hcsim(["guidance", "fit", "--world", str(tmp_path / "none.duckdb"), "--out", str(tmp_path)])
    assert stop.value.code == 2
