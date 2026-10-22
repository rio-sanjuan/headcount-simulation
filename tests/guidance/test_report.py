"""Tests for the guidance run report, its stamps, plots, and step offsets."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import duckdb
import pandas as pd
import pytest

from hcsim.cli import main as hcsim
from hcsim.guidance.config import GuidanceConfig
from hcsim.guidance.pipeline import PipelineResult, run
from hcsim.guidance.report import config_digest, markdown, steps, write_report
from hcsim.synth.config import preset
from hcsim.synth.world import generate_world, write_world


@pytest.fixture(scope="module")
def world(tmp_path_factory: pytest.TempPathFactory) -> Path:
    path = tmp_path_factory.mktemp("world") / "world.duckdb"
    write_world(generate_world(preset("tiny", seed=4)), path)
    return path


@pytest.fixture(scope="module")
def con(world: Path) -> Iterator[duckdb.DuckDBPyConnection]:
    with duckdb.connect(str(world), read_only=True) as connection:
        yield connection


UNCERTAIN = {
    "horizon": 6,
    "grouping": "TOPLINE",
    "models": ["native.NegativeBinomialGLM", "native.PoissonGLM"],
    "uncertainty": {"enabled": True, "paths": 50},
}


@pytest.fixture(scope="module")
def result(con: duckdb.DuckDBPyConnection) -> PipelineResult:
    return run(con, GuidanceConfig.from_mapping(UNCERTAIN))


def test_the_artifact_s_months_are_step_offsets_from_the_snapshot(result: PipelineResult) -> None:
    frame = result.artifact.frame
    for _, rows in frame.groupby(["TOPLINE", "category"]):
        assert sorted(rows["forecast_month"]) == list(range(6))
    assert steps(pd.Period("2025-12", "M"), 3) == [(0, "2026-01"), (1, "2026-02"), (2, "2026-03")]


def test_the_report_carries_the_run_s_stamps(result: PipelineResult) -> None:
    text, _ = markdown(result)
    assert f"| fingerprint | {result.artifact.fingerprint()} |" in text
    assert f"| configuration | {result.report['configuration']} |" in text
    assert "| step offsets | 0 = 2026-01 … 5 = 2026-06 |" in text
    assert "| source | duckdb:connection |" in text
    assert "## Reconciliation" in text
    assert "## Uncertainty" in text


def test_a_configuration_s_digest_follows_its_settings() -> None:
    one = GuidanceConfig.from_mapping(UNCERTAIN)
    assert config_digest(one) == config_digest(GuidanceConfig.from_mapping(UNCERTAIN))
    other = GuidanceConfig.from_mapping({**UNCERTAIN, "horizon": 7})
    assert config_digest(other) != config_digest(one)
    assert config_digest(one).startswith("gc1-")


def test_each_category_is_plotted_with_its_quantile_band(result: PipelineResult) -> None:
    _, figures = markdown(result)
    assert set(figures) == {c.value for c in result.artifact.categories}
    assert "history" in figures["headcount"]
    assert "q90" in figures["headcount"]


def test_the_report_is_the_same_every_time(
    con: duckdb.DuckDBPyConnection, result: PipelineResult, tmp_path: Path
) -> None:
    again = run(con, GuidanceConfig.from_mapping(UNCERTAIN))
    assert markdown(again) == markdown(result)
    path = write_report(result, tmp_path / "r")
    assert path.read_text() == markdown(result)[0]
    assert sorted(p.name for p in (tmp_path / "r" / "figures").iterdir()) == sorted(
        f"{c.value}.svg" for c in result.artifact.categories
    )


def test_the_command_writes_the_report_beside_the_artifact(world: Path, tmp_path: Path) -> None:
    out = tmp_path / "artifact"
    assert (
        hcsim(["guidance", "fit", "--world", str(world), "--out", str(out), "--horizon", "3"]) == 0
    )
    assert (out / "report.md").read_text().startswith("# Guidance run")
    assert (out / "figures" / "headcount.svg").is_file()
