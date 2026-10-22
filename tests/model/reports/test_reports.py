"""Tests for the report harness, its provenance, and the metrics sidecar."""

from __future__ import annotations

import copy
import datetime as dt
import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import duckdb
import pytest

from hcsim.engine.seeding import schema_from_world
from hcsim.model.data.population import REGULAR
from hcsim.model.ml.training import TemporalSplit, TrainingRun, train
from hcsim.model.reports.harness import (
    FIGURES,
    REPORT_FILE,
    Report,
    Section,
    number,
    table,
    training_report,
)
from hcsim.model.reports.sidecar import (
    SIDECAR_FILE,
    MetricsSidecar,
    Provenance,
    SidecarError,
    provenance,
)
from hcsim.synth.config import preset
from hcsim.synth.world import generate_world, write_world

SPLIT = TemporalSplit.holdout(dt.date(2021, 1, 31), dt.date(2025, 12, 31))
SVG = '<svg xmlns="http://www.w3.org/2000/svg" width="10" height="10"></svg>\n'


@pytest.fixture(scope="module")
def con(tmp_path_factory: pytest.TempPathFactory) -> Iterator[duckdb.DuckDBPyConnection]:
    path = tmp_path_factory.mktemp("world") / "w.duckdb"
    write_world(generate_world(preset("tiny", seed=8)), path)
    with duckdb.connect(str(path), read_only=True) as connection:
        yield connection


@pytest.fixture(scope="module")
def run(con: duckdb.DuckDBPyConnection, tmp_path_factory: pytest.TempPathFactory) -> TrainingRun:
    return train(con, schema_from_world(con), REGULAR, SPLIT, root=tmp_path_factory.mktemp("m"))


@pytest.fixture(scope="module")
def sidecar_data(con: duckdb.DuckDBPyConnection, run: TrainingRun) -> dict[str, Any]:
    return MetricsSidecar.of(run, provenance(con, run)).model_dump(mode="json")


def test_provenance_names_the_world_the_run_and_its_models(
    con: duckdb.DuckDBPyConnection, run: TrainingRun
) -> None:
    record = provenance(con, run)
    assert record.source == "synthetic_world"
    assert (record.world_preset, record.world_seed) == ("tiny", "8")
    assert record.contract_version == "1.0"
    assert record.population == str(REGULAR.stamp)
    assert record.schema_fingerprint == run.schema
    assert (record.split_start, record.split_cutoff, record.split_end) == (
        SPLIT.start,
        SPLIT.cutoff,
        SPLIT.end,
    )
    assert record.decision_z == run.z
    assert record.models == run.chosen
    # A warehouse without a truth schema is a warehouse, with no world to name.
    with duckdb.connect() as bare:
        assert provenance(bare, run).source == "warehouse"


def test_provenance_that_doesn_t_hold_is_refused(
    con: duckdb.DuckDBPyConnection, run: TrainingRun
) -> None:
    fields = provenance(con, run).model_dump()
    bad = fields | {
        "population": "regular",
        "schema_fingerprint": "abc",
        "split_cutoff": SPLIT.end,
        "models": {"attrition": "v2"},
        "world_seed": "",
    }
    with pytest.raises(ValueError, match="name@digest") as caught:
        Provenance(**bad)
    for reason in (
        "isn't a schema fingerprint",
        "start < cutoff < end",
        "for ['attrition']",
        "names its preset and seed",
    ):
        assert reason in str(caught.value)
    unsaved = TrainingRun(SPLIT, "regular", run.schema, {}, 0.0)
    with pytest.raises(SidecarError, match="decision_z: Input should be greater than 0"):
        provenance(con, unsaved)


def test_a_sidecar_round_trips_and_matches_the_run(
    con: duckdb.DuckDBPyConnection, run: TrainingRun, tmp_path: Path
) -> None:
    sidecar = MetricsSidecar.of(run, provenance(con, run))
    path = sidecar.write(tmp_path)
    assert path == tmp_path / SIDECAR_FILE
    assert MetricsSidecar.load(tmp_path) == sidecar
    for name, result in run.domains.items():
        domain = sidecar.domains[name]
        assert domain.champion == result.selection.champion
        assert domain.arms[domain.champion].version == run.chosen[name]
        assert domain.primary_metric in {"log_loss", "log_likelihood", "wape"}


def _broken(data: dict[str, Any]) -> dict[str, Any]:
    data = copy.deepcopy(data)
    attrition = data["domains"]["attrition"]
    attrition["primary_metric"] = "auc"
    attrition["champion"] = "attrition.missing"
    attrition["arms"]["attrition.constant"]["metrics"]["wape"] = 0.5
    attrition["arms"]["attrition.constant"]["version"] = "v9"
    attrition["decisions"][0] |= {"against": "attrition.other", "cleared": True, "mean_gain": -1}
    data["domains"]["growth"]["family"] = "regression"
    return data


def test_a_sidecar_that_doesn_t_hold_is_refused_with_every_reason(
    sidecar_data: dict[str, Any],
) -> None:
    with pytest.raises(SidecarError, match="invalid metrics sidecar") as caught:
        MetricsSidecar.validated(_broken(sidecar_data))
    message = str(caught.value)
    for reason in (
        "a classifier's primary metric is log_loss",
        "the champion 'attrition.missing' isn't among the arms",
        "attrition.constant has metrics outside its family: ['wape']",
        "attrition.constant's version 'v9' isn't a model version",
        "names other arms",
        "cleared the margin without gaining",
        "no model family 'regression'",
    ):
        assert reason in message, reason
    # Values must be finite, and nothing unknown rides along.
    odd = copy.deepcopy(sidecar_data)
    odd["domains"]["attrition"]["arms"]["attrition.constant"]["primary"] = float("inf")
    odd["notes"] = "extra"
    with pytest.raises(SidecarError, match="finite number") as caught:
        MetricsSidecar.validated(odd)
    assert "notes: Extra inputs are not permitted" in str(caught.value)


def test_a_sidecar_s_provenance_must_name_its_champions(sidecar_data: dict[str, Any]) -> None:
    data = copy.deepcopy(sidecar_data)
    data["provenance"]["models"]["attrition"] = "v1-000000000000"
    data["provenance"]["models"]["tenure"] = "v1-000000000000"
    with pytest.raises(SidecarError) as caught:
        MetricsSidecar.validated(data)
    assert "attrition's provenance version isn't its champion's" in str(caught.value)
    assert "a model for tenure, which has no metrics" in str(caught.value)


def test_an_unreadable_sidecar_is_refused(tmp_path: Path) -> None:
    with pytest.raises(SidecarError, match="can't read a metrics sidecar"):
        MetricsSidecar.load(tmp_path)
    (tmp_path / SIDECAR_FILE).write_text("{")
    with pytest.raises(SidecarError, match="can't read a metrics sidecar"):
        MetricsSidecar.load(tmp_path)


def test_the_training_report_puts_provenance_first_and_covers_every_domain(
    con: duckdb.DuckDBPyConnection, run: TrainingRun, tmp_path: Path
) -> None:
    report, sidecar = training_report(run, provenance(con, run))
    path = report.write(tmp_path / "report", sidecar)
    text = path.read_text()
    assert path.name == REPORT_FILE
    headings = [line for line in text.splitlines() if line.startswith("## ")]
    assert headings[:2] == ["## Provenance", "## Selection"]
    assert headings[2:] == [f"## Domain: {name}" for name in run.domains]
    assert f"| schema_fingerprint | {run.schema} |" in text
    for name, version in run.chosen.items():
        assert f"| model: {name} | {version} |" in text
    assert text.count("### Lift of the champion") == sum(
        r.lift is not None for r in run.domains.values()
    )
    assert json.loads((tmp_path / "report" / SIDECAR_FILE).read_text())["sidecar_version"] == 1
    # The same run writes the same report.
    again, sidecar_again = training_report(run, provenance(con, run))
    assert again.write(tmp_path / "again", sidecar_again).read_text() == text


def test_sections_carry_figures_and_refuse_clashes(
    con: duckdb.DuckDBPyConnection, run: TrainingRun, tmp_path: Path
) -> None:
    record = provenance(con, run)
    report = Report("Recovery", record).add(Section("Fit", "![fit](figures/fit.svg)", {"fit": SVG}))
    with pytest.raises(SidecarError, match=r"already in the report: \['fit'\]"):
        report.add(Section("Again", "", {"fit": SVG}))
    with pytest.raises(SidecarError, match="lowercase words"):
        Section("Bad", "", {"Fit Plot": SVG})
    sidecar = MetricsSidecar.of(run, record)
    report.write(tmp_path, sidecar)
    assert (tmp_path / FIGURES / "fit.svg").read_text() == SVG
    other = MetricsSidecar.of(run, record.model_copy(update={"world_seed": "9"}))
    with pytest.raises(SidecarError, match="isn't the report's"):
        report.write(tmp_path, other)
    # A report with no figures writes no figure folder.
    Report("Plain", record).write(tmp_path / "plain", sidecar)
    assert not (tmp_path / "plain" / FIGURES).exists()


def test_cells_format_numbers_flags_and_gaps() -> None:
    assert [number(v) for v in (3, 0.123456, True, None, float("nan"), "a")] == [
        "3",
        "0.1235",
        "yes",
        "—",
        "—",
        "a",
    ]
    assert table(["a", "b"], [["x|y", 1.0]]) == "| a | b |\n|---|---|\n| x\\|y | 1 |"
