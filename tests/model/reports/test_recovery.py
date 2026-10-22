"""Tests for ground-truth recovery and its SVG figures."""

from __future__ import annotations

import dataclasses
import datetime as dt
import json
from collections.abc import Iterator
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
import pytest

from hcsim.engine.seeding import schema_from_world
from hcsim.model.core.schema import SimulationSchema
from hcsim.model.data.population import EVERY_ACCOUNT, REGULAR, PopulationError
from hcsim.model.ml.arms import ArmMatrix
from hcsim.model.ml.training import TemporalSplit, TrainingRun, train
from hcsim.model.reports.harness import FIGURES, training_report
from hcsim.model.reports.recovery import ROW_COLUMNS, Recovery, recover
from hcsim.model.reports.sidecar import MetricsSidecar, SidecarError, provenance
from hcsim.model.reports.svg import ChartError, Series, chart, ticks
from hcsim.synth.config import preset
from hcsim.synth.world import generate_world, write_world

SPLIT = TemporalSplit.holdout(dt.date(2021, 1, 31), dt.date(2025, 12, 31))


def _connect(tmp_path_factory: pytest.TempPathFactory, name: str, seed: int) -> Path:
    path = tmp_path_factory.mktemp(name) / "w.duckdb"
    write_world(generate_world(preset(name, seed=seed)), path)
    return path


@pytest.fixture(scope="module")
def con(tmp_path_factory: pytest.TempPathFactory) -> Iterator[duckdb.DuckDBPyConnection]:
    with duckdb.connect(str(_connect(tmp_path_factory, "tiny", 8)), read_only=True) as connection:
        yield connection


@pytest.fixture(scope="module")
def schema(con: duckdb.DuckDBPyConnection) -> SimulationSchema:
    return schema_from_world(con)


@pytest.fixture(scope="module")
def run(con: duckdb.DuckDBPyConnection, schema: SimulationSchema) -> TrainingRun:
    return train(con, schema, REGULAR, SPLIT)


def test_every_domain_with_a_truth_is_recovered(
    con: duckdb.DuckDBPyConnection, schema: SimulationSchema, run: TrainingRun
) -> None:
    found = {r.name: r for r in recover(con, run, schema, REGULAR)}
    assert list(found) == ["attrition", "performance", "requisitions", "backfill"]
    (manifest,) = con.execute("SELECT manifest FROM truth.manifest").fetchone() or ("{}",)
    config = json.loads(manifest)["config"]
    for recovery in found.values():
        assert tuple(recovery.rows.columns) == ROW_COLUMNS
        assert np.isfinite(recovery.rows[["truth", "fitted"]].to_numpy(np.float64)).all()
        assert recovery.figures
        assert all(svg.startswith("<svg") for svg in recovery.figures.values())
    # The truth column is the world's own parameters.
    rows = found["performance"].rows.set_index("quantity")
    transitions = np.asarray(config["ratings"]["transitions"])
    assert rows.loc["R3 → R4", "truth"] == transitions[2, 3]
    assert rows.loc["entry R1", "truth"] == config["ratings"]["entry_distribution"][0]
    requisitions = found["requisitions"].rows.set_index("quantity")
    exits = config["requisitions"]["unfilled_exit_hazard"]
    assert requisitions.loc["unfilled exit, age 3+", "truth"] == exits[-1]
    assert set(found["requisitions"].figures) == {"cancellation", "external-fill"}
    levels = found["backfill"].rows["quantity"].str.contains("level")
    assert levels.sum() >= 2


def test_recovery_goes_in_the_report_and_its_sidecar(
    con: duckdb.DuckDBPyConnection,
    schema: SimulationSchema,
    run: TrainingRun,
    tmp_path: Path,
) -> None:
    recoveries = recover(con, run, schema, REGULAR)
    report, sidecar = training_report(run, provenance(con, run), recoveries)
    text = report.write(tmp_path, sidecar).read_text()
    for recovery in recoveries:
        assert f"## Recovery: {recovery.title}" in text
        assert len(sidecar.recovery[recovery.name]) == len(recovery.rows)
        for name in recovery.figures:
            assert f"![{name}](figures/{name}.svg)" in text
            assert (tmp_path / FIGURES / f"{name}.svg").is_file()
    assert MetricsSidecar.load(tmp_path) == sidecar


def test_recovery_needs_a_truth_and_the_run_s_population(
    con: duckdb.DuckDBPyConnection, schema: SimulationSchema, run: TrainingRun
) -> None:
    with pytest.raises(PopulationError, match="built for population regular"):
        recover(con, run, schema, EVERY_ACCOUNT)
    with duckdb.connect() as bare, pytest.raises(SidecarError, match="lacks"):
        recover(bare, run, schema, REGULAR)
    # A warehouse's sidecar can't carry recovery rows.
    data = MetricsSidecar.of(run, provenance(con, run)).model_dump(mode="json")
    data["provenance"] |= {"source": "warehouse"}
    data["recovery"] = {"x": [{"quantity": "q", "truth": 1.0, "fitted": 1.0}]}
    with pytest.raises(SidecarError, match="recovery needs a synthetic world's truth"):
        MetricsSidecar.validated(data)


def test_a_run_recovers_only_what_its_arms_allow(
    con: duckdb.DuckDBPyConnection, schema: SimulationSchema
) -> None:
    matrix = ArmMatrix.from_mapping(
        {
            "performance": [
                {
                    "name": "p.overall",
                    "family": "generator",
                    "algorithm": "conditional_categorical",
                    "null": True,
                }
            ],
            "external_fill": [
                {
                    "name": "x.constant",
                    "family": "classifier",
                    "algorithm": "constant",
                    "null": True,
                }
            ],
        }
    )
    run = train(con, schema, REGULAR, SPLIT, matrix=matrix)
    found = recover(con, run, schema, REGULAR)
    assert [r.name for r in found] == ["requisitions"]
    assert set(found[0].figures) == {"external-fill"}
    assert found[0].rows["quantity"].str.startswith("external fill").all()


def test_a_run_without_fills_skips_requisitions_and_a_classifier_needs_its_design(
    con: duckdb.DuckDBPyConnection, schema: SimulationSchema, run: TrainingRun
) -> None:
    only = TrainingRun(run.split, run.population, run.schema, {"backfill": run.domains["backfill"]})
    assert [r.name for r in recover(con, only, schema, REGULAR)] == ["backfill"]
    bare = dataclasses.replace(run.domains["backfill"], design=None)
    broken = TrainingRun(run.split, run.population, run.schema, {"backfill": bare})
    with pytest.raises(SidecarError, match="backfill has no design"):
        recover(con, broken, schema, REGULAR)


def test_recovery_rows_need_their_columns() -> None:
    with pytest.raises(SidecarError, match="need the columns"):
        Recovery("x", "x", pd.DataFrame({"quantity": ["a"]}), "")
    rows = pd.DataFrame([("a", 1.0, 0.5, None)], columns=list(ROW_COLUMNS))
    assert Recovery("x", "x", rows, "").records() == [
        {"quantity": "a", "truth": 1.0, "fitted": 0.5, "observed": None}
    ]


def test_ticks_are_round_and_cover_the_range() -> None:
    assert ticks(0.0, 1.0) == [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    assert ticks(0.013, 0.087) == [0.02, 0.04, 0.06, 0.08]
    assert ticks(-3.0, 3.0) == [-2.0, 0.0, 2.0]


def test_a_chart_draws_every_series_with_a_legend() -> None:
    svg = chart(
        [Series("fit <a>", [0.1, 0.2], [0.1, 0.3]), Series("line", [0, 1], [2, 2], "line")],
        title="T & U",
        x_label="x",
        y_label="y",
    )
    assert svg.startswith("<svg")
    assert svg.endswith("</svg>\n")
    assert "fit &lt;a&gt;" in svg
    assert "T &amp; U" in svg
    assert svg.count("<circle") == 4
    assert svg.count("<polyline") == 1
    assert "stroke-dasharray" not in svg
    same = chart([Series("s", [1.0], [1.0])], title="", x_label="", y_label="", diagonal=True)
    assert "stroke-dasharray" in same  # the identity line, on equal axes


def test_a_chart_refuses_what_it_can_t_draw() -> None:
    with pytest.raises(ChartError, match="at least one series"):
        chart([], title="", x_label="", y_label="")
    with pytest.raises(ChartError, match="one y per x"):
        Series("s", [1.0], [])
    with pytest.raises(ChartError, match="aren't finite"):
        Series("s", [1.0], [float("nan")])
    many = [Series(str(k), [0.0], [0.0]) for k in range(7)]
    with pytest.raises(ChartError, match="at most 6 series"):
        chart(many, title="", x_label="", y_label="")


@pytest.mark.acceptance
def test_fits_recover_the_truth_where_their_form_allows(
    tmp_path_factory: pytest.TempPathFactory,
) -> None:
    """M3 acceptance: on a small world's held-out months, fits land near the truth wherever
    the model can express it."""
    with duckdb.connect(str(_connect(tmp_path_factory, "small", 0)), read_only=True) as small:
        schema = schema_from_world(small)
        trained = train(small, schema, REGULAR, SPLIT)
        found = {
            r.name: r.rows.set_index("quantity") for r in recover(small, trained, schema, REGULAR)
        }
    attrition = found["attrition"]
    everyone = attrition.loc["monthly hazard, everyone"]
    assert abs(everyone["fitted"] / everyone["truth"] - 1) < 0.1
    levels = attrition[attrition.index.str.contains("level")]
    assert (abs(levels["fitted"] / levels["truth"] - 1) < 0.25).all()
    assert (abs(found["performance"]["fitted"] - found["performance"]["truth"]) < 0.06).all()
    external = found["requisitions"][found["requisitions"].index.str.startswith("external")]
    assert (abs(external["fitted"] - external["truth"]) < 0.06).all()
    backfill = found["backfill"].loc["backfill probability, everyone"]
    assert abs(backfill["fitted"] - backfill["truth"]) < 0.05
