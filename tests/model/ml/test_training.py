"""Tests for the temporal split, the design, and the training driver."""

from __future__ import annotations

import datetime as dt
import json
from collections.abc import Iterator
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
import pytest

from hcsim.engine.seeding import schema_from_world
from hcsim.model.core.encoding import CoverageWarning
from hcsim.model.core.schema import SimulationSchema
from hcsim.model.data.extracts import extract
from hcsim.model.data.population import REGULAR
from hcsim.model.ml.arms import ALGORITHMS, DEFAULT_MATRIX, ArmMatrix
from hcsim.model.ml.loader import load_models
from hcsim.model.ml.metrics import FAMILY_METRICS, primary
from hcsim.model.ml.persistence import load_model
from hcsim.model.ml.training import (
    RUN_FILE,
    Design,
    TemporalSplit,
    TrainingError,
    TrainingRun,
    _ClassifierTask,
    _CountTask,
    _GeneratorTask,
    train,
)
from hcsim.synth.config import preset
from hcsim.synth.world import generate_world, write_world

START, END = dt.date(2021, 1, 31), dt.date(2025, 12, 31)
SPLIT = TemporalSplit.holdout(START, END, 12)


def _world(tmp_path_factory: pytest.TempPathFactory, name: str, seed: int) -> Path:
    path = tmp_path_factory.mktemp(name) / "w.duckdb"
    write_world(generate_world(preset(name, seed=seed)), path)
    return path


@pytest.fixture(scope="module")
def con(tmp_path_factory: pytest.TempPathFactory) -> Iterator[duckdb.DuckDBPyConnection]:
    with duckdb.connect(str(_world(tmp_path_factory, "tiny", 8)), read_only=True) as connection:
        yield connection


@pytest.fixture(scope="module")
def schema(con: duckdb.DuckDBPyConnection) -> SimulationSchema:
    return schema_from_world(con)


@pytest.fixture(scope="module")
def saved(
    con: duckdb.DuckDBPyConnection,
    schema: SimulationSchema,
    tmp_path_factory: pytest.TempPathFactory,
) -> tuple[TrainingRun, Path]:
    root = tmp_path_factory.mktemp("models")
    return train(con, schema, REGULAR, SPLIT, root=root), root


def test_a_split_holds_out_the_last_months_and_keeps_its_dates_in_order() -> None:
    assert TemporalSplit(START, dt.date(2024, 12, 31), END) == SPLIT
    assert TemporalSplit.holdout(START, END, 3).cutoff == dt.date(2025, 9, 30)
    assert SPLIT.to_dict() == {"start": "2021-01-31", "cutoff": "2024-12-31", "end": "2025-12-31"}
    with pytest.raises(TrainingError, match="start < cutoff < end"):
        TemporalSplit(START, END, END)
    with pytest.raises(TrainingError, match="at least one month"):
        TemporalSplit.holdout(START, END, 0)


def test_a_split_cuts_an_extract_at_its_cutoff(con: duckdb.DuckDBPyConnection) -> None:
    growth = extract(con, "growth", REGULAR, START, END)
    early, late = SPLIT.cut(growth)
    assert early.rows + late.rows == growth.rows
    assert pd.to_datetime(early.frame["month_end"]).max() == pd.Timestamp(SPLIT.cutoff)
    assert pd.to_datetime(late.frame["month_end"]).min() > pd.Timestamp(SPLIT.cutoff)


def test_the_design_encodes_each_kind_of_column(schema: SimulationSchema) -> None:
    centers = schema.vocabulary("cost_center").labels
    frame = pd.DataFrame(
        {
            "job_level": [3, 1],
            "cost_center": [centers[2], centers[0]],
            "next_month": [1, 12],
            "departure": ["XFER_OUT", "TERM"],
            "is_manager": [True, False],
            "tenure_months": [7, 30],
        }
    )
    design = Design.fit(frame, list(frame.columns), schema)
    x = design.matrix(frame, schema)
    level = schema.vocabulary("job_level")
    np.testing.assert_array_equal(x[:, 0], [level.code(3), level.code(1)])
    np.testing.assert_array_equal(x[:, 1:], [[2, 0, 1, 1, 7], [0, 11, 0, 0, 30]])
    # Job level is ordinal; the cost center, the month ahead, and text columns are categories.
    assert [design.columns[i] for i in design.categorical] == [
        "cost_center",
        "next_month",
        "departure",
    ]
    assert design.levels["departure"] == 2
    narrow = design.mask(5)  # the tiny world's 10 cost centers and the 12 months are too wide
    assert narrow.columns == (0, 3, 4, 5)
    assert narrow.categorical == (3,)
    assert design.to_dict()["local"] == {"departure": ["TERM", "XFER_OUT"]}
    # A label outside the vocabulary encodes as -1, and the gap is reported.
    unknown = frame.assign(cost_center=["CC-NOWHERE", centers[0]], departure=["RIF", "TERM"])
    with pytest.warns(CoverageWarning, match="cost_center: 1 of 2 rows"):
        y = design.matrix(unknown, schema)
    np.testing.assert_array_equal(y[:, [1, 3]], [[-1, -1], [0, 0]])


def test_every_domain_is_trained_scored_and_selected(saved: tuple[TrainingRun, Path]) -> None:
    run, _ = saved
    assert sorted(run.domains) == DEFAULT_MATRIX.domains
    assert run.population == str(REGULAR.stamp)
    for name, result in run.domains.items():
        arms = DEFAULT_MATRIX.domain(name)
        assert list(result.arms) == [a.name for a in arms]
        assert result.train_rows > 0
        assert result.test_rows > 0
        assert len(result.selection.decisions) == len(arms) - 1
        assert result.champion.arm.name == result.selection.champion
        metric = primary(result.family)
        for arm in result.arms.values():
            assert set(arm.metrics) <= set(FAMILY_METRICS[result.family])
            assert np.isfinite(arm.primary)
        # A richer arm takes over only by doing better on the held-out rows.
        for decision in result.selection.decisions:
            better = metric.better(
                result.arms[decision.arm].primary, result.arms[decision.against].primary
            )
            assert not decision.cleared or better


def test_a_saved_run_loads_with_its_probes(saved: tuple[TrainingRun, Path]) -> None:
    run, root = saved
    assert json.loads((root / RUN_FILE).read_text()) == json.loads(json.dumps(run.to_dict()))
    assert sorted(run.chosen) == DEFAULT_MATRIX.domains
    report = load_models(root, run.chosen, DEFAULT_MATRIX, run.probes)
    assert report.failures == {}
    assert sorted(report.models) == DEFAULT_MATRIX.domains
    for name, result in run.domains.items():
        flags = []
        for arm in result.arms.values():
            assert arm.version is not None
            _, artifact = load_model(root / name / arm.version)
            assert artifact.config["arm"] == arm.arm.name
            assert artifact.config["split"] == SPLIT.to_dict()
            assert artifact.metadata["test_rows"] == result.test_rows
            flags.append(artifact.metadata["champion"])
        assert sum(flags) == 1


def test_training_is_reproducible(
    con: duckdb.DuckDBPyConnection,
    schema: SimulationSchema,
    saved: tuple[TrainingRun, Path],
    tmp_path: Path,
) -> None:
    again = train(con, schema, REGULAR, SPLIT, root=tmp_path)
    assert again.chosen == saved[0].chosen
    # Without a root nothing is saved, and nothing names a version.
    unsaved = train(con, schema, REGULAR, SPLIT, domains=["growth"])
    assert list(unsaved.domains) == ["growth"]
    assert unsaved.chosen == {}


def test_every_allowed_algorithm_has_a_fit() -> None:
    for task in (_ClassifierTask, _GeneratorTask, _CountTask):
        assert set(task.FITS) == ALGORITHMS[task.family]
        assert all(callable(getattr(task, method)) for method in task.FITS.values())


def test_a_metric_a_held_out_side_can_t_define_is_left_out(
    con: duckdb.DuckDBPyConnection, schema: SimulationSchema
) -> None:
    month = TemporalSplit(START, dt.date(2025, 11, 30), END)
    run = train(con, schema, REGULAR, month, domains=["cancellation"])
    result = run.domains["cancellation"]
    assert result.test_rows > 0
    for arm in result.arms.values():
        # One month holds no unfilled exit, so rank metrics have nothing to rank.
        assert arm.metrics["auc"] is None
        assert arm.metrics["log_loss"] is not None


def test_bad_domains_and_arms_are_refused(
    con: duckdb.DuckDBPyConnection, schema: SimulationSchema
) -> None:
    with pytest.raises(ValueError, match="no arms for domain 'tenure'"):
        train(con, schema, REGULAR, SPLIT, domains=["tenure"])
    odd = ArmMatrix.from_mapping(
        {
            "attrition": [
                {"name": "a.rate", "family": "classifier", "algorithm": "naive_rate", "null": True},
                {"name": "a.backfill", "family": "classifier", "algorithm": "backfill_rate"},
            ],
            "growth": [
                {"name": "g.markov", "family": "generator", "algorithm": "markov", "null": True}
            ],
            "performance": [
                {
                    "name": "p.roles",
                    "family": "generator",
                    "algorithm": "empirical_attributes",
                    "null": True,
                }
            ],
        }
    )
    run = train(con, schema, REGULAR, SPLIT, matrix=odd, domains=["attrition"])
    assert set(run.domains["attrition"].arms) == {"a.rate", "a.backfill"}
    with pytest.raises(TrainingError, match="no generator task for a cost_center_month"):
        train(con, schema, REGULAR, SPLIT, matrix=odd, domains=["growth"])
    with pytest.raises(TrainingError, match=r"p\.roles draws whole roles"):
        train(con, schema, REGULAR, SPLIT, matrix=odd, domains=["performance"])


@pytest.mark.parametrize("domain", ["attrition", "growth", "performance"])
def test_a_domain_needs_outcomes_on_both_sides_of_the_cutoff(
    con: duckdb.DuckDBPyConnection, schema: SimulationSchema, domain: str
) -> None:
    after_the_data = TemporalSplit(
        dt.date(2026, 6, 30), dt.date(2026, 9, 30), dt.date(2026, 12, 31)
    )
    with pytest.raises(TrainingError, match="both sides of the cutoff"):
        train(con, schema, REGULAR, after_the_data, domains=[domain])


@pytest.mark.acceptance
def test_trained_arms_beat_their_null_where_the_world_carries_signal(
    tmp_path_factory: pytest.TempPathFactory,
) -> None:
    """M3 acceptance: on held-out months of a small world, a richer arm clears the decision
    margin against its null arm in every domain whose truth varies across rows."""
    with duckdb.connect(str(_world(tmp_path_factory, "small", 0)), read_only=True) as small:
        run = train(small, schema_from_world(small), REGULAR, SPLIT)
    for name in ("attrition", "promotion", "external_fill", "job_family", "performance"):
        result = run.domains[name]
        assert result.selection.champion != DEFAULT_MATRIX.null(name).name, name
