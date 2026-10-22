"""Tests for saving and loading model artifacts."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from hcsim.model.ml.classifiers import BoostedModel, FeatureMask, LevelOnlyModel
from hcsim.model.ml.entity import ConstantModel
from hcsim.model.ml.persistence import (
    ArtifactError,
    load_model,
    save_model,
    version_of,
    versions,
)


def data() -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(0)
    x = np.column_stack([rng.integers(0, 4, 2000), rng.uniform(0, 60, 2000)]).astype(float)
    y = (rng.random(2000) < 0.05 + 0.03 * x[:, 0]).astype(float)
    return x, y


CONFIG = {"features": ["job_level", "tenure"], "population": "regular@pd1-0", "schema": "sc2-0"}


def test_a_saved_model_loads_back_and_scores_the_same(tmp_path: Path) -> None:
    x, y = data()
    model = BoostedModel(FeatureMask((0, 1)), max_iter=20).fit(x, y)
    artifact = save_model(model, tmp_path, "attrition", CONFIG, {"rows": len(y), "window": "2024"})
    assert artifact.version.startswith("v1-")
    assert artifact.path == tmp_path / "attrition" / artifact.version
    assert artifact.config["algorithm"] == "boosted"
    loaded, read = load_model(artifact.path)
    np.testing.assert_array_equal(loaded.compute_probability(x), model.compute_probability(x))  # type: ignore[attr-defined]
    assert read.metadata == {"rows": 2000, "window": "2024"}
    assert read.version == artifact.version
    assert versions(tmp_path, "attrition") == [artifact.version]
    assert versions(tmp_path, "growth") == []


def test_the_version_names_what_the_model_is(tmp_path: Path) -> None:
    x, y = data()
    first = save_model(
        BoostedModel(FeatureMask((0, 1)), max_iter=20).fit(x, y), tmp_path, "a", CONFIG
    )
    again = save_model(
        BoostedModel(FeatureMask((0, 1)), max_iter=20).fit(x, y),
        tmp_path,
        "a",
        CONFIG,
        {"trained": "later"},
    )
    assert again.version == first.version  # metadata doesn't change what it is
    other = save_model(
        BoostedModel(FeatureMask((0, 1)), max_iter=21).fit(x, y), tmp_path, "a", CONFIG
    )
    assert other.version != first.version
    moved = save_model(
        BoostedModel(FeatureMask((0, 1)), max_iter=20).fit(x, y),
        tmp_path,
        "a",
        CONFIG | {"population": "every_account@pd1-1"},
    )
    assert moved.version != first.version


def test_a_changed_artifact_is_refused(tmp_path: Path) -> None:
    artifact = save_model(LevelOnlyModel(0).fit(*data()), tmp_path, "attrition", CONFIG)
    model_file = artifact.path / "model.joblib"
    original = model_file.read_bytes()
    model_file.write_bytes(original + b"\0")
    with pytest.raises(ArtifactError, match="holds version v1-"):
        load_model(artifact.path)
    model_file.write_bytes(original)
    config_file = artifact.path / "model_config.json"
    config = json.loads(config_file.read_text())
    config_file.write_text(json.dumps(config | {"features": ["job_level"]}))
    with pytest.raises(ArtifactError, match="not the v1-"):
        load_model(artifact.path)
    config_file.write_text(json.dumps(config | {"format_version": 9}))
    with pytest.raises(ArtifactError, match="unsupported artifact format 9"):
        load_model(artifact.path)


def test_a_model_that_isn_t_what_its_config_says_is_refused(tmp_path: Path) -> None:
    artifact = save_model(ConstantModel(0.1), tmp_path, "attrition", CONFIG)
    config = json.loads((artifact.path / "model_config.json").read_text()) | {
        "algorithm": "boosted"
    }
    renamed = artifact.path.parent / version_of(
        config, (artifact.path / "model.joblib").read_bytes()
    )
    renamed.mkdir()
    for name in ("model.joblib", "metadata.json"):
        (renamed / name).write_bytes((artifact.path / name).read_bytes())
    (renamed / "model_config.json").write_text(json.dumps(config))
    with pytest.raises(ArtifactError, match="isn't 'boosted'"):
        load_model(renamed)


def test_bad_saves_and_missing_files_are_refused(tmp_path: Path) -> None:
    with pytest.raises(ArtifactError, match="needs an ALGORITHM_NAME"):
        save_model(object(), tmp_path, "attrition", CONFIG)
    with pytest.raises(ArtifactError, match="names 'logistic', but the model is 'constant'"):
        save_model(ConstantModel(0.1), tmp_path, "attrition", CONFIG | {"algorithm": "logistic"})
    with pytest.raises(ArtifactError, match="config must be plain JSON"):
        save_model(ConstantModel(0.1), tmp_path, "attrition", {"when": object()})
    with pytest.raises(ArtifactError, match="metadata must be plain JSON"):
        save_model(ConstantModel(0.1), tmp_path, "attrition", CONFIG, {"auc": float("nan")})
    with pytest.raises(ArtifactError, match="isn't a model artifact"):
        load_model(tmp_path / "nowhere")
