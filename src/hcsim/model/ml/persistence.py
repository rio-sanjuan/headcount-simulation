"""Model artifacts: a fitted model saved with its configuration and metadata.

An artifact is a directory of three files:

- `model_config.json`: what the model is: its algorithm, its parameters, the population it was
  fitted for, the schema it encodes through, and its feature names;
- `metadata.json`: how it came to be: its training window, row counts, and metrics;
- `model.joblib`: the fitted model.

Its version, `v1-` and 12 hex characters, hashes the configuration and the model's bytes, so
the same model fitted the same way gets the same version, and the version names it in run
records. Metadata stays out of the hash: when a model was trained doesn't change what it is.
Artifacts live under a root as `<domain>/<version>/`.

Loading re-checks the version, so an artifact whose files have changed since saving is refused.
`joblib` unpickles, which can run code: load only artifacts this project wrote.
"""

from __future__ import annotations

import hashlib
import io
import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

import joblib

CONFIG_FILE: Final = "model_config.json"
METADATA_FILE: Final = "metadata.json"
MODEL_FILE: Final = "model.joblib"
FORMAT_VERSION: Final = 1


class ArtifactError(ValueError):
    """A model artifact that can't be saved or trusted."""


@dataclass(frozen=True)
class ModelArtifact:
    domain: str
    version: str
    path: Path
    config: Mapping[str, Any]
    metadata: Mapping[str, Any]


def _canonical(data: Mapping[str, Any], what: str) -> str:
    try:
        return json.dumps(data, sort_keys=True, allow_nan=False)
    except (TypeError, ValueError) as error:
        raise ArtifactError(f"the {what} must be plain JSON: {error}") from None


def _model_bytes(model: object) -> bytes:
    buffer = io.BytesIO()
    joblib.dump(model, buffer)
    return buffer.getvalue()


def version_of(config: Mapping[str, Any], payload: bytes) -> str:
    digest = hashlib.sha256(_canonical(config, "config").encode() + b"\n" + payload)
    return "v1-" + digest.hexdigest()[:12]


def save_model(
    model: object,
    root: Path,
    domain: str,
    config: Mapping[str, Any],
    metadata: Mapping[str, Any] | None = None,
) -> ModelArtifact:
    """Save `model` under `root/<domain>/<version>/`. The config must name the algorithm."""
    algorithm = getattr(model, "ALGORITHM_NAME", None)
    if algorithm is None:
        raise ArtifactError("a model needs an ALGORITHM_NAME to be saved")
    if config.get("algorithm", algorithm) != algorithm:
        raise ArtifactError(
            f"the config names {config['algorithm']!r}, but the model is {algorithm!r}"
        )
    full = {"format_version": FORMAT_VERSION, "domain": domain, **config, "algorithm": algorithm}
    meta = dict(metadata or {})
    _canonical(meta, "metadata")
    payload = _model_bytes(model)
    version = version_of(full, payload)
    path = root / domain / version
    path.mkdir(parents=True, exist_ok=True)
    (path / MODEL_FILE).write_bytes(payload)
    (path / CONFIG_FILE).write_text(json.dumps(full, indent=2, sort_keys=True) + "\n")
    (path / METADATA_FILE).write_text(json.dumps(meta, indent=2, sort_keys=True) + "\n")
    return ModelArtifact(domain, version, path, full, meta)


def load_model(path: Path) -> tuple[object, ModelArtifact]:
    """Load an artifact directory, refusing one whose files don't match its version."""
    missing = [
        name for name in (CONFIG_FILE, METADATA_FILE, MODEL_FILE) if not (path / name).is_file()
    ]
    if missing:
        raise ArtifactError(f"{path} isn't a model artifact: it lacks {missing}")
    config = json.loads((path / CONFIG_FILE).read_text())
    if config.get("format_version") != FORMAT_VERSION:
        raise ArtifactError(f"unsupported artifact format {config.get('format_version')!r}")
    payload = (path / MODEL_FILE).read_bytes()
    version = version_of(config, payload)
    if version != path.name:
        raise ArtifactError(f"{path} holds version {version}, not the {path.name} it's named")
    model = joblib.load(io.BytesIO(payload))
    if getattr(model, "ALGORITHM_NAME", None) != config["algorithm"]:
        raise ArtifactError(f"{path} holds a model that isn't {config['algorithm']!r}")
    metadata = json.loads((path / METADATA_FILE).read_text())
    return model, ModelArtifact(str(config["domain"]), version, path, config, metadata)


def versions(root: Path, domain: str) -> list[str]:
    """The versions saved for a domain, sorted."""
    folder = root / domain
    return sorted(p.name for p in folder.iterdir() if p.is_dir()) if folder.is_dir() else []
