"""Loading a run's models: conformance to their family's protocol, and a probe of each.

A run names one artifact per domain. `load_models` loads each one and checks two things:

- **conformance**: the model has an algorithm name, belongs to its domain's family, and has
  the methods that family's protocol needs: a classifier scores rows (`EntityModel`), a count
  model gives a rate per group (`GroupRateModel`) and draws counts, and a generator draws
  outcomes;
- **a probe**: the model scores a small probe input, and its output has the right shape and is
  finite: probabilities in [0, 1], non-negative counts, and category codes in range.

Every problem is recorded against its domain instead of stopping the load, so one report names
everything wrong with a run's models at once. A domain with problems loads no model.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from hcsim.model.interface import EntityModel, GroupRateModel
from hcsim.model.ml.arms import ALGORITHMS, ArmMatrix
from hcsim.model.ml.persistence import ArtifactError, ModelArtifact, load_model


@dataclass(frozen=True)
class Probe:
    """A small input to score: feature rows for classifiers, group codes for count models,
    and (rows, levels) contexts for generators."""

    features: NDArray[np.float64] | None = None
    groups: NDArray[np.int64] | None = None
    contexts: NDArray[np.int64] | None = None
    n_categories: int | None = None


@dataclass
class LoadReport:
    models: dict[str, tuple[object, ModelArtifact]] = field(default_factory=dict)
    failures: dict[str, list[str]] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return not self.failures


def conformance(model: object, family: str) -> list[str]:
    """What keeps `model` from serving a domain of `family`."""
    problems = []
    algorithm = getattr(model, "ALGORITHM_NAME", None)
    if not isinstance(algorithm, str):
        problems.append("the model has no ALGORITHM_NAME")
    elif family in ALGORITHMS and algorithm not in ALGORITHMS[family]:
        problems.append(f"{algorithm!r} isn't a {family} algorithm")
    if family == "classifier" and not isinstance(model, EntityModel):
        problems.append("a classifier must score rows: it needs compute_probability")
    if family == "count" and not (
        isinstance(model, GroupRateModel) and callable(getattr(model, "sample", None))
    ):
        problems.append("a count model needs rate_for and sample")
    if family == "generator" and not callable(getattr(model, "sample", None)):
        problems.append("a generator needs sample")
    if family not in ALGORITHMS:
        problems.append(f"no model family {family!r}")
    return problems


def probe(model: Any, family: str, inputs: Probe) -> list[str]:
    """Score the probe and check the output, or say why it couldn't be scored."""
    rng = np.random.default_rng(0)
    try:
        if family == "classifier":
            if inputs.features is None:
                return ["a classifier probe needs features"]
            p = np.asarray(model.compute_probability(inputs.features), dtype=np.float64)
            if p.shape != (len(inputs.features),):
                return [f"scored {p.shape} for {len(inputs.features)} rows"]
            if not np.isfinite(p).all() or (p < 0).any() or (p > 1).any():
                return ["probabilities must be finite and lie in [0, 1]"]
            return []
        if family == "count":
            if inputs.groups is None:
                return ["a count probe needs groups"]
            rates = np.array([model.rate_for(int(g)) for g in inputs.groups], dtype=np.float64)
            drawn = np.asarray(model.sample(inputs.groups, rng))
            problems = []
            if not np.isfinite(rates).all() or (rates < 0).any():
                problems.append("rates must be finite and non-negative")
            if drawn.shape != inputs.groups.shape or (drawn < 0).any():
                problems.append("counts must be non-negative, one per group")
            return problems
        if inputs.contexts is None or inputs.n_categories is None:
            return ["a generator probe needs contexts and the number of categories"]
        drawn = np.asarray(model.sample(inputs.contexts, rng))
        if drawn.shape != (len(inputs.contexts),):
            return [f"drew {drawn.shape} for {len(inputs.contexts)} contexts"]
        if (drawn < 0).any() or (drawn >= inputs.n_categories).any():
            return [f"outcomes must lie in 0..{inputs.n_categories - 1}"]
        return []
    except Exception as error:
        return [f"the probe raised {type(error).__name__}: {error}"]


def load_models(
    root: Path,
    chosen: Mapping[str, str],
    matrix: ArmMatrix,
    probes: Mapping[str, Probe],
) -> LoadReport:
    """Load each domain's chosen version from `root`, check it, and probe it."""
    report = LoadReport()
    for domain, version in sorted(chosen.items()):
        problems: list[str] = []
        try:
            family = matrix.null(domain).family
        except ValueError as error:
            report.failures[domain] = [str(error)]
            continue
        try:
            model, artifact = load_model(root / domain / version)
        except (ArtifactError, OSError) as error:
            report.failures[domain] = [str(error)]
            continue
        problems += conformance(model, family)
        if domain not in probes:
            problems.append("no probe input for this domain")
        elif not problems:
            problems += probe(model, family, probes[domain])
        if problems:
            report.failures[domain] = problems
        else:
            report.models[domain] = (model, artifact)
    return report
