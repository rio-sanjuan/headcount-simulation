"""Model protocols. Any conforming model works in the simulation.

This module imports nothing else from hcsim, so every layer can import it without a cycle.
Eligibility rules are deliberately absent: they live in the simulation configuration, never in
a model or its saved state.
"""

from __future__ import annotations

from enum import StrEnum
from typing import ClassVar, Protocol, Self, runtime_checkable

import numpy as np
from numpy.typing import NDArray

Probabilities = NDArray[np.float64]
Mask = NDArray[np.bool_]


class ProbabilisticOutput(StrEnum):
    """What a time-series model can emit beyond a point forecast. The pipeline never fabricates
    uncertainty for a model that declares only `NONE`."""

    NONE = "none"
    QUANTILES = "quantiles"
    PARAMETRIC = "parametric"
    SAMPLE_PATHS = "sample_paths"


@runtime_checkable
class EntityModel(Protocol):
    """Scores individuals: an (N, F) feature array maps to N probabilities in [0, 1]."""

    ALGORITHM_NAME: ClassVar[str]

    def compute_probability(self, features: NDArray[np.float64]) -> Probabilities: ...


@runtime_checkable
class GroupRateModel(Protocol):
    """Returns a rate or count for a group, such as a cost center, when there's no per-entity
    row to score."""

    ALGORITHM_NAME: ClassVar[str]

    def rate_for(self, group: int) -> float: ...


@runtime_checkable
class TimeSeriesModel(Protocol):
    """Forecasts one aggregate monthly series."""

    NAME: ClassVar[str]
    OUTPUTS: ClassVar[frozenset[ProbabilisticOutput]]

    def fit(self, history: NDArray[np.float64]) -> Self: ...

    def predict(self, horizon: int) -> NDArray[np.float64]: ...


@runtime_checkable
class SamplingStrategy(Protocol):
    """Turns probabilities into a selection mask, optionally realizing an exact quota per
    group. It works on plain arrays, so it's usable without a simulation."""

    def select(
        self,
        probabilities: Probabilities,
        rng: np.random.Generator,
        *,
        groups: NDArray[np.int64] | None = None,
        quota: NDArray[np.int64] | None = None,
    ) -> Mask: ...
