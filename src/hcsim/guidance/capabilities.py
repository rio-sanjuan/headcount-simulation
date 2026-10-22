"""Capabilities: what each model can read beyond its own history, and emit beyond a point.

Every model declares, as class variables, so the declarations can be read without building
or fitting it:

- its **outputs** (`OUTPUTS`): `none` for a point forecast, and `quantiles`, `parametric`, or
  `sample_paths` for the uncertainty it can give. The pipeline never fabricates uncertainty
  for a model that declares only `none`;
- its **covariates** (`COVARIATES`): the calendar columns it reads (see `covariates`), such as
  `weekdays`. A covariate-aware model is a `CovariateModel`; it's given their values for its
  history and its horizon before it fits;
- whether it needs an **exposure**: a model of a share, such as attrition's share of
  headcount, is an `ExposureModel`.

A model that declares `sample_paths` is a `SamplingModel`: once fitted, it draws paths from
its predictive distribution, which is how a forecast carries uncertainty.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import numpy as np
from numpy.typing import NDArray

from hcsim.model.interface import ProbabilisticOutput

Floats = NDArray[np.float64]


@runtime_checkable
class ExposureModel(Protocol):
    """A model of a share of an exposure. Before it fits, it's given the exposure's history,
    as long as the series', and the exposure's values for the months it will forecast."""

    def expose(self, past: Floats, future: Floats | None) -> None: ...


@runtime_checkable
class CovariateModel(Protocol):
    """A model that reads calendar covariates. Before it fits, it's given their values, a
    column per covariate it declares, for its history and for the months it will forecast."""

    def provide(self, past: Floats, future: Floats | None) -> None: ...


@runtime_checkable
class SamplingModel(Protocol):
    """A fitted model that draws sample paths from its predictive distribution: `paths`
    draws of the `horizon` months ahead, one row per path."""

    def sample(self, horizon: int, paths: int, rng: np.random.Generator) -> Floats: ...


@dataclass(frozen=True)
class Capabilities:
    outputs: frozenset[ProbabilisticOutput]
    covariates: tuple[str, ...]
    exposure: bool

    @property
    def probabilistic(self) -> bool:
        return bool(self.outputs - {ProbabilisticOutput.NONE})

    def as_record(self) -> dict[str, object]:
        return {
            "outputs": sorted(o.value for o in self.outputs),
            "covariates": list(self.covariates),
            "exposure": self.exposure,
        }


def declared(model: type) -> Capabilities:
    """The capabilities a model class declares."""
    outputs = getattr(model, "OUTPUTS", frozenset({ProbabilisticOutput.NONE}))
    return Capabilities(
        frozenset(outputs),
        tuple(getattr(model, "COVARIATES", ())),
        issubclass(model, ExposureModel),
    )
