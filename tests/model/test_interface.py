"""Conformance tests for the model protocols."""

from __future__ import annotations

import ast
from pathlib import Path
from typing import ClassVar, Self

import numpy as np
from numpy.typing import NDArray

import hcsim.model.interface as interface
from hcsim.model.interface import (
    EntityModel,
    GroupRateModel,
    ProbabilisticOutput,
    SamplingStrategy,
    TimeSeriesModel,
)


class Half:
    ALGORITHM_NAME: ClassVar[str] = "half"

    def compute_probability(self, features: NDArray[np.float64]) -> NDArray[np.float64]:
        return np.full(len(features), 0.5)


class FlatRate:
    ALGORITHM_NAME: ClassVar[str] = "flat"

    def rate_for(self, group: int) -> float:
        return 0.1


class LastValue:
    NAME: ClassVar[str] = "last_value"
    OUTPUTS: ClassVar[frozenset[ProbabilisticOutput]] = frozenset({ProbabilisticOutput.NONE})

    def fit(self, history: NDArray[np.float64]) -> Self:
        self.last = float(history[-1])
        return self

    def predict(self, horizon: int) -> NDArray[np.float64]:
        return np.full(horizon, self.last)


class Everyone:
    def select(
        self,
        probabilities: NDArray[np.float64],
        rng: np.random.Generator,
        *,
        groups: NDArray[np.int64] | None = None,
        quota: NDArray[np.int64] | None = None,
    ) -> NDArray[np.bool_]:
        return probabilities > 0


def test_conforming_classes_satisfy_their_protocols() -> None:
    assert isinstance(Half(), EntityModel)
    assert isinstance(FlatRate(), GroupRateModel)
    assert isinstance(LastValue(), TimeSeriesModel)
    assert isinstance(Everyone(), SamplingStrategy)


def test_a_class_missing_the_method_doesnt_conform() -> None:
    assert not isinstance(FlatRate(), EntityModel)
    assert not isinstance(Half(), TimeSeriesModel)


def test_a_time_series_model_fits_then_predicts() -> None:
    model = LastValue().fit(np.array([1.0, 2.0, 3.0]))
    assert model.predict(2).tolist() == [3.0, 3.0]


def test_the_interface_module_imports_nothing_else_from_hcsim() -> None:
    tree = ast.parse(Path(interface.__file__).read_text())
    imported = {
        node.module if isinstance(node, ast.ImportFrom) else alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.Import | ast.ImportFrom)
        for alias in node.names
    }
    assert not {name for name in imported if name and name.startswith("hcsim")}
