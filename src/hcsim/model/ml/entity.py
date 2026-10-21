"""Entity models: per-row event probabilities.

`ConstantModel` gives every row one probability: the null arm, and the model a replay run is
allowed to use. `NaiveRateModel` fits an event rate for every prefix of a stratification key,
such as level then cost center, and serves each row from the longest prefix whose training cell
holds at least `min_population` rows. The global rate, prefix zero, always qualifies.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import ClassVar, Self

import numpy as np
from numpy.typing import NDArray

from hcsim.model.interface import Probabilities


class ModelNotFittedError(RuntimeError):
    """A model was asked to score before it was fitted."""


class ConstantModel:
    ALGORITHM_NAME: ClassVar[str] = "constant"

    def __init__(self, probability: float) -> None:
        if not 0.0 <= probability <= 1.0:
            raise ValueError("probability must lie in [0, 1]")
        self.probability = probability

    def compute_probability(self, features: NDArray[np.float64]) -> Probabilities:
        return np.full(len(features), self.probability, dtype=np.float64)


def _cell_ids(
    train: NDArray[np.int64], query: NDArray[np.int64]
) -> tuple[NDArray[np.int64], NDArray[np.int64]]:
    """Give each distinct key row one id across both arrays."""
    stacked = np.concatenate([train, query])
    if stacked.shape[1] == 0:
        ids = np.zeros(len(stacked), dtype=np.int64)
    else:
        _, inverse = np.unique(stacked, axis=0, return_inverse=True)
        ids = inverse.reshape(-1).astype(np.int64)
    return ids[: len(train)], ids[len(train) :]


class NaiveRateModel:
    ALGORITHM_NAME: ClassVar[str] = "naive_rate"

    def __init__(self, key_columns: Sequence[int], *, min_population: int = 30) -> None:
        if min_population < 1:
            raise ValueError("min_population must be at least 1")
        self.key_columns = tuple(int(c) for c in key_columns)
        self.min_population = min_population
        self._keys: NDArray[np.int64] | None = None
        self._labels: NDArray[np.float64] | None = None

    def fit(
        self, features: NDArray[np.float64], labels: NDArray[np.float64] | NDArray[np.bool_]
    ) -> Self:
        y = np.asarray(labels, dtype=np.float64)
        if len(features) != len(y):
            raise ValueError("features and labels need the same number of rows")
        if len(y) == 0:
            raise ValueError("can't fit a rate on zero rows")
        if ((y != 0) & (y != 1)).any():
            raise ValueError("labels must be 0 or 1")
        self._keys = features[:, list(self.key_columns)].astype(np.int64)
        self._labels = y
        return self

    @property
    def global_rate(self) -> float:
        if self._labels is None:
            raise ModelNotFittedError("fit the model first")
        return float(self._labels.mean())

    def compute_probability(self, features: NDArray[np.float64]) -> Probabilities:
        if self._keys is None or self._labels is None:
            raise ModelNotFittedError("fit the model first")
        query = features[:, list(self.key_columns)].astype(np.int64)
        rate = np.full(len(features), self.global_rate, dtype=np.float64)
        resolved = np.zeros(len(features), dtype=bool)
        for length in range(len(self.key_columns), 0, -1):
            train_ids, query_ids = _cell_ids(self._keys[:, :length], query[:, :length])
            n_cells = int(max(train_ids.max(initial=-1), query_ids.max(initial=-1))) + 1
            counts = np.bincount(train_ids, minlength=n_cells)
            events = np.bincount(train_ids, weights=self._labels, minlength=n_cells)
            usable = (counts[query_ids] >= self.min_population) & ~resolved
            rate[usable] = events[query_ids[usable]] / counts[query_ids[usable]]
            resolved |= usable
        return rate
