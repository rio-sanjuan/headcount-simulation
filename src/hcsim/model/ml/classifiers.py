"""Entity classifiers: a level-only baseline, logistic regression, and gradient boosting.

Every model reads one feature matrix, (rows, features), and a `FeatureMask` says which of its
columns the model uses and which of those are categorical codes. The rest are ignored, so
models with different features score the same rows without rebuilding them.

- `LevelOnlyModel` is the baseline: an event rate per level, pooled where a level is thin.
- `LogisticModel` one-hot encodes its categorical columns, standardizes the rest, and fits a
  regularized logistic regression.
- `BoostedModel` wraps a histogram gradient-boosting classifier, with categorical columns as
  native categories. An unattributed code (-1) reads as missing.

`logit_offset` solves for the shift on every row's log-odds that makes a model's mean probability
match a target rate, and `calibrated` applies it. A model fitted on one window can be moved to
another's base rate this way without changing how it ranks rows.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import ClassVar, Final, Self

import numpy as np
from numpy.typing import NDArray
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression

from hcsim.model.interface import Probabilities
from hcsim.model.ml.entity import ModelNotFittedError, NaiveRateModel

Floats = NDArray[np.float64]
#: A histogram booster holds at most this many categories per feature.
MAX_CATEGORIES: Final = 254
_EPS: Final = 1e-12


class ModelError(ValueError):
    """Inputs a model can't fit or score."""


@dataclass(frozen=True)
class FeatureMask:
    """Which feature columns a model uses, and which of those are categorical codes."""

    columns: tuple[int, ...]
    categorical: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        if not self.columns:
            raise ModelError("a model needs at least one feature column")
        if len(set(self.columns)) != len(self.columns):
            raise ModelError(f"feature columns repeat: {list(self.columns)}")
        stray = sorted(set(self.categorical) - set(self.columns))
        if stray:
            raise ModelError(f"categorical columns {stray} aren't among the model's columns")

    def select(self, features: NDArray[np.float64]) -> Floats:
        array = np.asarray(features, dtype=np.float64)
        if array.ndim != 2:
            raise ModelError("features must be (rows, features)")
        if self.columns and max(self.columns) >= array.shape[1]:
            raise ModelError(
                f"features have {array.shape[1]} columns; the mask reads {max(self.columns)}"
            )
        result: Floats = array[:, list(self.columns)]
        return result


def _labels(labels: NDArray[np.float64] | NDArray[np.bool_], rows: int) -> Floats:
    y = np.asarray(labels, dtype=np.float64)
    if y.shape != (rows,):
        raise ModelError("labels need one entry per row")
    if rows == 0:
        raise ModelError("can't fit on zero rows")
    if ((y != 0) & (y != 1)).any():
        raise ModelError("labels must be 0 or 1")
    if y.min() == y.max():
        raise ModelError("labels need both outcomes to fit a classifier")
    return y


class LevelOnlyModel:
    """An event rate per level: the null arm's richer sibling, and every family's floor."""

    ALGORITHM_NAME: ClassVar[str] = "level_only"

    def __init__(self, level_column: int, *, min_population: int = 30) -> None:
        self.level_column = level_column
        self._rates = NaiveRateModel([level_column], min_population=min_population)

    def fit(self, features: NDArray[np.float64], labels: NDArray[np.float64]) -> Self:
        self._rates.fit(np.asarray(features, dtype=np.float64), labels)
        return self

    def compute_probability(self, features: NDArray[np.float64]) -> Probabilities:
        return self._rates.compute_probability(np.asarray(features, dtype=np.float64))


class _Masked:
    def __init__(self, mask: FeatureMask) -> None:
        self.mask = mask
        self._fitted = False

    @property
    def _numeric(self) -> list[int]:
        return [i for i, c in enumerate(self.mask.columns) if c not in self.mask.categorical]

    @property
    def _categorical(self) -> list[int]:
        return [i for i, c in enumerate(self.mask.columns) if c in self.mask.categorical]

    def _require(self) -> None:
        if not self._fitted:
            raise ModelNotFittedError("fit the model first")


class LogisticModel(_Masked):
    ALGORITHM_NAME: ClassVar[str] = "logistic"

    def __init__(self, mask: FeatureMask, *, regularization: float = 1.0) -> None:
        super().__init__(mask)
        self.regularization = regularization

    def _design(self, x: Floats) -> Floats:
        numeric = (x[:, self._numeric] - self._mean) / self._scale
        blocks = [numeric]
        for k, levels in zip(self._categorical, self._levels, strict=True):
            codes = x[:, k]
            blocks.append((codes[:, None] == levels[None, :]).astype(np.float64))
        return np.hstack(blocks) if blocks else np.zeros((len(x), 0))

    def fit(self, features: NDArray[np.float64], labels: NDArray[np.float64]) -> Self:
        x = self.mask.select(features)
        y = _labels(labels, len(x))
        numeric = x[:, self._numeric]
        self._mean = numeric.mean(axis=0)
        self._scale = np.where(numeric.std(axis=0) > 0, numeric.std(axis=0), 1.0)
        self._levels = [np.unique(x[:, k][x[:, k] >= 0]) for k in self._categorical]
        self._model = LogisticRegression(C=self.regularization, max_iter=1000)
        self._model.fit(self._design(x), y)
        self._fitted = True
        return self

    def compute_probability(self, features: NDArray[np.float64]) -> Probabilities:
        self._require()
        x = self.mask.select(features)
        result: Probabilities = self._model.predict_proba(self._design(x))[:, 1]
        return result


class BoostedModel(_Masked):
    ALGORITHM_NAME: ClassVar[str] = "boosted"

    def __init__(
        self,
        mask: FeatureMask,
        *,
        max_iter: int = 150,
        learning_rate: float = 0.05,
        max_leaf_nodes: int = 8,
        min_samples_leaf: int = 100,
        l2_regularization: float = 1.0,
        seed: int = 0,
    ) -> None:
        # Event rates are low and noisy, so the defaults are shallow and slow: deeper, faster
        # boosting overfits workforce-sized data. Tuning (M8) can move them.
        super().__init__(mask)
        self.params: dict[str, float | int] = {
            "max_iter": max_iter,
            "learning_rate": learning_rate,
            "max_leaf_nodes": max_leaf_nodes,
            "min_samples_leaf": min_samples_leaf,
            "l2_regularization": l2_regularization,
        }
        self.seed = seed
        self.offset = 0.0

    def _prepared(self, x: Floats) -> Floats:
        out = x.copy()
        for k in self._categorical:
            column = out[:, k]
            column[column < 0] = np.nan  # unattributed reads as missing
        return out

    def fit(self, features: NDArray[np.float64], labels: NDArray[np.float64]) -> Self:
        x = self.mask.select(features)
        y = _labels(labels, len(x))
        wide = [
            self.mask.columns[k]
            for k in self._categorical
            if x[:, k].max(initial=-1) >= MAX_CATEGORIES
        ]
        if wide:
            raise ModelError(
                f"categorical columns {wide} have codes past {MAX_CATEGORIES - 1}; a booster "
                "holds at most that many categories, so give it a coarser column"
            )
        flags = np.zeros(x.shape[1], dtype=bool)
        flags[self._categorical] = True
        self._model = HistGradientBoostingClassifier(
            categorical_features=flags,
            early_stopping=False,
            random_state=self.seed,
            **self.params,
        )
        self._model.fit(self._prepared(x), y)
        self._fitted = True
        self.offset = 0.0
        return self

    def raw_probability(self, features: NDArray[np.float64]) -> Probabilities:
        self._require()
        result: Probabilities = self._model.predict_proba(
            self._prepared(self.mask.select(features))
        )[:, 1]
        return result

    def calibrate(self, features: NDArray[np.float64], target_rate: float) -> Self:
        """Shift the log-odds so the mean probability over `features` is `target_rate`."""
        self.offset = logit_offset(self.raw_probability(features), target_rate)
        return self

    def compute_probability(self, features: NDArray[np.float64]) -> Probabilities:
        return calibrated(self.raw_probability(features), self.offset)


def _logit(p: Floats) -> Floats:
    clipped = np.clip(p, _EPS, 1 - _EPS)
    result: Floats = np.log(clipped / (1 - clipped))
    return result


def calibrated(p: Sequence[float] | Floats, offset: float) -> Probabilities:
    """Probabilities with every row's log-odds shifted by `offset`."""
    result: Probabilities = 1.0 / (
        1.0 + np.exp(-(_logit(np.asarray(p, dtype=np.float64)) + offset))
    )
    return result


def logit_offset(
    p: Sequence[float] | Floats, target_rate: float, *, iterations: int = 100
) -> float:
    """The shift on every row's log-odds that makes the mean probability `target_rate`.
    The mean rises with the shift, so bisection finds it."""
    probabilities = np.asarray(p, dtype=np.float64)
    if probabilities.size == 0:
        raise ModelError("calibration needs at least one row")
    if not 0.0 < target_rate < 1.0:
        raise ModelError("a target rate must lie strictly between 0 and 1")
    low, high = -50.0, 50.0
    for _ in range(iterations):
        middle = (low + high) / 2
        if calibrated(probabilities, middle).mean() < target_rate:
            low = middle
        else:
            high = middle
    return (low + high) / 2
