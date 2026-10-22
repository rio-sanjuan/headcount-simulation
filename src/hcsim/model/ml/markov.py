"""A Markov model with an entry distribution, and a similarity matrix from co-occurrence.

`MarkovModel` describes a state that moves at each cycle, such as a rating from one calibration
to the next. Its transition matrix comes from observed consecutive pairs, with optional additive
smoothing, and a state never seen as a source reads the overall mix of next states. Its entry
distribution gives a newcomer's first state. Draws come from a seeded generator, one uniform per
row.

`SimilarityMatrix` measures how closely categories go together, from how often they co-occur:
moves between job families, say. It's the cosine of the co-occurrence counts, symmetric, with a
unit diagonal for every category seen at all, and gives each category's nearest neighbors.
"""

from __future__ import annotations

from typing import ClassVar, Self

import numpy as np
from numpy.typing import NDArray

from hcsim.model.ml.entity import ModelNotFittedError

Ints = NDArray[np.int64]
Floats = NDArray[np.float64]


def _codes(values: Ints, n: int, what: str) -> Ints:
    codes = np.asarray(values, dtype=np.int64)
    if codes.size and (codes.min() < 0 or codes.max() >= n):
        raise ValueError(f"{what} must lie in 0..{n - 1}")
    return codes


def _draw(probabilities: Floats, uniform: Floats) -> Ints:
    cumulative = np.cumsum(probabilities, axis=-1)
    result: Ints = np.minimum(
        (uniform[..., None] > cumulative).sum(axis=-1), probabilities.shape[-1] - 1
    )
    return result


class MarkovModel:
    ALGORITHM_NAME: ClassVar[str] = "markov"

    def __init__(self, n_states: int, *, smoothing: float = 0.0) -> None:
        if n_states < 1 or smoothing < 0:
            raise ValueError("a Markov model needs a state and non-negative smoothing")
        self.n_states = n_states
        self.smoothing = smoothing
        self._transitions: Floats | None = None
        self._entry: Floats | None = None

    def fit(self, previous: Ints, following: Ints, entries: Ints) -> Self:
        """`previous` and `following` are consecutive pairs; `entries` are first states."""
        n = self.n_states
        a, b = _codes(previous, n, "states"), _codes(following, n, "states")
        first = _codes(entries, n, "entry states")
        if a.shape != b.shape:
            raise ValueError("each transition needs a state before and after")
        if a.size == 0 or first.size == 0:
            raise ValueError("a Markov model needs transitions and entries to fit")
        counts = np.zeros((n, n))
        np.add.at(counts, (a, b), 1.0)
        overall = counts.sum(axis=0) + self.smoothing
        overall /= overall.sum()
        smoothed = counts + self.smoothing
        totals = smoothed.sum(axis=1, keepdims=True)
        self._transitions = np.where(
            totals > 0, smoothed / np.where(totals > 0, totals, 1), overall
        )
        entry = np.bincount(first, minlength=n) + self.smoothing
        self._entry = entry / entry.sum()
        return self

    def _require(self) -> tuple[Floats, Floats]:
        if self._transitions is None or self._entry is None:
            raise ModelNotFittedError("fit the model first")
        return self._transitions, self._entry

    @property
    def transitions(self) -> Floats:
        return self._require()[0].copy()

    @property
    def entry(self) -> Floats:
        return self._require()[1].copy()

    def stationary(self) -> Floats:
        """The long-run share of each state: the transition matrix's left unit eigenvector."""
        transitions = self._require()[0]
        values, vectors = np.linalg.eig(transitions.T)
        vector = np.real(vectors[:, np.argmin(np.abs(values - 1))])
        result: Floats = np.abs(vector) / np.abs(vector).sum()
        return result

    def next_states(self, current: Ints, rng: np.random.Generator) -> Ints:
        transitions = self._require()[0]
        state = _codes(current, self.n_states, "states")
        return _draw(transitions[state], rng.random(len(state)))

    def entry_states(self, n: int, rng: np.random.Generator) -> Ints:
        entry = self._require()[1]
        return _draw(np.broadcast_to(entry, (n, self.n_states)), rng.random(n))


class SimilarityMatrix:
    ALGORITHM_NAME: ClassVar[str] = "cooccurrence_similarity"

    def __init__(self, n_categories: int) -> None:
        if n_categories < 1:
            raise ValueError("a similarity matrix needs at least one category")
        self.n_categories = n_categories
        self._matrix: Floats | None = None

    def fit(self, first: Ints, second: Ints) -> Self:
        """Co-occurring pairs, such as moves from one category to another. Order doesn't
        matter: a pair counts for both directions."""
        n = self.n_categories
        a, b = _codes(first, n, "categories"), _codes(second, n, "categories")
        if a.shape != b.shape:
            raise ValueError("each co-occurrence needs two categories")
        counts = np.zeros((n, n))
        np.add.at(counts, (a, b), 1.0)
        counts = counts + counts.T
        np.fill_diagonal(counts, 0.0)
        occurrences = np.bincount(np.concatenate([a, b]), minlength=n).astype(np.float64)
        with np.errstate(invalid="ignore", divide="ignore"):
            norm = np.sqrt(np.outer(occurrences, occurrences))
            matrix = np.where(norm > 0, counts / np.where(norm > 0, norm, 1), 0.0)
        np.fill_diagonal(matrix, (occurrences > 0).astype(np.float64))
        self._matrix = matrix
        return self

    @property
    def matrix(self) -> Floats:
        if self._matrix is None:
            raise ModelNotFittedError("fit the matrix first")
        return self._matrix.copy()

    def neighbors(self, category: int, k: int) -> Ints:
        """The `k` other categories most similar to `category`, closest first; ties go to the
        lower code."""
        matrix = self.matrix
        _codes(np.array([category]), self.n_categories, "categories")
        row = matrix[category].copy()
        row[category] = -np.inf
        order = np.lexsort((np.arange(len(row)), -row))
        result: Ints = order[: min(k, self.n_categories - 1)]
        return result
