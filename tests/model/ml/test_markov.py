"""Tests for the Markov model and the co-occurrence similarity matrix."""

from __future__ import annotations

import numpy as np
import pytest

from hcsim.model.ml.entity import ModelNotFittedError
from hcsim.model.ml.markov import MarkovModel, SimilarityMatrix
from tests.support.stat_check import stat_check

TRUE = np.array([[0.7, 0.3, 0.0], [0.2, 0.6, 0.2], [0.0, 0.4, 0.6]])


def chains(n: int = 4000, seed: int = 3) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    entries = rng.choice(3, size=n, p=[0.5, 0.3, 0.2])
    previous, following = [], []
    state = entries.copy()
    for _ in range(5):
        nxt = np.array([rng.choice(3, p=TRUE[s]) for s in state])
        previous.append(state)
        following.append(nxt)
        state = nxt
    return np.concatenate(previous), np.concatenate(following), entries


def test_transitions_and_entries_are_recovered() -> None:
    model = MarkovModel(3).fit(*chains())
    np.testing.assert_allclose(model.transitions, TRUE, atol=0.02)
    np.testing.assert_allclose(model.entry, [0.5, 0.3, 0.2], atol=0.03)
    stationary = model.stationary()
    np.testing.assert_allclose(stationary @ model.transitions, stationary, atol=1e-9)
    assert stationary.sum() == pytest.approx(1.0)


def test_draws_follow_the_fitted_matrix() -> None:
    model = MarkovModel(3).fit(*chains())
    rng = np.random.default_rng(4)
    nxt = model.next_states(np.full(20_000, 1), rng)
    for state in range(3):
        p = model.transitions[1, state]
        stat_check(
            float((nxt == state).mean()),
            p,
            float(np.sqrt(p * (1 - p) / 20_000)),
            label=f"to {state}",
        )
    first = model.entry_states(20_000, rng)
    stat_check(
        float((first == 0).mean()),
        model.entry[0],
        float(np.sqrt(model.entry[0] * (1 - model.entry[0]) / 20_000)),
        label="entry",
    )


def test_an_unseen_state_reads_the_overall_mix_and_smoothing_fills_gaps() -> None:
    model = MarkovModel(3).fit(np.array([0, 0, 1]), np.array([1, 1, 0]), np.array([0]))
    np.testing.assert_allclose(model.transitions[2], [1 / 3, 2 / 3, 0.0])  # never a source
    smooth = MarkovModel(3, smoothing=1.0).fit(np.array([0]), np.array([1]), np.array([0]))
    assert (smooth.transitions > 0).all()
    np.testing.assert_allclose(smooth.transitions.sum(axis=1), 1.0)


def test_bad_markov_inputs_are_refused() -> None:
    with pytest.raises(ValueError, match="non-negative smoothing"):
        MarkovModel(0)
    with pytest.raises(ValueError, match=r"states must lie in 0\.\.2"):
        MarkovModel(3).fit(np.array([3]), np.array([0]), np.array([0]))
    with pytest.raises(ValueError, match="a state before and after"):
        MarkovModel(3).fit(np.array([0, 1]), np.array([0]), np.array([0]))
    with pytest.raises(ValueError, match="transitions and entries"):
        MarkovModel(3).fit(np.zeros(0, np.int64), np.zeros(0, np.int64), np.array([0]))
    with pytest.raises(ModelNotFittedError):
        MarkovModel(3).stationary()


def test_similarity_is_the_cosine_of_co_occurrence() -> None:
    # Families 0 and 1 trade people often, 1 and 2 rarely, and 3 never moves.
    first = np.array([0] * 8 + [1] * 2 + [1])
    second = np.array([1] * 8 + [0] * 2 + [2])
    similarity = SimilarityMatrix(4).fit(first, second)
    m = similarity.matrix
    np.testing.assert_allclose(m, m.T)
    assert m.diagonal().tolist() == [1.0, 1.0, 1.0, 0.0]  # family 3 was never seen
    assert m[0, 1] == pytest.approx(10 / np.sqrt(10 * 11))
    assert m[1, 2] == pytest.approx(1 / np.sqrt(11 * 1))
    assert m[0, 2] == 0.0
    assert similarity.neighbors(1, 2).tolist() == [0, 2]
    assert similarity.neighbors(3, 5).tolist() == [0, 1, 2]  # ties: lower codes first


def test_bad_similarity_inputs_are_refused() -> None:
    with pytest.raises(ValueError, match="at least one category"):
        SimilarityMatrix(0)
    with pytest.raises(ValueError, match="two categories"):
        SimilarityMatrix(2).fit(np.array([0]), np.array([0, 1]))
    with pytest.raises(ModelNotFittedError):
        SimilarityMatrix(2).neighbors(0, 1)
    with pytest.raises(ValueError, match=r"categories must lie in 0\.\.1"):
        SimilarityMatrix(2).fit(np.array([0]), np.array([1])).neighbors(2, 1)
