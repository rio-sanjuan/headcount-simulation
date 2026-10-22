"""Tests for the family metrics."""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.metrics import average_precision_score, roc_auc_score

from hcsim.model.ml.metrics import (
    FAMILY_METRICS,
    METRICS,
    Direction,
    MetricError,
    auc,
    average_precision,
    brier,
    ece,
    jensen_shannon,
    ks,
    log_likelihood,
    log_loss,
    primary,
    top_k_agreement,
    wape,
    wasserstein_1,
)

Y = np.array([0, 0, 1, 1, 0, 1, 0, 0, 1, 0], dtype=float)
P = np.array([0.1, 0.4, 0.35, 0.8, 0.4, 0.7, 0.2, 0.4, 0.9, 0.05])


def test_ranking_metrics_match_scikit_learn_with_ties() -> None:
    assert auc(Y, P) == pytest.approx(roc_auc_score(Y, P))
    assert average_precision(Y, P) == pytest.approx(average_precision_score(Y, P))
    rng = np.random.default_rng(0)
    y = (rng.random(500) < 0.2).astype(float)
    s = np.round(rng.random(500) + y * 0.3, 1)  # coarse scores, many ties
    assert auc(y, s) == pytest.approx(roc_auc_score(y, s))
    assert average_precision(y, s) == pytest.approx(average_precision_score(y, s))


def test_probability_metrics_known_answers() -> None:
    assert log_loss([1, 0], [0.5, 0.5]) == pytest.approx(np.log(2))
    assert brier([1, 0], [0.75, 0.25]) == pytest.approx(0.0625)
    assert ks([0, 0, 1, 1], [0.1, 0.2, 0.3, 0.4]) == pytest.approx(1.0)
    assert ks([0, 1, 0, 1], [0.5, 0.5, 0.5, 0.5]) == pytest.approx(0.0)
    assert ece([0, 1], [0.0, 1.0]) == 0.0
    assert ece([0, 0, 1, 1], [0.5, 0.5, 0.5, 0.5], bins=1) == 0.0
    assert ece([0, 0, 0, 1], [0.1, 0.2, 0.3, 0.9], bins=2) == pytest.approx(
        (2 * 0.15 + 2 * abs(0.5 - 0.6)) / 4
    )


def test_distribution_metrics_known_answers() -> None:
    assert wasserstein_1([1, 0, 0], [0, 0, 1]) == pytest.approx(2.0)
    assert wasserstein_1([0.5, 0.5], [0.5, 0.5]) == 0.0
    assert jensen_shannon([1, 0], [0, 1]) == pytest.approx(1.0)
    assert jensen_shannon([0.3, 0.7], [0.3, 0.7]) == pytest.approx(0.0)
    p = np.array([[0.5, 0.5], [0.9, 0.1]])
    assert log_likelihood(p, [1, 0]) == pytest.approx((np.log(0.5) + np.log(0.9)) / 2)


def test_top_k_agreement_compares_nearest_neighbours() -> None:
    a = np.array([[1, 0.9, 0.1], [0.9, 1, 0.2], [0.1, 0.2, 1]])
    b = np.array([[1, 0.1, 0.9], [0.1, 1, 0.2], [0.9, 0.2, 1]])
    assert top_k_agreement(a, a, 1) == 1.0
    assert top_k_agreement(a, b, 1) == 0.0  # no category keeps its nearest neighbour
    c = a.copy()
    c[2] = [0.9, 0.2, 1]  # category 2's nearest is now 0; the others are unchanged
    assert top_k_agreement(a, c, 1) == pytest.approx(2 / 3)
    with pytest.raises(MetricError, match=r"k must lie in 1\.\.2"):
        top_k_agreement(a, b, 3)


def test_wape_and_its_zero_case() -> None:
    assert wape([10, 0, 5], [8, 1, 5]) == pytest.approx(3 / 15)
    assert wape([0, 0], [0, 0]) == 0.0
    assert wape([0, 0], [1, 0]) == float("inf")


def test_each_family_has_a_primary_metric_with_a_direction() -> None:
    assert set(FAMILY_METRICS) == {"classifier", "generator", "count"}
    assert primary("classifier").name == "log_loss"
    assert primary("classifier").direction is Direction.LOWER
    assert primary("generator").direction is Direction.HIGHER
    assert METRICS["auc"].better(0.8, 0.7)
    assert METRICS["log_loss"].better(0.2, 0.3)
    assert all(name in METRICS for names in FAMILY_METRICS.values() for name in names)
    with pytest.raises(MetricError, match="no model family 'tree'; known: classifier"):
        primary("tree")


@pytest.mark.parametrize(
    ("call", "message"),
    [
        (lambda: log_loss([0, 1], [0.5]), "one value each per row"),
        (lambda: log_loss([0, 2], [0.5, 0.5]), "0 or 1"),
        (lambda: log_loss([0, 1], [0.5, 1.5]), "lie in \\[0, 1\\]"),
        (lambda: auc([0, 1], [np.nan, 0.5]), "finite"),
        (lambda: auc([1, 1], [0.2, 0.5]), "both outcomes"),
        (lambda: average_precision([0, 0], [0.2, 0.5]), "at least one event"),
        (lambda: ece([0, 1], [0.5, 0.5], bins=0), "at least one bin"),
        (lambda: wasserstein_1([0.5, 0.5], [1.0]), "same categories"),
        (lambda: jensen_shannon([0.5, 0.6], [0.5, 0.5]), "sum to 1"),
        (lambda: jensen_shannon([0.5], [1.0, 0.0]), "sum to 1"),
        (lambda: jensen_shannon([1.0], [1.0, 0.0]), "same categories"),
        (lambda: wasserstein_1([-0.5, 1.5], [0.5, 0.5]), "non-negative distribution"),
        (lambda: log_likelihood([[0.5, 0.5]], [2]), "lie in 0..1"),
        (lambda: log_likelihood([[0.5, 0.6]], [0]), "must be a distribution"),
        (lambda: log_likelihood([0.5, 0.5], [0]), "\\(rows, categories\\)"),
        (lambda: top_k_agreement(np.ones((2, 3)), np.ones((2, 3)), 1), "square"),
        (lambda: wape([1.0], [1.0, 2.0]), "same cells"),
    ],
)
def test_bad_inputs_are_refused(call, message: str) -> None:  # type: ignore[no-untyped-def]
    with pytest.raises(MetricError, match=message):
        call()
