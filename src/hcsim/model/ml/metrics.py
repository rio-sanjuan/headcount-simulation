"""Model metrics, grouped by model family, each with the direction that's better.

- `classifier`: average precision, AUC, log loss, Brier score, KS, and equal-count ECE, with
  log loss primary;
- `generator`: log-likelihood, Wasserstein-1, Jensen-Shannon divergence, and top-k neighbour
  agreement, with log-likelihood primary;
- `count`: WAPE.

A family's primary metric is the one its arms are selected on: a proper scoring rule where the
family has one, since a rule like that can't be gamed by a model that's merely overconfident.
The others go in reports. Every metric validates its inputs and raises on what it can't score.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from enum import StrEnum
from types import MappingProxyType
from typing import Final

import numpy as np
from numpy.typing import ArrayLike, NDArray

Floats = NDArray[np.float64]
_EPS: Final = 1e-15


class MetricError(ValueError):
    """Inputs a metric can't score."""


class Direction(StrEnum):
    HIGHER = "higher"
    LOWER = "lower"


def _binary(y_true: ArrayLike, scores: ArrayLike, *, probabilities: bool) -> tuple[Floats, Floats]:
    y = np.asarray(y_true, dtype=np.float64)
    s = np.asarray(scores, dtype=np.float64)
    if y.ndim != 1 or y.shape != s.shape or y.size == 0:
        raise MetricError("labels and scores need one value each per row")
    if ((y != 0) & (y != 1)).any():
        raise MetricError("labels must be 0 or 1")
    if not np.isfinite(s).all():
        raise MetricError("scores must be finite")
    if probabilities and ((s < 0) | (s > 1)).any():
        raise MetricError("probabilities must lie in [0, 1]")
    return y, s


def _both_classes(y: Floats, name: str) -> None:
    if y.min() == y.max():
        raise MetricError(f"{name} needs both outcomes among the labels")


def _ranks(values: Floats) -> Floats:
    """Average ranks, from 1, with ties sharing their mean rank."""
    order = np.argsort(values, kind="stable")
    sorted_values = values[order]
    ranks = np.empty(len(values))
    starts = np.flatnonzero(np.r_[True, sorted_values[1:] != sorted_values[:-1]])
    ends = np.r_[starts[1:], len(values)]
    for start, end in zip(starts, ends, strict=True):
        ranks[order[start:end]] = (start + end + 1) / 2
    return ranks


def auc(y_true: ArrayLike, scores: ArrayLike) -> float:
    """ROC AUC: the chance a random event outranks a random non-event, ties counting half."""
    y, s = _binary(y_true, scores, probabilities=False)
    _both_classes(y, "AUC")
    positives = y.sum()
    negatives = len(y) - positives
    return float(
        (_ranks(s)[y == 1].sum() - positives * (positives + 1) / 2) / (positives * negatives)
    )


def average_precision(y_true: ArrayLike, scores: ArrayLike) -> float:
    """Precision averaged over recall steps, rows with equal scores taken together."""
    y, s = _binary(y_true, scores, probabilities=False)
    if y.sum() == 0:
        raise MetricError("average precision needs at least one event")
    order = np.argsort(-s, kind="stable")
    y, s = y[order], s[order]
    last = np.r_[s[1:] != s[:-1], True]  # the last row of each run of equal scores
    hits = np.cumsum(y)[last]
    taken = (np.arange(len(y)) + 1)[last]
    recall_step = np.diff(np.r_[0.0, hits]) / y.sum()
    return float((recall_step * hits / taken).sum())


def log_loss(y_true: ArrayLike, p: ArrayLike) -> float:
    y, q = _binary(y_true, p, probabilities=True)
    q = np.clip(q, _EPS, 1 - _EPS)
    return float(-(y * np.log(q) + (1 - y) * np.log(1 - q)).mean())


def brier(y_true: ArrayLike, p: ArrayLike) -> float:
    y, q = _binary(y_true, p, probabilities=True)
    return float(((q - y) ** 2).mean())


def ks(y_true: ArrayLike, scores: ArrayLike) -> float:
    """The largest gap between the score distributions of events and non-events."""
    y, s = _binary(y_true, scores, probabilities=False)
    _both_classes(y, "KS")
    grid = np.unique(s)
    events = np.searchsorted(np.sort(s[y == 1]), grid, side="right") / y.sum()
    others = np.searchsorted(np.sort(s[y == 0]), grid, side="right") / (len(y) - y.sum())
    return float(np.abs(events - others).max())


def ece(y_true: ArrayLike, p: ArrayLike, bins: int = 10) -> float:
    """Expected calibration error over bins of equal count, sorted by probability."""
    y, q = _binary(y_true, p, probabilities=True)
    if bins < 1:
        raise MetricError("ECE needs at least one bin")
    order = np.argsort(q, kind="stable")
    total = 0.0
    for chunk in np.array_split(order, min(bins, len(order))):
        total += len(chunk) * abs(y[chunk].mean() - q[chunk].mean())
    return total / len(y)


def _distribution(p: ArrayLike, name: str) -> Floats:
    d = np.asarray(p, dtype=np.float64)
    if d.ndim != 1 or d.size == 0 or not np.isfinite(d).all() or (d < 0).any():
        raise MetricError(f"{name} must be a non-negative distribution")
    if not np.isclose(d.sum(), 1.0, atol=1e-9):
        raise MetricError(f"{name} must sum to 1")
    return d


def log_likelihood(probabilities: ArrayLike, outcomes: ArrayLike) -> float:
    """Mean log-probability of each row's observed outcome; (rows, categories) in."""
    p = np.asarray(probabilities, dtype=np.float64)
    k = np.asarray(outcomes, dtype=np.int64)
    if p.ndim != 2 or k.shape != (len(p),) or len(p) == 0:
        raise MetricError("probabilities are (rows, categories), with one outcome per row")
    if k.min() < 0 or k.max() >= p.shape[1]:
        raise MetricError(f"outcomes must lie in 0..{p.shape[1] - 1}")
    if not np.allclose(p.sum(axis=1), 1.0, atol=1e-9) or (p < 0).any():
        raise MetricError("each row of probabilities must be a distribution")
    return float(np.log(np.clip(p[np.arange(len(p)), k], _EPS, None)).mean())


def wasserstein_1(p: ArrayLike, q: ArrayLike) -> float:
    """Earth-mover distance between two distributions over ordered, unit-spaced categories."""
    a, b = _distribution(p, "the first distribution"), _distribution(q, "the second distribution")
    if a.shape != b.shape:
        raise MetricError("distributions need the same categories")
    return float(np.abs(np.cumsum(a) - np.cumsum(b)).sum())


def jensen_shannon(p: ArrayLike, q: ArrayLike) -> float:
    """Jensen-Shannon divergence, in bits: 0 for equal distributions, 1 for disjoint ones."""
    a, b = _distribution(p, "the first distribution"), _distribution(q, "the second distribution")
    if a.shape != b.shape:
        raise MetricError("distributions need the same categories")
    m = (a + b) / 2

    def kl(x: Floats) -> float:
        support = x > 0
        return float((x[support] * np.log2(x[support] / m[support])).sum())

    return (kl(a) + kl(b)) / 2


def top_k_agreement(fitted: ArrayLike, reference: ArrayLike, k: int) -> float:
    """The mean share of each category's `k` nearest neighbours that two similarity matrices
    agree on."""
    a, b = np.asarray(fitted, dtype=np.float64), np.asarray(reference, dtype=np.float64)
    if a.ndim != 2 or a.shape != b.shape or a.shape[0] != a.shape[1]:
        raise MetricError("similarity matrices must be square and the same size")
    n = len(a)
    if not 1 <= k < n:
        raise MetricError(f"k must lie in 1..{n - 1}")

    def neighbours(m: Floats, i: int) -> set[int]:
        row = m[i].copy()
        row[i] = -np.inf
        return set(np.lexsort((np.arange(n), -row))[:k].tolist())

    return float(np.mean([len(neighbours(a, i) & neighbours(b, i)) / k for i in range(n)]))


def wape(actual: ArrayLike, forecast: ArrayLike) -> float:
    """Absolute error summed over cells, divided by summed absolute actuals. All-zero actuals
    give 0 for a perfect forecast and infinity otherwise."""
    a, f = np.asarray(actual, dtype=np.float64), np.asarray(forecast, dtype=np.float64)
    if a.shape != f.shape or a.size == 0 or not (np.isfinite(a).all() and np.isfinite(f).all()):
        raise MetricError("actuals and forecasts need finite values for the same cells")
    error = float(np.abs(a - f).sum())
    scale = float(np.abs(a).sum())
    if scale == 0:
        return 0.0 if error == 0 else float("inf")
    return error / scale


@dataclass(frozen=True)
class Metric:
    name: str
    direction: Direction
    compute: Callable[..., float]

    def better(self, a: float, b: float) -> bool:
        """Whether `a` is strictly better than `b`."""
        return a > b if self.direction is Direction.HIGHER else a < b


_ALL: Final = {
    m.name: m
    for m in (
        Metric("average_precision", Direction.HIGHER, average_precision),
        Metric("auc", Direction.HIGHER, auc),
        Metric("log_loss", Direction.LOWER, log_loss),
        Metric("brier", Direction.LOWER, brier),
        Metric("ks", Direction.HIGHER, ks),
        Metric("ece", Direction.LOWER, ece),
        Metric("log_likelihood", Direction.HIGHER, log_likelihood),
        Metric("wasserstein_1", Direction.LOWER, wasserstein_1),
        Metric("jensen_shannon", Direction.LOWER, jensen_shannon),
        Metric("top_k_agreement", Direction.HIGHER, top_k_agreement),
        Metric("wape", Direction.LOWER, wape),
    )
}
METRICS: Final[Mapping[str, Metric]] = MappingProxyType(_ALL)

#: Each family's metrics, primary first.
FAMILY_METRICS: Final[Mapping[str, tuple[str, ...]]] = MappingProxyType(
    {
        "classifier": ("log_loss", "average_precision", "auc", "brier", "ks", "ece"),
        "generator": ("log_likelihood", "wasserstein_1", "jensen_shannon", "top_k_agreement"),
        "count": ("wape",),
    }
)


def primary(family: str) -> Metric:
    try:
        return METRICS[FAMILY_METRICS[family][0]]
    except KeyError:
        known = ", ".join(FAMILY_METRICS)
        raise MetricError(f"no model family {family!r}; known: {known}") from None
