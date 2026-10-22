"""Combination: weights for the models whose forecasts combine into one.

Every method weighs the `top_k` best-scoring models, the candidates, and gives the rest no
weight. The methods:

- `inverse_error`: weights in proportion to the inverse of each score. Perfect scores share
  the weight equally; with no finite score, every candidate shares it;
- `equal`: the candidates share the weight equally;
- `rank`: the best candidate weighs `k`, the next `k - 1`, down to 1, normalized;
- `softmax`: weights in proportion to `exp(-score / temperature)`, so a lower temperature
  favours the best candidate more;
- `bates_granger`: weights in proportion to the inverse of each candidate's mean squared error
  over the backtest folds, as Bates and Granger combine forecasts by their error variances;
- `ridge`: a stacked meta-learner. It regresses the actuals of the folds every candidate
  forecast on the candidates' fold forecasts, with a ridge penalty that pulls the weights
  towards equal, then clips negative weights to zero and normalizes.

**Fallbacks.** A method that can't weigh the candidates refuses (`CombinationError`), as
`ridge` does with fewer shared fold months than candidates, `bates_granger` with no candidate
error, and `softmax` and `rank` with no finite score. The configuration's fallbacks are tried
in order, and the method that weighed the forecast is reported. With none left, the
candidates share the weight equally.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Final, Protocol

import numpy as np
from numpy.typing import NDArray

from hcsim.guidance.evaluation import FoldScore
from hcsim.guidance.registry import Entry, Registry

Floats = NDArray[np.float64]
Weights = dict[str, float]


class CombinationError(ValueError):
    """A method that can't weigh these candidates; the next fallback tries."""


@dataclass(frozen=True)
class CombinationInputs:
    """What a method weighs: each model's score and backtest folds, and the series."""

    scores: Mapping[str, float]
    folds: Mapping[str, Sequence[FoldScore]]
    series: Floats
    top_k: int = 2
    temperature: float = 0.1
    ridge_alpha: float = 1.0

    @property
    def candidates(self) -> list[str]:
        """The `top_k` best scores, ties broken by name."""
        ranked = sorted(self.scores, key=lambda name: (self.scores[name], name))
        return ranked[: max(1, self.top_k)]


class Combiner(Protocol):
    def __call__(self, inputs: CombinationInputs) -> Weights: ...


def _normalized(raw: Mapping[str, float]) -> Weights:
    total = sum(raw.values())
    if not np.isfinite(total) or total <= 0:
        raise CombinationError("the weights don't sum to a positive total")
    return {name: w / total for name, w in raw.items() if w > 0}


def inverse_error(inputs: CombinationInputs) -> Weights:
    scores, ranked = inputs.scores, inputs.candidates
    perfect = [name for name in ranked if scores[name] == 0.0]
    if perfect:
        return {name: 1.0 / len(perfect) for name in perfect}
    finite = [name for name in ranked if np.isfinite(scores[name])]
    if not finite:
        return {name: 1.0 / len(ranked) for name in ranked}
    return _normalized({name: 1.0 / scores[name] for name in finite})


def equal(inputs: CombinationInputs) -> Weights:
    ranked = inputs.candidates
    return {name: 1.0 / len(ranked) for name in ranked}


def _finite(inputs: CombinationInputs) -> list[str]:
    finite = [n for n in inputs.candidates if np.isfinite(inputs.scores[n])]
    if not finite:
        raise CombinationError("no candidate has a finite score")
    return finite


def rank(inputs: CombinationInputs) -> Weights:
    finite = _finite(inputs)
    return _normalized({name: float(len(finite) - k) for k, name in enumerate(finite)})


def softmax(inputs: CombinationInputs) -> Weights:
    finite = _finite(inputs)
    if inputs.temperature <= 0:
        raise CombinationError("softmax needs a positive temperature")
    scores = np.array([inputs.scores[n] for n in finite])
    logits = -(scores - scores.min()) / inputs.temperature
    return _normalized(dict(zip(finite, np.exp(logits).tolist(), strict=True)))


def bates_granger(inputs: CombinationInputs) -> Weights:
    errors = {}
    for name in inputs.candidates:
        folds = inputs.folds.get(name, ())
        squared = [f.squared_error(inputs.series) for f in folds if len(f.predicted)]
        if squared and np.isfinite(np.mean(squared)):
            errors[name] = float(np.mean(squared))
    if not errors:
        raise CombinationError("no candidate has a backtest error to weigh")
    perfect = [name for name, e in errors.items() if e == 0.0]
    if perfect:
        return {name: 1.0 / len(perfect) for name in perfect}
    return _normalized({name: 1.0 / e for name, e in errors.items()})


def ridge(inputs: CombinationInputs) -> Weights:
    """Ridge weights on the fold forecasts the candidates share, pulled towards equal."""
    names = inputs.candidates
    shared: set[tuple[int, int]] | None = None
    for name in names:
        forecast = {
            (f.train_end, f.test_end) for f in inputs.folds.get(name, ()) if len(f.predicted)
        }
        shared = forecast if shared is None else shared & forecast
    windows = sorted(shared or ())
    months = sum(end - start for start, end in windows)
    if months < len(names):
        raise CombinationError(
            f"ridge needs at least {len(names)} fold months every candidate forecast; "
            f"they share {months}"
        )
    by_window = {
        name: {(f.train_end, f.test_end): f.predicted for f in inputs.folds[name]} for name in names
    }
    design = np.column_stack(
        [np.concatenate([by_window[name][w] for w in windows]) for name in names]
    )
    actual = np.concatenate([inputs.series[start:end] for start, end in windows])
    prior = np.full(len(names), 1.0 / len(names))
    # Minimize |actual - design w|^2 + alpha |w - prior|^2.
    gram = design.T @ design + inputs.ridge_alpha * np.eye(len(names))
    weights = np.linalg.solve(gram, design.T @ actual + inputs.ridge_alpha * prior)
    return _normalized(dict(zip(names, np.clip(weights, 0.0, None).tolist(), strict=True)))


#: The combination methods a configuration can name.
COMBINERS: Final[Registry[Combiner]] = Registry(
    "combination method",
    (
        Entry("inverse_error", lambda: inverse_error),
        Entry("equal", lambda: equal),
        Entry("rank", lambda: rank),
        Entry("softmax", lambda: softmax),
        Entry("bates_granger", lambda: bates_granger),
        Entry("ridge", lambda: ridge),
    ),
)


def combine(
    inputs: CombinationInputs, method: str, fallbacks: Sequence[str] = ()
) -> tuple[Weights, str]:
    """The weights from `method`, or the first of `fallbacks` that can weigh the candidates,
    and the method that did; the candidates share equally when none can."""
    for name in (method, *fallbacks):
        try:
            return COMBINERS.build(name)(inputs), name
        except CombinationError:
            continue
    return equal(inputs), "equal"
