"""Tuning: a model's hyperparameters searched by Optuna's TPE sampler.

A tunable model declares its search space (`SPACE`): each hyperparameter's range, an integer
or a float, on a linear or log scale. It's built with the values chosen, which override its
defaults.

Tuning is nested inside evaluation, so a model's score never sees the months it was tuned
on. For each outer fold, the model is tuned on that fold's training window alone, by its own
rolling-origin folds (`inner_splits` of them, each training on at least
`inner_min_train_months`); the trial with the lowest mean inner WAPE wins, and the outer fold
scores the model fitted with it. The final forecast tunes on the whole history the same way.

Every study is seeded, so tuning is reproducible: the same history, configuration, and seed
always choose the same values. A training window too short for any inner fold keeps the
model's defaults.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

Floats = NDArray[np.float64]


@dataclass(frozen=True)
class IntParam:
    low: int
    high: int
    log: bool = False

    def suggest(self, trial: Any, name: str) -> int:
        return int(trial.suggest_int(name, self.low, self.high, log=self.log))


@dataclass(frozen=True)
class FloatParam:
    low: float
    high: float
    log: bool = False

    def suggest(self, trial: Any, name: str) -> float:
        return float(trial.suggest_float(name, self.low, self.high, log=self.log))


Param = IntParam | FloatParam
Space = Mapping[str, Param]


@dataclass(frozen=True)
class TuningSettings:
    trials: int = 20
    seed: int = 0
    inner_splits: int = 2
    inner_min_train_months: int = 24


def tune(
    space: Space,
    objective: Callable[[dict[str, Any]], float],
    settings: TuningSettings,
) -> dict[str, Any]:
    """The values in `space` that minimize `objective`, by a seeded TPE search. An objective
    that's infinite for every trial keeps the defaults: no values are returned."""
    import optuna  # imported when a model is first tuned

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    sampler = optuna.samplers.TPESampler(seed=settings.seed)
    study = optuna.create_study(direction="minimize", sampler=sampler)

    def trial_score(trial: Any) -> float:
        values = {name: param.suggest(trial, name) for name, param in space.items()}
        score = objective(values)
        return score if np.isfinite(score) else float("inf")

    study.optimize(trial_score, n_trials=settings.trials, n_jobs=1)
    if not np.isfinite(study.best_value):
        return {}
    return dict(study.best_params)
