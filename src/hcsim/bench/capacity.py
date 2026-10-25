"""The capacity model: a run's time and peak memory, predicted from its size.

The model is fitted per forecast scope from the scale harness's measurements of simulations,
since SBC dominates both time and memory where a run computes it. Each cost is linear in the
run's size:

- preparing takes `a + b · employees` seconds: reading the world, fitting, and seeding;
- simulating takes `a + b · employee-months` seconds, where employee-months are the employees
  seeded times months times trajectories;
- the peak is `base + per_employee · employees + per_held · employees · held` MiB, where a run
  holds every trajectory, or one at a time when they stream.

Every coefficient is fitted by least squares on the cases' relative errors, so a small case
counts as much as one a hundred times its size, and kept nonnegative: where the unconstrained
fit would make one negative, the best fit over the remaining terms is used instead. Each fit
reports its worst relative error over the cases it was fitted to, so a prediction says how far
to trust it. The error is relative to the case's value or the measure's noise floor, whichever
is larger, so a case that takes hundredths of a second can't dominate it. Outside the sizes it
was fitted on, a prediction is an extrapolation.

**Planning a run** (`plan`): its predicted time and peak, and, for a memory budget, the most
trajectories it can hold at once. That says whether it fits as it stands, has to stream its
trajectories, or doesn't fit even streaming.
"""

from __future__ import annotations

import itertools
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Final

import numpy as np

from hcsim.bench.scale import SIMULATED, Measurement

FIT_VERSION: Final = 1
#: Each measure's noise floor: below it, an error is relative to the floor, not the value.
FLOORS: Final = {"prepare": 0.25, "simulate": 0.25, "memory": 32.0}


class CapacityError(ValueError):
    """A model that can't be fitted, or a prediction it can't make."""


def nonnegative_fit(features: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Least-squares coefficients for `features @ x ≈ target` with every `x >= 0`, by the best
    unconstrained fit over each subset of the features whose coefficients come out nonnegative.
    Exact for the few features the model has."""
    n = features.shape[1]
    best, best_error = np.zeros(n), float(np.sum(target**2))
    for k in range(1, n + 1):
        for subset in itertools.combinations(range(n), k):
            columns = list(subset)
            x, *_ = np.linalg.lstsq(features[:, columns], target, rcond=None)
            if (x < 0).any():
                continue
            full = np.zeros(n)
            full[columns] = x
            error = float(np.sum((features @ full - target) ** 2))
            if error < best_error - 1e-12:
                best, best_error = full, error
    return best


def _relative_fit(features: np.ndarray, target: np.ndarray, floor: float) -> np.ndarray:
    """A nonnegative fit that weighs each case by its own size, floored at the measure's noise,
    so it minimizes the cases' relative errors rather than their absolute ones."""
    weight = 1 / np.maximum(np.abs(target), floor)
    return nonnegative_fit(features * weight[:, None], target * weight)


def _worst(
    features: np.ndarray, coefficients: np.ndarray, target: np.ndarray, floor: float
) -> float:
    fitted = features @ coefficients
    return float(np.max(np.abs(fitted - target) / np.maximum(np.abs(target), floor)))


@dataclass(frozen=True)
class Fit:
    """One scope's coefficients, and the worst relative error of each over its cases."""

    prepare: tuple[float, float]
    simulate: tuple[float, float]
    memory: tuple[float, float, float]
    cases: int
    worst: Mapping[str, float]

    @classmethod
    def of(cls, measurements: Sequence[Measurement]) -> Fit:
        if len(measurements) < 3:
            raise CapacityError(f"a scope's fit needs three cases or more, not {len(measurements)}")
        employees = np.array([m.employees for m in measurements], dtype=np.float64)
        months = np.array([m.employee_months for m in measurements], dtype=np.float64)
        held = np.array([m.held for m in measurements], dtype=np.float64)
        ones = np.ones(len(measurements))
        prepare_x = np.column_stack([ones, employees])
        simulate_x = np.column_stack([ones, months])
        memory_x = np.column_stack([ones, employees, employees * held])
        prepare_y = np.array([m.prepare_seconds for m in measurements])
        simulate_y = np.array([m.simulate_seconds for m in measurements])
        memory_y = np.array([m.peak_mib for m in measurements])
        prepare = _relative_fit(prepare_x, prepare_y, FLOORS["prepare"])
        simulate = _relative_fit(simulate_x, simulate_y, FLOORS["simulate"])
        memory = _relative_fit(memory_x, memory_y, FLOORS["memory"])
        worst = {
            "prepare": _worst(prepare_x, prepare, prepare_y, FLOORS["prepare"]),
            "simulate": _worst(simulate_x, simulate, simulate_y, FLOORS["simulate"]),
            "memory": _worst(memory_x, memory, memory_y, FLOORS["memory"]),
        }
        return cls(
            (float(prepare[0]), float(prepare[1])),
            (float(simulate[0]), float(simulate[1])),
            (float(memory[0]), float(memory[1]), float(memory[2])),
            len(measurements),
            {k: round(v, 4) for k, v in worst.items()},
        )


@dataclass(frozen=True)
class Prediction:
    prepare_seconds: float
    simulate_seconds: float
    peak_mib: float


@dataclass(frozen=True)
class Plan:
    """A run's predicted costs within a memory budget, holding its trajectories and streaming
    them, and the most it can hold at once: so whether it fits as it is, must stream, or can't
    fit."""

    held: Prediction
    streamed: Prediction
    budget_mib: float
    most_held: int

    @property
    def fits(self) -> bool:
        return self.held.peak_mib <= self.budget_mib

    @property
    def advice(self) -> str:
        if self.fits:
            return "fits as it is"
        if self.streamed.peak_mib <= self.budget_mib:
            return f"stream its trajectories: it can hold {self.most_held} at once"
        return "doesn't fit, even streaming its trajectories"


@dataclass(frozen=True)
class CapacityModel:
    """A fit per forecast scope."""

    fits: Mapping[str, Fit]

    @classmethod
    def fit(cls, measurements: Sequence[Measurement]) -> CapacityModel:
        scopes = sorted({m.scope for m in measurements if m.scope in SIMULATED})
        return cls({s: Fit.of([m for m in measurements if m.scope == s]) for s in scopes})

    def _scope(self, scope: str) -> Fit:
        if scope not in self.fits:
            raise CapacityError(f"no fit for the scope {scope!r}; fitted: {', '.join(self.fits)}")
        return self.fits[scope]

    def predict(
        self, employees: int, trajectories: int, months: int, scope: str, streaming: bool = False
    ) -> Prediction:
        if min(employees, trajectories, months) < 1:
            raise CapacityError("a run has at least one employee, trajectory, and month")
        f = self._scope(scope)
        held = 1 if streaming else trajectories
        work = employees * months * trajectories
        return Prediction(
            round(f.prepare[0] + f.prepare[1] * employees, 3),
            round(f.simulate[0] + f.simulate[1] * work, 3),
            round(f.memory[0] + f.memory[1] * employees + f.memory[2] * employees * held, 1),
        )

    def most_held(self, employees: int, scope: str, budget_mib: float) -> int:
        """The most trajectories a run of `employees` can hold at once within `budget_mib`."""
        base, per_employee, per_held = self._scope(scope).memory
        room = budget_mib - base - per_employee * employees
        if room < per_held * employees:
            return 0
        if per_held == 0:
            return 2**31 - 1
        return math.floor(room / (per_held * employees))

    def plan(
        self, employees: int, trajectories: int, months: int, scope: str, budget_mib: float
    ) -> Plan:
        return Plan(
            self.predict(employees, trajectories, months, scope),
            self.predict(employees, trajectories, months, scope, streaming=True),
            budget_mib,
            min(self.most_held(employees, scope, budget_mib), trajectories),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": FIT_VERSION,
            "fits": {
                s: {
                    "prepare": list(f.prepare),
                    "simulate": list(f.simulate),
                    "memory": list(f.memory),
                    "cases": f.cases,
                    "worst": dict(f.worst),
                }
                for s, f in self.fits.items()
            },
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> CapacityModel:
        if data.get("version") != FIT_VERSION:
            raise CapacityError(f"a capacity model is version {FIT_VERSION}")
        return cls(
            {
                s: Fit(
                    (f["prepare"][0], f["prepare"][1]),
                    (f["simulate"][0], f["simulate"][1]),
                    (f["memory"][0], f["memory"][1], f["memory"][2]),
                    f["cases"],
                    f["worst"],
                )
                for s, f in data["fits"].items()
            }
        )
