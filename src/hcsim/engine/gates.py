"""Gates: whether a domain takes part in a run, and whether it runs in a given month.

- A **registration gate** decides once, when a run starts. A domain registers unless one of its
  gates closes it: switched off, or limited to run modes the run isn't in. Other movement, for
  instance, registers only in replay.
- A **step gate** decides every month. A domain runs only in the calendar months its gate lists
  (every month, if it lists none), and then only with its probability. The draw comes from the
  gate's own seeded stream, so a stochastic gate is reproducible and perturbs no other stream. A
  domain with several step gates runs when all of them open.

Every month a step gate keeps a registered domain from running is a *bypass*, counted per
domain in each trajectory's results, and a domain no registration gate admits is reported once
as unregistered. Gates serialize to plain data, so a configuration's record carries them.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from hcsim.engine.modes import RunMode


class GateError(ValueError):
    """Gates that can't be declared or registered, every problem listed."""

    def __init__(self, problems: list[str]) -> None:
        self.problems = problems
        super().__init__("bad gates: " + "; ".join(problems))


@dataclass(frozen=True)
class RegistrationGate:
    domain: str
    enabled: bool = True
    #: The run modes the domain takes part in; none means every mode.
    modes: tuple[RunMode, ...] = ()
    reason: str = ""

    def admits(self, mode: RunMode) -> bool:
        return self.enabled and (not self.modes or mode in self.modes)

    def problems(self) -> list[str]:
        return []

    def to_dict(self) -> dict[str, Any]:
        return {
            "gate": "registration",
            "domain": self.domain,
            "enabled": self.enabled,
            "modes": [m.value for m in self.modes],
            "reason": self.reason,
        }


@dataclass(frozen=True)
class StepGate:
    domain: str
    #: The calendar months the domain may run in; none means every month.
    months: tuple[int, ...] = ()
    probability: float = 1.0
    reason: str = ""

    def opens(self, month: int, rng: np.random.Generator) -> bool:
        if self.months and month not in self.months:
            return False
        return self.probability >= 1.0 or bool(rng.random() < self.probability)

    def problems(self) -> list[str]:
        found = []
        if any(not 1 <= m <= 12 for m in self.months) or len(set(self.months)) != len(self.months):
            found.append(f"{self.domain}: months must be distinct, 1 to 12: {list(self.months)}")
        if not 0.0 <= self.probability <= 1.0:
            found.append(f"{self.domain}: a probability must lie in [0, 1]")
        return found

    def to_dict(self) -> dict[str, Any]:
        return {
            "gate": "step",
            "domain": self.domain,
            "months": list(self.months),
            "probability": self.probability,
            "reason": self.reason,
        }


Gate = RegistrationGate | StepGate


def gate_from_dict(data: Mapping[str, Any]) -> Gate:
    try:
        kind = data["gate"]
        if kind == "registration":
            modes = tuple(RunMode(m) for m in data["modes"])
            return RegistrationGate(
                str(data["domain"]), bool(data["enabled"]), modes, str(data["reason"])
            )
        if kind == "step":
            months = tuple(int(m) for m in data["months"])
            return StepGate(
                str(data["domain"]), months, float(data["probability"]), str(data["reason"])
            )
        raise ValueError(f"no gate kind {kind!r}")
    except (KeyError, TypeError, ValueError) as error:
        raise GateError([f"can't read gate {dict(data)!r}: {error}"]) from None


def check(gates: Iterable[Gate]) -> list[str]:
    return [p for g in gates for p in g.problems()]


@dataclass(frozen=True)
class Registered:
    """The domains a run registered, in step order, and their step gates."""

    domains: tuple[str, ...]
    unregistered: tuple[str, ...]
    steps: Mapping[str, tuple[StepGate, ...]]

    def opens(self, domain: str, month: int, rng: np.random.Generator) -> bool:
        return all(g.opens(month, rng) for g in self.steps.get(domain, ()))


def register(domains: Sequence[str], gates: Iterable[Gate], mode: RunMode) -> Registered:
    """Register `domains`, in order, under `gates`. A gate for a domain the engine doesn't
    run is refused."""
    gates = tuple(gates)
    unknown = sorted({g.domain for g in gates} - set(domains))
    if unknown:
        raise GateError(
            [f"no domain {d!r} to gate; the engine runs {', '.join(domains)}" for d in unknown]
        )
    admitted = [
        d
        for d in domains
        if all(g.admits(mode) for g in gates if isinstance(g, RegistrationGate) and g.domain == d)
    ]
    steps = {
        d: tuple(g for g in gates if isinstance(g, StepGate) and g.domain == d) for d in admitted
    }
    return Registered(
        tuple(admitted),
        tuple(d for d in domains if d not in admitted),
        {d: s for d, s in steps.items() if s},
    )
