"""The run-mode ladder, in its own module so the configuration and gates can both read it.

A study varies one thing inside an otherwise pinned run. Each rung of the ladder holds some of a
run's inputs at their realized values, its **pins**, and releases one more than the rung above,
so a regression can be traced to the input that caused it:

| Mode | Pinned | Isolates |
|---|---|---|
| `replay` | movements, volumes, and quota | orchestration, data fidelity, and the Financial Layer |
| `guided_actuals` | volumes and quota | quota sampling, given correct volumes |
| `guided_forecast` | quota | the guidance forecast |
| `unguided` | nothing | the domain models' own rates |

- **Movements** are who moves where and when: realized history, replayed as known events.
- **Volumes** are how many events of each kind happen per group and month, at their realized
  counts.
- **Quota** means the counts come from guidance, which domains realize exactly by quota
  sampling.

**Derived fields.** Each rung's other properties follow from its pins (`Rung`):
- it **reads guidance** when quota is pinned and movements aren't: replay's quota is history's
  own;
- that guidance is **realized** when volumes are pinned too, as `guided_actuals` reads the
  window's actual counts;
- it **replays history** when movements are pinned;
- it **needs actuals**, and so a closed window, when volumes are pinned;
- it **releases** the one pin the rung above holds and it doesn't.

**Import-time checks.** The ladder is checked when this module is imported (`check_ladder`):
every mode appears once, the top rung pins everything and the bottom nothing, and each rung pins
exactly the rung above's pins less one. A ladder that breaks any of these refuses to import, so
no run can read an inconsistent one.
"""

from __future__ import annotations

import itertools
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from types import MappingProxyType
from typing import Final


class RunMode(StrEnum):
    """The run-mode ladder, from most pinned to least."""

    REPLAY = "replay"
    GUIDED_ACTUALS = "guided_actuals"
    GUIDED_FORECAST = "guided_forecast"
    UNGUIDED = "unguided"


class Pin(StrEnum):
    """An input a run mode holds at its realized value."""

    MOVEMENTS = "movements"
    VOLUMES = "volumes"
    QUOTA = "quota"


class LadderError(RuntimeError):
    """A ladder that doesn't hold together: a defect, refused at import."""


@dataclass(frozen=True)
class Rung:
    """One run mode: the inputs it pins, what it isolates, and the properties that follow."""

    mode: RunMode
    pins: frozenset[Pin]
    isolates: str

    @property
    def reads_guidance(self) -> bool:
        """Whether its counts come from guidance: quota pinned, movements free."""
        return Pin.QUOTA in self.pins and Pin.MOVEMENTS not in self.pins

    @property
    def realized_guidance(self) -> bool:
        """Whether the guidance it reads is the window's realized counts."""
        return self.reads_guidance and Pin.VOLUMES in self.pins

    @property
    def replays_history(self) -> bool:
        return Pin.MOVEMENTS in self.pins

    @property
    def needs_actuals(self) -> bool:
        """Whether it reads the window's actuals, and so needs a closed window."""
        return Pin.VOLUMES in self.pins


#: The ladder, from most pinned to least.
LADDER: Final[tuple[Rung, ...]] = (
    Rung(
        RunMode.REPLAY,
        frozenset({Pin.MOVEMENTS, Pin.VOLUMES, Pin.QUOTA}),
        "orchestration, data fidelity, and the Financial Layer",
    ),
    Rung(
        RunMode.GUIDED_ACTUALS,
        frozenset({Pin.VOLUMES, Pin.QUOTA}),
        "quota sampling, given correct volumes",
    ),
    Rung(RunMode.GUIDED_FORECAST, frozenset({Pin.QUOTA}), "the guidance forecast"),
    Rung(RunMode.UNGUIDED, frozenset(), "the domain models' own rates"),
)


def check_ladder(ladder: Sequence[Rung]) -> list[str]:
    """Every way `ladder` fails to be one: a mode missing or repeated, a top rung that doesn't
    pin everything or a bottom one that pins something, or a rung that doesn't release exactly
    one of the rung above's pins."""
    problems = []
    modes = [r.mode for r in ladder]
    missing = sorted(m.value for m in RunMode if m not in modes)
    repeated = sorted({m.value for m in modes if modes.count(m) > 1})
    if missing:
        problems.append(f"the ladder is missing {missing}")
    if repeated:
        problems.append(f"the ladder repeats {repeated}")
    if ladder and ladder[0].pins != frozenset(Pin):
        problems.append(f"the top rung, {ladder[0].mode}, must pin everything")
    if ladder and ladder[-1].pins:
        problems.append(f"the bottom rung, {ladder[-1].mode}, must pin nothing")
    for above, below in itertools.pairwise(ladder):
        if not (below.pins < above.pins and len(above.pins - below.pins) == 1):
            problems.append(f"{below.mode} must pin {above.mode}'s pins less exactly one")
    return problems


def releases(mode: RunMode) -> Pin | None:
    """The pin `mode`'s rung releases from the rung above, or None for the top rung."""
    k = next(i for i, r in enumerate(LADDER) if r.mode is mode)
    if k == 0:
        return None
    (released,) = LADDER[k - 1].pins - LADDER[k].pins
    return released


def rung(mode: RunMode | str) -> Rung:
    """`mode`'s rung, by the mode or its name."""
    return RUNGS[RunMode(mode)]


_problems = check_ladder(LADDER)
if _problems:  # pragma: no cover - a defect in the table above
    raise LadderError("; ".join(_problems))
RUNGS: Final[Mapping[RunMode, Rung]] = MappingProxyType({r.mode: r for r in LADDER})
