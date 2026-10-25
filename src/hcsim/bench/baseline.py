"""Benchmark baselines: a suite's measurements, stored, and later runs compared with them.

A **baseline** (`Baseline`) keeps one run of the scale harness's suite: the machine it ran on,
the package version, the date, every case's measurement, and the capacity model fitted to
them. The committed one is `benchmarks/baseline.json`; `make bench-baseline` regenerates it.

**Comparing** (`compare`) sets a later run beside a baseline, case by case and measure by
measure: the seconds to prepare, the seconds to simulate, and the peak memory. A measure that
grew by more than the threshold, 20% by default, is **flagged**, unless it grew by less than a
floor that's noise at that scale: a quarter of a second, or 32 MiB. A case the baseline lacks is
noted, and so is a run on another machine, whose timings don't compare.

Nothing here gates CI: shared runners are noisy, so a regression is flagged for a person to
read. `hcsim bench compare` exits nonzero when anything is flagged.
"""

from __future__ import annotations

import dataclasses
import datetime as dt
import json
import os
import platform
import subprocess
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from importlib.metadata import version
from pathlib import Path
from typing import Any, Final

from hcsim.bench.capacity import CapacityModel
from hcsim.bench.scale import Measurement

BASELINE_VERSION: Final = 1
#: The committed baseline, relative to the repository's root.
BASELINE_PATH: Final = Path("benchmarks") / "baseline.json"
THRESHOLD: Final = 0.20
#: Each measure, and the growth below which it's noise.
MEASURES: Final[Mapping[str, float]] = {
    "prepare_seconds": 0.25,
    "simulate_seconds": 0.25,
    "peak_mib": 32.0,
}


class BaselineError(ValueError):
    """A baseline that can't be read or compared."""


def _memory_gib() -> float:
    if platform.system() == "Darwin":
        out = subprocess.run(
            ["sysctl", "-n", "hw.memsize"], capture_output=True, text=True, check=False
        ).stdout.strip()
        return round(int(out) / 2**30, 1) if out.isdigit() else 0.0
    try:
        for line in Path("/proc/meminfo").read_text().splitlines():
            if line.startswith("MemTotal:"):
                return round(int(line.split()[1]) / 2**20, 1)
    except OSError:
        pass
    return 0.0


def machine() -> dict[str, str]:
    """What the harness ran on: enough to say whether two runs' timings compare."""
    return {
        "system": platform.system(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "cpus": str(os.cpu_count() or 0),
        "memory_gib": str(_memory_gib()),
        "python": platform.python_version(),
    }


@dataclass(frozen=True)
class Baseline:
    machine: Mapping[str, str]
    hcsim_version: str
    measured_on: str
    measurements: tuple[Measurement, ...]
    capacity: CapacityModel

    @classmethod
    def of(cls, measurements: Sequence[Measurement]) -> Baseline:
        """A baseline of `measurements`, run here, today, with its capacity model fitted."""
        return cls(
            machine(),
            version("hcsim"),
            dt.date.today().isoformat(),
            tuple(measurements),
            CapacityModel.fit(measurements),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": BASELINE_VERSION,
            "machine": dict(self.machine),
            "hcsim_version": self.hcsim_version,
            "measured_on": self.measured_on,
            "measurements": [dataclasses.asdict(m) for m in self.measurements],
            "capacity": self.capacity.to_dict(),
        }

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2) + "\n")

    @classmethod
    def load(cls, path: Path) -> Baseline:
        try:
            data = json.loads(path.read_text())
        except (OSError, ValueError) as error:
            raise BaselineError(f"can't read the baseline at {path}: {error}") from None
        if data.get("version") != BASELINE_VERSION:
            raise BaselineError(f"a baseline is version {BASELINE_VERSION}")
        return cls(
            data["machine"],
            data["hcsim_version"],
            data["measured_on"],
            tuple(Measurement(**m) for m in data["measurements"]),
            CapacityModel.from_dict(data["capacity"]),
        )


@dataclass(frozen=True)
class Flag:
    """A measure that grew past the threshold."""

    case: str
    measure: str
    now: float
    then: float

    @property
    def growth(self) -> float:
        return self.now / self.then - 1 if self.then else float("inf")

    def __str__(self) -> str:
        return f"{self.case}: {self.measure} {self.now:g}, from {self.then:g} (+{self.growth:.0%})"


@dataclass(frozen=True)
class Comparison:
    flags: tuple[Flag, ...]
    #: Cases the run measured that the baseline lacks.
    new: tuple[str, ...]
    #: Whether the run and the baseline ran on the same kind of machine.
    same_machine: bool

    @property
    def flagged(self) -> bool:
        return bool(self.flags)


def compare(
    run: Sequence[Measurement],
    baseline: Baseline,
    threshold: float = THRESHOLD,
    here: Mapping[str, str] | None = None,
) -> Comparison:
    """Every measure of `run` that grew past `threshold` over `baseline`'s same case."""
    if threshold <= 0:
        raise BaselineError(f"a threshold is a positive fraction, not {threshold}")
    then = {m.case: m for m in baseline.measurements}
    flags, new = [], []
    for m in run:
        old = then.get(m.case)
        if old is None:
            new.append(m.case)
            continue
        for measure, floor in MEASURES.items():
            a, b = float(getattr(m, measure)), float(getattr(old, measure))
            if a > b * (1 + threshold) and a - b > floor:
                flags.append(Flag(m.case, measure, a, b))
    same = dict(here if here is not None else machine()) == dict(baseline.machine)
    return Comparison(tuple(flags), tuple(new), same)
