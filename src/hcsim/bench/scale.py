"""The scale harness: how a run's time and memory grow with its size.

A **case** (`Case`) is one run: a world preset and its seed, a forecast scope, the run's
trajectories and months, and whether its trajectories stream. Two suites hold the cases:

- **`SUITE`** is what the stored baseline measures. Its cases are sized to run on a laptop in
  about a minute and a half once their worlds are cached.
- **`LARGE`** runs the `large` preset, for profiling at 500k employees.

Each case runs in a fresh process, so its peak memory is its own, under the world bench's
watchdog (`world.run_child`), which stops one that passes a memory limit. In it, the harness:

- prepares the run as a study's preflight does (`studies.harness.preflight`): it reads the
  world, fits the models, and seeds the state, timed;
- simulates the run unguided, timed, with its trajectories held, or streamed to a sink that
  keeps only what it counts.

A case of the `guidance` scope fits guidance on its world instead, as the explorer's valves do,
and aligns it to a plan: its preparing is the fit, and its simulating the alignment and the
artifact's writing. The capacity model fits only the cases that simulate.

A **measurement** (`Measurement`) records the run's size (the employees and openings seeded,
its trajectories and months, the tranches it tracked) and its costs: each phase's seconds, and
the process's peak memory once prepared and at the end. A world is generated once into the
cache directory and reused.

Timings are machine-specific and noisy, so they're recorded and compared, never gated in CI.
"""

from __future__ import annotations

import dataclasses
import json
import sys
import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Final

from hcsim.bench.world import peak_rss_mib, run_child
from hcsim.synth.config import preset as world_preset
from hcsim.synth.world import generate_world, write_world


@dataclass(frozen=True)
class Case:
    """One benchmark run."""

    name: str
    preset: str
    scope: str
    trajectories: int
    months: int = 12
    streaming: bool = False
    #: The world's seed.
    seed: int = 0


@dataclass(frozen=True)
class Measurement:
    """One case's run: its size and its costs."""

    case: str
    preset: str
    scope: str
    trajectories: int
    months: int
    streaming: bool
    employees: int
    openings: int
    tranches: int
    prepare_seconds: float
    simulate_seconds: float
    peak_mib: float
    #: The process's peak once prepared, before simulating: the world read, fitted, and seeded.
    prepare_peak_mib: float = 0.0

    @property
    def employee_months(self) -> int:
        """The work the run did: employees seeded, times months, times trajectories."""
        return self.employees * self.months * self.trajectories

    @property
    def held(self) -> int:
        """The trajectories the run held at once: all of them, or one when they stream."""
        return 1 if self.streaming else self.trajectories


_HCF, _SBC = "hcf", "hcf+sbc"
#: A case of this scope fits guidance instead of simulating (see `measure`).
GUIDANCE: Final = "guidance"
#: The scopes whose cases simulate, which the capacity model fits.
SIMULATED: Final = (_HCF, _SBC)
#: The cases the stored baseline measures.
SUITE: Final = (
    Case("tiny-hcf-2", "tiny", _HCF, 2),
    Case("small-hcf-2", "small", _HCF, 2),
    Case("small-hcf-8", "small", _HCF, 8),
    Case("small-sbc-2", "small", _SBC, 2),
    Case("small-sbc-8", "small", _SBC, 8),
    Case("small-sbc-8-streamed", "small", _SBC, 8, streaming=True),
    Case("medium-hcf-2", "medium", _HCF, 2),
    Case("medium-hcf-8", "medium", _HCF, 8),
    Case("medium-sbc-2", "medium", _SBC, 2),
    Case("small-guidance", "small", GUIDANCE, 1, months=6),
    Case("medium-guidance", "medium", GUIDANCE, 1, months=6),
)
#: The cases at 500k employees.
LARGE: Final = (
    Case("large-hcf-2", "large", _HCF, 2),
    Case("large-sbc-1-streamed", "large", _SBC, 1, streaming=True),
)
#: Every case, by name.
CASES: Final = {c.name: c for c in (*SUITE, *LARGE)}


def world_path(case: Case, cache: Path) -> Path:
    return cache / f"{case.preset}-{case.seed}.duckdb"


def ensure_world(case: Case, cache: Path) -> Path:
    """The case's world, generated into `cache` if it isn't there yet."""
    path = world_path(case, cache)
    if not path.is_file():
        cache.mkdir(parents=True, exist_ok=True)
        partial = path.with_suffix(".partial")
        write_world(generate_world(world_preset(case.preset, case.seed)), partial)
        partial.rename(path)
    return path


def measure(case: Case, world: Path) -> Measurement:
    """Run `case` on `world` in this process, timing each phase."""
    if case.scope == GUIDANCE:
        return _guidance(case, world)
    # Imported here, so the harness's own process never loads the engine.
    from hcsim.engine.loop import simulate
    from hcsim.studies.harness import ITSELF, StudyError, preflight
    from hcsim.studies.profile import StudyProfile

    profile = StudyProfile.from_mapping(
        {
            "name": case.name,
            "seed": 1,
            "trajectories": case.trajectories,
            "months": case.months,
            "mode": "unguided",
            "scope": case.scope,
            "evaluators": ["invariants"],
        }
    )
    started = time.perf_counter()
    checked = preflight(profile, world)
    if not checked.ok or checked.inputs is None:
        raise StudyError(list(checked.problems))
    prepared = time.perf_counter()
    prepare_peak = peak_rss_mib()
    config = checked.configs[ITSELF.name]
    tranches = 0
    if case.streaming:

        def count(t: object) -> None:
            nonlocal tranches
            held = getattr(t, "tranches", None)
            tranches += len(held) if held is not None else 0

        config = dataclasses.replace(config, results_streaming=True)
        simulate(config, checked.inputs.state, sink=count)
    else:
        result = simulate(config, checked.inputs.state)
        tranches = sum(len(t.tranches) for t in result.trajectories if t.tranches is not None)
    simulated = time.perf_counter()
    seed = checked.inputs.seed
    return Measurement(
        case=case.name,
        preset=case.preset,
        scope=case.scope,
        trajectories=case.trajectories,
        months=case.months,
        streaming=case.streaming,
        employees=seed.employees_seeded,
        openings=seed.openings_seeded,
        tranches=tranches,
        prepare_seconds=round(prepared - started, 3),
        simulate_seconds=round(simulated - prepared, 3),
        peak_mib=round(peak_rss_mib(), 1),
        prepare_peak_mib=round(prepare_peak, 1),
    )


def _guidance(case: Case, world: Path) -> Measurement:
    """Fit guidance on `world` over `months`, as the explorer does, and align it greedily to a
    plan holding each cost center at its last headcount. Fitting is timed as preparing, and
    aligning and writing as simulating; the employees are the headcount the series end on."""
    from hcsim.guidance.config import GuidanceConfig
    from hcsim.guidance.pipeline import finish, fit
    from hcsim.model.core.alignment_builders import synthetic
    from hcsim.model.core.categories import ANCHOR

    config = GuidanceConfig.from_mapping(
        {"horizon": case.months, "alignment": {"method": "greedy"}}
    )
    started = time.perf_counter()
    fitted = fit(world, config)
    prepared = time.perf_counter()
    prepare_peak = peak_rss_mib()
    series = fitted.context["series"]
    heads = {g: float(series.history(g, ANCHOR)[-1]) for g, c in series.keys if c is ANCHOR}
    plan = synthetic(
        "constant", series.grouping, heads, start=series.snapshot + 1, months=case.months
    )
    aligning = time.perf_counter()
    finish(fitted, plan)
    finished = time.perf_counter()
    return Measurement(
        case=case.name,
        preset=case.preset,
        scope=case.scope,
        trajectories=case.trajectories,
        months=case.months,
        streaming=case.streaming,
        employees=round(sum(heads.values())),
        openings=0,
        tranches=0,
        prepare_seconds=round(prepared - started, 3),
        simulate_seconds=round(finished - aligning, 3),
        peak_mib=round(peak_rss_mib(), 1),
        prepare_peak_mib=round(prepare_peak, 1),
    )


def run(cases: Sequence[Case], cache: Path, max_rss_gib: float | None = None) -> list[Measurement]:
    """Measure each case in a fresh process, in order, each world generated once."""
    measurements = []
    for case in cases:
        path = world_path(case, cache)
        if not path.is_file():
            command = [
                sys.executable,
                "-m",
                "hcsim.bench.scale",
                "world",
                json.dumps(dataclasses.asdict(case)),
                str(cache),
            ]
            run_child(command, f"{case.preset!r} world", max_rss_gib)
        command = [
            sys.executable,
            "-m",
            "hcsim.bench.scale",
            "case",
            json.dumps(dataclasses.asdict(case)),
            str(path),
        ]
        line = run_child(command, f"{case.name!r} case", max_rss_gib)
        measurements.append(Measurement(**json.loads(line)))
    return measurements


def main(argv: list[str]) -> int:
    """The child's entry point: `world CASE CACHE`, `case CASE WORLD`, or `profile CASE WORLD
    OUT TOP` (see `profiling`)."""
    action, spec, where, *rest = argv
    case = Case(**json.loads(spec))
    if action == "world":
        print(ensure_world(case, Path(where)))
    elif action == "profile":
        from hcsim.bench.profiling import profile_here

        out, top = rest
        print(json.dumps(dataclasses.asdict(profile_here(case, Path(where), Path(out), int(top)))))
    else:
        print(json.dumps(dataclasses.asdict(measure(case, Path(where)))))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
