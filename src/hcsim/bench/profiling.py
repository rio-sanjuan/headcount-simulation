"""Profiling a case: where its time goes, function by function, and which phase sets its peak.

`profile` runs one of the scale harness's cases in its own process, as the harness does, but
under the standard library's profiler (`cProfile`). It writes two files:

- `<case>.prof`: the raw statistics, which `pstats` and viewers such as snakeviz read;
- `<case>.txt`: a report. It gives the case's measurement, the process's peak memory once
  prepared and at the end (so which phase set the peak), and the functions that took the most
  time, cumulatively and on their own, each list with hcsim's functions first.

The profiler slows a run, so a profiled case's seconds don't compare with the harness's own.
"""

from __future__ import annotations

import cProfile
import dataclasses
import io
import json
import pstats
import sys
from dataclasses import dataclass
from pathlib import Path

from hcsim.bench.scale import Case, Measurement, measure, world_path
from hcsim.bench.world import run_child


@dataclass(frozen=True)
class Profiled:
    """A profiled case: its measurement and the files it wrote."""

    measurement: Measurement
    stats: Path
    report: Path


def _table(path: Path, key: str, top: int, ours: bool) -> str:
    out = io.StringIO()
    stats = pstats.Stats(str(path), stream=out)
    stats.sort_stats(key)
    if ours:
        stats.print_stats(r"hcsim[/\\]", top)
    else:
        stats.print_stats(top)
    lines = out.getvalue().splitlines()
    start = next((k for k, line in enumerate(lines) if line.lstrip().startswith("ncalls")), 0)
    return "\n".join(lines[start:])


def report(m: Measurement, stats: Path, top: int) -> str:
    """The text report of a profiled case, from its statistics at `stats`."""
    peak = "preparing" if m.prepare_peak_mib >= m.peak_mib else "simulating"
    head = [
        f"case {m.case}: {m.preset}, {m.scope}, {m.trajectories} trajectories of {m.months} "
        f"months{', streamed' if m.streaming else ''}",
        f"size: {m.employees:,} employees, {m.openings:,} openings, {m.tranches:,} tranches",
        f"time under the profiler: prepare {m.prepare_seconds:.2f}s, "
        f"simulate {m.simulate_seconds:.2f}s",
        f"peak memory: {m.prepare_peak_mib:,.0f} MiB once prepared, {m.peak_mib:,.0f} MiB at the "
        f"end; {peak} set it",
    ]
    sections = [
        ("hcsim's functions, by cumulative time", "cumulative", True),
        ("hcsim's functions, by their own time", "tottime", True),
        ("every function, by its own time", "tottime", False),
    ]
    body = [f"\n## {title}\n\n{_table(stats, key, top, ours)}" for title, key, ours in sections]
    return "\n".join(head) + "\n" + "\n".join(body) + "\n"


def profile_here(case: Case, world: Path, out: Path, top: int = 30) -> Measurement:
    """Profile `case` on `world` in this process, writing its statistics and report to `out`."""
    out.mkdir(parents=True, exist_ok=True)
    profiler = cProfile.Profile()
    profiler.enable()
    m = measure(case, world)
    profiler.disable()
    stats_path = out / f"{case.name}.prof"
    profiler.dump_stats(stats_path)
    (out / f"{case.name}.txt").write_text(report(m, stats_path, top))
    return m


def profile(
    case: Case, cache: Path, out: Path, top: int = 30, max_rss_gib: float | None = None
) -> Profiled:
    """Profile `case` in a fresh process, its world generated into `cache` if it isn't there."""
    spec = json.dumps(dataclasses.asdict(case))
    path = world_path(case, cache)
    if not path.is_file():
        command = [sys.executable, "-m", "hcsim.bench.scale", "world", spec, str(cache)]
        run_child(command, f"{case.preset!r} world", max_rss_gib)
    command = [
        sys.executable,
        "-m",
        "hcsim.bench.scale",
        "profile",
        spec,
        str(path),
        str(out),
        str(top),
    ]
    line = run_child(command, f"{case.name!r} profile", max_rss_gib)
    m = Measurement(**json.loads(line))
    return Profiled(m, out / f"{case.name}.prof", out / f"{case.name}.txt")
