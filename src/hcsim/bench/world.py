"""Preset timings for the synthetic world.

Each preset runs in its own process, so its peak memory is its own: generate the world, write
it to DuckDB, and run the coherence checks, timing each step. A watchdog stops a run whose
memory passes a limit, so a preset too big for the machine fails fast rather than swapping.

Timings are recorded, not gated: shared machines are noisy.
"""

from __future__ import annotations

import dataclasses
import json
import resource
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

import duckdb

from hcsim.synth.config import preset as world_preset
from hcsim.synth.validate import validate
from hcsim.synth.world import digest, generate_world, write_world

#: How often the watchdog reads a run's memory, in seconds.
POLL_SECONDS = 0.5


class BenchError(RuntimeError):
    """A run that failed or passed its memory limit."""


@dataclass(frozen=True)
class PresetRun:
    preset: str
    seed: int
    employees: int
    generate_seconds: float
    write_seconds: float
    validate_seconds: float
    peak_rss_mib: float
    rows: int
    checks_passed: bool
    digest: str

    @property
    def total_seconds(self) -> float:
        return self.generate_seconds + self.write_seconds + self.validate_seconds


def peak_rss_mib() -> float:
    """This process's peak resident memory so far. macOS reports bytes, Linux kibibytes."""
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return peak / 2**20 if sys.platform == "darwin" else peak / 2**10


def run_one(name: str, seed: int, directory: Path) -> PresetRun:
    """Run one preset in this process."""
    config = world_preset(name, seed)
    started = time.perf_counter()
    world = generate_world(config)
    generated = time.perf_counter()
    path = directory / f"{name}-{seed}.duckdb"
    write_world(world, path)
    written = time.perf_counter()
    with duckdb.connect(str(path), read_only=True) as con:
        results = validate(con, examples=0)
        rows = 0
        for (table,) in con.execute(
            "SELECT table_name FROM information_schema.tables "
            "WHERE table_schema = 'main' AND table_type = 'BASE TABLE'"
        ).fetchall():
            (count,) = con.execute(f'SELECT count(*) FROM "{table}"').fetchone() or (0,)
            rows += int(count)
    checked = time.perf_counter()
    return PresetRun(
        preset=name,
        seed=seed,
        employees=config.initial_employees,
        generate_seconds=round(generated - started, 3),
        write_seconds=round(written - generated, 3),
        validate_seconds=round(checked - written, 3),
        peak_rss_mib=round(peak_rss_mib(), 1),
        rows=rows,
        checks_passed=all(r.passed for r in results),
        digest=digest(world),
    )


def _rss_mib(pid: int) -> float:
    out = subprocess.run(
        ["ps", "-o", "rss=", "-p", str(pid)], capture_output=True, text=True, check=False
    ).stdout.strip()
    return int(out) / 1024 if out else 0.0


def run_child(command: list[str], label: str, max_rss_gib: float | None = None) -> str:
    """Run `command` in a fresh process, watching its memory, and return its last line of
    output. A process that passes `max_rss_gib` is killed; one that fails is reported with its
    error. `label` names it in either message."""
    with subprocess.Popen(
        command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
    ) as child:
        peak = 0.0
        while child.poll() is None:
            peak = max(peak, _rss_mib(child.pid))
            if max_rss_gib is not None and peak > max_rss_gib * 1024:
                child.kill()
                child.communicate()
                raise BenchError(f"the {label} passed its memory limit of {max_rss_gib} GiB")
            time.sleep(POLL_SECONDS)
        out, err = child.communicate()
        if child.returncode != 0:
            raise BenchError(f"the {label} failed:\n{err.strip()}")
    return out.strip().splitlines()[-1]


def run(presets: list[str], seed: int = 0, max_rss_gib: float | None = None) -> list[PresetRun]:
    """Run each preset in a fresh process, in order, and stop at the first that fails."""
    runs = []
    for name in presets:
        with tempfile.TemporaryDirectory() as directory:
            command = [sys.executable, "-m", "hcsim.bench.world", name, str(seed), directory]
            line = run_child(command, f"{name!r} preset", max_rss_gib)
        runs.append(PresetRun(**json.loads(line)))
    return runs


def main(argv: list[str]) -> int:
    name, seed, directory = argv
    print(json.dumps(dataclasses.asdict(run_one(name, int(seed), Path(directory)))))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
