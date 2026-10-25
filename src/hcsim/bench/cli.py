"""`hcsim bench`: time the synthetic world's presets, and measure how runs scale.

- `world` generates, writes, and checks each preset, timed.
- `scale` runs the scale harness's suite, or named cases, each in its own process, and writes
  the measurements; with `--baseline-out`, it writes them as a baseline with its capacity
  model fitted.
- `compare` flags every measure of a run that grew more than 20% over the baseline's, and exits
  3 when anything is flagged.
- `capacity` predicts a run's time and peak memory from the baseline's capacity model, and says
  whether it fits a memory budget as it stands, has to stream its trajectories, or can't fit.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
from collections.abc import Sequence
from pathlib import Path

from hcsim.bench import scale
from hcsim.bench.baseline import BASELINE_PATH, THRESHOLD, Baseline, BaselineError, compare
from hcsim.bench.capacity import CapacityError
from hcsim.bench.world import BenchError, run
from hcsim.synth.config import PRESETS

EXIT_FLAGGED = 3


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="hcsim bench", description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    commands = parser.add_subparsers(dest="command", required=True)
    world = commands.add_parser("world", help="generate, write, and check each preset, timed")
    world.add_argument(
        "--preset", action="append", choices=sorted(PRESETS), help="repeat for several"
    )
    world.add_argument("--seed", type=int, default=0)
    world.add_argument("--max-rss-gib", type=float, help="stop a run whose memory passes this")
    world.add_argument("--out", type=Path, help="write the runs as JSON")

    measure = commands.add_parser("scale", help="run the scale harness's cases")
    measure.add_argument(
        "--suite", choices=["default", "large"], default="default", help="the cases to run"
    )
    measure.add_argument(
        "--case", action="append", choices=sorted(scale.CASES), help="run only these cases"
    )
    measure.add_argument("--cache-dir", type=Path, required=True, help="where worlds are kept")
    measure.add_argument("--max-rss-gib", type=float, help="stop a case whose memory passes this")
    measure.add_argument("--out", type=Path, help="write the measurements as JSON")
    measure.add_argument("--baseline-out", type=Path, help="write them as a baseline")

    against = commands.add_parser("compare", help="flag a run's regressions over a baseline")
    against.add_argument("--run", type=Path, required=True, help="measurements, as `scale` wrote")
    against.add_argument("--baseline", type=Path, default=BASELINE_PATH)
    against.add_argument("--threshold", type=float, default=THRESHOLD)

    capacity = commands.add_parser("capacity", help="predict a run's time and peak memory")
    capacity.add_argument("--baseline", type=Path, default=BASELINE_PATH)
    capacity.add_argument("--employees", type=int, required=True, help="employees seeded")
    capacity.add_argument("--trajectories", type=int, default=20)
    capacity.add_argument("--months", type=int, default=12)
    capacity.add_argument("--scope", choices=["hcf", "hcf+sbc"], default="hcf+sbc")
    capacity.add_argument("--memory-gib", type=float, default=16.0, help="the memory budget")
    return parser


def _world(args: argparse.Namespace) -> int:
    try:
        runs = run(args.preset or ["tiny"], args.seed, args.max_rss_gib)
    except BenchError as error:
        print(f"hcsim bench: {error}")
        return 1
    print(
        f"{'preset':8} {'employees':>9} {'generate':>9} {'write':>7} {'validate':>8} "
        f"{'peak MiB':>9} {'rows':>11} checks"
    )
    for r in runs:
        print(
            f"{r.preset:8} {r.employees:>9,} {r.generate_seconds:>8.1f}s {r.write_seconds:>6.1f}s "
            f"{r.validate_seconds:>7.1f}s {r.peak_rss_mib:>9,.0f} {r.rows:>11,} "
            f"{'ok' if r.checks_passed else 'FAIL'}"
        )
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps([dataclasses.asdict(r) for r in runs], indent=2) + "\n")
    return 0 if all(r.checks_passed for r in runs) else 1


def _scale(args: argparse.Namespace) -> int:
    if args.case:
        cases = [scale.CASES[name] for name in args.case]
    else:
        cases = list(scale.SUITE if args.suite == "default" else scale.LARGE)
    try:
        measurements = scale.run(cases, args.cache_dir, args.max_rss_gib)
    except BenchError as error:
        print(f"hcsim bench: {error}")
        return 1
    print(
        f"{'case':22} {'employees':>9} {'tranches':>10} {'prepare':>8} {'simulate':>9} "
        f"{'peak MiB':>9}"
    )
    for m in measurements:
        print(
            f"{m.case:22} {m.employees:>9,} {m.tranches:>10,} {m.prepare_seconds:>7.2f}s "
            f"{m.simulate_seconds:>8.2f}s {m.peak_mib:>9,.0f}"
        )
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        rows = [dataclasses.asdict(m) for m in measurements]
        args.out.write_text(json.dumps(rows, indent=2) + "\n")
    if args.baseline_out is not None:
        try:
            Baseline.of(measurements).save(args.baseline_out)
        except CapacityError as error:
            print(f"hcsim bench: can't fit a capacity model: {error}")
            return 1
        print(f"wrote the baseline to {args.baseline_out}")
    return 0


def _compare(args: argparse.Namespace) -> int:
    try:
        baseline = Baseline.load(args.baseline)
        rows = json.loads(args.run.read_text())
        measured = [scale.Measurement(**row) for row in rows]
        result = compare(measured, baseline, args.threshold)
    except (OSError, ValueError, TypeError) as error:
        print(f"hcsim bench: {error}")
        return 2
    if not result.same_machine:
        print("note: the baseline ran on another machine, so its timings may not compare")
    for case in result.new:
        print(f"  new      {case}: the baseline has no such case")
    for flag in result.flags:
        print(f"  FLAGGED  {flag}")
    verdict = f"{len(result.flags)} flagged" if result.flagged else "no regressions"
    print(f"{verdict} past {args.threshold:.0%} over the baseline of {baseline.measured_on}")
    return EXIT_FLAGGED if result.flagged else 0


def _capacity(args: argparse.Namespace) -> int:
    try:
        model = Baseline.load(args.baseline).capacity
        plan = model.plan(
            args.employees, args.trajectories, args.months, args.scope, args.memory_gib * 1024
        )
    except (BaselineError, CapacityError) as error:
        print(f"hcsim bench: {error}")
        return 2
    fit = model.fits[args.scope]
    print(
        f"{args.employees:,} employees, {args.trajectories} trajectories, {args.months} months, "
        f"{args.scope}:"
    )
    for label, p in (("holding them", plan.held), ("streaming them", plan.streamed)):
        print(
            f"  {label:15} prepare {p.prepare_seconds:,.1f}s, simulate {p.simulate_seconds:,.1f}s, "
            f"peak {p.peak_mib / 1024:,.1f} GiB"
        )
    worst = ", ".join(f"{k} {v:.0%}" for k, v in fit.worst.items())
    print(f"  fitted on {fit.cases} cases; worst error over them: {worst}")
    print(f"  within {args.memory_gib:g} GiB: {plan.advice}")
    return 0


def main(argv: Sequence[str]) -> int:
    args = _parser().parse_args(list(argv))
    handlers = {"world": _world, "scale": _scale, "compare": _compare, "capacity": _capacity}
    return handlers[args.command](args)
