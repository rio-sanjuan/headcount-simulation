# Measure performance

The scale harness measures how a run's time and memory grow with its size, and a stored baseline
flags regressions.

```bash
make bench           # the suite, compared with benchmarks/baseline.json; flags growth past 20%
make bench-large     # the 500k cases (needs about 16 GiB)
make bench-baseline  # remeasures every case and stores the baseline
```

Timings are machine-specific, so they're flagged for a person to read and never gate CI.

## Will a run fit?

The capacity model, fitted to the baseline, predicts a run's time and peak memory, and whether
it fits a memory budget as it stands, has to stream its trajectories, or can't fit:

```bash
uv run hcsim bench capacity --employees 733000 --trajectories 20 --scope hcf+sbc --memory-gib 16
```

## Where the time goes

`hcsim bench profile` runs one case under the profiler and reports its hottest functions and
which phase set its peak:

```bash
uv run hcsim bench profile --case medium-sbc-2 --cache-dir .bench/worlds --out profiles
```
