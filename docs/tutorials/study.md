# Run a study

A study is a declared experiment: a window, a population, a run mode, arms, evaluators, and
gates, written as a YAML profile. `hcsim study run` runs one and writes its scorecard.

```yaml
# studies/backtest.yaml
name: backtest
seed: 7
trajectories: 5
months: 6
mode: unguided
scope: hcf
run_date: 2025-07-01
evaluators: [invariants, fidelity, domain, dispersion]
arms:
  - {name: forecast, null: true}
  - {name: history, reference: true, mode: replay}
```

```bash
uv run hcsim study run --profile studies/backtest.yaml --world worlds/tiny.duckdb --out runs/backtest
```

It writes `scorecard.csv`, one row per pass, arm, evaluator, and metric, each carrying the
provenance of the run behind it, and `run.json`, everything else about the run.

## The run-mode ladder

`mode` is a rung of the ladder, each pinning one more input than the next:

| Mode | Pins | A regression traces to |
|---|---|---|
| `replay` | movements, volumes, and quota | orchestration and data fidelity |
| `guided_actuals` | volumes and quota | quota sampling, given correct volumes |
| `guided_forecast` | quota | the guidance forecast |
| `unguided` | nothing | the domain models' own rates |

A mode that reads its window's actuals (`replay`, `guided_actuals`) runs only once the window is
wholly closed.

## Options and exit codes

- `--preflight-only` checks everything the study needs and writes nothing.
- `--offline` scores the study's inputs on the offline backend, with no simulation.
- `--schema-path` encodes with a saved schema; `--cache-dir` keeps the extracts it reads.

| Code | Meaning |
|---|---|
| 0 | the study passed every gate |
| 2 | a usage, profile, or input error |
| 3 | stopped: a stop gate failed, or a metric failed its threshold |
| 4 | provisional: a provisional gate failed, or its inputs include a fixture |
| 5 | nothing can run yet: its mode reads a window that isn't wholly closed |
