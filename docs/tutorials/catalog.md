# Run the study catalog

The catalog holds the studies that measure the simulator: the fidelity ceiling, an SBC accuracy
replay, variation, dispersion, a regime comparison, a backtest, domain sweeps, subsets, seven
alignment studies, award sizing, and *why quota sampling*, against independent draws.

```bash
uv run hcsim study catalog list
uv run hcsim study catalog write --world worlds/tiny.duckdb --out catalog
uv run hcsim study catalog run --world worlds/tiny.duckdb --catalog catalog --out catalog-runs
```

`write` sets each study's window from the world's last snapshot and writes the inputs it reads:
the plans, the fitted guidance, and guidance aligned to a plan. `run` runs every study through
`hcsim study run` and checks each exit code against the one the catalog documents.

## Checkpoints

The SBC accuracy replay's accuracy can be recorded and later runs checked against it:

```bash
uv run hcsim study checkpoint save --run catalog-runs/sbc_accuracy_replay --out sbc.json
uv run hcsim study checkpoint check --run catalog-runs/sbc_accuracy_replay --against sbc.json
```

Accuracy may improve, but a metric that regresses past the tolerance fails the check, and a run
of another study is refused.
