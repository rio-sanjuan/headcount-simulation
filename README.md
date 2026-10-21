# headcount-simulation

An employee-level Monte Carlo forecast of headcount and stock-based compensation (SBC) expense, driven by an aggregate guidance forecast. Everything runs on a seeded synthetic company the repository generates, so no real data, credentials, or external systems are involved.

> **Status:** pre-alpha. Milestone M0 (foundations) is in progress. See the [implementation plan](docs/implementation-plan.md) and the [project board](https://github.com/users/rio-sanjuan/projects/4).

## What it does

The repository has three components that share one contract surface.

- **Headcount forecast.** A monthly simulation in which every employee and every open requisition is a row in a preallocated array. Each month runs a fixed sequence of domains, such as attrition, promotion, backfill, fills, cancellation, and growth. Each domain scores eligible rows with a model and returns a declarative change, and a single function applies every change.
- **Guidance forecast.** An offline time-series pipeline that forecasts event counts per group and month and reconciles them to a stock-flow identity. The simulation realizes those counts exactly through quota sampling, so totals match the forecast by construction and only *who* experiences each event stays random.
- **SBC expense.** A financial layer that sizes equity grants, lays out their vesting, and tracks each tranche by holder. After the simulation it amortizes every tranche under ASC 718, with attrition, rate-change, and vesting true-ups. The vesting true-up is the residual, so a per-tranche accounting identity holds by construction.

## Design principles

- Three layers with a one-way dependency, enforced in CI.
- One writer for simulation state, and one frozen configuration per run.
- Exact-count quota sampling where a guidance forecast exists, and independent sampling where it doesn't.
- Separate valves for plan targets, event flows, and requisition vetoes.
- A run-mode ladder that releases one pinned input per rung, so a regression can be traced to its input.
- Reproducible runs: a configuration and a seed reproduce a run bit for bit.
- A synthetic world with known ground truth, so every model can report how well it recovers the true parameters.

## Development

Requires [uv](https://docs.astral.sh/uv/). The project targets Python 3.12–3.14, and runs locally only.

```bash
uv sync
uv run pre-commit install
uv run pytest
```

Optional extras add time-series libraries: `uv sync --extra darts`, `--extra sktime`, or `--extra prophet`.

## Repository layout

```
src/hcsim/
├── model/       Model Layer: protocols, schema, models, samplers (the leaf)
├── guidance/    Guidance forecast pipeline
├── financial/   Financial Layer: award sizing, vesting, ASC 718 expense
├── engine/      Orchestration: configuration, state, step loop, handlers
├── studies/     Run modes, profiles, evaluators, the CLI
├── synth/       Synthetic world generator
└── bench/       Scale harness
docs/            Design documents and the implementation plan
tools/           Development utilities
```

## Contributing

Issues are welcome: bug reports, questions, and ideas. Pull requests aren't accepted. See [CONTRIBUTING.md](CONTRIBUTING.md).

## License

[Apache License 2.0](LICENSE). All data in this repository is synthetic.
