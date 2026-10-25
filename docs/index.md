# hcsim

An employee-level Monte Carlo forecast of headcount and stock-based compensation (SBC) expense,
driven by an aggregate guidance forecast. Everything runs on a seeded synthetic company the
repository generates, so no real data, credentials, or external systems are involved.

It has three components that share one contract surface:

- **The headcount forecast**: a monthly simulation in which every employee and every open
  requisition is a row. Each month runs a fixed sequence of [domains](domains/index.md), each
  scoring its eligible rows with a model and returning a declarative change.
- **The guidance forecast**: an offline time-series pipeline that forecasts event counts per
  group and month. The simulation realizes those counts exactly, so its totals match the
  forecast by construction, and only *who* experiences each event stays random.
- **SBC expense**: a financial layer that sizes equity grants, lays out their vesting, tracks
  each tranche by holder, and amortizes it after the simulation.

## Where to start

- [The design overview](design/overview.md), [its notation](design/notation.md), and
  [the glossary](glossary.md) explain the model.
- [The domains](domains/index.md) describe each month's steps, from the code itself, with what
  fitting and running them on the synthetic world showed.
- [The tutorials](tutorials/world.md) walk through generating a world, fitting guidance,
  running studies, exploring the engine, and measuring its performance.
- [The worked examples](notebooks/quota-sampling.ipynb) are notebooks that run as this site
  builds.
- [The data contract](data-contract.md) and [the decisions](adr/README.md) are the reference.

## Building this site

```bash
uv sync --extra docs
make docs        # builds into site/, strictly: a broken link or reference fails it
make docs-serve  # serves it at http://127.0.0.1:8000
```
