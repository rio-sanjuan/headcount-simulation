# Design overview

This page describes how the system is put together and the rules that hold it together. The [notation](notation.md) defines the symbols, the [glossary](../glossary.md) defines the terms, and the [decision records](../adr/) explain why each load-bearing choice was made.

## What the system forecasts

The system forecasts two linked quantities for a workforce, month by month:

- **Headcount:** active employees by cost center and attributes, plus the transactions that move it, such as hires, terminations, transfers, reorganizations, conversions, and promotions.
- **Stock-based compensation (SBC) expense:** monthly recognized expense for restricted stock units under ASC 718, by cost center, award type, and level, including the true-ups that reconcile expense to forfeitures and vesting.

Most of a year's SBC comes from awards already granted to the current workforce, so departures move expense more than new grants do. A grant needs a person, a date, and a level: sizing reads the holder's own outstanding vests, and the gap-fill rule is nonlinear, so averaging a cohort first would understate grants. That's why the forecast simulates individual employees rather than cohort rates.

## Three components

**Headcount forecast (HCF).** A Monte Carlo simulation with one row per active employee and one row per open requisition, held in preallocated arrays. Each simulated month runs a fixed sequence of *domains*, such as attrition, promotion, backfill, fills, cancellation, and growth. A domain reads state, scores its eligible rows with a model, selects the rows that experience an event, and returns a declarative description of the change. One function applies every change.

**Guidance forecast.** An offline time-series pipeline that forecasts how many events of each kind happen per group and month. It backtests a suite of models, selects and combines them, reconciles the result to a stock-flow identity, and can align it to plan targets. The simulation reads these counts as exact quotas.

**SBC expense.** The Financial Layer sizes each equity grant the simulation emits, lays out its vesting, and tracks every tranche by holder. After the last month it amortizes each tranche into monthly expense and computes three true-ups: attrition, forfeiture-rate change, and vesting.

## Layers and the dependency rule

The code splits into layers, and dependencies run one way.

```
studies   run modes, profiles, inputs, evaluators, gates, the CLI
   │
engine    configuration, state, the apply step, the step loop, domain handlers
   │   └──────────────┐
   │              financial   award sizing, vesting, the RSU tracker, ASC 718 expense
   │                  │
model ◄───────────────┘       protocols, schema, models, samplers (the leaf)
   ▲
guidance  the offline time-series pipeline

synth     the synthetic world (imports nothing from hcsim)
```

- **The Model Layer is the leaf.** It imports nothing else in the package, so a model can be trained without the engine.
- **The Financial Layer runs without a simulation.** It reads only the Model Layer's contract surface (the schema and buffer layouts), so an expense calculation can be checked in a notebook from hand-built inputs.
- **The guidance pipeline never sees the simulation schema.** It groups by plain column names and writes an artifact that the engine validates against the schema when it loads.
- **The synthetic world shares no code** with anything fitted to it.

`import-linter` contracts in `pyproject.toml` enforce these rules on every PR.

## Invariants

Every change preserves these rules. Each one closes off a class of failure.

| Invariant | What it prevents |
|---|---|
| **One writer.** Only the apply step mutates simulation state. Handlers are pure and return results. | Missed updates, and state changes no one can trace |
| **Frozen configuration.** One validated, frozen configuration holds every parameter of a run. | A run record that doesn't describe its run |
| **One encoding authority.** The schema assigns every categorical code, for training and simulation alike. A code is a position in a vocabulary, and `-1` means unattributed. | Silent misreads when a model is scored through a different vocabulary |
| **Eligibility is simulation-level.** Eligibility rules live in the configuration, never in a model. | Hidden policy baked into a saved model |
| **Known events before sampled ones.** Known events fire first, and an employee with a future known event is excluded from earlier draws. | Double counting, such as sampling the departure of someone already scheduled to leave |
| **Three separate valves.** Plan targets size requisition supply, guidance governs event counts, and capacity caps veto requisition creation. | One number being asked to mean two things |
| **The accounting identity.** For every tranche, cumulative net expense plus true-ups equals the value of the shares that vest. | Expense that doesn't tie to vested value |

## One simulated month

Each month runs the domains in a fixed order, and the order is behavior.

1. Known-event domains: reorganizations, other movements, and conversions out.
2. **Performance**, in its cycle month, so attrition and promotion in the same month read the new ratings.
3. **Attrition**, then **backfill** for the departures, so backfill requisitions can fill this month.
4. **Promotion** in cycle months, and the **annual award** in its month.
5. **Pending starts**, which consume their share of hiring first.
6. **Fills** through every channel: external, conversion, transfer, and campus.
7. **Cancellation** of requisitions that exit unfilled.
8. **Backfill** for roles vacated by transfers.
9. **Growth** last, because it fills the residual of a shared requisitions-opened quota and needs a settled month.

## Sampling

Where a guidance forecast covers a group, a domain samples by **quota**. It selects exactly `min(K, eligible)` rows per group, by Gumbel-top-k on the model's probabilities, so totals match the forecast by construction and only *who* experiences each event stays random. Where no forecast exists, each row draws independently and the count is an output of the run.

Before selection, a row's probability passes through the model, then any planner tilts, then any rate-calibration shift, then the eligibility rules and known-event exclusions, which always win.

## Expense after the simulation

SBC follows two phases. During the simulation, domains emit grants and the apply step sizes them and records their vesting. After the last month, a stateless calculator amortizes every tranche, in proportion to calendar days, and computes the true-ups. The vesting true-up is the residual, so the accounting identity holds by construction, and a failed check signals a defect.

A run's *forecast scope* is `hcf+sbc` by default, or `hcf` for headcount alone, which skips sizing and expense.

## The run-mode ladder

A study varies one thing inside an otherwise pinned run. The ladder releases one pinned input per rung, so a regression can be traced to the input that caused it.

| Mode | Pinned | Isolates |
|---|---|---|
| `replay` | Movements, volumes, and quota | Orchestration, data fidelity, and the Financial Layer |
| `guided_actuals` | Volumes and quota | Quota sampling, given correct volumes |
| `guided_forecast` | Quota | The guidance forecast |
| `unguided` | Nothing | The domain models' own rates |

## Data and provenance

All data comes from `hcsim.synth`, a seeded generator that simulates a company with its own ground-truth processes and writes source tables to DuckDB. Its processes differ in form from the engine's models, so fitted models have to learn the world rather than mirror it. Because the truth is known, every model can report how well it recovers it.

Every input records whether it comes from the synthetic world's actuals or from a test fixture, and that provenance propagates to every result built on it.
