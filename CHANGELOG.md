# Changelog

Every notable change is recorded here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and versions follow
[Semantic Versioning](https://semver.org/spec/v2.0.0.html). A release's section is dated when
it's prepared (`tools/release.py prepare`), and its text becomes the release's notes.

## [Unreleased]

## [0.1.0] - Unreleased

The first release: the headcount forecast, the guidance forecast, and SBC expense, end to end
on a synthetic company, with the studies that measure them.

### Added

- **Foundations.** The package, its tooling and CI, the layer contracts, the sanitization
  guardrails, and the design baseline: the overview, the notation, the glossary, and the ADRs.
- **A walking skeleton.** A tiny world through guidance, the headcount forecast, SBC, and a
  study, end to end in CI.
- **The synthetic world** (`hcsim synth`). Every source table, each quirk behind its own
  toggle, the equity ledger, and a ground-truth manifest, in four presets from 200 to 500,000
  employees, checked for coherence and written to DuckDB.
- **The Model Layer.** The contract surface, every model family, training drivers, the arm
  matrix, and recovery reports that compare fits with the world's truth.
- **The headcount forecast engine.** Every domain, known events, fills over the fill tree,
  transfers, and replay; quota sampling wherever guidance covers a group; and streaming runs
  bit-identical to standard ones.
- **The guidance pipeline** (`hcsim guidance`). Ingestion through formatting, reconciliation,
  alignment solvers, disaggregation, realized totals, and the guidance ladder.
- **SBC expense.** Sizing, vesting, timing, the holder ledger, prices, the seed's and the
  offline subsidiaries' equity, and the expense calculator, whose per-tranche identity holds by
  construction.
- **Alignment and planner controls.** Five alignment methods, supply rules, credit and floor,
  rate calibration, fill-share targets, probability tilts, and capacity caps.
- **Studies** (`hcsim study`). Profiles, the run-mode ladder, inputs with provenance, the
  harness, simulation and offline backends, nine evaluators, gates, a versioned scorecard, a
  catalog of seventeen studies, and checkpoints.
- **Performance** (`hcsim bench`). A scale harness, a capacity model, a stored baseline, 20%
  regression flags, and profiling at 500,000 employees.
- **The explorer** (`hcsim app`). A local demo of quota sampling, the three valves, the
  requisition lifecycle, and the SBC layer, with a React frontend.
- **A docs site** (`make docs`). A page per domain, tutorials, and worked examples.
- **Releases.** Semantic versions, this changelog, and release workflows that check a tag
  against them before anything ships.
