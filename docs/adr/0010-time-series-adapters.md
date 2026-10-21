# ADR 0010: Time-series libraries behind one adapter protocol

**Status:** Accepted, 2026-10-21

## Context

The guidance pipeline benefits from many forecasters, and the strongest ones are spread across statsforecast, darts, and sktime, each with its own data model.

## Decision

Every forecaster implements the `TimeSeriesModel` protocol through an adapter per library. Models are registered under namespaced names, such as `statsforecast.AutoARIMA`, validated when the configuration loads, and installed through optional extras. Each adapter declares the covariates it consumes and the probabilistic outputs it can produce.

## Alternatives considered

- **One library only.** Simpler, but it limits the model suite.
- **Using each library's API directly in the pipeline.** No adapter code, but the pipeline would couple to every library.

## Consequences

- A model from an uninstalled extra fails at load, with the install command.
- Parity tests check that the same model family agrees across libraries.
- The core install stays light, because the heavy libraries are optional.
