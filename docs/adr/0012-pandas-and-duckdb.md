# ADR 0012: pandas for frames, DuckDB for heavy relational work

**Status:** Accepted, 2026-10-21

## Context

Frames appear at the boundaries: extraction, training extracts, guidance series, and reports. The heavy relational work, such as populations, joins, and realized flows, is SQL over the synthetic warehouse.

## Decision

DuckDB runs the relational work in SQL, and pandas 2.x is the frame type everywhere else. The time-series libraries require pandas, so it's installed regardless.

## Alternatives considered

- **Polars.** It's faster and stricter, but it would add a second frame library rather than replace pandas, duplicate what DuckDB already does, and need conversions inside the guidance pipeline.

## Consequences

- One frame library, with no conversion bugs between two.
- The project stays on pandas 2.x until statsforecast supports pandas 3.
