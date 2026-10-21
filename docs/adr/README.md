# Architecture decision records

Each record states one load-bearing decision: the context, the decision, the alternatives considered, and the consequences. A superseded record stays in place with a pointer to its replacement.

| ADR | Decision |
|---|---|
| [0001](0001-layered-architecture.md) | Layered architecture with a one-way dependency rule |
| [0002](0002-quota-sampling.md) | Exact-count quota sampling for aggregate alignment |
| [0003](0003-single-writer.md) | Pure handlers and a single writer |
| [0004](0004-three-valves.md) | Three separate valves |
| [0005](0005-post-simulation-expense.md) | Amortize expense after the simulation |
| [0006](0006-known-events-capability.md) | Known events as a capability, not a run mode |
| [0007](0007-frozen-configuration.md) | One frozen configuration |
| [0008](0008-synthetic-world.md) | A synthetic world as the only data source |
| [0009](0009-numpy-backend.md) | NumPy arrays, with performance-critical kernels in one module |
| [0010](0010-time-series-adapters.md) | Time-series libraries behind one adapter protocol |
| [0011](0011-local-execution.md) | Local-only execution, with cloud deferred |
| [0012](0012-pandas-and-duckdb.md) | pandas for frames, DuckDB for heavy relational work |
