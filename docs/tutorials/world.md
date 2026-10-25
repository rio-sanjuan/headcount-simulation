# Generate a world

Every model, study, and example runs on a synthetic company: a seeded world with a known ground
truth, written to a DuckDB file. Four presets set its size: `tiny` (200 employees), `small`
(5,000), `medium` (50,000), and `large` (500,000), each with about 20 employees per cost center.

```bash
uv run hcsim synth generate --preset small --seed 0 --out worlds/small.duckdb
uv run hcsim synth validate --db worlds/small.duckdb
uv run hcsim synth profile --db worlds/small.duckdb --out worlds/small.md
```

`validate` checks the world's coherence: every stock and flow agrees with every other. `profile`
writes a Markdown summary of what the world holds. The same preset and seed always give the same
world, bit for bit.

The world's tables follow [the data contract](../data-contract.md), which is all the rest of
hcsim reads: no model or study imports the generator's code, so none can share its
assumptions.

!!! note "Sizes and time"
    The small world generates in a few seconds. The large one takes about six minutes and
    10 GiB; `hcsim bench world` times each preset (see [Measure performance](benchmarks.md)).
