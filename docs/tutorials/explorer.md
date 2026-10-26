# Explore the engine

The explorer is an interactive demo of the engine, run on your machine. It's one page that tells
four of the engine's ideas as a story, a chapter each: quota sampling against independent draws,
the three valves that drive a plan, the requisition lifecycle, and the SBC layer. Its server is
`hcsim app`, and its React frontend lives in `apps/explorer`.

```bash
uv sync --extra app
WORLD=worlds/tiny.duckdb make explorer
```

`make explorer` builds the frontend (it needs Node and npm) and serves it, with the API, at
`http://127.0.0.1:8765`. The API alone runs with
`uv run hcsim app serve --world worlds/tiny.duckdb`, and documents itself at `/api/docs`.

Every demo is a seeded run on the world, so the same settings always show the same answer. Each
chart ends with a caption naming the run behind it.

## Prepare before you show it

Each chapter offers three presets, and the page asks for all of them as it loads. The first time,
the engine runs them: a few seconds on tiny, and under a minute on small. Run them ahead, and the
explorer keeps their answers, so a first reader never waits:

```bash
uv run hcsim app prepare --world worlds/small.duckdb --cache .cache/explorer
uv run hcsim app serve --world worlds/small.duckdb --cache .cache/explorer
```

An answer is kept under the world's contents and the engine's version, so a changed world or a
new version runs again. A released version keeps answers in `~/.cache/hcsim/explorer` without
`--cache`. A development version keeps them only where `--cache` says, so a code change never
serves a stale answer.
