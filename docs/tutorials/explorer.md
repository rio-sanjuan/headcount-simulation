# Explore the engine

The explorer is an interactive demo, run on your machine, of four of the engine's ideas: quota
sampling against independent draws, the three valves that drive a plan, the requisition
lifecycle, and the SBC layer. Its server is `hcsim app`, and its React frontend lives in
`apps/explorer`.

```bash
uv sync --extra app
WORLD=worlds/tiny.duckdb make explorer
```

`make explorer` builds the frontend (it needs Node and npm) and serves it, with the API, at
`http://127.0.0.1:8765`. The API alone runs with
`uv run hcsim app serve --world worlds/tiny.duckdb`, and documents itself at `/api/docs`.

Every demo is a seeded run on the world, so the same settings always show the same answer.
