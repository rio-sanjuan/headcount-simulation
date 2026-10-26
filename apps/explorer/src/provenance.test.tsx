import { describe, expect, it } from "vitest";

import { caption } from "./provenance";

const WORLD = {
  world: "tiny5.duckdb",
  run_date: "2026-01-01",
  employees: 320,
  cost_centers: 10,
  version: "0.1.0.dev0",
  chapters: {},
};

describe("a chart's caption", () => {
  it("names the world, its size, the start, the trajectories, the seed, and the engine", () => {
    expect(caption(WORLD, { trajectories: 10, seed: 7 })).toBe(
      "Run on tiny5.duckdb (320 employees, 10 cost centers) from 2026-01 · 10 trajectories · seed 7 · hcsim 0.1.0.dev0",
    );
  });

  it("says synthetic rows aren't the world's", () => {
    expect(caption(WORLD, { trajectories: 500, seed: 0, rows: 200 })).toBe(
      "200 synthetic rows · 500 trajectories · seed 0 · hcsim 0.1.0.dev0",
    );
  });

  it("writes large counts with commas, and one of a thing alone", () => {
    const big = { ...WORLD, employees: 512_340, cost_centers: 1 };
    expect(caption(big, { trajectories: 1, seed: 7 })).toBe(
      "Run on tiny5.duckdb (512,340 employees, 1 cost center) from 2026-01 · 1 trajectory · seed 7 · hcsim 0.1.0.dev0",
    );
  });

  it("names the run alone until the world is known", () => {
    expect(caption(undefined, { trajectories: 10, seed: 7 })).toBe("10 trajectories · seed 7");
  });
});
