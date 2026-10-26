import { describe, expect, it } from "vitest";

import { doing, timeLeft } from "./ChapterState";

describe("a waiting chapter", () => {
  it("says what the engine is doing", () => {
    expect(doing("sampling", { rows: 200, trajectories: 500 })).toBe("Drawing 200 rows 500 times");
    expect(doing("lifecycle", { months: 12, trajectories: 10 }, "tiny5.duckdb")).toBe(
      "Running 10 trajectories of 12 months on tiny5.duckdb",
    );
    expect(doing("sbc", { months: 36, trajectories: 10 })).toBe("Running 10 trajectories of 36 months, with SBC");
    expect(doing("valves", { growth: 0.05, trajectories: 10 }, "small.duckdb")).toBe(
      "Fitting guidance on small.duckdb, then running 10 trajectories of each valve",
    );
  });

  it("says about how long is left, rounded", () => {
    expect(timeLeft(null, 3)).toBe("");
    expect(timeLeft(undefined, 3)).toBe("");
    expect(timeLeft(200, 20)).toBe("about 3 min left");
    expect(timeLeft(40, 2)).toBe("about 40 s left");
    expect(timeLeft(40, 7)).toBe("about 35 s left");
    expect(timeLeft(6, 2.5)).toBe("about 4 s left");
    expect(timeLeft(6, 6)).toBe("almost done");
    expect(timeLeft(6, 12)).toBe("taking longer than expected");
  });
});
