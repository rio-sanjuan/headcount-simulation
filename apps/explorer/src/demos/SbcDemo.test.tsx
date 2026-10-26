import { cleanup, render } from "@testing-library/react";
import { afterEach, describe, expect, it } from "vitest";

import type { Sbc } from "../api";
import { StackedBars, stack } from "../charts";
import { drawn, sbcSummary } from "./SbcDemo";

afterEach(cleanup);

const spread = (values: number[]) => ({ mean: values, p10: values.map((v) => v - 5), p50: values, p90: values.map((v) => v + 5) });

/** Two months: net expense, a vesting true-up in the first, an attrition true-up in both. */
const YEAR: Sbc = {
  trajectories: 10,
  seed: 7,
  months: 2,
  run_date: "2026-01-01",
  total: spread([260_000, 220_000]),
  parts: { net: [250_000, 260_000], vesting: [40_000, 0], attrition: [-30_000, -40_000], rate: [0, 0] },
  origins: { seed: [255_000, 110_000], sim: [5_000, 110_000] },
};

describe("the SBC chapter", () => {
  it("draws the parts with something to draw, leaving out a flat rate's true-up", () => {
    expect(drawn(YEAR).map((p) => p.name)).toEqual(["net expense", "vesting true-up", "attrition true-up"]);
    const revised = { ...YEAR, parts: { ...YEAR.parts, rate: [0, -3_000] } };
    expect(drawn(revised).map((p) => p.name)).toContain("rate true-up");
  });

  it("sums up the average month and the parts it adds up from", () => {
    expect(sbcSummary(YEAR)).toBe(
      "Over 2 months, recognized expense averages 240k a month: net expense 255k, plus vesting true-ups of 20k, " +
        "less attrition true-ups of 35k.",
    );
  });
});

describe("the stacked bars", () => {
  it("stack positive parts up from zero, and negative parts down from it", () => {
    const segments = stack(
      [
        { name: "net", values: [10, 10], shade: 2 },
        { name: "vesting", values: [5, -2], shade: 3 },
        { name: "attrition", values: [-3, -1], shade: 1 },
      ],
      2,
    );
    expect(segments).toEqual([
      [
        { from: 0, to: 10 },
        { from: 0, to: 10 },
      ],
      [
        { from: 10, to: 15 },
        { from: 0, to: -2 },
      ],
      [
        { from: 0, to: -3 },
        { from: -2, to: -3 },
      ],
    ]);
  });

  it("draw a bar of parts per month, the total as the subject's line and band, each labelled", () => {
    const { container, getByText } = render(
      <StackedBars
        title="Expense"
        x="month"
        y="expense"
        xs={[0, 1]}
        parts={drawn(YEAR)}
        total={{ name: "recognized", role: "subject", values: YEAR.total.mean, band: { low: YEAR.total.p10, high: YEAR.total.p90 } }}
        summary={sbcSummary(YEAR)}
      />,
    );
    expect(container.querySelectorAll("rect.bar")).toHaveLength(5); // the vesting true-up is nothing in month 1
    expect(container.querySelectorAll("rect.shade-1")).toHaveLength(2);
    expect(container.querySelector("line.zero")).not.toBeNull();
    expect(container.querySelector("path.band.subject")).not.toBeNull();
    expect(container.querySelector("polyline.line.subject")).not.toBeNull();
    expect(getByText("recognized").getAttribute("class")).toBe("end-label subject");
    expect(getByText("attrition true-up").getAttribute("class")).toBe("end-label context");
    // The attrition true-up hangs below the zero line.
    const zero = Number(container.querySelector("line.zero")?.getAttribute("y1"));
    const below = [...container.querySelectorAll("rect.shade-1")].map((r) => Number(r.getAttribute("y")));
    expect(Math.min(...below)).toBeGreaterThanOrEqual(zero);
  });
});
