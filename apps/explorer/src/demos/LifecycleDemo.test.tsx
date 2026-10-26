import { cleanup, render } from "@testing-library/react";
import { afterEach, describe, expect, it } from "vitest";

import type { Lifecycle } from "../api";
import { FlowDiagram, ribbonPath } from "../charts";
import { flows, identity, listed, stockSummary } from "./LifecycleDemo";

afterEach(cleanup);

const spread = (values: number[]) => ({ mean: values, p10: values.map((v) => v - 2), p50: values, p90: values.map((v) => v + 3) });

/** A year on a world like tiny: growth and backfill in, external fills and unfilled exits out. */
const YEAR: Lifecycle = {
  months: 3,
  run_date: "2026-01-01",
  open: spread([18, 19, 20, 21]),
  opened: { G: [2, 3, 2], G_camp: [0, 0, 0], B_att: [3, 2, 3], B_tr: [0, 0, 0] },
  filled: { F_ext: [3, 3, 3], F_conv: [0, 0, 0], F_tr: [0, 0, 0], F_camp: [0, 0, 0], F_pend: [0, 0, 0] },
  unfilled: [1, 0, 0],
  identity_residual: 0,
};

describe("the lifecycle chapter", () => {
  it("lists in plain words", () => {
    expect(listed(["a"])).toBe("a");
    expect(listed(["a", "b"])).toBe("a or b");
    expect(listed(["a", "b", "c"], "and")).toBe("a, b and c");
    expect(listed(["a", "b", "c"], "or", true)).toBe("a, b, or c");
  });

  it("draws a ribbon for each flow that carries something, and names the rest", () => {
    const { inflows, outflows, empty } = flows(YEAR);
    expect(inflows).toEqual([
      { name: "growth", total: 7 },
      { name: "backfill", total: 8 },
    ]);
    expect(outflows).toEqual([
      { name: "filled", total: 9 },
      { name: "unfilled", total: 1 },
    ]);
    expect(empty).toEqual([
      "campus openings",
      "backfill for transfers",
      "the conversion, transfer, campus and pending-start fills",
    ]);
  });

  it("gives each fill channel and each backfill its own ribbon when there are more than one", () => {
    const busy = {
      ...YEAR,
      opened: { ...YEAR.opened, B_tr: [1, 0, 0] },
      filled: { ...YEAR.filled, F_conv: [1, 0, 0] },
    };
    const { inflows, outflows } = flows(busy);
    expect(inflows.map((f) => f.name)).toEqual(["growth", "backfill: departures", "backfill: transfers"]);
    expect(outflows.map((f) => f.name)).toEqual(["filled: external", "filled: conversion", "unfilled"]);
  });

  it("writes the identity in words: what opened, what closed, and the stock's change", () => {
    const { inflows, outflows } = flows(YEAR);
    expect(identity(inflows, outflows)).toBe("15.0 opened = 10.0 closed + 5.0 more open");
    expect(identity([{ name: "growth", total: 2 }], [{ name: "filled", total: 3 }])).toBe(
      "2.0 opened = 3.0 closed + 1.0 fewer open",
    );
  });

  it("sums up the stock's line and band", () => {
    expect(stockSummary(YEAR.open)).toBe(
      "The median stock goes from 18 open openings to 21 over 3 months; at the end, 80% of trajectories hold 19 to 24.",
    );
  });
});

describe("the flow diagram", () => {
  it("draws each ribbon as wide as its total, the stock between, and the identity above", () => {
    const { container, getByText } = render(
      <FlowDiagram
        title="Openings"
        inflows={[
          { name: "growth", total: 20 },
          { name: "backfill", total: 10 },
        ]}
        outflows={[{ name: "filled", total: 30 }]}
        stock={{ name: "open openings", start: 18, end: 18 }}
        summary="30.0 opened = 30.0 closed + 0.0 more open"
      />,
    );
    const ribbons = [...container.querySelectorAll("path.ribbon")].map((r) => r.getAttribute("d") ?? "");
    expect(ribbons).toHaveLength(3);
    const thickness = (d: string) => {
      const v = /V([\d.]+)/.exec(d)?.[1];
      const y1 = /C[\d.]+,[\d.]+ [\d.]+,[\d.]+ [\d.]+,([\d.]+)/.exec(d)?.[1];
      return Number(v) - Number(y1);
    };
    expect(thickness(ribbons[0]!) / thickness(ribbons[1]!)).toBeCloseTo(2);
    expect(thickness(ribbons[2]!)).toBeCloseTo(thickness(ribbons[0]!) + thickness(ribbons[1]!));
    expect(container.querySelector("rect.stock")).not.toBeNull();
    expect(getByText("18.0 → 18.0")).toBeTruthy();
    expect(getByText("20.0")).toBeTruthy(); // each ribbon's total, as text
    const summary = container.querySelector(".summary.above");
    expect(summary?.textContent).toBe("30.0 opened = 30.0 closed + 0.0 more open");
    expect(summary?.nextElementSibling?.classList.contains("plot")).toBe(true); // above the drawing
  });

  it("curves a ribbon between its ends, keeping its thickness", () => {
    expect(ribbonPath(0, 10, 100, 50, 5)).toBe("M0,10C50,10 50,50 100,50V55C50,55 50,15 0,15Z");
  });
});
