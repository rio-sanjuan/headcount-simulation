import { render } from "@testing-library/react";
import { describe, expect, it } from "vitest";

import { BarChart, LineChart, labelled, scale, ticks } from "./charts";

describe("scale", () => {
  it("maps a domain onto a range, and a flat one onto its middle", () => {
    const y = scale(0, 10, 100, 0);
    expect(y(0)).toBe(100);
    expect(y(5)).toBe(50);
    expect(scale(3, 3, 0, 10)(3)).toBe(5);
  });
});

describe("ticks", () => {
  it("covers the domain in round steps", () => {
    expect(ticks(0, 10)).toEqual([0, 2, 4, 6, 8, 10]);
    expect(ticks(0.1, 0.5)).toEqual([0.1, 0.2, 0.3, 0.4, 0.5]);
    expect(ticks(4, 4)).toEqual([4]);
    expect(ticks(Number.NaN, 1)).toEqual([]);
  });
});

describe("labelled", () => {
  it("labels about a dozen positions however many there are", () => {
    const every = labelled(200);
    expect([0, 1, 17, 34].map(every)).toEqual([true, false, true, true]);
    expect([0, 1, 2].map(labelled(3))).toEqual([true, true, true]);
  });
});

describe("charts", () => {
  it("leaves a value that isn't finite out of its line", () => {
    const { container } = render(
      <LineChart title="t" x="x" y="y" xs={[0, 1, 2]} series={[{ name: "plan", values: [Number.NaN, 2, 3] }]} />,
    );
    const points = container.querySelector("polyline")?.getAttribute("points") ?? "";
    expect(points.split(" ")).toHaveLength(2);
    expect(points).not.toContain("NaN");
  });

  it("draws a line per series, its title, and a legend", () => {
    const { container, getByText } = render(
      <LineChart
        title="Headcount"
        x="month"
        y="heads"
        xs={[0, 1, 2]}
        series={[
          { name: "plan", values: [1, 2, 3], dashed: true },
          { name: "run", values: [1, 1.5, 2.5] },
        ]}
      />,
    );
    expect(container.querySelectorAll("polyline")).toHaveLength(2);
    expect(container.querySelector("polyline")?.getAttribute("stroke-dasharray")).toBe("6 4");
    expect(getByText("Headcount")).toBeTruthy();
    expect(getByText("run")).toBeTruthy();
  });

  it("draws a bar per category and series", () => {
    const { container } = render(
      <BarChart
        title="Draws"
        x="rows"
        y="draws"
        categories={[18, 19, 20]}
        series={[
          { name: "quota", values: [0, 0, 5] },
          { name: "independent", values: [1, 2, 2] },
        ]}
      />,
    );
    const bars = container.querySelectorAll("rect");
    expect(bars).toHaveLength(6);
    expect(Number(bars[0]?.getAttribute("height"))).toBe(0); // a zero is no bar
  });
});
