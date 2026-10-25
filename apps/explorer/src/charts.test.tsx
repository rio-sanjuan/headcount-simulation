import { render } from "@testing-library/react";
import { describe, expect, it } from "vitest";

import { BarChart, LineChart, barPath, format, labelled, nudge, scale, spreadSeries, ticks } from "./charts";

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
    expect(ticks(312, 336)).toEqual([310, 315, 320, 325, 330, 335, 340]);
    expect(ticks(0, 0.55)).toEqual([0, 0.2, 0.4, 0.6]);
    expect(ticks(-38, 99)).toEqual([-50, 0, 50, 100]);
    expect(ticks(4, 4)).toEqual([4]);
    expect(ticks(Number.NaN, 1)).toEqual([]);
  });
});

describe("format", () => {
  it("writes a tick short, compact from a thousand up", () => {
    expect([0.25, 12, 999, 1500, 256_000, -38_000, 1_250_000].map(format)).toEqual([
      "0.25",
      "12",
      "999",
      "1.5k",
      "256k",
      "-38k",
      "1.25M",
    ]);
  });
});

describe("labelled", () => {
  it("labels about a dozen positions however many there are", () => {
    const every = labelled(200);
    expect([0, 1, 17, 34].map(every)).toEqual([true, false, true, true]);
    expect([0, 1, 2].map(labelled(3))).toEqual([true, true, true]);
  });
});

describe("nudge", () => {
  it("leaves labels that don't collide where they are", () => {
    expect(nudge([10, 50, 30], 14, 0, 100)).toEqual([10, 50, 30]);
  });

  it("spreads labels that collide around where they wanted to be, keeping their order", () => {
    const placed = nudge([52, 50, 48], 14, 0, 100);
    expect(placed).toEqual([64, 50, 36]);
    const [a, b] = nudge([20, 21], 14, 0, 100);
    expect((a ?? 0) - (b ?? 0)).toBeLessThan(0); // the first wanted to sit higher, and does
    expect((b ?? 0) - (a ?? 0)).toBe(14);
  });

  it("keeps labels inside the chart while they fit", () => {
    expect(nudge([0, 1], 14, 5, 100)).toEqual([5, 19]);
    expect(nudge([99, 100], 14, 0, 100)).toEqual([86, 100]);
  });

  it("gathers a run of labels that a nudge pushes into another", () => {
    const placed = nudge([10, 12, 30], 14, 0, 100);
    const sorted = [...placed].sort((p, q) => p - q);
    expect(sorted[1]! - sorted[0]!).toBeGreaterThanOrEqual(14);
    expect(sorted[2]! - sorted[1]!).toBeGreaterThanOrEqual(14);
  });
});

describe("barPath", () => {
  it("rounds a bar's end, not its base, up or down from the baseline", () => {
    expect(barPath(0, 10, 100, 60)).toBe("M0,100V64Q0,60 4,60H6Q10,60 10,64V100Z");
    expect(barPath(0, 10, 100, 140)).toBe("M0,100V136Q0,140 4,140H6Q10,140 10,136V100Z");
    expect(barPath(0, 10, 100, 100)).toBe(""); // a zero is no bar
  });
});

describe("the line chart", () => {
  it("leaves a value that isn't finite out of its line", () => {
    const { container } = render(
      <LineChart title="t" x="x" y="y" xs={[0, 1, 2]} series={[{ name: "plan", values: [Number.NaN, 2, 3] }]} />,
    );
    const points = container.querySelector("polyline")?.getAttribute("points") ?? "";
    expect(points.split(" ")).toHaveLength(2);
    expect(points).not.toContain("NaN");
  });

  it("draws the subject over its band and its context, each labelled at its end, with no legend", () => {
    const { container, getByText } = render(
      <LineChart
        title="Headcount"
        x="month"
        y="heads"
        xs={[0, 1, 2]}
        series={[
          spreadSeries("Supply", { p10: [1, 1.2, 2], p50: [1, 1.5, 2.5], p90: [1, 1.8, 3] }),
          { name: "Plan", values: [Number.NaN, 2, 3], role: "plan" },
          { name: "No valve", values: [1, 1.1, 1.2] },
        ]}
      />,
    );
    const lines = [...container.querySelectorAll("polyline")].map((l) => l.getAttribute("class"));
    expect(lines).toEqual(["line context", "line plan", "line subject"]); // the subject on top
    expect(container.querySelectorAll("path.band")).toHaveLength(1);
    expect(container.querySelector("path.band")?.getAttribute("d")).toMatch(/^M.*Z$/);
    expect(getByText("Supply").getAttribute("class")).toBe("end-label subject");
    expect(getByText("No valve").getAttribute("class")).toBe("end-label context");
    expect(getByText("Headcount")).toBeTruthy();
    expect(container.querySelector(".legend")).toBeNull();
  });

  it("nudges end labels apart where their lines end close together", () => {
    const { getByText } = render(
      <LineChart
        title="t"
        x="x"
        y="y"
        xs={[0, 1]}
        series={[
          { name: "a", values: [0, 100] },
          { name: "b", values: [0, 99.5] },
          { name: "c", values: [100, 0] },
        ]}
      />,
    );
    const y = (name: string) => Number(getByText(name).getAttribute("y"));
    expect(Math.abs(y("a") - y("b"))).toBeGreaterThanOrEqual(14);
  });
});

describe("the bar chart", () => {
  it("draws a bar per category and series, the subject in the accent, each series labelled", () => {
    const { container, getByText } = render(
      <BarChart
        title="Draws"
        x="rows"
        y="draws"
        categories={[18, 19, 20]}
        series={[
          { name: "quota", values: [0, 0, 5], role: "subject" },
          { name: "independent", values: [1, 2, 2] },
        ]}
      />,
    );
    const bars = [...container.querySelectorAll("path.bar")];
    expect(bars).toHaveLength(6);
    expect(bars[0]?.getAttribute("d")).toBe(""); // a zero is no bar
    expect(bars.filter((b) => b.classList.contains("subject"))).toHaveLength(3);
    expect(getByText("quota").getAttribute("class")).toBe("end-label subject");
    expect(getByText("independent")).toBeTruthy();
  });

  it("caps a bar's width, so a few categories don't fill the chart", () => {
    const { container } = render(
      <BarChart title="t" x="x" y="y" categories={["a"]} series={[{ name: "n", values: [3] }]} />,
    );
    const d = container.querySelector("path.bar")?.getAttribute("d") ?? "";
    const xs = [...d.matchAll(/(-?[\d.]+),-?[\d.]+/g)].map((m) => Number(m[1]));
    expect(Math.max(...xs) - Math.min(...xs)).toBeLessThanOrEqual(24);
  });
});
