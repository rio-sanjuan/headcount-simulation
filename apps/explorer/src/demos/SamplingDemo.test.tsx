import { cleanup, render } from "@testing-library/react";
import { afterEach, describe, expect, it } from "vitest";

import type { Sampling } from "../api";
import { Histogram, LineChart } from "../charts";
import { countSummary, grouped, rateSummary } from "./SamplingDemo";

afterEach(cleanup);

const DATA: Sampling = {
  rows: 40,
  trajectories: 500,
  seed: 0,
  probabilities: Array.from({ length: 40 }, (_, i) => (i < 2 ? 1 : 0.5)),
  expected: 21,
  bins: [8, 9, 10, 20, 31],
  quota: { histogram: [0, 0, 0, 500, 0], mean: 20, sd: 0, rates: Array.from({ length: 40 }, (_, i) => (i < 2 ? 0.71 : 0.5)) },
  independent: { histogram: [3, 0, 90, 300, 107], mean: 20, sd: 4, rates: Array.from({ length: 40 }, (_, i) => (i < 2 ? 1 : 0.5)) },
};

describe("the sampling chapter", () => {
  it("groups the rows' rates in about 20 groups of neighbours, each its mean", () => {
    expect(grouped([1, 3, 5, 7])).toEqual({ starts: [1, 2, 3, 4], means: [1, 3, 5, 7] });
    const { starts, means } = grouped(Array.from({ length: 200 }, (_, i) => i));
    expect(starts.slice(0, 3)).toEqual([1, 11, 21]);
    expect(means).toHaveLength(20);
    expect(means[0]).toBe(4.5);
  });

  it("sums up each chart in a line", () => {
    expect(countSummary(DATA)).toBe(
      "Quota sampling drew 20 rows in all 500 trajectories; independent draws took 8 to 31.",
    );
    expect(rateSummary(DATA)).toBe(
      "The likeliest 2 rows: quota sampling drew each in 71% of trajectories, and independent draws in 100%, " +
        "against the model's 100%.",
    );
  });
});

describe("the histogram", () => {
  it("caps its height, so the subject's bar breaks through the top, labelled with its count", () => {
    const { container, getByText } = render(
      <Histogram
        title="Draws"
        x="rows"
        y="trajectories"
        bins={DATA.bins}
        subject={{ name: "Quota sampling", counts: DATA.quota.histogram }}
        context={{ name: "Independent draws", counts: [3, 0, 40, 60, 7] }}
        cap={80}
        summary={countSummary(DATA)}
      />,
    );
    const ticks = [...container.querySelectorAll("text.tick")].map((t) => t.textContent);
    expect(ticks).toContain("80");
    expect(ticks).not.toContain("500");
    expect(container.querySelectorAll("path.bar.subject")).toHaveLength(1);
    expect(container.querySelectorAll("path.break")).toHaveLength(1);
    expect(getByText("Quota sampling: all 500").getAttribute("class")).toBe("end-label subject");
    expect(getByText("Independent draws")).toBeTruthy();
    const svg = container.querySelector("svg");
    const summary = container.querySelector(".summary");
    expect(summary?.textContent).toMatch(/^Quota sampling drew 20 rows/);
    expect(svg?.getAttribute("aria-describedby")).toBe(summary?.id);
  });

  it("draws the subject's bars narrower than the context's, in front of them", () => {
    const { container } = render(
      <Histogram
        title="t"
        x="x"
        y="y"
        bins={[1, 2]}
        subject={{ name: "s", counts: [0, 10] }}
        context={{ name: "c", counts: [5, 10] }}
        cap={80}
      />,
    );
    const width = (selector: string) => {
      const d = container.querySelector(selector)?.getAttribute("d") ?? "";
      const xs = [...d.matchAll(/(-?[\d.]+),-?[\d.]+/g)].map((m) => Number(m[1]));
      return Math.max(...xs) - Math.min(...xs);
    };
    expect(width("path.bar.subject")).toBeLessThan(width("path.bar.context"));
    expect(container.querySelector("path.break")).toBeNull(); // nothing taller than the cap
    const paths = [...container.querySelectorAll("path.bar")];
    expect(paths.at(-1)?.classList.contains("subject")).toBe(true);
  });

  it("raises its cap to the context's tallest bar, so only the subject breaks through", () => {
    const { container } = render(
      <Histogram title="t" x="x" y="y" bins={[1]} subject={{ name: "s", counts: [0] }} context={{ name: "c", counts: [130] }} cap={80} />,
    );
    const ticks = [...container.querySelectorAll("text.tick")].map((t) => Number(t.textContent));
    expect(Math.max(...ticks)).toBeGreaterThanOrEqual(130);
  });
});

describe("labels at a line's start", () => {
  it("sit left of the plot, with the y axis at its right", () => {
    const { getByText, container } = render(
      <LineChart
        title="t"
        x="x"
        y="share"
        xs={[1, 2, 3]}
        labelAt="start"
        series={[
          { name: "Quota sampling", values: [0.7, 0.3, 0], role: "subject" },
          { name: "Probability", values: [1, 0.3, 0], role: "plan" },
        ]}
      />,
    );
    const label = getByText("Quota sampling");
    expect(label.getAttribute("text-anchor")).toBe("end");
    const firstPoint = Number(container.querySelector("polyline.subject")?.getAttribute("points")?.split(",")[0]);
    expect(Number(label.getAttribute("x"))).toBeLessThan(firstPoint);
    const tick = container.querySelector("text.tick[text-anchor='start']");
    expect(tick).not.toBeNull(); // the y ticks, at the right
  });
});
