import { cleanup, fireEvent, render, screen, waitFor, within } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";

import { CHAPTERS } from "../story";
import { forget } from "../useDemo";
import { ValvesDemo, ending } from "./ValvesDemo";

const spread = (values: number[]) => ({ mean: values, p10: values.map((v) => v - 1), p50: values, p90: values.map((v) => v + 1) });

const ANSWER = {
  months: 2,
  run_date: "2026-01-01",
  plan: [330, 336],
  runs: {
    none: { valve: "no valve", headcount: spread([320, 320, 319.5]), bias_mean: 1 },
    supply: { valve: "supply", headcount: spread([320, 318, 315]), bias_mean: 2 },
    demand: { valve: "flow", headcount: spread([320, 328, 331.5]), bias_mean: 1 },
    rate_calibration: { valve: "rate", headcount: spread([320, 329, 336]), bias_mean: 0 },
  },
};

const valves = CHAPTERS.find((c) => c.key === "valves")!;

afterEach(() => {
  cleanup();
  forget();
  vi.unstubAllGlobals();
});

describe("the valves chapter", () => {
  it("says where a run ends against the plan", () => {
    expect(ending(331.5, 336)).toBe("It ends at 331.5, 4.5 under the plan's 336.");
    expect(ending(340.04, 336)).toBe("It ends at 340, 4 over the plan's 336.");
    expect(ending(336, 336)).toBe("It ends at 336, on the plan's 336.");
  });

  it("puts one valve in focus at a time, Supply first, without running the engine again", async () => {
    const fetch = vi.fn(async () => ({ ok: true, statusText: "OK", json: async () => ANSWER }) as Response);
    vi.stubGlobal("fetch", fetch);
    const { container } = render(<ValvesDemo name="The three valves" presets={valves.presets} />);
    const chips = screen.getByRole("radiogroup", { name: "Valve in focus" });
    const checked = () => (within(chips).getAllByRole("radio") as HTMLInputElement[]).map((r) => r.checked);
    expect(checked()).toEqual([false, true, false, false]);
    await waitFor(() => expect(container.querySelector(".end-label.subject")?.textContent).toBe("Supply"));
    const summary = container.querySelector(".summary") as HTMLElement;
    expect(summary.getAttribute("aria-live")).toBe("polite");
    expect(summary.textContent).toMatch(/^Supply opens the requisitions.*It ends at 315, 21 under the plan's 336\./);
    expect(container.querySelectorAll("path.band")).toHaveLength(1);
    expect(container.querySelectorAll("polyline.context")).toHaveLength(3);
    expect(container.querySelector("polyline.plan")).not.toBeNull();
    const calls = fetch.mock.calls.length;
    fireEvent.click(within(chips).getByLabelText("Flow"));
    expect(checked()).toEqual([false, false, true, false]);
    expect(container.querySelector(".end-label.subject")?.textContent).toBe("Flow");
    expect(summary.textContent).toMatch(/It ends at 331.5, 4.5 under the plan's 336\./);
    expect(fetch.mock.calls.length).toBe(calls);
    expect(container.querySelector("table")).toBeNull(); // the bias table is gone
  });
});
