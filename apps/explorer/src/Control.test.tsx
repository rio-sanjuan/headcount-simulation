import { act, cleanup, fireEvent, render, screen, within } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";

import { SETTLE_MS } from "./Control";
import { SamplingDemo } from "./demos/SamplingDemo";
import { CHAPTERS, presetsToFetch } from "./story";
import { ask, forget, keyOf } from "./useDemo";

const sampling = CHAPTERS.find((c) => c.key === "sampling")!;

/** An answer whose quota sampling draws ten times the spread, plus one, so a test can tell which
 * spread a chart shows. */
function answer(spread: number) {
  const drawn = Math.round(spread * 10);
  return {
    rows: 2,
    trajectories: 10,
    seed: 0,
    probabilities: [0.3, 0.2],
    expected: spread,
    bins: [drawn, drawn + 1],
    quota: { histogram: [0, 10], mean: 1, sd: 0, rates: [0.6, 0.4] },
    independent: { histogram: [5, 5], mean: 0.5, sd: 0.5, rates: [0.3, 0.2] },
  };
}

/** A fetch that answers the sampling demo, its expected count the spread it was asked for. It
 * holds back its answer for a spread in `held` until `release` is called. */
function serving(held: number[] = []) {
  let release = () => {};
  const waiting = new Promise<void>((resolve) => (release = resolve));
  const fetch = vi.fn(async (url: string) => {
    const spread = Number(new URL(url, "http://here").searchParams.get("spread"));
    if (held.includes(spread)) {
      await waiting;
    }
    return { ok: true, statusText: "OK", json: async () => answer(spread) } as Response;
  });
  return Object.assign(fetch, { release: () => release() });
}

afterEach(() => {
  cleanup();
  forget();
  vi.unstubAllGlobals();
  vi.useRealTimers();
});

describe("the kept answers", () => {
  it("key a query the same whatever its order", () => {
    expect(keyOf("sbc", { months: 6, trajectories: 10 })).toBe(keyOf("sbc", { trajectories: 10, months: 6 }));
    expect(keyOf("sbc", { months: 6 })).not.toBe(keyOf("lifecycle", { months: 6 }));
  });

  it("ask the API once for each query, and again after a failure", async () => {
    const fetch = vi.fn(async () => ({ ok: false, statusText: "Bad", json: async () => ({ detail: "no" }) }) as Response);
    vi.stubGlobal("fetch", fetch);
    await expect(ask("sbc", { months: 6 })).rejects.toThrow("sbc: no");
    await expect(ask("sbc", { months: 6 })).rejects.toThrow("sbc: no");
    expect(fetch).toHaveBeenCalledTimes(2);
    vi.stubGlobal("fetch", serving());
    await ask("sampling", { spread: 1 });
    await ask("sampling", { spread: 1 });
    expect(vi.mocked(fetch)).toHaveBeenCalledTimes(2);
  });
});

describe("the presets", () => {
  it("are fetched as the page loads, each chapter's first preset before the rest", () => {
    const asks = presetsToFetch(CHAPTERS);
    expect(asks).toHaveLength(12);
    expect(asks.slice(0, 4).map((a) => a.demo)).toEqual(["sampling", "valves", "lifecycle", "sbc"]);
    expect(asks.slice(0, 4).map((a) => Object.values(a.query)[0])).toEqual([200, 0.05, 12, 12]);
    expect(CHAPTERS.map((c) => c.presets.length)).toEqual([3, 3, 3, 3]);
    expect(CHAPTERS.find((c) => c.key === "valves")?.presets.map((p) => p.query.growth)).toEqual([0, 0.05, 0.2]);
    expect(CHAPTERS.find((c) => c.key === "sbc")?.presets.map((p) => p.query.months)).toEqual([6, 12, 36]);
  });

  it("are a radio group named for the chapter, the middle one chosen, and answer at once from what was fetched", async () => {
    const fetch = serving();
    vi.stubGlobal("fetch", fetch);
    await Promise.all(sampling.presets.map((p) => ask("sampling", p.query)));
    render(<SamplingDemo name="Quota sampling" presets={sampling.presets} />);
    const group = screen.getByRole("radiogroup", { name: "Quota sampling" });
    const radios = within(group).getAllByRole("radio") as HTMLInputElement[];
    expect(radios.map((r) => r.checked)).toEqual([false, true, false]);
    expect(screen.getByText(/drew 11 rows/)).toBeTruthy(); // drawn at once, with no tick to wait
    fireEvent.click(within(group).getByLabelText("Very uneven rows"));
    expect(screen.getByText(/drew 21 rows/)).toBeTruthy();
    expect(screen.getByRole("status").textContent).toBe("");
    expect(fetch).toHaveBeenCalledTimes(3);
  });
});

describe("Adjust", () => {
  it("opens the sliders, and runs once a drag settles, showing the last answer while it runs", async () => {
    vi.useFakeTimers();
    const fetch = serving([1.5]);
    vi.stubGlobal("fetch", fetch);
    await Promise.all(sampling.presets.map((p) => ask("sampling", p.query)));
    render(<SamplingDemo name="Quota sampling" presets={sampling.presets} />);
    const adjust = screen.getByRole("button", { name: "Adjust: Quota sampling" });
    expect(adjust.getAttribute("aria-expanded")).toBe("false");
    fireEvent.click(adjust);
    expect(adjust.getAttribute("aria-expanded")).toBe("true");
    const spread = screen.getByLabelText("Spread of probabilities");
    fireEvent.change(spread, { target: { value: "1.4" } });
    fireEvent.change(spread, { target: { value: "1.5" } });
    const radios = screen.getAllByRole("radio") as HTMLInputElement[];
    expect(radios.some((r) => r.checked)).toBe(false); // no longer a preset
    await act(async () => vi.advanceTimersByTime(SETTLE_MS - 1));
    expect(fetch).toHaveBeenCalledTimes(3);
    await act(async () => vi.advanceTimersByTime(1));
    expect(fetch).toHaveBeenCalledTimes(4); // once, for the value it settled on
    expect(screen.getByRole("status").textContent).toBe("Running…");
    expect(screen.getByText(/drew 11 rows/)).toBeTruthy(); // the last answer, while it runs
    await act(async () => fetch.release());
    expect(screen.getByText(/drew 16 rows/)).toBeTruthy();
    expect(screen.getByRole("status").textContent).toBe("");
  });

  it("returns to a preset when the sliders reach its values", () => {
    vi.stubGlobal("fetch", serving());
    render(<SamplingDemo name="Quota sampling" presets={sampling.presets} />);
    fireEvent.click(screen.getByRole("button", { name: "Adjust: Quota sampling" }));
    fireEvent.change(screen.getByLabelText("Spread of probabilities"), { target: { value: "0" } });
    const radios = screen.getAllByRole("radio") as HTMLInputElement[];
    expect(radios.map((r) => r.checked)).toEqual([true, false, false]);
  });
});
