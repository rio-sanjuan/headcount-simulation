import { act, cleanup, fireEvent, render, screen, waitFor, within } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { get } from "./api";
import { App, SERVE, holdAnchor } from "./App";
import { CHAPTERS } from "./story";
import { forget } from "./useDemo";

const spread = (values: number[]) => ({ mean: values, p10: values, p50: values, p90: values });

/** An answer for each demo, as the API gives it. */
const ANSWERS: Record<string, unknown> = {
  health: { world: "tiny5.duckdb", run_date: "2026-01-01", employees: 320, cost_centers: 10, version: "0.1.0", chapters: {} },
  sampling: {
    rows: 2,
    trajectories: 10,
    seed: 0,
    probabilities: [0.3, 0.2],
    expected: 0.5,
    bins: [0, 1],
    quota: { histogram: [0, 10], mean: 1, sd: 0, rates: [0.6, 0.4] },
    independent: { histogram: [5, 5], mean: 0.5, sd: 0.5, rates: [0.3, 0.2] },
  },
  valves: {
    trajectories: 10,
    seed: 7,
    months: 2,
    run_date: "2026-01-01",
    plan: [101, 102],
    runs: { supply: { valve: "supply", headcount: spread([100, 100.5, 101]), bias_mean: 1 } },
  },
  lifecycle: {
    trajectories: 10,
    seed: 7,
    months: 2,
    run_date: "2026-01-01",
    open: spread([5, 6, 7]),
    opened: { G: [2, 2] },
    filled: { F_ext: [1, 1] },
    unfilled: [0, 0],
    identity_residual: 0,
  },
  sbc: {
    trajectories: 10,
    seed: 7,
    months: 2,
    run_date: "2026-01-01",
    total: spread([10, 12]),
    parts: { net: [11, 11], vesting: [1, 3], attrition: [-2, -2], rate: [0, 0] },
    origins: { seed: [8, 9], sim: [2, 3] },
  },
};

function answering(body: unknown, ok = true) {
  return vi.fn(async () => ({ ok, statusText: "Unprocessable", json: async () => body }) as Response);
}

/** A fetch that answers each demo with its own answer. */
function serving() {
  return vi.fn(async (url: string) => {
    const demo = /\/api\/(\w+)/.exec(url)?.[1] ?? "";
    return { ok: true, statusText: "OK", json: async () => ANSWERS[demo] } as Response;
  });
}

afterEach(() => {
  cleanup();
  forget();
  vi.unstubAllGlobals();
  vi.restoreAllMocks();
  window.history.replaceState(null, "", "/");
});

describe("the API client", () => {
  it("sends a demo's parameters and reads its answer", async () => {
    const fetch = answering({ value: 1 });
    vi.stubGlobal("fetch", fetch);
    expect(await get("sampling", { rows: 5, spread: 0.5 })).toEqual({ value: 1 });
    expect(fetch).toHaveBeenCalledWith("/api/sampling?rows=5&spread=0.5");
  });

  it("says why an answer was refused", async () => {
    vi.stubGlobal("fetch", answering({ detail: "a quota lies between 1 and the rows" }, false));
    await expect(get("sampling", {})).rejects.toThrow("sampling: a quota lies between 1 and the rows");
  });
});

describe("the story", () => {
  it("tells four chapters in order, each a section with its heading and its anchor", async () => {
    vi.stubGlobal("fetch", serving());
    render(<App />);
    const sections = screen.getAllByRole("region");
    expect(sections.map((s) => s.id)).toEqual(["sampling", "valves", "lifecycle", "sbc"]);
    expect(sections.map((s) => within(s).getByRole("heading", { level: 2 }).textContent)).toEqual([
      "Quota sampling",
      "The three valves",
      "The requisition lifecycle",
      "The SBC layer",
    ]);
    await waitFor(() => expect(screen.getByText(/Quota sampling drew 1 row in all 10 trajectories/)).toBeTruthy());
    await waitFor(() => expect(screen.getByText("4.0 opened = 2.0 closed + 2.0 more open")).toBeTruthy());
    expect(screen.getByRole("heading", { level: 1 }).textContent).toBe("hcsim explorer");
  });

  it("ends every chart with a caption naming the run behind it", async () => {
    vi.stubGlobal("fetch", serving());
    const { container } = render(<App />);
    await waitFor(() => expect(container.querySelectorAll("figure.chart")).toHaveLength(6));
    await waitFor(() => expect(container.querySelector("#valves .provenance")?.textContent).toMatch(/^Run on/));
    const captions = [...container.querySelectorAll("figure.chart")].map((f) => f.lastElementChild?.textContent);
    expect(captions.filter((c) => c?.startsWith("Run on tiny5.duckdb (320 employees, 10 cost centers) from 2026-01"))).toHaveLength(4);
    expect(captions.filter((c) => c === "2 synthetic rows · 10 trajectories · seed 0 · hcsim 0.1.0")).toHaveLength(2);
  });

  it("opens each chapter with its idea, before its demo, and ends with a line to the next", () => {
    vi.stubGlobal("fetch", serving());
    render(<App />);
    for (const chapter of CHAPTERS) {
      const section = document.getElementById(chapter.key) as HTMLElement;
      const intro = section.querySelector(".intro");
      expect(intro?.textContent?.length).toBeGreaterThan(100);
      expect(intro?.nextElementSibling?.classList.contains("demo")).toBe(true);
      expect(section.lastElementChild?.textContent).toBe(chapter.handOn);
    }
    expect(screen.getByText(/drawn 71% of the time/)).toBeTruthy();
  });

  it("keeps a failed run in its own chapter, which can try again, while the others carry on", async () => {
    let sampling = false; // the sampling demo fails until it's fixed
    vi.stubGlobal(
      "fetch",
      vi.fn(async (url: string) => {
        const demo = /\/api\/(\w+)/.exec(url)?.[1] ?? "";
        const ok = demo !== "sampling" || sampling;
        const body = ok ? ANSWERS[demo] : { detail: "the engine stopped" };
        return { ok, statusText: "Error", json: async () => body } as Response;
      }),
    );
    const { container } = render(<App />);
    const chapter = document.getElementById("sampling") as HTMLElement;
    const alert = await within(chapter).findByRole("alert");
    expect(alert.textContent).toMatch(/sampling: the engine stopped/);
    await waitFor(() => expect(container.querySelector("#valves .provenance")).not.toBeNull()); // carries on
    await waitFor(() => expect(screen.getByRole("link", { name: /Quota sampling \(failed\)/ })).toBeTruthy());
    sampling = true;
    fireEvent.click(within(alert).getByRole("button", { name: "Try again" }));
    await waitFor(() => expect(within(chapter).getByText(/Quota sampling drew 1 row/)).toBeTruthy());
    expect(within(chapter).queryByRole("alert")).toBeNull();
  });

  it("says the server isn't running, and how to start it, when it doesn't answer", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn(async () => {
        throw new TypeError("Failed to fetch");
      }),
    );
    render(<App />);
    const alert = await screen.findByRole("alert");
    expect(alert.textContent).toMatch(/server isn't running/);
    expect(alert.textContent).toContain(SERVE);
    expect(screen.queryAllByRole("region")).toHaveLength(0);
  });

  it("waits in each chapter, saying what the engine is doing and about how long is left", async () => {
    const health = { ...(ANSWERS.health as object), chapters: { valves: { seconds: 40 } } };
    vi.stubGlobal(
      "fetch",
      vi.fn((url: string) =>
        url.startsWith("/api/health")
          ? Promise.resolve({ ok: true, statusText: "OK", json: async () => health } as Response)
          : new Promise<Response>(() => {}), // the engine, still running
      ),
    );
    render(<App />);
    const valves = document.getElementById("valves") as HTMLElement;
    const waiting = () => valves.querySelector(".chart-state.waiting p")?.textContent;
    await waitFor(() => expect(waiting()).toMatch(/on tiny5.duckdb, then running 10 trajectories of each valve · about 40 s left/));
    // Only what the engine is doing is announced, not the time left as it ticks.
    expect(within(valves).getByRole("status").textContent).toBe(
      "Fitting guidance on tiny5.duckdb, then running 10 trajectories of each valve",
    );
    expect(within(valves).getByRole("progressbar").getAttribute("aria-valuenow")).toBe("0");
    expect(screen.getByRole("link", { name: /The three valves \(still running\)/ })).toBeTruthy();
  });
});

describe("the chapter rail", () => {
  /** Places each chapter's top `tops[k]` pixels down the window, as jsdom lays nothing out. */
  function place(tops: number[]) {
    const original = Element.prototype.getBoundingClientRect;
    vi.spyOn(Element.prototype, "getBoundingClientRect").mockImplementation(function (this: Element) {
      const k = CHAPTERS.findIndex((c) => c.key === this.id);
      return k < 0 ? original.call(this) : ({ top: tops[k] ?? 0 } as DOMRect);
    });
  }

  beforeEach(() => place([40, 900, 1800, 2700]));

  it("lists the numbered chapters, each a link to its anchor", () => {
    vi.stubGlobal("fetch", serving());
    render(<App />);
    const rail = screen.getByRole("navigation", { name: "Chapters" });
    const links = within(rail).getAllByRole("link").filter((l) => l.getAttribute("href") !== "#top");
    expect(links.map((l) => l.getAttribute("href"))).toEqual(["#sampling", "#valves", "#lifecycle", "#sbc"]);
    expect(links[1]?.textContent).toMatch(/^2 The three valves/);
  });

  it("marks the chapter in view as the page scrolls", async () => {
    vi.stubGlobal("fetch", serving());
    render(<App />);
    const current = () => screen.getByRole("navigation").querySelector("[aria-current]")?.getAttribute("href");
    expect(current()).toBe("#sampling");
    place([-900, -100, 600, 1400]); // the valves chapter's top has passed under the line
    await act(async () => {
      window.dispatchEvent(new Event("scroll"));
      await new Promise((resolve) => requestAnimationFrame(resolve));
    });
    expect(current()).toBe("#valves");
    expect(screen.getByRole("button", { name: "2 of 4: The three valves" })).toBeTruthy();
  });

  it("folds into a bar that names the chapter and opens the list", () => {
    vi.stubGlobal("fetch", serving());
    render(<App />);
    const bar = screen.getByRole("button", { name: "1 of 4: Quota sampling" });
    expect(bar.getAttribute("aria-expanded")).toBe("false");
    fireEvent.click(bar);
    expect(bar.getAttribute("aria-expanded")).toBe("true");
    expect(screen.getByRole("navigation").className).toContain("open");
    fireEvent.click(screen.getByRole("link", { name: /^3 The requisition lifecycle/ }));
    expect(bar.getAttribute("aria-expanded")).toBe("false");
    fireEvent.click(bar);
    fireEvent.keyDown(window, { key: "Escape" });
    expect(bar.getAttribute("aria-expanded")).toBe("false");
  });

  it("keeps the address's chapter in view as the chapters above fill in, until the reader scrolls", () => {
    vi.stubGlobal("fetch", serving());
    render(<App />);
    let resized = () => {};
    vi.stubGlobal(
      "ResizeObserver",
      class {
        constructor(callback: () => void) {
          resized = callback;
        }
        observe() {}
        disconnect() {
          resized = () => {};
        }
      },
    );
    const scrolls = vi.fn();
    (document.getElementById("sbc") as HTMLElement).scrollIntoView = scrolls;
    const stop = holdAnchor("#sbc");
    resized(); // a chapter above grew
    expect(scrolls).toHaveBeenCalledTimes(2);
    window.dispatchEvent(new Event("wheel"));
    resized();
    expect(scrolls).toHaveBeenCalledTimes(2);
    stop();
    expect(holdAnchor("#nowhere")).toBeTypeOf("function");
  });

  it("opens the page at the chapter its address names", () => {
    vi.stubGlobal("fetch", serving());
    const scrolled: string[] = [];
    const original = HTMLElement.prototype.scrollIntoView; // jsdom doesn't scroll
    HTMLElement.prototype.scrollIntoView = function (this: HTMLElement) {
      scrolled.push(this.id);
    };
    try {
      window.history.replaceState(null, "", "/#lifecycle");
      render(<App />);
      expect(scrolled).toEqual(["lifecycle"]);
    } finally {
      HTMLElement.prototype.scrollIntoView = original;
    }
  });
});
