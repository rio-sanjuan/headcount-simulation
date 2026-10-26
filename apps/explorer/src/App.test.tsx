import { act, cleanup, fireEvent, render, screen, waitFor, within } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { get } from "./api";
import { App, holdAnchor } from "./App";
import { CHAPTERS } from "./story";
import { forget } from "./useDemo";

const spread = (values: number[]) => ({ mean: values, p10: values, p50: values, p90: values });

/** An answer for each demo, as the API gives it. */
const ANSWERS: Record<string, unknown> = {
  sampling: {
    probabilities: [0.3, 0.2],
    expected: 0.5,
    bins: [0, 1],
    quota: { histogram: [0, 10], mean: 1, sd: 0, rates: [0.6, 0.4] },
    independent: { histogram: [5, 5], mean: 0.5, sd: 0.5, rates: [0.3, 0.2] },
  },
  valves: {
    months: 2,
    run_date: "2026-01-01",
    plan: [101, 102],
    runs: { supply: { valve: "supply", headcount: spread([100, 100.5, 101]), bias_mean: 1 } },
  },
  lifecycle: {
    months: 2,
    run_date: "2026-01-01",
    open: spread([5, 6, 7]),
    opened: { G: [2, 2] },
    filled: { F_ext: [1, 1] },
    unfilled: [0, 0],
    identity_residual: 0,
  },
  sbc: {
    months: 2,
    run_date: "2026-01-01",
    total: spread([10, 12]),
    recognized: { seed: [8, 9], sim: [2, 3] },
    grants: { new_hire: 2 },
    example_grant: {},
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
    await waitFor(() => expect(screen.getByText(/largest residual: 0 openings/)).toBeTruthy());
    expect(screen.getByRole("heading", { level: 1 }).textContent).toBe("hcsim explorer");
  });

  it("marks the text that waits for review as a draft", () => {
    vi.stubGlobal("fetch", serving());
    render(<App />);
    const drafts = CHAPTERS.filter((c) => c.handOn.draft).length;
    expect(screen.getAllByText("draft")).toHaveLength(drafts);
  });

  it("shows a demo's error in its own chapter", async () => {
    vi.stubGlobal("fetch", answering({ detail: "no world" }, false));
    render(<App />);
    const sampling = document.getElementById("sampling") as HTMLElement;
    await waitFor(() => expect(within(sampling).getByText("sampling: no world")).toBeTruthy());
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
    expect(links[1]?.textContent).toBe("2The three valves");
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
    fireEvent.click(screen.getByRole("link", { name: "3The requisition lifecycle" }));
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
