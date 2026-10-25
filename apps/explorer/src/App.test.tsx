import { cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";

import { get } from "./api";
import { App, DEMOS } from "./App";

const SAMPLING = {
  probabilities: [0.3, 0.2],
  expected: 0.5,
  bins: [0, 1],
  quota: { histogram: [0, 10], mean: 1, sd: 0, rates: [0.6, 0.4] },
  independent: { histogram: [5, 5], mean: 0.5, sd: 0.5, rates: [0.3, 0.2] },
};

function answering(body: unknown, ok = true) {
  return vi.fn(async () => ({ ok, statusText: "Unprocessable", json: async () => body }) as Response);
}

afterEach(() => {
  cleanup();
  vi.unstubAllGlobals();
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

describe("the app", () => {
  it("shows the first demo, with its answer, and switches tabs", async () => {
    vi.stubGlobal("fetch", answering(SAMPLING));
    render(<App />);
    expect(screen.getAllByRole("tab")).toHaveLength(DEMOS.length);
    await waitFor(() => expect(screen.getByText(/quota sampling drew 1 ± 0/)).toBeTruthy());
    expect(screen.getByText("How many rows each draw took")).toBeTruthy();
    fireEvent.click(screen.getByRole("tab", { name: "The requisition lifecycle" }));
    expect(screen.getByRole("tab", { name: "The requisition lifecycle" }).getAttribute("aria-selected")).toBe(
      "true",
    );
  });

  it("shows a demo's error", async () => {
    vi.stubGlobal("fetch", answering({ detail: "no world" }, false));
    render(<App />);
    await waitFor(() => expect(screen.getByText("sampling: no world")).toBeTruthy());
  });
});
