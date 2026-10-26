// The explorer's API, as `hcsim app serve` serves it under /api.

export type Draws = { histogram: number[]; mean: number; sd: number; rates: number[] };

/** The page's world, the engine that runs on it, and about how many seconds each chapter's first
 * preset takes to run: none where it's kept, and null where there's no estimate. */
export type Health = {
  world: string;
  run_date: string;
  employees: number;
  cost_centers: number;
  version: string;
  chapters: Record<string, { seconds: number | null }>;
};

/** What every run says about itself: how many trajectories it took, its seed, and how many
 * seconds it took to make. */
export type Ran = { trajectories: number; seed: number; seconds?: number };

export type Sampling = Ran & {
  rows: number;
  probabilities: number[];
  expected: number;
  bins: number[];
  quota: Draws;
  independent: Draws;
};

/** A value across trajectories, month by month: its mean, and its 10th, 50th and 90th
 * percentiles. */
export type Spread = { mean: number[]; p10: number[]; p50: number[]; p90: number[] };

export type ValveRun = { valve: string; headcount: Spread; bias_mean: number };

export type Valves = Ran & {
  months: number;
  run_date: string;
  plan: number[];
  runs: Record<string, ValveRun>;
};

export type Lifecycle = Ran & {
  months: number;
  run_date: string;
  open: Spread;
  opened: Record<string, number[]>;
  filled: Record<string, number[]>;
  unfilled: number[];
  identity_residual: number;
};

/** The parts the books record SBC in: they add up to what's recognized. */
export type SbcPart = "net" | "vesting" | "attrition" | "rate";

export type Sbc = Ran & {
  months: number;
  run_date: string;
  /** The expense recognized each month, across trajectories. */
  total: Spread;
  /** Each part's mean, month by month. */
  parts: Record<SbcPart, number[]>;
  /** The mean expense from the grants made before the run (seed), by it (sim), and offline. */
  origins: Record<string, number[]>;
};

export type Query = Record<string, number>;

/** The page's world, or nothing if the server doesn't answer. */
export async function health(): Promise<Health | undefined> {
  try {
    const response = await fetch("/api/health");
    return response.ok ? ((await response.json()) as Health) : undefined;
  } catch {
    return undefined;
  }
}

/** GET one demo, with its parameters, refusing an answer that isn't OK with the API's reason. */
export async function get<T>(demo: string, query: Query): Promise<T> {
  const params = new URLSearchParams(Object.entries(query).map(([k, v]) => [k, String(v)]));
  const response = await fetch(`/api/${demo}?${params}`);
  if (!response.ok) {
    let reason = response.statusText;
    try {
      const body = (await response.json()) as { detail?: unknown };
      if (body.detail !== undefined) {
        reason = typeof body.detail === "string" ? body.detail : JSON.stringify(body.detail);
      }
    } catch {
      // The body wasn't JSON: the status says enough.
    }
    throw new Error(`${demo}: ${reason}`);
  }
  return (await response.json()) as T;
}
