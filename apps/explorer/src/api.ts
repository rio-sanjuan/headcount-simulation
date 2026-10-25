// The explorer's API, as `hcsim app serve` serves it under /api.

export type Draws = { histogram: number[]; mean: number; sd: number; rates: number[] };

export type Sampling = {
  probabilities: number[];
  expected: number;
  bins: number[];
  quota: Draws;
  independent: Draws;
};

export type ValveRun = { valve: string; headcount: number[]; bias_mean: number };

export type Valves = {
  months: number;
  run_date: string;
  plan: number[];
  runs: Record<string, ValveRun>;
};

export type Lifecycle = {
  months: number;
  run_date: string;
  open: number[];
  opened: Record<string, number[]>;
  filled: Record<string, number[]>;
  unfilled: number[];
  identity_residual: number;
};

export type Sbc = {
  months: number;
  run_date: string;
  recognized: Record<string, number[]>;
  grants: Record<string, number>;
  example_grant: { award_month?: number; vest_months?: number[]; shares?: number[] };
};

export type Query = Record<string, number>;

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
