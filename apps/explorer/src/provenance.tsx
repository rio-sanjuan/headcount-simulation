// The run behind each chart (the design's E9): a caption every chart's frame ends with, naming the
// world and its size, the month its runs start, the trajectories, the seed, and the engine's
// version, so a chart carries its provenance wherever it goes.

import { createContext, useContext } from "react";

import type { Health } from "./api";

/** The page's world, as `/api/health` names it, once it has answered. */
export const WorldContext = createContext<Health | undefined>(undefined);

/** A chapter's run: its trajectories and seed, and, for synthetic rows rather than the world's,
 * how many rows. */
export type Run = { trajectories: number; seed: number; rows?: number };

/** The run a chapter's charts draw, which their captions name. */
export const RunContext = createContext<Run | undefined>(undefined);

/** A caption naming a run, such as "Run on tiny5.duckdb (320 employees, 10 cost centers) from
 * 2026-01 · 10 trajectories · seed 7 · hcsim 0.1.0", or, for synthetic rows, "200 synthetic rows ·
 * 500 trajectories · seed 0 · hcsim 0.1.0". Before the world is known, it names the run alone. */
export function caption(world: Health | undefined, run: Run): string {
  const count = (n: number, one: string, many: string) => `${n.toLocaleString("en-US")} ${n === 1 ? one : many}`;
  const where =
    run.rows !== undefined
      ? [count(run.rows, "synthetic row", "synthetic rows")]
      : world === undefined
        ? []
        : [
            `Run on ${world.world} (${count(world.employees, "employee", "employees")}, ` +
              `${count(world.cost_centers, "cost center", "cost centers")}) from ${world.run_date.slice(0, 7)}`,
          ];
  const engine = world === undefined ? [] : [`hcsim ${world.version}`];
  return [...where, count(run.trajectories, "trajectory", "trajectories"), `seed ${run.seed}`, ...engine].join(" · ");
}

/** The caption a chart's frame ends with, for the chapter's run; none outside a chapter. */
export function Provenance() {
  const world = useContext(WorldContext);
  const run = useContext(RunContext);
  return run === undefined ? null : <p className="provenance">{caption(world, run)}</p>;
}
