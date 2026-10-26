// A chapter's states (the design's E10): waiting while its first answer runs, saying what the
// engine is doing and about how long is left; failed, with the server's message and "Try again";
// and ready, its charts, dimmed with a polite "Running…" while a new answer runs.

import { useContext, useEffect, useState, type ReactNode } from "react";

import type { Health, Query } from "./api";
import { Status } from "./Control";
import { WorldContext } from "./provenance";
import type { Loaded } from "./useDemo";

/** What the engine is doing for a demo's query, in a few words. */
export function doing(demo: string, query: Query, world?: string): string {
  const on = world === undefined ? "" : ` on ${world}`;
  const { trajectories = 10, months = 6, rows = 200 } = query;
  switch (demo) {
    case "sampling":
      return `Drawing ${rows} rows ${trajectories} times`;
    case "valves":
      return `Fitting guidance${on}, then running ${trajectories} trajectories of each valve`;
    case "sbc":
      return `Running ${trajectories} trajectories of ${months} months${on}, with SBC`;
    default:
      return `Running ${trajectories} trajectories of ${months} months${on}`;
  }
}

/** About how long is left of an estimate, after `elapsed` seconds: rounded, as "about 40 s left".
 * Nothing when there's no estimate. */
export function timeLeft(estimate: number | null | undefined, elapsed: number): string {
  if (estimate === null || estimate === undefined) {
    return "";
  }
  const left = estimate - elapsed;
  if (left > 90) {
    return `about ${Math.round(left / 60)} min left`;
  }
  if (left > 10) {
    return `about ${Math.round(left / 5) * 5} s left`;
  }
  if (left > 1) {
    return `about ${Math.ceil(left)} s left`;
  }
  return elapsed < estimate * 1.5 ? "almost done" : "taking longer than expected";
}

/** Seconds since the component mounted, ticking while `running`. */
function useElapsed(running: boolean): number {
  const [start] = useState(() => performance.now());
  const [now, setNow] = useState(start);
  useEffect(() => {
    if (!running) {
      return;
    }
    const tick = setInterval(() => setNow(performance.now()), 500);
    return () => clearInterval(tick);
  }, [running]);
  return (now - start) / 1000;
}

function Waiting({ demo, query, world }: { demo: string; query: Query; world?: Health }) {
  const elapsed = useElapsed(true);
  const estimate = world?.chapters?.[demo]?.seconds;
  const left = timeLeft(estimate, elapsed);
  // A bar that fills as the estimate runs out, and stops short of full until the answer comes.
  const share = estimate ? Math.min(0.95, elapsed / estimate) : undefined;
  return (
    <div className="chart-state waiting">
      {/* What the engine is doing is announced once; the time left, which ticks, isn't. */}
      <p>
        <span role="status">{doing(demo, query, world?.world)}</span>
        {left && <span> · {left}</span>}
      </p>
      <div
        className={share === undefined ? "progress unknown" : "progress"}
        role="progressbar"
        aria-label="The engine's progress"
        aria-valuemin={0}
        aria-valuemax={100}
        aria-valuenow={share === undefined ? undefined : Math.round(share * 100)}
      >
        <div style={share === undefined ? undefined : { width: `${share * 100}%` }} />
      </div>
    </div>
  );
}

function Failed({ error, retry }: { error: string; retry: () => void }) {
  return (
    <div className="chart-state failed" role="alert">
      <p>The engine couldn&apos;t run this chapter: {error}</p>
      <button type="button" onClick={retry}>
        Try again
      </button>
    </div>
  );
}

/** A chapter's charts once its answer has come, and its waiting and failed states before. */
export function ChapterState<T>({
  demo,
  query,
  loaded,
  children,
}: {
  demo: string;
  query: Query;
  loaded: Loaded<T>;
  children: (data: T) => ReactNode;
}) {
  const world = useContext(WorldContext);
  const { data, error, loading, retry } = loaded;
  if (data === undefined) {
    return error === undefined ? <Waiting demo={demo} query={query} world={world} /> : <Failed error={error} retry={retry} />;
  }
  return (
    <>
      <div className={loading ? "running" : undefined}>{children(data)}</div>
      <Status loading={loading} error={error} />
    </>
  );
}
