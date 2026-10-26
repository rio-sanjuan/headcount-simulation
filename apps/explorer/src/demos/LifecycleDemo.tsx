// The lifecycle chapter (the design's E7): flows through a stock. Growth and backfill flow into
// the stock of open openings, and fills and unfilled exits flow out, each ribbon as wide as its
// total over the run. The stock is the subject, and the flows are its context. The stock's line
// and band sit under the diagram.

import type { Lifecycle } from "../api";
import { FlowDiagram, LineChart, spreadSeries, type Flow } from "../charts";
import { ChapterState } from "../ChapterState";
import { Adjust, Control, Presets, useControls } from "../Control";
import type { DemoProps } from "../story";
import { RunContext } from "../provenance";
import { useDemo } from "../useDemo";

/** The flows in: each requisition type that opens openings, its ribbon's name, and how the line
 * of empty channels names it. Backfill is "backfill" alone, as a single fill channel is "filled". */
const OPENED: Record<string, { name: string; alone?: string; empty: string }> = {
  G: { name: "growth", empty: "growth openings" },
  G_camp: { name: "campus", empty: "campus openings" },
  B_att: { name: "backfill: departures", alone: "backfill", empty: "backfill for departures" },
  B_tr: { name: "backfill: transfers", alone: "backfill", empty: "backfill for transfers" },
};
/** The fill channels, each a flow out. */
const FILLED: Record<string, string> = {
  F_ext: "external",
  F_conv: "conversion",
  F_tr: "transfer",
  F_camp: "campus",
  F_pend: "pending-start",
};
/** A total that rounds to nothing at one decimal carries nothing. */
const NOTHING = 0.05;

function sum(values: number[]): number {
  return values.reduce((a, b) => a + b, 0);
}

/** "a", "a or b", or "a, b or c", with a comma before the last of three or more if `serial`. */
export function listed(items: string[], last = "or", serial = false): string {
  if (items.length < 2) {
    return items[0] ?? "";
  }
  const comma = serial && items.length > 2 ? "," : "";
  return `${items.slice(0, -1).join(", ")}${comma} ${last} ${items.at(-1)}`;
}

/** The ribbons in and out of the stock over the run, and the channels that carried nothing. Each
 * fill channel is its own ribbon, named "filled" alone when it's the only one. */
export function flows(data: Lifecycle): { inflows: Flow[]; outflows: Flow[]; empty: string[] } {
  const opened = Object.entries(data.opened).map(([k, v]) => ({ key: k, total: sum(v) }));
  const filled = Object.entries(data.filled).map(([k, v]) => ({ key: k, name: FILLED[k] ?? k, total: sum(v) }));
  const carries = (f: { total: number }) => f.total >= NOTHING;
  const fills = filled.filter(carries);
  const backfills = opened.filter((f) => carries(f) && f.key.startsWith("B_")).length;
  const inflows = opened.filter(carries).map(({ key, total }) => {
    const names = OPENED[key];
    return { name: (backfills === 1 ? names?.alone : undefined) ?? names?.name ?? key, total };
  });
  const outflows = [
    ...fills.map(({ name, total }) => ({ name: fills.length === 1 ? "filled" : `filled: ${name}`, total })),
    { name: "unfilled", total: sum(data.unfilled) },
  ].filter(carries);
  const emptyFills = filled.filter((f) => !carries(f)).map((f) => f.name);
  const empty = [
    ...opened.filter((f) => !carries(f)).map((f) => OPENED[f.key]?.empty ?? f.key),
    ...(emptyFills.length > 0 ? [`the ${listed(emptyFills, "and")} fills`] : []),
    ...(sum(data.unfilled) < NOTHING ? ["unfilled exits"] : []),
  ];
  return { inflows, outflows, empty };
}

/** The identity, in words: what opened equals what closed plus the change in the stock, such as
 * "46.7 opened = 38.4 closed + 3.0 more open". */
export function identity(inflows: Flow[], outflows: Flow[]): string {
  const opened = sum(inflows.map((f) => f.total));
  const closed = sum(outflows.map((f) => f.total));
  const change = opened - closed;
  const more = change < 0 ? `${Math.abs(change).toFixed(1)} fewer open` : `${change.toFixed(1)} more open`;
  return `${opened.toFixed(1)} opened = ${closed.toFixed(1)} closed + ${more}`;
}

/** The stock's summary: its median at the start and the end, and its band at the end. */
export function stockSummary(open: Lifecycle["open"]): string {
  const last = open.p50.length - 1;
  const at = (values: number[], i: number) => Number((values[i] ?? Number.NaN).toFixed(1));
  return (
    `The median stock goes from ${at(open.p50, 0)} open openings to ${at(open.p50, last)} over ${last} months; ` +
    `at the end, 80% of trajectories hold ${at(open.p10, last)} to ${at(open.p90, last)}.`
  );
}

export function LifecycleDemo({ name, presets }: DemoProps) {
  const controls = useControls(presets);
  const months = controls.values.months ?? 12;
  const loaded = useDemo<Lifecycle>("lifecycle", controls.query);
  return (
    <div className="demo">
      <ChapterState demo="lifecycle" query={controls.query} loaded={loaded}>
        {(data) => {
          const through = flows(data);
          return (
            <RunContext.Provider value={{ trajectories: data.trajectories, seed: data.seed }}>
              <FlowDiagram
                title={`Openings over ${data.months} months, per trajectory`}
                inflows={through.inflows}
                outflows={through.outflows}
                stock={{ name: "open openings", start: data.open.mean[0] ?? 0, end: data.open.mean.at(-1) ?? 0 }}
                summary={identity(through.inflows, through.outflows)}
                after={
                  through.empty.length > 0 && (
                    <p className="summary">Nothing flowed through {listed(through.empty, "or", true)} on this world.</p>
                  )
                }
              />
              <LineChart
                title="Open openings at each month-end"
                x="month-end"
                y="openings"
                xs={Array.from({ length: data.months + 1 }, (_, i) => i)}
                series={[spreadSeries("open", data.open)]}
                summary={stockSummary(data.open)}
              />
            </RunContext.Provider>
          );
        }}
      </ChapterState>
      <Presets name={name} presets={presets} chosen={controls.chosen} onChoose={controls.choose} />
      <Adjust name={name} controls={controls} seconds={loaded.data?.seconds}>
        <Control label="Months" value={months} min={3} max={36} onChange={(v) => controls.adjust({ months: v })} />
      </Adjust>
    </div>
  );
}
