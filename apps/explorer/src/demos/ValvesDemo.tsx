// The valves chapter (the design's E6): one chart, a valve at a time. Chips choose the valve in
// focus, the subject with its band; the plan and the other runs are its context. A line under the
// chart names the valve's lever and where it ends against the plan.

import { useState } from "react";

import type { Valves } from "../api";
import { LineChart, spreadSeries, type Series } from "../charts";
import { Adjust, Choices, Control, Presets, Status, useControls } from "../Control";
import { DraftMark, type DemoProps, type Line } from "../story";
import { useDemo } from "../useDemo";

/** The valves, in the order the chapter explains them: each method's name in the plan's terms,
 * and the lever it pulls. The levers are placeholders, for the maintainer's review in #660. */
export const VALVES: readonly { method: string; name: string; lever: Line }[] = [
  {
    method: "none",
    name: "No valve",
    lever: { text: "With no valve, the run follows the world's own rates and ignores the plan.", draft: true },
  },
  {
    method: "supply",
    name: "Supply",
    lever: {
      text: "Supply opens the requisitions the plan's growth needs, and heads arrive only as they fill, months later.",
      draft: true,
    },
  },
  {
    method: "demand",
    name: "Flow",
    lever: { text: "Flow aligns guidance's hires and exits to the plan, month by month.", draft: true },
  },
  {
    method: "rate_calibration",
    name: "Rate",
    lever: { text: "Rate calibrates the domains' rates, so an unguided run drifts toward the plan.", draft: true },
  },
];
/** The valve in focus at first: Supply, the first the chapter explains. */
const FIRST = 1;

function round(v: number): number {
  return Number(v.toFixed(1));
}

/** Where a run ends against the plan, such as "It ends at 331.5, 4.5 under the plan's 336." */
export function ending(end: number, plan: number): string {
  const gap = round(end - plan);
  const against = gap === 0 ? "on" : `${Math.abs(gap)} ${gap < 0 ? "under" : "over"}`;
  return `It ends at ${round(end)}, ${against} the plan's ${round(plan)}.`;
}

export function ValvesDemo({ name, presets }: DemoProps) {
  const controls = useControls(presets);
  const [focus, setFocus] = useState(FIRST);
  const growth = Math.round((controls.values.growth ?? 0.05) * 100);
  const { data, error, loading } = useDemo<Valves>("valves", controls.query);
  const valve = VALVES[focus] ?? VALVES[FIRST];
  const run = data?.runs[valve.method];
  return (
    <div className="demo">
      <p className="lead">
        A plan for every cost center, rising evenly over six months, and the valve each alignment
        method drives toward it: the requisitions growth opens (supply), guidance aligned to the
        plan (flow), or calibrated domain rates (rate).
      </p>
      <Choices
        label="Valve in focus"
        options={VALVES.map((v) => v.name)}
        chosen={focus}
        onChoose={setFocus}
        className="chips"
      />
      {data && run && (
        <div className={loading ? "running" : undefined}>
          <LineChart
            title={`Headcount from ${data.run_date}, every cost center together`}
            x="month-end"
            y="headcount"
            xs={Array.from({ length: data.months + 1 }, (_, i) => i)}
            series={[
              { name: "Plan", values: [Number.NaN, ...data.plan], role: "plan" },
              ...VALVES.flatMap(({ method, name }): Series[] => {
                const r = data.runs[method];
                if (r === undefined) {
                  return [];
                }
                return [method === valve.method ? spreadSeries(name, r.headcount) : { name, values: r.headcount.p50 }];
              }),
            ]}
            summary={
              <>
                {valve.lever.text} {ending(run.headcount.p50.at(-1) ?? Number.NaN, data.plan.at(-1) ?? Number.NaN)}
                {valve.lever.draft && <DraftMark />}
              </>
            }
            announce
          />
        </div>
      )}
      <Status loading={loading} error={error} />
      <Presets name={name} presets={presets} chosen={controls.chosen} onChoose={controls.choose} />
      <Adjust name={name}>
        <Control label="Plan's growth (%)" value={growth} min={-10} max={20} onChange={(v) => controls.adjust({ growth: v / 100 })} />
      </Adjust>
    </div>
  );
}
