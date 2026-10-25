import { useState } from "react";

import { TRAJECTORIES, type Valves } from "../api";
import { LineChart, spreadSeries, type Series } from "../charts";
import { Control, Status } from "../Control";
import { useDemo } from "../useDemo";

/** Each method's valve, as the plan names it. */
const NAMES: Record<string, string> = {
  none: "No valve",
  supply: "Supply",
  demand: "Flow",
  rate_calibration: "Rate",
};
/** The valve in focus. */
const FOCUS = "supply";

export function ValvesDemo() {
  const [growth, setGrowth] = useState(5);
  const { data, error, loading } = useDemo<Valves>("valves", { growth: growth / 100, trajectories: TRAJECTORIES });
  return (
    <div className="demo">
      <p className="lead">
        A plan for every cost center, rising evenly over six months, and the valve each alignment
        method drives toward it: the requisitions growth opens (supply), guidance aligned to the
        plan (flow), or calibrated domain rates (rate). Bias is the mean gap per cost center where
        the plan is scored, at its last month.
      </p>
      <div className="controls">
        <Control label="Plan's growth (%)" value={growth} min={-10} max={20} onChange={setGrowth} />
      </div>
      <Status loading={loading} error={error} />
      {data && (
        <>
          <LineChart
            title={`Headcount from ${data.run_date}, every cost center together`}
            x="month-end"
            y="headcount"
            xs={Array.from({ length: data.months + 1 }, (_, i) => i)}
            series={[
              { name: "Plan", values: [Number.NaN, ...data.plan], role: "plan" },
              ...Object.entries(data.runs).map(([method, run]): Series => {
                const name = NAMES[method] ?? method;
                return method === FOCUS
                  ? spreadSeries(name, run.headcount)
                  : { name, values: run.headcount.p50 };
              }),
            ]}
          />
          <table>
            <thead>
              <tr>
                <th>Method</th>
                <th>Valve</th>
                <th>Mean bias at the plan's last month</th>
              </tr>
            </thead>
            <tbody>
              {Object.entries(data.runs).map(([method, run]) => (
                <tr key={method}>
                  <td>{method}</td>
                  <td>{run.valve}</td>
                  <td>{run.bias_mean} heads</td>
                </tr>
              ))}
            </tbody>
          </table>
        </>
      )}
    </div>
  );
}
