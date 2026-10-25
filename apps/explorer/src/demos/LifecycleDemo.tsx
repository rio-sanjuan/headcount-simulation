import { useState } from "react";

import type { Lifecycle } from "../api";
import { BarChart, LineChart } from "../charts";
import { Control, Status } from "../Control";
import { useDemo } from "../useDemo";

const FLOWS: Record<string, string> = {
  G: "growth",
  G_camp: "campus",
  B_att: "backfill for departures",
  B_tr: "backfill for transfers",
  F_ext: "external",
  F_conv: "conversion",
  F_tr: "transfer",
  F_camp: "campus",
  F_pend: "pending start",
};

function total(parts: Record<string, number[]>, months: number): number[] {
  return Array.from({ length: months }, (_, i) =>
    Object.values(parts).reduce((sum, values) => sum + (values[i] ?? 0), 0),
  );
}

export function LifecycleDemo() {
  const [months, setMonths] = useState(12);
  const { data, error, loading } = useDemo<Lifecycle>("lifecycle", { months, trajectories: 4 });
  const steps = data ? Array.from({ length: data.months }, (_, i) => i + 1) : [];
  return (
    <section>
      <p className="lead">
        A requisition opens (for growth, a campus class, or a backfill), waits as open openings,
        and leaves either filled, by one of five channels, or unfilled. Every month the stock
        changes by exactly what opened, less what filled and what left unfilled.
      </p>
      <div className="controls">
        <Control label="Months" value={months} min={3} max={36} onChange={setMonths} />
      </div>
      <Status loading={loading} error={error} />
      {data && (
        <>
          <p className="stats">
            The opening identity's largest residual: {data.identity_residual} openings.
          </p>
          <LineChart
            title="Open openings at each month-end"
            x="month-end"
            y="openings"
            xs={Array.from({ length: data.months + 1 }, (_, i) => i)}
            series={[{ name: "open", values: data.open }]}
          />
          <BarChart
            title="Each month's flows, per trajectory"
            x="month"
            y="openings"
            categories={steps}
            series={[
              { name: "opened", values: total(data.opened, data.months) },
              { name: "filled", values: total(data.filled, data.months) },
              { name: "left unfilled", values: data.unfilled },
            ]}
          />
          <BarChart
            title="Fills by channel, over the run"
            x="channel"
            y="openings"
            categories={Object.keys(data.filled).map((k) => FLOWS[k] ?? k)}
            series={[
              {
                name: "filled",
                values: Object.values(data.filled).map((v) => v.reduce((a, b) => a + b, 0)),
              },
            ]}
          />
        </>
      )}
    </section>
  );
}
