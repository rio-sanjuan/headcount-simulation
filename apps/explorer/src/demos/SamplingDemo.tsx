import { useState } from "react";

import type { Sampling } from "../api";
import { BarChart, LineChart } from "../charts";
import { Control, Status } from "../Control";
import { useDemo } from "../useDemo";

export function SamplingDemo() {
  const [rows, setRows] = useState(200);
  const [quota, setQuota] = useState(20);
  const [spread, setSpread] = useState(1);
  const { data, error, loading } = useDemo<Sampling>("sampling", {
    rows,
    quota: Math.min(quota, rows),
    spread,
    trajectories: 500,
  });
  return (
    <div className="demo">
      <p className="lead">
        Where guidance sets a count, the engine draws exactly that many rows, the model choosing
        which (quota sampling). Drawing every row on its own at the same rates expects the count,
        but misses it from draw to draw.
      </p>
      <div className="controls">
        <Control label="Rows" value={rows} min={20} max={1000} step={10} onChange={setRows} />
        <Control label="Quota" value={Math.min(quota, rows)} min={1} max={rows} onChange={setQuota} />
        <Control label="Spread of probabilities" value={spread} min={0} max={2} step={0.1} onChange={setSpread} />
      </div>
      <Status loading={loading} error={error} />
      {data && (
        <>
          <p className="stats">
            Expected count {data.expected}: quota sampling drew {data.quota.mean} ± {data.quota.sd}; independent
            draws {data.independent.mean} ± {data.independent.sd}, over 500 draws.
          </p>
          <BarChart
            title="How many rows each draw took"
            x="rows drawn"
            y="draws"
            categories={data.bins}
            series={[
              { name: "quota sampling", values: data.quota.histogram, role: "subject" },
              { name: "independent draws", values: data.independent.histogram },
            ]}
          />
          <LineChart
            title="How often each row was drawn, likeliest first"
            x="row"
            y="share of draws"
            xs={data.probabilities.map((_, i) => i)}
            series={[
              { name: "quota sampling", values: data.quota.rates, role: "subject" },
              { name: "independent draws", values: data.independent.rates },
              { name: "probability", values: data.probabilities, role: "plan" },
            ]}
          />
        </>
      )}
    </div>
  );
}
