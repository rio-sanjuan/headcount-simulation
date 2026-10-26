import type { Sampling } from "../api";
import { BarChart, LineChart } from "../charts";
import { Adjust, Control, Presets, Status, useControls } from "../Control";
import type { DemoProps } from "../story";
import { useDemo } from "../useDemo";

export function SamplingDemo({ name, presets }: DemoProps) {
  const controls = useControls(presets);
  const { rows = 200, quota = 20, spread = 1 } = controls.values;
  const { data, error, loading } = useDemo<Sampling>("sampling", controls.query);
  return (
    <div className="demo">
      <p className="lead">
        Where guidance sets a count, the engine draws exactly that many rows, the model choosing
        which (quota sampling). Drawing every row on its own at the same rates expects the count,
        but misses it from draw to draw.
      </p>
      {data && (
        <div className={loading ? "running" : undefined}>
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
        </div>
      )}
      <Status loading={loading} error={error} />
      <Presets name={name} presets={presets} chosen={controls.chosen} onChoose={controls.choose} />
      <Adjust name={name}>
        <Control label="Rows" value={rows} min={20} max={1000} step={10} onChange={(v) => controls.adjust({ rows: v, quota: Math.min(quota, v) })} />
        <Control label="Quota" value={Math.min(quota, rows)} min={1} max={rows} onChange={(v) => controls.adjust({ quota: v })} />
        <Control label="Spread of probabilities" value={spread} min={0} max={2} step={0.1} onChange={(v) => controls.adjust({ spread: v })} />
      </Adjust>
    </div>
  );
}
