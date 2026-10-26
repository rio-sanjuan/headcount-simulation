// The sampling chapter (the design's E5): the count, then who. The first chart shows how many rows
// each trajectory drew, and the second how often each row was drawn, against the model's
// probability. Quota sampling is the subject, and independent draws are its context.

import type { Sampling } from "../api";
import { Histogram, LineChart } from "../charts";
import { Adjust, Control, Presets, Status, useControls } from "../Control";
import type { DemoProps } from "../story";
import { RunContext } from "../provenance";
import { useDemo } from "../useDemo";

/** The histogram's height, in trajectories, so the independent draws' spread reads clearly. */
const CAP = 80;
/** The groups the rows' rates are averaged in, so their lines read smoothly. */
const GROUPS = 20;

/** How many neighbouring rows a group holds, so `rows` make about `GROUPS` groups. */
function groupSize(rows: number): number {
  return Math.max(1, Math.ceil(rows / GROUPS));
}

/** `values` in groups of `groupSize` neighbours, each the group's mean, and where each group
 * starts, counting from 1. */
export function grouped(values: number[]): { starts: number[]; means: number[] } {
  const size = groupSize(values.length);
  const starts: number[] = [];
  const means: number[] = [];
  for (let start = 0; start < values.length; start += size) {
    const part = values.slice(start, start + size);
    starts.push(start + 1);
    means.push(part.reduce((a, b) => a + b, 0) / part.length);
  }
  return { starts, means };
}

function percent(share: number): string {
  return `${Math.round(share * 100)}%`;
}

/** The counts a histogram drew, as a range: "20", or "8 to 31". */
function drawn(bins: number[], histogram: number[]): string {
  const seen = bins.filter((_, i) => (histogram[i] ?? 0) > 0);
  const [low, high] = [seen[0], seen.at(-1)];
  return low === high ? `${low}` : `${low} to ${high}`;
}

/** The histogram's summary, such as "Quota sampling drew 20 rows in all 500 trajectories;
 * independent draws took 8 to 31." */
export function countSummary(data: Sampling): string {
  const trajectories = data.quota.histogram.reduce((a, b) => a + b, 0);
  const quota = drawn(data.bins, data.quota.histogram);
  return (
    `Quota sampling drew ${quota} ${quota === "1" ? "row" : "rows"} in all ${trajectories} trajectories; ` +
    `independent draws took ${drawn(data.bins, data.independent.histogram)}.`
  );
}

/** The rate chart's summary: how often each sampler drew the likeliest group of rows, against the
 * model's probability. */
export function rateSummary(data: Sampling): string {
  const size = groupSize(data.probabilities.length);
  const first = (values: number[]) => grouped(values).means[0] ?? 0;
  return (
    `The likeliest ${size} rows: quota sampling drew each in ${percent(first(data.quota.rates))} of trajectories, ` +
    `and independent draws in ${percent(first(data.independent.rates))}, against the model's ` +
    `${percent(first(data.probabilities))}.`
  );
}

export function SamplingDemo({ name, presets }: DemoProps) {
  const controls = useControls(presets);
  const { rows = 200, quota = 20, spread = 1 } = controls.values;
  const { data, error, loading } = useDemo<Sampling>("sampling", controls.query);
  const rates = data && {
    quota: grouped(data.quota.rates),
    independent: grouped(data.independent.rates),
    probability: grouped(data.probabilities),
  };
  return (
    <div className="demo">
      <p className="lead">
        Where guidance sets a count, the engine draws exactly that many rows, the model choosing
        which (quota sampling). Drawing every row on its own at the same rates expects the count,
        but misses it from draw to draw.
      </p>
      {data && rates && (
        <RunContext.Provider value={{ trajectories: data.trajectories, seed: data.seed, rows: data.rows }}>
          <div className={loading ? "running" : undefined}>
            <Histogram
              title="How many rows each trajectory drew"
              x="rows drawn"
              y="trajectories"
              bins={data.bins}
              subject={{ name: "Quota sampling", counts: data.quota.histogram }}
              context={{ name: "Independent draws", counts: data.independent.histogram }}
              cap={CAP}
              summary={countSummary(data)}
            />
            <LineChart
              title="How often each row was drawn, likeliest first"
              x={`row, in groups of ${groupSize(data.probabilities.length)}`}
              y="share of trajectories"
              xs={rates.quota.starts}
              labelAt="start"
              series={[
                { name: "Quota sampling", values: rates.quota.means, role: "subject" },
                { name: "Independent draws", values: rates.independent.means },
                { name: "Probability", values: rates.probability.means, role: "plan" },
              ]}
              summary={rateSummary(data)}
            />
          </div>
        </RunContext.Provider>
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
