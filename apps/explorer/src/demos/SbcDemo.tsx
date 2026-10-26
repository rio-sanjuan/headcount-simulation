// The SBC chapter (the design's E8): net and true-ups. A bar per month split into the parts the
// books record: the net expense and the vesting true-up above the line, and the attrition true-up
// below it. The recognized total is the subject, as the accent line with its band, and its parts
// are context.

import type { Sbc, SbcPart } from "../api";
import { StackedBars, format, type Part } from "../charts";
import { Adjust, Control, Presets, Status, useControls } from "../Control";
import { DraftMark, type DemoProps } from "../story";
import { useDemo } from "../useDemo";

/** Each part, as the chart draws it: its name, and its gray, lighter for net expense and lightest
 * for the attrition true-up below the line. */
const PARTS: readonly { key: SbcPart; name: string; shade: Part["shade"] }[] = [
  { key: "net", name: "net expense", shade: 2 },
  { key: "vesting", name: "vesting true-up", shade: 3 },
  { key: "rate", name: "rate true-up", shade: 3 },
  { key: "attrition", name: "attrition true-up", shade: 1 },
];
/** A part this small in every month carries nothing worth a bar. */
const NOTHING = 0.5;

function mean(values: number[]): number {
  return values.length === 0 ? 0 : values.reduce((a, b) => a + b, 0) / values.length;
}

/** The parts with something to draw. The rate true-up is drawn only where the forfeiture rate is
 * revised. */
export function drawn(data: Sbc): Part[] {
  return PARTS.filter((p) => data.parts[p.key].some((v) => Math.abs(v) >= NOTHING)).map((p) => ({
    name: p.name,
    values: data.parts[p.key],
    shade: p.shade,
  }));
}

/** The chart's summary: the run's average month, and the parts it adds up from, such as "Over 12
 * months, recognized expense averages 256k a month: net expense 261k, plus vesting true-ups of
 * 33k, less attrition true-ups of 38k." */
export function sbcSummary(data: Sbc): string {
  const parts = drawn(data)
    .map((p, k) => {
      const v = mean(p.values);
      const name = p.name === "net expense" ? "net expense" : `${p.name}s of`;
      const sign = k === 0 ? "" : v < 0 ? "less " : "plus ";
      return `${sign}${name} ${format(Math.abs(v))}`;
    })
    .join(", ");
  return `Over ${data.months} months, recognized expense averages ${format(mean(data.total.mean))} a month: ${parts}.`;
}

/** How much of a month's expense comes from grants made before the run, as a whole percent. */
function inherited(data: Sbc, month: number): number {
  const seed = data.origins.seed?.[month] ?? 0;
  const total = data.total.mean[month] ?? 0;
  return total === 0 ? 0 : Math.round((seed / total) * 100);
}

export function SbcDemo({ name, presets }: DemoProps) {
  const controls = useControls(presets);
  const months = controls.values.months ?? 12;
  const { data, error, loading } = useDemo<Sbc>("sbc", controls.query);
  const rateless = data !== undefined && !drawn(data).some((p) => p.name === "rate true-up");
  return (
    <div className="demo">
      <p className="lead">
        Every grant becomes tranches that vest over the years after it, and each tranche&apos;s
        value is recognized as expense month by month. The run&apos;s expense comes from the
        equity granted before it (seed), the grants it makes (sim), and offline plans.
      </p>
      {data && (
        <div className={loading ? "running" : undefined}>
          <StackedBars
            title="Expense recognized each month, and its parts"
            x="month"
            y="expense"
            xs={Array.from({ length: data.months }, (_, i) => i)}
            parts={drawn(data)}
            total={{
              name: "recognized",
              role: "subject",
              // The bars are means, so the line they add up to is the mean, in its band.
              values: data.total.mean,
              band: { low: data.total.p10, high: data.total.p90 },
            }}
            summary={sbcSummary(data)}
            after={
              rateless && (
                <p className="summary">The forfeiture rate isn&apos;t revised on this world, so there&apos;s no rate true-up.</p>
              )
            }
          />
          <p className="lead">
            Grants made before the run carry {inherited(data, 0)}% of the first month&apos;s expense, and{" "}
            {inherited(data, data.months - 1)}% of the last month&apos;s, as the run&apos;s own grants take over.
            <DraftMark />
          </p>
        </div>
      )}
      <Status loading={loading} error={error} />
      <Presets name={name} presets={presets} chosen={controls.chosen} onChoose={controls.choose} />
      <Adjust name={name}>
        <Control label="Months" value={months} min={3} max={36} onChange={(v) => controls.adjust({ months: v })} />
      </Adjust>
    </div>
  );
}
