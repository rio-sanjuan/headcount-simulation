import type { Sbc } from "../api";
import { BarChart, LineChart } from "../charts";
import { Adjust, Control, Presets, Status, useControls } from "../Control";
import type { DemoProps } from "../story";
import { useDemo } from "../useDemo";

export function SbcDemo({ name, presets }: DemoProps) {
  const controls = useControls(presets);
  const months = controls.values.months ?? 12;
  const { data, error, loading } = useDemo<Sbc>("sbc", controls.query);
  const grant = data?.example_grant;
  return (
    <div className="demo">
      <p className="lead">
        Every grant becomes tranches that vest over the years after it, and each tranche&apos;s
        value is recognized as expense month by month. The run&apos;s expense comes from the
        equity granted before it (seed), the grants it makes (sim), and offline plans.
      </p>
      {data && (
        <div className={loading ? "running" : undefined}>
          <LineChart
            title="Expense recognized each month, by part"
            x="month"
            y="recognized"
            xs={Array.from({ length: data.months }, (_, i) => i)}
            series={[
              // The parts are means, so the total they add up to is the mean too, in its band.
              {
                name: "total",
                role: "subject",
                values: data.total.mean,
                band: { low: data.total.p10, high: data.total.p90 },
              },
              ...Object.entries(data.recognized).map(([part, values]) => ({ name: part, values })),
            ]}
          />
          <BarChart
            title="Grants the run makes, per trajectory"
            x="award type"
            y="grants"
            categories={Object.keys(data.grants)}
            series={[{ name: "grants", values: Object.values(data.grants), role: "subject" }]}
          />
          {grant?.vest_months && grant.shares && (
            <BarChart
              title={`One new-hire grant, awarded in month ${grant.award_month}: shares by vest month`}
              x="vest month"
              y="shares"
              categories={grant.vest_months}
              series={[{ name: "shares", values: grant.shares, role: "subject" }]}
            />
          )}
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
