// Small SVG charts, so the explorer needs no charting library.

import type { ReactNode } from "react";

export type Series = { name: string; values: number[]; dashed?: boolean };

const WIDTH = 640;
const HEIGHT = 260;
const PAD = { left: 56, right: 16, top: 12, bottom: 32 };
/** A colour per series, in order; they repeat past the last. */
export const COLOURS = ["#2563eb", "#dc2626", "#16a34a", "#9333ea", "#ea580c", "#0891b2", "#4b5563"];

/** Maps `[lo, hi]` onto `[from, to]`; a flat domain maps to the range's middle. */
export function scale(lo: number, hi: number, from: number, to: number): (v: number) => number {
  if (hi === lo) {
    return () => (from + to) / 2;
  }
  return (v) => from + ((v - lo) / (hi - lo)) * (to - from);
}

/** Round, evenly spaced ticks covering `[lo, hi]`. */
export function ticks(lo: number, hi: number, count = 5): number[] {
  if (!Number.isFinite(lo) || !Number.isFinite(hi)) {
    return [];
  }
  if (hi === lo) {
    return [lo];
  }
  const raw = (hi - lo) / count;
  const magnitude = 10 ** Math.floor(Math.log10(raw));
  const step = [1, 2, 5, 10].map((m) => m * magnitude).find((s) => s >= raw) ?? raw;
  const out = [];
  for (let t = Math.ceil(lo / step) * step; t <= hi + step * 1e-9; t += step) {
    out.push(Number(t.toPrecision(12)));
  }
  return out;
}

function format(v: number): string {
  return Math.abs(v) >= 1e4 ? v.toExponential(1) : String(Number(v.toPrecision(4)));
}

function Legend({ names }: { names: string[] }) {
  return (
    <div className="legend">
      {names.map((name, k) => (
        <span key={name}>
          <i style={{ background: COLOURS[k % COLOURS.length] }} />
          {name}
        </span>
      ))}
    </div>
  );
}

type Axes = { title: string; x: string; y: string };

function Frame({
  axes,
  yTicks,
  y,
  children,
}: {
  axes: Axes;
  yTicks: number[];
  y: (v: number) => number;
  children: ReactNode;
}) {
  return (
    <svg viewBox={`0 0 ${WIDTH} ${HEIGHT}`} role="img" aria-label={axes.title}>
      {yTicks.map((t) => (
        <g key={t}>
          <line className="grid" x1={PAD.left} x2={WIDTH - PAD.right} y1={y(t)} y2={y(t)} />
          <text className="tick" x={PAD.left - 6} y={y(t) + 4} textAnchor="end">
            {format(t)}
          </text>
        </g>
      ))}
      {children}
      <text className="label" x={(PAD.left + WIDTH - PAD.right) / 2} y={HEIGHT - 4} textAnchor="middle">
        {axes.x}
      </text>
      <text className="label" x={12} y={HEIGHT / 2} transform={`rotate(-90 12 ${HEIGHT / 2})`} textAnchor="middle">
        {axes.y}
      </text>
    </svg>
  );
}

/** Labels for no more than about a dozen of `count` positions, evenly spaced. */
export function labelled(count: number): (i: number) => boolean {
  const every = Math.max(1, Math.ceil(count / 12));
  return (i) => i % every === 0;
}

/** Lines over a shared x axis, `xs[k]` the k-th point of every series. A value that isn't
 * finite, such as a month a plan has no target for, is left out of its line. */
export function LineChart({ series, xs, ...axes }: Axes & { series: Series[]; xs: number[] }) {
  const all = series.flatMap((s) => s.values).filter(Number.isFinite);
  const lo = Math.min(...all);
  const hi = Math.max(...all);
  const yTicks = ticks(lo, hi);
  const y = scale(Math.min(lo, yTicks[0] ?? lo), Math.max(hi, yTicks.at(-1) ?? hi), HEIGHT - PAD.bottom, PAD.top);
  const x = scale(xs[0] ?? 0, xs.at(-1) ?? 1, PAD.left, WIDTH - PAD.right);
  const shown = labelled(xs.length);
  return (
    <figure>
      <figcaption>{axes.title}</figcaption>
      <Frame axes={axes} yTicks={yTicks} y={y}>
        {xs.map((v, i) =>
          shown(i) ? (
            <text key={v} className="tick" x={x(v)} y={HEIGHT - PAD.bottom + 14} textAnchor="middle">
              {v}
            </text>
          ) : null,
        )}
        {series.map((s, k) => (
          <polyline
            key={s.name}
            className="line"
            stroke={COLOURS[k % COLOURS.length]}
            strokeDasharray={s.dashed ? "6 4" : undefined}
            points={s.values
              .flatMap((v, i) => (Number.isFinite(v) ? [`${x(xs[i] ?? i)},${y(v)}`] : []))
              .join(" ")}
          />
        ))}
      </Frame>
      <Legend names={series.map((s) => s.name)} />
    </figure>
  );
}

/** Bars per category, one bar for each series side by side. */
export function BarChart({ series, categories, ...axes }: Axes & { series: Series[]; categories: (string | number)[] }) {
  const all = series.flatMap((s) => s.values);
  const hi = Math.max(0, ...all);
  const lo = Math.min(0, ...all);
  const yTicks = ticks(lo, hi);
  const y = scale(lo, Math.max(hi, yTicks.at(-1) ?? hi), HEIGHT - PAD.bottom, PAD.top);
  const band = (WIDTH - PAD.left - PAD.right) / Math.max(categories.length, 1);
  const bar = (band * 0.8) / Math.max(series.length, 1);
  const shown = labelled(categories.length);
  return (
    <figure>
      <figcaption>{axes.title}</figcaption>
      <Frame axes={axes} yTicks={yTicks} y={y}>
        {categories.map((c, i) => (
          <g key={String(c)}>
            {series.map((s, k) => {
              const v = s.values[i] ?? 0;
              return (
                <rect
                  key={s.name}
                  className="bar"
                  fill={COLOURS[k % COLOURS.length]}
                  x={PAD.left + i * band + band * 0.1 + k * bar}
                  y={Math.min(y(v), y(0))}
                  width={bar}
                  height={Math.abs(y(0) - y(v))}
                />
              );
            })}
            {shown(i) && (
              <text className="tick" x={PAD.left + (i + 0.5) * band} y={HEIGHT - PAD.bottom + 14} textAnchor="middle">
                {c}
              </text>
            )}
          </g>
        ))}
      </Frame>
      <Legend names={series.map((s) => s.name)} />
    </figure>
  );
}
