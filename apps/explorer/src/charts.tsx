// Small SVG charts, so the explorer needs no charting library. Every chart speaks one language
// (the design's E4): the chapter's subject in the accent, with its spread across trajectories
// as a band behind it; everything else as context, in gray; the plan dashed, in the text
// colour; and each line labelled at its end rather than in a legend. The colours are the
// tokens in styles.css, so a chart follows the page's light or dark theme.

import { useLayoutEffect, useRef, useState, type ReactNode, type RefObject } from "react";

/** What a series is to its chart: the subject the chapter is about, context beside it, or the
 * plan it's measured against. */
export type Role = "subject" | "context" | "plan";

/** A spread across trajectories: its low and high edges, a value per point. */
export type Band = { low: number[]; high: number[] };

export type Series = {
  name: string;
  values: number[];
  /** Context unless said otherwise. */
  role?: Role;
  /** The subject's spread, drawn as a tint of the accent behind its line. */
  band?: Band;
  /** For context bars: 1 (lightest) to 3 (darkest), so parts side by side stay apart. */
  shade?: 1 | 2 | 3;
};

/** A series drawn as a spread's median, with its band from the 10th to the 90th percentile. */
export function spreadSeries(
  name: string,
  spread: { p10: number[]; p50: number[]; p90: number[] },
  role: Role = "subject",
): Series {
  return { name, role, values: spread.p50, band: { low: spread.p10, high: spread.p90 } };
}

/** The width a chart takes when its column can't be measured, as in tests: the text column. */
export const WIDTH = 620;
const HEIGHT = 240;
const PAD = { left: 52, top: 14, bottom: 34 };
/** The vertical room an end label takes, so labels closer than this are nudged apart. */
const LABEL_GAP = 14;
/** About how wide a character of an end label is, to size the gutter the labels sit in. */
const CHAR_WIDTH = 6.8;
const BAR_MAX = 24;
const BAR_RADIUS = 4;

/** Maps `[lo, hi]` onto `[from, to]`; a flat domain maps to the range's middle. */
export function scale(lo: number, hi: number, from: number, to: number): (v: number) => number {
  if (hi === lo) {
    return () => (from + to) / 2;
  }
  return (v) => from + ((v - lo) / (hi - lo)) * (to - from);
}

/** Round, evenly spaced ticks covering `[lo, hi]`: the first at or below `lo`, the last at or
 * above `hi`. */
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
  const first = Math.floor(lo / step + 1e-9) * step;
  const last = Math.ceil(hi / step - 1e-9) * step;
  for (let t = first; t <= last + step * 1e-9; t += step) {
    out.push(Number(t.toPrecision(12)));
  }
  return out;
}

/** A tick's value, short: compact from a thousand up (12k, 1.5M). */
export function format(v: number): string {
  const size = Math.abs(v);
  if (size >= 1e6) {
    return `${Number((v / 1e6).toPrecision(3))}M`;
  }
  if (size >= 1e3) {
    return `${Number((v / 1e3).toPrecision(3))}k`;
  }
  return String(Number(v.toPrecision(4)));
}

/** Labels for no more than about a dozen of `count` positions, evenly spaced. */
export function labelled(count: number): (i: number) => boolean {
  const every = Math.max(1, Math.ceil(count / 12));
  return (i) => i % every === 0;
}

/** Where labels wanted at `wanted` sit so that none is closer than `gap` to another, each as
 * near its own place as it can be and in the same order, within `[top, bottom]` while they fit.
 * Labels that would collide are gathered and centred on where they wanted to be. */
export function nudge(wanted: number[], gap: number, top: number, bottom: number): number[] {
  const order = wanted.map((_, i) => i).sort((a, b) => (wanted[a] ?? 0) - (wanted[b] ?? 0));
  // Runs of labels that touch, each starting at `start` and spaced by `gap`.
  const runs: { start: number; members: number[] }[] = [];
  for (const i of order) {
    runs.push({ start: wanted[i] ?? 0, members: [i] });
    for (;;) {
      const last = runs.at(-1);
      const before = runs.at(-2);
      if (last === undefined || before === undefined) {
        break;
      }
      if (before.start + before.members.length * gap <= last.start) {
        break;
      }
      before.members.push(...last.members);
      runs.pop();
      const centre = before.members.reduce((sum, m) => sum + (wanted[m] ?? 0), 0) / before.members.length;
      before.start = centre - ((before.members.length - 1) * gap) / 2;
    }
  }
  const placed = new Array<number>(wanted.length);
  const sorted = runs.flatMap((r) => r.members.map((m, k) => ({ m, at: r.start + k * gap })));
  // Keep them inside the bounds: down from the top, then up from the bottom.
  for (let k = 0; k < sorted.length; k++) {
    const floor = k === 0 ? top : (sorted[k - 1]?.at ?? top) + gap;
    const entry = sorted[k];
    if (entry !== undefined) {
      entry.at = Math.max(entry.at, floor);
    }
  }
  for (let k = sorted.length - 1; k >= 0; k--) {
    const ceiling = k === sorted.length - 1 ? bottom : (sorted[k + 1]?.at ?? bottom) - gap;
    const entry = sorted[k];
    if (entry !== undefined) {
      entry.at = Math.min(entry.at, ceiling);
    }
  }
  for (const { m, at } of sorted) {
    placed[m] = at;
  }
  return placed;
}

/** The width of the element `ref` points at, following it as it resizes, or `WIDTH` where it
 * can't be measured. */
export function useWidth(ref: RefObject<HTMLElement | null>): number {
  const [width, setWidth] = useState(WIDTH);
  useLayoutEffect(() => {
    const element = ref.current;
    if (element === null || typeof ResizeObserver === "undefined") {
      return;
    }
    const measure = () => {
      const measured = element.getBoundingClientRect().width;
      if (measured > 0) {
        setWidth(Math.round(measured));
      }
    };
    measure();
    const observer = new ResizeObserver(measure);
    observer.observe(element);
    return () => observer.disconnect();
  }, [ref]);
  return width;
}

type Axes = { title: string; x: string; y: string };

/** A chart's frame: its title, the drawing at the column's width, and what follows it. */
function Figure({
  title,
  children,
  after,
}: {
  title: string;
  children: (width: number) => ReactNode;
  after?: ReactNode;
}) {
  const ref = useRef<HTMLDivElement>(null);
  const width = useWidth(ref);
  return (
    <figure className="chart">
      <figcaption>{title}</figcaption>
      <div className="plot" ref={ref}>
        <svg viewBox={`0 0 ${width} ${HEIGHT}`} width={width} height={HEIGHT} role="img" aria-label={title}>
          {children(width)}
        </svg>
      </div>
      {after}
    </figure>
  );
}

/** The y axis's gridlines and ticks, and both axes' titles. */
function Axis({ axes, yTicks, y, right }: { axes: Axes; yTicks: number[]; y: (v: number) => number; right: number }) {
  return (
    <g>
      {yTicks.map((t) => (
        <g key={t}>
          <line className="grid" x1={PAD.left} x2={right} y1={y(t)} y2={y(t)} />
          <text className="tick" x={PAD.left - 6} y={y(t) + 4} textAnchor="end">
            {format(t)}
          </text>
        </g>
      ))}
      <text className="axis-title" x={(PAD.left + right) / 2} y={HEIGHT - 4} textAnchor="middle">
        {axes.x}
      </text>
      <text className="axis-title" x={12} y={HEIGHT / 2} transform={`rotate(-90 12 ${HEIGHT / 2})`} textAnchor="middle">
        {axes.y}
      </text>
    </g>
  );
}

/** The gutter end labels need: wide enough for the longest, but never most of the chart. */
function gutter(names: string[], width: number): number {
  const longest = Math.max(0, ...names.map((n) => n.length));
  return Math.min(width * 0.32, 14 + longest * CHAR_WIDTH);
}

function roleOf(s: Series): Role {
  return s.role ?? "context";
}

/** The subject drawn last, so it sits on top; the plan before it; context first. */
const DRAW_ORDER: Record<Role, number> = { context: 0, plan: 1, subject: 2 };

function lastFinite(values: number[]): number | undefined {
  for (let i = values.length - 1; i >= 0; i--) {
    const v = values[i];
    if (v !== undefined && Number.isFinite(v)) {
      return v;
    }
  }
  return undefined;
}

function points(values: number[], x: (i: number) => number, y: (v: number) => number): string {
  return values.flatMap((v, i) => (Number.isFinite(v) ? [`${x(i)},${y(v)}`] : [])).join(" ");
}

/** A band's outline: along its high edge, then back along its low edge. */
function bandPath(band: Band, x: (i: number) => number, y: (v: number) => number): string {
  const along = band.high.flatMap((v, i) => (Number.isFinite(v) && Number.isFinite(band.low[i] ?? NaN) ? [i] : []));
  if (along.length < 2) {
    return "";
  }
  const top = along.map((i) => `${x(i)},${y(band.high[i] ?? 0)}`);
  const bottom = [...along].reverse().map((i) => `${x(i)},${y(band.low[i] ?? 0)}`);
  return `M${top.join("L")}L${bottom.join("L")}Z`;
}

/** Lines over a shared x axis, `xs[k]` the k-th point of every series, each labelled at its end.
 * A value that isn't finite, such as a month a plan has no target for, is left out of its line. */
export function LineChart({
  series,
  xs,
  after,
  ...axes
}: Axes & { series: Series[]; xs: number[]; after?: ReactNode }) {
  const all = series
    .flatMap((s) => [...s.values, ...(s.band?.low ?? []), ...(s.band?.high ?? [])])
    .filter(Number.isFinite);
  const lo = Math.min(...all);
  const hi = Math.max(...all);
  const yTicks = ticks(lo, hi);
  const shown = labelled(xs.length);
  const ordered = [...series].sort((a, b) => DRAW_ORDER[roleOf(a)] - DRAW_ORDER[roleOf(b)]);
  return (
    <Figure title={axes.title} after={after}>
      {(width) => {
        const right = width - gutter(series.map((s) => s.name), width);
        const y = scale(Math.min(lo, yTicks[0] ?? lo), Math.max(hi, yTicks.at(-1) ?? hi), HEIGHT - PAD.bottom, PAD.top);
        const xv = scale(xs[0] ?? 0, xs.at(-1) ?? 1, PAD.left, right);
        const x = (i: number) => xv(xs[i] ?? i);
        const ends = ordered.flatMap((s) => {
          const last = lastFinite(s.values);
          return last === undefined ? [] : [{ s, wanted: y(last) }];
        });
        const placed = nudge(
          ends.map((e) => e.wanted),
          LABEL_GAP,
          PAD.top,
          HEIGHT - PAD.bottom,
        );
        return (
          <>
            <Axis axes={axes} yTicks={yTicks} y={y} right={right} />
            {xs.map((v, i) =>
              shown(i) ? (
                <text key={v} className="tick" x={x(i)} y={HEIGHT - PAD.bottom + 16} textAnchor="middle">
                  {v}
                </text>
              ) : null,
            )}
            {ordered.map((s) =>
              s.band ? <path key={`${s.name}-band`} className={`band ${roleOf(s)}`} d={bandPath(s.band, x, y)} /> : null,
            )}
            {ordered.map((s) => (
              <polyline key={s.name} className={`line ${roleOf(s)}`} points={points(s.values, x, y)} />
            ))}
            {ends.map((e, k) => {
              const at = placed[k] ?? e.wanted;
              return (
                <g key={`${e.s.name}-label`}>
                  {Math.abs(at - e.wanted) > 3 && (
                    <line className="leader" x1={right + 2} y1={e.wanted} x2={right + 8} y2={at} />
                  )}
                  <text className={`end-label ${roleOf(e.s)}`} x={right + 10} y={at + 4}>
                    {e.s.name}
                  </text>
                </g>
              );
            })}
          </>
        );
      }}
    </Figure>
  );
}

/** A bar from the baseline `y0` to `y1`, `width` wide, its end away from the baseline rounded. */
export function barPath(x: number, width: number, y0: number, y1: number): string {
  const height = Math.abs(y1 - y0);
  if (height === 0 || width <= 0) {
    return "";
  }
  const r = Math.min(BAR_RADIUS, width / 2, height);
  const up = y1 < y0; // in SVG, a positive value rises
  const end = up ? y0 - height : y0 + height;
  const turn = up ? r : -r;
  return [
    `M${x},${y0}`,
    `V${end + turn}`,
    `Q${x},${end} ${x + r},${end}`,
    `H${x + width - r}`,
    `Q${x + width},${end} ${x + width},${end + turn}`,
    `V${y0}`,
    "Z",
  ].join("");
}

/** Bars per category, one bar for each series side by side. Each series is labelled over its
 * tallest bar, and labels that would collide are lifted apart. */
export function BarChart({
  series,
  categories,
  after,
  ...axes
}: Axes & { series: Series[]; categories: (string | number)[]; after?: ReactNode }) {
  const all = series.flatMap((s) => s.values);
  const hi = Math.max(0, ...all);
  const lo = Math.min(0, ...all);
  const yTicks = ticks(lo, hi);
  const shown = labelled(categories.length);
  return (
    <Figure title={axes.title} after={after}>
      {(width) => {
        const right = width - 12;
        // Room above the tallest bar for the series' labels.
        const y = scale(lo, Math.max(hi, yTicks.at(-1) ?? hi), HEIGHT - PAD.bottom, PAD.top + LABEL_GAP * series.length);
        const slot = (right - PAD.left) / Math.max(categories.length, 1);
        const bar = Math.min(BAR_MAX, (slot * 0.8) / Math.max(series.length, 1));
        const group = bar * series.length;
        const left = (i: number, k: number) => PAD.left + i * slot + (slot - group) / 2 + k * bar;
        // Each series' label over its tallest bar, lifted over any label it would sit on.
        const labels: { x: number; y: number; half: number }[] = [];
        series.forEach((s, k) => {
          const i = s.values.reduce((best, v, j) => (Math.abs(v) > Math.abs(s.values[best] ?? 0) ? j : best), 0);
          const half = (s.name.length * CHAR_WIDTH) / 2;
          const centre = Math.min(Math.max(left(i, k) + bar / 2, PAD.left + half), right - half);
          let at = Math.min(y(s.values[i] ?? 0), y(0)) - 6;
          for (let moved = true; moved; ) {
            moved = false;
            for (const other of labels) {
              if (Math.abs(other.x - centre) < other.half + half + 4 && Math.abs(other.y - at) < LABEL_GAP) {
                at = other.y - LABEL_GAP;
                moved = true;
              }
            }
          }
          labels.push({ x: centre, y: at, half });
        });
        return (
          <>
            <Axis axes={axes} yTicks={yTicks} y={y} right={right} />
            {categories.map((c, i) => (
              <g key={String(c)}>
                {series.map((s, k) => (
                  <path
                    key={s.name}
                    className={`bar ${roleOf(s)} shade-${s.shade ?? 3}`}
                    d={barPath(left(i, k) + 1, bar - 2, y(0), y(s.values[i] ?? 0))}
                  />
                ))}
                {shown(i) && (
                  <text className="tick" x={PAD.left + (i + 0.5) * slot} y={HEIGHT - PAD.bottom + 16} textAnchor="middle">
                    {c}
                  </text>
                )}
              </g>
            ))}
            {series.length > 1 &&
              series.map((s, k) => (
                <text
                  key={`${s.name}-label`}
                  className={`end-label ${roleOf(s)}`}
                  x={labels[k]?.x}
                  y={labels[k]?.y}
                  textAnchor="middle"
                >
                  {s.name}
                </text>
              ))}
          </>
        );
      }}
    </Figure>
  );
}
