// Small SVG charts, so the explorer needs no charting library. Every chart speaks one language
// (the design's E4): the chapter's subject in the accent, with its spread across trajectories
// as a band behind it; everything else as context, in gray; the plan dashed, in the text
// colour; and each line labelled at its end rather than in a legend. The colours are the
// tokens in styles.css, so a chart follows the page's light or dark theme.

import { useId, useLayoutEffect, useRef, useState, type ReactNode, type RefObject } from "react";

import { Provenance } from "./provenance";

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

/** The room an x label needs, so a narrow chart labels fewer positions. */
const X_LABEL_ROOM = 36;

/** Labels for no more than `most` of `count` positions, evenly spaced. */
export function labelled(count: number, most = 12): (i: number) => boolean {
  const every = Math.max(1, Math.ceil(count / Math.max(1, most)));
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

/** What every chart says beside its drawing: its title, its axes' titles, and a line of text
 * that sums up what it shows, for anyone who can't see the drawing. */
type Axes = {
  title: string;
  x: string;
  y: string;
  summary?: ReactNode;
  /** Whether a change to the summary is announced, as when a control changes what's in focus. */
  announce?: boolean;
  after?: ReactNode;
};

/** A chart's frame: its title, the drawing at the column's width, its summary, and what follows
 * it. */
function Figure({
  title,
  summary,
  summaryAbove = false,
  announce,
  height = HEIGHT,
  children,
  after,
}: {
  title: string;
  summary?: ReactNode;
  /** Whether the summary leads the drawing rather than follows it. */
  summaryAbove?: boolean;
  announce?: boolean;
  /** The drawing's height, or how it follows the width. */
  height?: number | ((width: number) => number);
  children: (width: number) => ReactNode;
  after?: ReactNode;
}) {
  const ref = useRef<HTMLDivElement>(null);
  const width = useWidth(ref);
  const id = useId();
  const tall = typeof height === "number" ? height : height(width);
  const said = summary !== undefined && (
    <p className={summaryAbove ? "summary above" : "summary"} id={id} aria-live={announce ? "polite" : undefined}>
      {summary}
    </p>
  );
  return (
    <figure className="chart">
      <figcaption>{title}</figcaption>
      {summaryAbove && said}
      <div className="plot" ref={ref}>
        <svg
          viewBox={`0 0 ${width} ${tall}`}
          width={width}
          height={tall}
          role="img"
          aria-label={title}
          aria-describedby={summary === undefined ? undefined : id}
        >
          {children(width)}
        </svg>
      </div>
      {!summaryAbove && said}
      {after}
      <Provenance />
    </figure>
  );
}

/** The y axis's gridlines and ticks, at the plot's left or right, and both axes' titles. */
function Axis({
  axes,
  yTicks,
  y,
  left = PAD.left,
  right,
  width,
  side = "left",
}: {
  axes: Axes;
  yTicks: number[];
  y: (v: number) => number;
  left?: number;
  right: number;
  width?: number;
  side?: "left" | "right";
}) {
  const atLeft = side === "left";
  const title = atLeft ? 12 : (width ?? right + PAD.left) - 6;
  return (
    <g>
      {yTicks.map((t) => (
        <g key={t}>
          <line className="grid" x1={left} x2={right} y1={y(t)} y2={y(t)} />
          <text className="tick" x={atLeft ? left - 6 : right + 6} y={y(t) + 4} textAnchor={atLeft ? "end" : "start"}>
            {format(t)}
          </text>
        </g>
      ))}
      <text className="axis-title" x={(left + right) / 2} y={HEIGHT - 4} textAnchor="middle">
        {axes.x}
      </text>
      <text className="axis-title" x={title} y={HEIGHT / 2} transform={`rotate(-90 ${title} ${HEIGHT / 2})`} textAnchor="middle">
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

/** A series' last finite value, or its first. */
function endOf(values: number[], at: "start" | "end"): number | undefined {
  const finite = values.filter(Number.isFinite);
  return at === "end" ? finite.at(-1) : finite[0];
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

/** Lines over a shared x axis, `xs[k]` the k-th point of every series, each labelled at its end,
 * or at its start where the lines differ most there (`labelAt`), with the y axis then at the right.
 * A value that isn't finite, such as a month a plan has no target for, is left out of its line. */
export function LineChart({
  series,
  xs,
  labelAt = "end",
  ...axes
}: Axes & { series: Series[]; xs: number[]; labelAt?: "start" | "end" }) {
  const all = series
    .flatMap((s) => [...s.values, ...(s.band?.low ?? []), ...(s.band?.high ?? [])])
    .filter(Number.isFinite);
  const lo = Math.min(...all);
  const hi = Math.max(...all);
  const yTicks = ticks(lo, hi);
  const ordered = [...series].sort((a, b) => DRAW_ORDER[roleOf(a)] - DRAW_ORDER[roleOf(b)]);
  return (
    <Figure title={axes.title} summary={axes.summary} announce={axes.announce} after={axes.after}>
      {(width) => {
        const room = gutter(series.map((s) => s.name), width);
        const atEnd = labelAt === "end";
        const left = atEnd ? PAD.left : room;
        const right = atEnd ? width - room : width - PAD.left;
        const y = scale(Math.min(lo, yTicks[0] ?? lo), Math.max(hi, yTicks.at(-1) ?? hi), HEIGHT - PAD.bottom, PAD.top);
        const xv = scale(xs[0] ?? 0, xs.at(-1) ?? 1, left, right);
        const x = (i: number) => xv(xs[i] ?? i);
        const shown = labelled(xs.length, Math.min(12, (right - left) / X_LABEL_ROOM));
        // Labels sit in the gutter, beside the plot's edge, with a leader if nudged off their line.
        const edge = atEnd ? right : left;
        const toward = atEnd ? 1 : -1;
        const ends = ordered.flatMap((s) => {
          const end = endOf(s.values, labelAt);
          return end === undefined ? [] : [{ s, wanted: y(end) }];
        });
        const placed = nudge(
          ends.map((e) => e.wanted),
          LABEL_GAP,
          PAD.top,
          HEIGHT - PAD.bottom,
        );
        return (
          <>
            <Axis axes={axes} yTicks={yTicks} y={y} left={left} right={right} width={width} side={atEnd ? "left" : "right"} />
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
                    <line className="leader" x1={edge + 2 * toward} y1={e.wanted} x2={edge + 8 * toward} y2={at} />
                  )}
                  <text
                    className={`end-label ${roleOf(e.s)}`}
                    x={edge + 10 * toward}
                    y={at + 4}
                    textAnchor={atEnd ? "start" : "end"}
                  >
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

/** Places labels centred over their marks, each at its `y` unless it would sit on a label placed
 * before it, then lifted over that one, and each kept between `left` and `right`. */
export function liftApart(
  wanted: readonly { x: number; y: number; text: string }[],
  left: number,
  right: number,
): { x: number; y: number }[] {
  const placed: { x: number; y: number; half: number }[] = [];
  for (const w of wanted) {
    const half = (w.text.length * CHAR_WIDTH) / 2;
    const x = Math.min(Math.max(w.x, left + half), right - half);
    let y = w.y;
    for (let moved = true; moved; ) {
      moved = false;
      for (const other of placed) {
        if (Math.abs(other.x - x) < other.half + half + 4 && Math.abs(other.y - y) < LABEL_GAP) {
          y = other.y - LABEL_GAP;
          moved = true;
        }
      }
    }
    placed.push({ x, y, half });
  }
  return placed;
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
export function BarChart({ series, categories, ...axes }: Axes & { series: Series[]; categories: (string | number)[] }) {
  const all = series.flatMap((s) => s.values);
  const hi = Math.max(0, ...all);
  const lo = Math.min(0, ...all);
  const yTicks = ticks(lo, hi);
  return (
    <Figure title={axes.title} summary={axes.summary} announce={axes.announce} after={axes.after}>
      {(width) => {
        const right = width - 12;
        // Room above the tallest bar for the series' labels.
        const y = scale(lo, Math.max(hi, yTicks.at(-1) ?? hi), HEIGHT - PAD.bottom, PAD.top + LABEL_GAP * series.length);
        const slot = (right - PAD.left) / Math.max(categories.length, 1);
        const shown = labelled(categories.length, Math.min(12, (right - PAD.left) / X_LABEL_ROOM));
        const bar = Math.min(BAR_MAX, (slot * 0.8) / Math.max(series.length, 1));
        const group = bar * series.length;
        const left = (i: number, k: number) => PAD.left + i * slot + (slot - group) / 2 + k * bar;
        // Each series' label over its tallest bar.
        const labels = liftApart(
          series.map((s, k) => {
            const i = s.values.reduce((best, v, j) => (Math.abs(v) > Math.abs(s.values[best] ?? 0) ? j : best), 0);
            return { x: left(i, k) + bar / 2, y: Math.min(y(s.values[i] ?? 0), y(0)) - 6, text: s.name };
          }),
          PAD.left,
          right,
        );
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

/** How many trajectories drew each count: the context's bars in gray, and the subject's in front,
 * narrower, in the accent. The height is capped at `cap` trajectories, or the context's tallest
 * bar if that's taller, so the context's spread reads clearly. A subject bar taller than that runs
 * off the top with a break mark, labelled with its count. */
export function Histogram({
  bins,
  subject,
  context,
  cap,
  ...axes
}: Axes & {
  bins: number[];
  subject: { name: string; counts: number[] };
  context: { name: string; counts: number[] };
  cap: number;
}) {
  const total = subject.counts.reduce((a, b) => a + b, 0);
  const yTicks = ticks(0, Math.max(cap, ...context.counts));
  const ceiling = yTicks.at(-1) ?? cap;
  return (
    <Figure title={axes.title} summary={axes.summary} announce={axes.announce} after={axes.after}>
      {(width) => {
        const right = width - 12;
        const top = PAD.top + LABEL_GAP * 2; // room above the plot for the labels
        const y = scale(0, ceiling, HEIGHT - PAD.bottom, top);
        const slot = (right - PAD.left) / Math.max(bins.length, 1);
        const bar = Math.min(BAR_MAX, slot * 0.8);
        const narrow = Math.max(3, bar / 2);
        const centre = (i: number) => PAD.left + (i + 0.5) * slot;
        const shown = labelled(bins.length, Math.min(12, (right - PAD.left) / X_LABEL_ROOM));
        const tallest = context.counts.reduce((best, v, i) => (v > (context.counts[best] ?? 0) ? i : best), 0);
        const peaks = subject.counts.flatMap((v, i) => (v > 0 ? [i] : []));
        const end = (v: number) => (v > ceiling ? top - 6 : y(v));
        // The centre of a label `half` wide over the bar at `at`, running away from the subject's
        // bar at `subject`, so it doesn't cross it.
        const awayFrom = (subject: number, at: number, half: number) =>
          at >= subject ? Math.max(at - bar / 2, subject + narrow / 2 + 4) + half : Math.min(at + bar / 2, subject - narrow / 2 - 4) - half;
        const wanted = [
          ...peaks.map((i) => {
            const v = subject.counts[i] ?? 0;
            const text = `${subject.name}: ${v === total ? `all ${total}` : v}`;
            return { x: centre(i), y: end(v) - 6, text, role: "subject" };
          }),
          {
            x: awayFrom(centre(peaks[0] ?? tallest), centre(tallest), (context.name.length * CHAR_WIDTH) / 2),
            y: y(context.counts[tallest] ?? 0) - 6,
            text: context.name,
            role: "context",
          },
        ];
        const labels = liftApart(wanted, PAD.left, right);
        return (
          <>
            <Axis axes={axes} yTicks={yTicks} y={y} right={right} />
            {bins.map((b, i) => (
              <g key={b}>
                <path className="bar context shade-3" d={barPath(centre(i) - bar / 2 + 1, bar - 2, y(0), y(context.counts[i] ?? 0))} />
                {shown(i) && (
                  <text className="tick" x={centre(i)} y={HEIGHT - PAD.bottom + 16} textAnchor="middle">
                    {b}
                  </text>
                )}
              </g>
            ))}
            {peaks.map((i) => {
              const v = subject.counts[i] ?? 0;
              const x0 = centre(i) - narrow / 2;
              const broken = v > ceiling;
              return (
                <g key={`${subject.name}-${i}`}>
                  <path className="bar subject" d={barPath(x0, narrow, y(0), end(v))} />
                  {broken && (
                    // A break across the bar, under the plot's top: it runs further than drawn.
                    <path
                      className="break"
                      d={`M${x0 - 3},${top + 9}L${x0 + narrow + 3},${top + 3}M${x0 - 3},${top + 15}L${x0 + narrow + 3},${top + 9}`}
                    />
                  )}
                </g>
              );
            })}
            {wanted.map((w, k) => (
              <text key={w.text} className={`end-label ${w.role}`} x={labels[k]?.x} y={labels[k]?.y} textAnchor="middle">
                {w.text}
              </text>
            ))}
          </>
        );
      }}
    </Figure>
  );
}

/** A flow into or out of a stock, and its total over the run. */
export type Flow = { name: string; total: number };

/** The vertical gap between ribbons, which holds each one's label: one line, or two in a narrow
 * chart, the total under the name, so a long name doesn't run into the stock. */
const RIBBON_GAP = 18;
const RIBBON_GAP_NARROW = 32;
const NARROW = 480;
/** The stock's box, between the ribbons in and the ribbons out. */
const STOCK_WIDTH = 16;

/** One flow's ribbon: a band `thickness` thick, from `(x0, y0)` to `(x1, y1)`, its edges curved. */
export function ribbonPath(x0: number, y0: number, x1: number, y1: number, thickness: number): string {
  const mid = (x0 + x1) / 2;
  return [
    `M${x0},${y0}`,
    `C${mid},${y0} ${mid},${y1} ${x1},${y1}`,
    `V${y1 + thickness}`,
    `C${mid},${y1 + thickness} ${mid},${y0 + thickness} ${x0},${y0 + thickness}`,
    "Z",
  ].join("");
}

/** Where each of `totals` sits in a stack `scale` high per unit, from `top`, `gap` apart. */
function stacked(totals: number[], scale: number, top: number, gap: number): number[] {
  const out: number[] = [];
  let at = top;
  for (const t of totals) {
    out.push(at);
    at += t * scale + gap;
  }
  return out;
}

/** Flows through a stock: what flowed in at the left, and out at the right, each a ribbon as wide
 * as its total, and the stock as the accent box between, labelled with its count at the start
 * and at the end. The scale is set by the larger side, so the diagram keeps its size, and each
 * ribbon's total is printed beside it. Its height grows with the ribbons it draws. */
export function FlowDiagram({
  title,
  inflows,
  outflows,
  stock,
  summary,
  after,
}: {
  title: string;
  inflows: Flow[];
  outflows: Flow[];
  stock: { name: string; start: number; end: number };
  summary?: ReactNode;
  after?: ReactNode;
}) {
  const sum = (flows: Flow[]) => flows.reduce((a, f) => a + f.total, 0);
  const most = Math.max(inflows.length, outflows.length, 1);
  const top = 44; // room for the stock's labels
  const band = 120 + 24 * Math.max(0, most - 2); // the stock's height, which the larger side fills
  const gapFor = (width: number) => (width < NARROW ? RIBBON_GAP_NARROW : RIBBON_GAP);
  const height = (width: number) => top + gapFor(width) + band + (gapFor(width) * (most - 1)) / 2 + 12;
  const scale = band / Math.max(sum(inflows), sum(outflows), 1e-9);
  const total = (v: number) => v.toFixed(1);
  return (
    <Figure title={title} summary={summary} summaryAbove after={after} height={height}>
      {(width) => {
        const gap = gapFor(width);
        const twoLines = width < NARROW;
        const centre = width / 2;
        const boxLeft = centre - STOCK_WIDTH / 2;
        const boxRight = centre + STOCK_WIDTH / 2;
        // At the box, each side's ribbons meet it in one stack; at the edges, each starts on its
        // own, with its label in the gap above it. Both stacks are centred on the box.
        const boxTop = top + gap;
        const centred = (flows: Flow[], gap: number) => {
          const tall = sum(flows) * scale + gap * (flows.length - 1);
          return stacked(
            flows.map((f) => f.total),
            scale,
            boxTop + (band - tall) / 2,
            gap,
          );
        };
        const side = (flows: Flow[], into: boolean) => {
          const outer = centred(flows, gap);
          const inner = centred(flows, 0);
          return flows.map((f, k) => {
            const y0 = outer[k] ?? 0;
            const y1 = inner[k] ?? 0;
            const thickness = f.total * scale;
            const d = into ? ribbonPath(0, y0, boxLeft, y1, thickness) : ribbonPath(boxRight, y1, width, y0, thickness);
            return (
              <g key={f.name}>
                <path className="ribbon" d={d} />
                {twoLines ? (
                  <text className="ribbon-label" x={into ? 0 : width} y={y0 - 18} textAnchor={into ? "start" : "end"}>
                    {f.name}
                    <tspan className="ribbon-total" x={into ? 0 : width} dy={13}>
                      {total(f.total)}
                    </tspan>
                  </text>
                ) : (
                  <text className="ribbon-label" x={into ? 0 : width} y={y0 - 5} textAnchor={into ? "start" : "end"}>
                    {f.name} <tspan className="ribbon-total">{total(f.total)}</tspan>
                  </text>
                )}
              </g>
            );
          });
        };
        return (
          <>
            {side(inflows, true)}
            {side(outflows, false)}
            <rect className="stock" x={boxLeft} y={boxTop} width={STOCK_WIDTH} height={band} rx={3} />
            <text className="end-label" x={centre} y={top - 22} textAnchor="middle">
              {stock.name}
            </text>
            <text className="end-label subject" x={centre} y={top - 6} textAnchor="middle">
              {total(stock.start)} → {total(stock.end)}
            </text>
          </>
        );
      }}
    </Figure>
  );
}

/** A part of a whole, stacked in a bar per month: its values, and its shade of gray. */
export type Part = { name: string; values: number[]; shade: 1 | 2 | 3 };

/** Where each part's segment lies in each month's bar: positive parts stacked up from zero, and
 * negative parts down from it, each in the order given. */
export function stack(parts: Part[], months: number): { from: number; to: number }[][] {
  const up = new Array<number>(months).fill(0);
  const down = new Array<number>(months).fill(0);
  return parts.map((p) =>
    Array.from({ length: months }, (_, i) => {
      const v = p.values[i] ?? 0;
      const base = v >= 0 ? (up[i] ?? 0) : (down[i] ?? 0);
      if (v >= 0) {
        up[i] = base + v;
      } else {
        down[i] = base + v;
      }
      return { from: base, to: base + v };
    }),
  );
}

/** A bar per month split into its parts, in grays, positive parts above the zero line and
 * negative ones below, with the total they add up to as the subject's line and band. Each part
 * and the total are labelled at the right. */
export function StackedBars({
  parts,
  total,
  xs,
  ...axes
}: Axes & { parts: Part[]; total: Series; xs: number[] }) {
  const months = xs.length;
  const stacks = stack(parts, months);
  const all = [
    0,
    ...stacks.flatMap((segments) => segments.flatMap((s) => [s.from, s.to])),
    ...total.values,
    ...(total.band?.low ?? []),
    ...(total.band?.high ?? []),
  ].filter(Number.isFinite);
  const yTicks = ticks(Math.min(...all), Math.max(...all));
  return (
    <Figure title={axes.title} summary={axes.summary} after={axes.after}>
      {(width) => {
        const right = width - gutter([total.name, ...parts.map((p) => p.name)], width);
        const y = scale(yTicks[0] ?? 0, yTicks.at(-1) ?? 1, HEIGHT - PAD.bottom, PAD.top);
        const slot = (right - PAD.left) / Math.max(months, 1);
        const bar = Math.min(BAR_MAX, slot * 0.7);
        const x = (i: number) => PAD.left + (i + 0.5) * slot;
        const shown = labelled(months, Math.min(12, (right - PAD.left) / X_LABEL_ROOM));
        // Each part's label beside its segment in the last month, and the total's beside its end.
        const last = months - 1;
        const wanted = [
          ...parts.map((p, k) => {
            const s = stacks[k]?.[last] ?? { from: 0, to: 0 };
            return { name: p.name, role: "context", at: y((s.from + s.to) / 2) };
          }),
          { name: total.name, role: "subject", at: y(total.values[last] ?? 0) },
        ];
        const placed = nudge(
          wanted.map((w) => w.at),
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
            {parts.map((p, k) => (
              <g key={p.name}>
                {(stacks[k] ?? []).map((s, i) => {
                  // A 2 px gap of the surface between touching segments: each gives up 1 px.
                  const top = Math.min(y(s.from), y(s.to)) + 1;
                  const height = Math.abs(y(s.to) - y(s.from)) - 2;
                  return height > 0 ? (
                    <rect key={i} className={`bar context shade-${p.shade}`} x={x(i) - bar / 2} y={top} width={bar} height={height} />
                  ) : null;
                })}
              </g>
            ))}
            <line className="zero" x1={PAD.left} x2={right} y1={y(0)} y2={y(0)} />
            {total.band && <path className="band subject" d={bandPath(total.band, x, y)} />}
            <polyline className="line subject" points={points(total.values, x, y)} />
            {wanted.map((w, k) => {
              const at = placed[k] ?? w.at;
              return (
                <g key={`${w.name}-label`}>
                  {Math.abs(at - w.at) > 3 && <line className="leader" x1={right + 2} y1={w.at} x2={right + 8} y2={at} />}
                  <text className={`end-label ${w.role}`} x={right + 10} y={at + 4}>
                    {w.name}
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
