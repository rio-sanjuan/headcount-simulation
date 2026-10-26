// The explorer's story (the design's E1): four chapters in the engine's order of ideas, each one
// demo with the line that hands on to the next.

import type { ComponentType } from "react";

import { TRAJECTORIES } from "./api";
import type { Preset } from "./Control";
import { LifecycleDemo } from "./demos/LifecycleDemo";
import { SamplingDemo } from "./demos/SamplingDemo";
import { SbcDemo } from "./demos/SbcDemo";
import { ValvesDemo } from "./demos/ValvesDemo";

/** A line of the story's text. A draft is a placeholder that waits for the maintainer's review
 * (#660), and the page marks it so. */
export type Line = { text: string; draft?: boolean };

/** What a chapter gives its demo: its name, to label the presets, and the presets. */
export type DemoProps = { name: string; presets: readonly Preset[] };

export type Chapter = {
  /** The chapter's anchor, so a link can open the page at it, and the demo it runs. */
  key: string;
  title: string;
  Demo: ComponentType<DemoProps>;
  /** Three named scenarios that make the chapter's point; the middle one shows at first. Their
   * labels are the design's proposed ones (E3), for the maintainer's review in #660. */
  presets: readonly Preset[];
  /** The line that hands on to the next chapter. */
  handOn: Line;
};

/** Six months, a year, and three years of a run. */
function horizons(): Preset[] {
  return [
    { label: "Six months", query: { months: 6, trajectories: TRAJECTORIES } },
    { label: "A year", query: { months: 12, trajectories: TRAJECTORIES } },
    { label: "Three years", query: { months: 36, trajectories: TRAJECTORIES } },
  ];
}

export const CHAPTERS: readonly Chapter[] = [
  {
    key: "sampling",
    title: "Quota sampling",
    Demo: SamplingDemo,
    presets: [
      { label: "Even rows", query: { rows: 200, quota: 20, spread: 0, trajectories: 500 } },
      { label: "Uneven rows", query: { rows: 200, quota: 20, spread: 1, trajectories: 500 } },
      { label: "Very uneven rows", query: { rows: 200, quota: 20, spread: 2, trajectories: 500 } },
    ],
    handOn: {
      text: "Quota sampling lets a guided month hit its counts. Next, how a run is steered toward a plan.",
      draft: true,
    },
  },
  {
    key: "valves",
    title: "The three valves",
    Demo: ValvesDemo,
    presets: [
      { label: "No growth", query: { growth: 0, trajectories: TRAJECTORIES } },
      { label: "A modest plan, 5%", query: { growth: 0.05, trajectories: TRAJECTORIES } },
      { label: "An aggressive plan, 20%", query: { growth: 0.2, trajectories: TRAJECTORIES } },
    ],
    handOn: {
      text: "The supply valve works through the requisitions growth opens. Next, how a requisition opens, waits, and closes.",
      draft: true,
    },
  },
  {
    key: "lifecycle",
    title: "The requisition lifecycle",
    Demo: LifecycleDemo,
    presets: horizons(),
    handOn: {
      text: "A filled opening is a hire, and a hire brings a grant. Next, how grants become expense.",
      draft: true,
    },
  },
  {
    key: "sbc",
    title: "The SBC layer",
    Demo: SbcDemo,
    presets: horizons(),
    handOn: {
      text: "That's the engine's story. The studies and their scorecards compare runs in depth.",
      draft: true,
    },
  },
];

/** What the page asks for as it loads: the preset each chapter shows first, so every chapter
 * fills in as soon as it can, then the rest, so a click on any preset answers at once. */
export function presetsToFetch(chapters: readonly Chapter[]): { demo: string; query: Preset["query"] }[] {
  const middle = (c: Chapter) => Math.floor(c.presets.length / 2);
  const first = chapters.flatMap((c) => c.presets.filter((_, i) => i === middle(c)).map((p) => ({ demo: c.key, query: p.query })));
  const rest = chapters.flatMap((c) => c.presets.filter((_, i) => i !== middle(c)).map((p) => ({ demo: c.key, query: p.query })));
  return [...first, ...rest];
}

/** A paragraph of the story, marked when it's a draft. */
export function Prose({ line, className }: { line: Line; className?: string }) {
  return (
    <p className={className}>
      {line.text}
      {line.draft && (
        <>
          {" "}
          <span className="draft" title="Placeholder text, for the maintainer's review in #660">
            draft
          </span>
        </>
      )}
    </p>
  );
}
