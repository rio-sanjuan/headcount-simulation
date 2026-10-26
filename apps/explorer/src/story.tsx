// The explorer's story (the design's E1): four chapters in the engine's order of ideas, each one
// demo with the line that hands on to the next.

import type { ComponentType } from "react";

import type { Query } from "./api";
import type { Preset } from "./Control";
import { LifecycleDemo } from "./demos/LifecycleDemo";
import { SamplingDemo } from "./demos/SamplingDemo";
import { SbcDemo } from "./demos/SbcDemo";
import { ValvesDemo } from "./demos/ValvesDemo";
import PRESETS from "./presets.json";

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
  /** Three named scenarios that make the chapter's point; the middle one shows at first. */
  presets: readonly Preset[];
  /** The line that hands on to the next chapter. */
  handOn: Line;
};

/** A demo's presets: their parameters, from presets.json, which the server keeps answers for
 * too, each with its label. The labels are the design's proposed ones (E3), for the
 * maintainer's review in #660. */
function named(demo: keyof typeof PRESETS, labels: readonly string[]): Preset[] {
  const queries: Query[] = PRESETS[demo];
  if (queries.length !== labels.length) {
    throw new Error(`${demo} has ${queries.length} presets, and ${labels.length} labels`);
  }
  return queries.map((query, i) => ({ label: labels[i] ?? "", query }));
}

const HORIZONS = ["Six months", "A year", "Three years"];

export const CHAPTERS: readonly Chapter[] = [
  {
    key: "sampling",
    title: "Quota sampling",
    Demo: SamplingDemo,
    presets: named("sampling", ["Even rows", "Uneven rows", "Very uneven rows"]),
    handOn: {
      text: "Quota sampling lets a guided month hit its counts. Next, how a run is steered toward a plan.",
      draft: true,
    },
  },
  {
    key: "valves",
    title: "The three valves",
    Demo: ValvesDemo,
    presets: named("valves", ["No growth", "A modest plan, 5%", "An aggressive plan, 20%"]),
    handOn: {
      text: "The supply valve works through the requisitions growth opens. Next, how a requisition opens, waits, and closes.",
      draft: true,
    },
  },
  {
    key: "lifecycle",
    title: "The requisition lifecycle",
    Demo: LifecycleDemo,
    presets: named("lifecycle", HORIZONS),
    handOn: {
      text: "A filled opening is a hire, and a hire brings a grant. Next, how grants become expense.",
      draft: true,
    },
  },
  {
    key: "sbc",
    title: "The SBC layer",
    Demo: SbcDemo,
    presets: named("sbc", HORIZONS),
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

/** The mark after a draft's text. */
export function DraftMark() {
  return (
    <>
      {" "}
      <span className="draft" title="Placeholder text, for the maintainer's review in #660">
        draft
      </span>
    </>
  );
}

/** A paragraph of the story, marked when it's a draft. */
export function Prose({ line, className }: { line: Line; className?: string }) {
  return (
    <p className={className}>
      {line.text}
      {line.draft && <DraftMark />}
    </p>
  );
}
