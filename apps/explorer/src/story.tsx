// The explorer's story (the design's E1): four chapters in the engine's order of ideas. Each
// opens with a few lines on its idea, runs one demo, and ends with a line that hands on to the
// next. Every word a reader sees outside the charts is here, or in a demo's line under its chart.

import type { ComponentType, ReactNode } from "react";

import type { Query } from "./api";
import type { Preset } from "./Control";
import { LifecycleDemo } from "./demos/LifecycleDemo";
import { SamplingDemo } from "./demos/SamplingDemo";
import { SbcDemo } from "./demos/SbcDemo";
import { ValvesDemo } from "./demos/ValvesDemo";
import PRESETS from "./presets.json";

/** What a chapter gives its demo: its name, to label the presets, and the presets. */
export type DemoProps = { name: string; presets: readonly Preset[] };

export type Chapter = {
  /** The chapter's anchor, so a link can open the page at it, and the demo it runs. */
  key: string;
  title: string;
  Demo: ComponentType<DemoProps>;
  /** A few lines on the chapter's idea, before its demo. */
  intro: ReactNode;
  /** Three named scenarios that make the chapter's point; the middle one shows at first. */
  presets: readonly Preset[];
  /** The line that hands on to the next chapter. */
  handOn: string;
};

/** A demo's presets: their parameters, from presets.json, which the server keeps answers for
 * too, each with its label, which names a scenario rather than a number. */
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
    intro: (
      <>
        <p>
          A guided month knows how many people to hire, move, or let go, but not who. The engine
          asks its model how likely each person is, draws exactly the count guidance set, and lets
          the model choose who. That&apos;s quota sampling. Drawing each person on their own, at the
          same rates, would hit the count only on average, and miss it from draw to draw.
        </p>
        <p>
          The charts draw 200 synthetic rows whose probabilities add up to a quota of 20. Quota
          sampling draws the rows the model rates higher more often, but not in exact proportion.
          On very uneven rows, a row the model rates as certain is drawn 71% of the time, where
          independent draws always draw it. That&apos;s what drawing rows one at a time, without
          replacement, does.
        </p>
      </>
    ),
    presets: named("sampling", ["Even rows", "Uneven rows", "Very uneven rows"]),
    handOn: "Quota sampling settles a month's counts. Next, how a run is steered toward a plan.",
  },
  {
    key: "valves",
    title: "The three valves",
    Demo: ValvesDemo,
    intro: (
      <>
        <p>
          A plan says how many people each cost center should have, month by month. A run can be
          steered toward it through one of three valves. Supply opens the requisitions the
          plan&apos;s growth calls for. Flow aligns guidance&apos;s hires and exits to the plan.
          Rate calibrates the domains&apos; rates, so a run without guidance drifts toward it.
        </p>
        <p>
          Choose a valve to see its run against the plan, with the others beside it. Supply starts
          slowly: its requisitions take months to fill, so at 5% it ends below no valve. Flow still
          falls short where a plan grows faster than the openings can fill, and slides about 4
          heads below a flat plan over six months.
        </p>
      </>
    ),
    presets: named("valves", ["No growth", "A modest plan, 5%", "An aggressive plan, 20%"]),
    handOn: "Supply works through requisitions, so its runs turn on how they open and fill. Next, the requisition lifecycle.",
  },
  {
    key: "lifecycle",
    title: "The requisition lifecycle",
    Demo: LifecycleDemo,
    intro: (
      <p>
        A requisition opens (for growth, a campus class, or a backfill), waits as open openings,
        and leaves either filled, by one of five channels, or unfilled. Every month the stock
        changes by exactly what opened, less what filled and what left unfilled.
      </p>
    ),
    presets: named("lifecycle", HORIZONS),
    handOn: "A filled opening is a hire, and a hire brings a grant. Next, how grants become expense.",
  },
  {
    key: "sbc",
    title: "The SBC layer",
    Demo: SbcDemo,
    intro: (
      <>
        <p>
          A grant becomes tranches that vest over the years after it, and the books recognize
          each tranche&apos;s value as expense, month by month. A month&apos;s expense adds up
          from four parts:
        </p>
        <ul>
          <li>
            <b>net expense:</b> each tranche&apos;s value, spread from its award to its vest, less
            the forfeitures expected;
          </li>
          <li>
            <b>the vesting true-up:</b> at a vest, the gap between the value and what was
            recognized;
          </li>
          <li>
            <b>the attrition true-up:</b> when someone leaves, their unvested tranches reverse what
            they recognized;
          </li>
          <li>
            <b>the rate true-up:</b> when the forfeiture rate is revised.
          </li>
        </ul>
        <p>
          The spikes are the point, not noise: the vesting true-up jumps in the months tranches
          vest in bulk.
        </p>
      </>
    ),
    presets: named("sbc", HORIZONS),
    handOn:
      "That's the engine, from a month's draws to a year's expense. To compare runs in depth, the studies and their scorecards take it from here.",
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

/** A paragraph of the story. */
export function Prose({ text, className }: { text: string; className?: string }) {
  return <p className={className}>{text}</p>;
}
