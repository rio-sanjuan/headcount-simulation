// The explorer's story (the design's E1): four chapters in the engine's order of ideas, each one
// demo with the line that hands on to the next.

import type { ComponentType } from "react";

import { LifecycleDemo } from "./demos/LifecycleDemo";
import { SamplingDemo } from "./demos/SamplingDemo";
import { SbcDemo } from "./demos/SbcDemo";
import { ValvesDemo } from "./demos/ValvesDemo";

/** A line of the story's text. A draft is a placeholder that waits for the maintainer's review
 * (#660), and the page marks it so. */
export type Line = { text: string; draft?: boolean };

export type Chapter = {
  /** The chapter's anchor, so a link can open the page at it. */
  key: string;
  title: string;
  Demo: ComponentType;
  /** The line that hands on to the next chapter. */
  handOn: Line;
};

export const CHAPTERS: readonly Chapter[] = [
  {
    key: "sampling",
    title: "Quota sampling",
    Demo: SamplingDemo,
    handOn: {
      text: "Quota sampling lets a guided month hit its counts. Next, how a run is steered toward a plan.",
      draft: true,
    },
  },
  {
    key: "valves",
    title: "The three valves",
    Demo: ValvesDemo,
    handOn: {
      text: "The supply valve works through the requisitions growth opens. Next, how a requisition opens, waits, and closes.",
      draft: true,
    },
  },
  {
    key: "lifecycle",
    title: "The requisition lifecycle",
    Demo: LifecycleDemo,
    handOn: {
      text: "A filled opening is a hire, and a hire brings a grant. Next, how grants become expense.",
      draft: true,
    },
  },
  {
    key: "sbc",
    title: "The SBC layer",
    Demo: SbcDemo,
    handOn: {
      text: "That's the engine's story. The studies and their scorecards compare runs in depth.",
      draft: true,
    },
  },
];

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
