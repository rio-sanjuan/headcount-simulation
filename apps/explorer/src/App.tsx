import { useState } from "react";

import { LifecycleDemo } from "./demos/LifecycleDemo";
import { SamplingDemo } from "./demos/SamplingDemo";
import { SbcDemo } from "./demos/SbcDemo";
import { ValvesDemo } from "./demos/ValvesDemo";

/** The explorer's demos, each a tab. */
export const DEMOS = [
  { key: "sampling", title: "Quota vs independent sampling", Demo: SamplingDemo },
  { key: "valves", title: "The three valves", Demo: ValvesDemo },
  { key: "lifecycle", title: "The requisition lifecycle", Demo: LifecycleDemo },
  { key: "sbc", title: "The SBC layer", Demo: SbcDemo },
] as const;

export function App() {
  const [active, setActive] = useState<string>(DEMOS[0].key);
  const current = DEMOS.find((d) => d.key === active) ?? DEMOS[0];
  return (
    <main>
      <header>
        <h1>hcsim explorer</h1>
        <p>An interactive look at how the headcount and SBC simulator works, on a synthetic world.</p>
      </header>
      <nav role="tablist">
        {DEMOS.map((d) => (
          <button
            key={d.key}
            role="tab"
            aria-selected={d.key === active}
            className={d.key === active ? "active" : ""}
            onClick={() => setActive(d.key)}
          >
            {d.title}
          </button>
        ))}
      </nav>
      <current.Demo />
    </main>
  );
}
