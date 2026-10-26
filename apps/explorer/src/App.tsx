import { useEffect, useState } from "react";

import { health, type Health } from "./api";
import { ChapterRail, useChapterInView } from "./ChapterRail";
import { WorldContext } from "./provenance";
import { CHAPTERS, Prose, presetsToFetch } from "./story";
import { prefetch, useAnswerStates } from "./useDemo";

const NAME = "hcsim explorer";
const KEYS = CHAPTERS.map((c) => c.key);
/** Each chapter's first preset, whose answer the rail waits for. */
const FIRSTS = CHAPTERS.map((c) => ({ demo: c.key, query: c.presets[Math.floor(c.presets.length / 2)]?.query ?? {} }));
/** The command that starts the server, for a page that finds none. */
export const SERVE = "uv run hcsim app serve --world path/to/world.duckdb";

/** How long the page keeps the address's chapter in view while the chapters above it fill in. */
const HOLD_MS = 10_000;

/** Opens the page at the chapter `hash` names. The page renders after the browser looks for the
 * anchor, so it goes there itself, and the chapters above fill in as their runs arrive, so it
 * stays there until the reader scrolls, or a while has passed. Returns what stops it. */
export function holdAnchor(hash: string): () => void {
  const key = decodeURIComponent(hash.slice(1));
  const target = KEYS.includes(key) ? document.getElementById(key) : null;
  if (target === null) {
    return () => {};
  }
  const go = () => target.scrollIntoView();
  go();
  const main = target.closest("main");
  if (main === null || typeof ResizeObserver === "undefined") {
    return () => {};
  }
  const observer = new ResizeObserver(go);
  observer.observe(main);
  const reader = ["wheel", "touchstart", "keydown", "pointerdown"];
  const stop = () => {
    observer.disconnect();
    clearTimeout(timer);
    reader.forEach((event) => window.removeEventListener(event, stop));
  };
  const timer = setTimeout(stop, HOLD_MS);
  reader.forEach((event) => window.addEventListener(event, stop, { passive: true }));
  return stop;
}

/** The explorer: one page that tells its story in four chapters, with the rail beside them. */
export function App() {
  const current = useChapterInView(KEYS);
  useEffect(() => holdAnchor(window.location.hash), []);
  // The world, once the server says; "down" if it doesn't answer. It's asked for first, before
  // the presets take the browser's connections.
  const [world, setWorld] = useState<Health | "down" | undefined>(undefined);
  useEffect(() => {
    let current = true;
    void health().then((answer) => current && setWorld(answer ?? "down"));
    return () => {
      current = false;
    };
  }, []);
  useEffect(() => prefetch(presetsToFetch(CHAPTERS)), []);
  const states = useAnswerStates(FIRSTS);
  const chapters = CHAPTERS.map((c, i) => ({ ...c, state: states[i] }));
  if (world === "down") {
    return <NoServer />;
  }
  return (
    <WorldContext.Provider value={world}>
      <div className="page">
        <ChapterRail name={NAME} chapters={chapters} current={current} />
        <main>
          <header id="top">
            <h1>{NAME}</h1>
            <p className="lead">An interactive look at how the headcount and SBC simulator works, on a synthetic world.</p>
          </header>
          {CHAPTERS.map(({ key, title, intro, Demo, presets, handOn }, i) => (
            <section key={key} id={key} className="chapter" aria-labelledby={`${key}-title`}>
              <p className="chapter-number">Chapter {i + 1}</p>
              <h2 id={`${key}-title`}>{title}</h2>
              <div className="intro">{intro}</div>
              <Demo name={title} presets={presets} />
              <Prose text={handOn} className="hand-on" />
            </section>
          ))}
        </main>
      </div>
    </WorldContext.Provider>
  );
}

/** What the page says when the server doesn't answer: that it isn't running, and how to start it. */
function NoServer() {
  return (
    <main className="no-server">
      <h1>{NAME}</h1>
      <div className="chart-state failed" role="alert">
        <p>The explorer&apos;s server isn&apos;t running, so there&apos;s nothing to show yet. Start it from the repository:</p>
        <pre>
          <code>{SERVE}</code>
        </pre>
        <button type="button" onClick={() => window.location.reload()}>
          Try again
        </button>
      </div>
    </main>
  );
}
