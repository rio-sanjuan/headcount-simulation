import { useEffect } from "react";

import { ChapterRail, useChapterInView } from "./ChapterRail";
import { CHAPTERS, Prose, presetsToFetch } from "./story";
import { prefetch } from "./useDemo";

const NAME = "hcsim explorer";
const KEYS = CHAPTERS.map((c) => c.key);

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
  useEffect(() => prefetch(presetsToFetch(CHAPTERS)), []);
  return (
    <div className="page">
      <ChapterRail name={NAME} chapters={CHAPTERS} current={current} />
      <main>
        <header id="top">
          <h1>{NAME}</h1>
          <p className="lead">An interactive look at how the headcount and SBC simulator works, on a synthetic world.</p>
        </header>
        {CHAPTERS.map(({ key, title, Demo, presets, handOn }, i) => (
          <section key={key} id={key} className="chapter" aria-labelledby={`${key}-title`}>
            <p className="chapter-number">Chapter {i + 1}</p>
            <h2 id={`${key}-title`}>{title}</h2>
            <Demo name={title} presets={presets} />
            <Prose line={handOn} className="hand-on" />
          </section>
        ))}
      </main>
    </div>
  );
}
