// The chapter rail (the design's E2): the explorer's name and its numbered chapters, the one in
// view marked. In a wide window it stays at the left as the page scrolls. In a narrow one it
// folds into a bar across the top that names the chapter in view and opens the list.

import { useEffect, useState } from "react";

/** How far down the window a chapter's top must reach for it to count as the one in view. */
const IN_VIEW = 0.3;

/** The key of the chapter in view: the last whose top has reached `IN_VIEW` of the way down the
 * window, or the last of all once the page is scrolled to its end. */
export function useChapterInView(keys: readonly string[]): string {
  const first = keys[0] ?? "";
  const [current, setCurrent] = useState(first);
  useEffect(() => {
    let frame = 0;
    const update = () => {
      frame = 0;
      const line = window.innerHeight * IN_VIEW;
      let found = first;
      for (const key of keys) {
        const top = document.getElementById(key)?.getBoundingClientRect().top;
        if (top !== undefined && top <= line) {
          found = key;
        }
      }
      const end = document.documentElement.scrollHeight - 2;
      if (window.scrollY > 0 && window.innerHeight + window.scrollY >= end) {
        found = keys.at(-1) ?? found;
      }
      setCurrent(found);
    };
    const later = () => {
      if (frame === 0) {
        frame = requestAnimationFrame(update);
      }
    };
    update();
    window.addEventListener("scroll", later, { passive: true });
    window.addEventListener("resize", later);
    return () => {
      cancelAnimationFrame(frame);
      window.removeEventListener("scroll", later);
      window.removeEventListener("resize", later);
    };
  }, [keys, first]);
  return current;
}

export function ChapterRail({
  name,
  chapters,
  current,
}: {
  name: string;
  chapters: readonly { key: string; title: string; state?: "waiting" | "ready" | "failed" }[];
  current: string;
}) {
  const [open, setOpen] = useState(false);
  const at = Math.max(0, chapters.findIndex((c) => c.key === current));
  useEffect(() => {
    if (!open) {
      return;
    }
    const close = (event: KeyboardEvent) => {
      if (event.key === "Escape") {
        setOpen(false);
      }
    };
    window.addEventListener("keydown", close);
    return () => window.removeEventListener("keydown", close);
  }, [open]);
  return (
    <nav className={open ? "rail open" : "rail"} aria-label="Chapters">
      <a className="rail-name" href="#top">
        {name}
      </a>
      <button
        type="button"
        className="rail-toggle"
        aria-expanded={open}
        aria-controls="rail-chapters"
        onClick={() => setOpen((o) => !o)}
      >
        {at + 1} of {chapters.length}: {chapters[at]?.title}
      </button>
      <ol id="rail-chapters">
        {chapters.map((c, i) => (
          <li key={c.key}>
            <a
              href={`#${c.key}`}
              aria-current={c.key === current ? "location" : undefined}
              onClick={() => setOpen(false)}
            >
              <span className="rail-number">{i + 1}</span> {c.title}
              {c.state !== undefined && c.state !== "ready" && (
                <>
                  <span className={`rail-state ${c.state}`} aria-hidden="true" />{" "}
                  <span className="hidden-text">{c.state === "waiting" ? "(still running)" : "(failed)"}</span>
                </>
              )}
            </a>
          </li>
        ))}
      </ol>
    </nav>
  );
}
