import { useEffect, useState, useSyncExternalStore } from "react";

import { get, type Query } from "./api";

export type Loaded<T> = {
  data?: T;
  error?: string;
  loading: boolean;
  /** Asks again for an answer that failed. */
  retry: () => void;
};

/** Every answer asked for, by demo and query, so asking again, as a preset does, answers at once.
 * Adjust's answers are kept too, in memory only. */
const kept = new Map<string, Promise<unknown>>();
/** The answers that have arrived, so a page can draw one without waiting a tick. */
const settled = new Map<string, unknown>();
/** The answers that failed, until they're asked for again. */
const failed = new Set<string>();
/** Who wants to know when an answer arrives or fails. */
const listeners = new Set<() => void>();

function changed(): void {
  listeners.forEach((listener) => listener());
}

/** A demo's query as a key, the same whatever order its parameters come in. */
export function keyOf(demo: string, query: Query): string {
  const params = Object.keys(query)
    .sort()
    .map((k) => `${k}=${query[k]}`);
  return `${demo}?${params.join("&")}`;
}

/** One demo's answer for `query`: asked for once, then kept. A failed answer isn't kept, so
 * asking again runs it again. */
export function ask<T>(demo: string, query: Query): Promise<T> {
  const key = keyOf(demo, query);
  let answer = kept.get(key);
  if (answer === undefined) {
    answer = get<T>(demo, query);
    kept.set(key, answer);
    if (failed.delete(key)) {
      changed();
    }
    answer.then(
      (value) => {
        settled.set(key, value);
        changed();
      },
      () => {
        kept.delete(key);
        failed.add(key);
        changed();
      },
    );
  }
  return answer as Promise<T>;
}

/** How many prefetched answers are asked for at once. A browser opens about six connections to
 * a server, so this leaves room for the page's own questions, and a reader's clicks. */
export const PREFETCH_AT_ONCE = 4;

/** Asks for every query, in order, a few at a time. */
export function prefetch(asks: readonly { demo: string; query: Query }[]): void {
  const queue = [...asks];
  const next = (): void => {
    const item = queue.shift();
    if (item !== undefined) {
      ask(item.demo, item.query)
        .catch(() => {
          // The chapter that shows this answer says why it failed, and can ask again.
        })
        .finally(next);
    }
  };
  for (let k = 0; k < PREFETCH_AT_ONCE; k++) {
    next();
  }
}

/** Forgets every answer, as a test needs between pages. */
export function forget(): void {
  kept.clear();
  settled.clear();
  failed.clear();
  changed();
}

export type AnswerState = "waiting" | "ready" | "failed";

function subscribe(listener: () => void): () => void {
  listeners.add(listener);
  return () => listeners.delete(listener);
}

/** Whether each query's answer is still running, has arrived, or failed, following them as they
 * change, as the rail does for each chapter's first preset. */
export function useAnswerStates(asks: readonly { demo: string; query: Query }[]): AnswerState[] {
  const keys = asks.map((a) => keyOf(a.demo, a.query));
  const snapshot = useSyncExternalStore(subscribe, () =>
    keys.map((k) => (settled.has(k) ? "ready" : failed.has(k) ? "failed" : "waiting")).join(","),
  );
  return snapshot.split(",") as AnswerState[];
}

/** One demo's answer for `query`. While a new query runs, the last answer stays, with `loading`
 * set, so a chart can show it until the new one arrives. */
export function useDemo<T>(demo: string, query: Query): Loaded<T> {
  const key = keyOf(demo, query);
  const json = JSON.stringify(query);
  const [last, setLast] = useState<{ key: string; attempt: number; data?: T; error?: string }>({
    key: "",
    attempt: 0,
  });
  const [attempt, setAttempt] = useState(0);
  const retry = () => setAttempt((a) => a + 1);
  useEffect(() => {
    let current = true;
    ask<T>(demo, JSON.parse(json) as Query).then(
      (data) => current && setLast({ key, attempt, data }),
      (error: unknown) => {
        const message = error instanceof Error ? error.message : String(error);
        if (current) {
          setLast({ key, attempt, error: message });
        }
      },
    );
    return () => {
      current = false;
    };
  }, [demo, key, json, attempt]);
  if (settled.has(key)) {
    return { data: settled.get(key) as T, loading: false, retry };
  }
  if (last.key === key && last.attempt === attempt) {
    return { data: last.data, error: last.error, loading: false, retry };
  }
  // The last answer shown, while this one runs; none if the last one failed.
  return { data: last.data, loading: true, retry };
}
