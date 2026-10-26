import { useEffect, useState } from "react";

import { get, type Query } from "./api";

export type Loaded<T> = { data?: T; error?: string; loading: boolean };

/** Every answer asked for, by demo and query, so asking again, as a preset does, answers at once.
 * Adjust's answers are kept too, in memory only. */
const kept = new Map<string, Promise<unknown>>();
/** The answers that have arrived, so a page can draw one without waiting a tick. */
const settled = new Map<string, unknown>();

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
    answer.then(
      (value) => settled.set(key, value),
      () => kept.delete(key),
    );
  }
  return answer as Promise<T>;
}

/** Asks for every query, in order, without waiting for any. */
export function prefetch(asks: readonly { demo: string; query: Query }[]): void {
  for (const { demo, query } of asks) {
    ask(demo, query).catch(() => {
      // The chapter that shows this answer asks again, and says why it failed.
    });
  }
}

/** Forgets every answer, as a test needs between pages. */
export function forget(): void {
  kept.clear();
  settled.clear();
}

/** One demo's answer for `query`. While a new query runs, the last answer stays, with `loading`
 * set, so a chart can show it until the new one arrives. */
export function useDemo<T>(demo: string, query: Query): Loaded<T> {
  const key = keyOf(demo, query);
  const json = JSON.stringify(query);
  const [last, setLast] = useState<{ key: string; data?: T; error?: string }>({ key: "" });
  useEffect(() => {
    let current = true;
    ask<T>(demo, JSON.parse(json) as Query).then(
      (data) => current && setLast({ key, data }),
      (error: unknown) => {
        const message = error instanceof Error ? error.message : String(error);
        if (current) {
          setLast({ key, error: message });
        }
      },
    );
    return () => {
      current = false;
    };
  }, [demo, key, json]);
  if (settled.has(key)) {
    return { data: settled.get(key) as T, loading: false };
  }
  if (last.key === key) {
    return { data: last.data, error: last.error, loading: false };
  }
  return { data: last.data, loading: true };
}
