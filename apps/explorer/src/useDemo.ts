import { useEffect, useState } from "react";

import { get, type Query } from "./api";

export type Loaded<T> = { data?: T; error?: string; loading: boolean };

/** One demo's answer for `query`, fetched again whenever the query changes. */
export function useDemo<T>(demo: string, query: Query): Loaded<T> {
  const [state, setState] = useState<Loaded<T>>({ loading: true });
  const key = JSON.stringify(query);
  useEffect(() => {
    let current = true;
    setState((s) => ({ ...s, loading: true }));
    get<T>(demo, JSON.parse(key) as Query)
      .then((data) => current && setState({ data, loading: false }))
      .catch((error: unknown) => {
        const message = error instanceof Error ? error.message : String(error);
        if (current) {
          setState({ error: message, loading: false });
        }
      });
    return () => {
      current = false;
    };
  }, [demo, key]);
  return state;
}
