// A chapter's controls (the design's E3): three named presets under its chart, then "Adjust",
// which opens the sliders for anyone who wants to go further.

import { useEffect, useId, useRef, useState, type ReactNode } from "react";

import type { Query } from "./api";

/** A named scenario: the parameters a chapter's demo runs with. */
export type Preset = { label: string; query: Query };

/** How long a slider must rest before the engine runs on its value. */
export const SETTLE_MS = 250;

function same(a: Query, b: Query): boolean {
  const keys = Object.keys(a);
  return keys.length === Object.keys(b).length && keys.every((k) => a[k] === b[k]);
}

/** A chapter's parameters: a preset's, the middle one at first, or the sliders' once adjusted.
 * `values` follow the sliders at once; `query`, what runs, follows once they settle. */
export function useControls(presets: readonly Preset[]) {
  const first = presets[Math.floor(presets.length / 2)]?.query ?? {};
  const [values, setValues] = useState<Query>(first);
  const [query, setQuery] = useState<Query>(first);
  const timer = useRef<ReturnType<typeof setTimeout>>(undefined);
  useEffect(() => () => clearTimeout(timer.current), []);
  return {
    values,
    query,
    /** The preset the parameters are, if they're one. */
    chosen: presets.findIndex((p) => same(p.query, values)),
    choose(i: number) {
      const preset = presets[i];
      if (preset !== undefined) {
        clearTimeout(timer.current);
        setValues(preset.query);
        setQuery(preset.query);
      }
    },
    adjust(change: Query) {
      const next = { ...values, ...change };
      setValues(next);
      clearTimeout(timer.current);
      timer.current = setTimeout(() => setQuery(next), SETTLE_MS);
    },
  };
}

/** A radio group of named choices, drawn as a row of pills. Native radio inputs, so the arrow
 * keys move through them. */
export function Choices({
  label,
  options,
  chosen,
  onChoose,
  className,
}: {
  label: string;
  options: readonly string[];
  chosen: number;
  onChoose: (i: number) => void;
  className: string;
}) {
  const group = useId();
  return (
    <div className={`choices ${className}`} role="radiogroup" aria-label={label}>
      {options.map((option, i) => (
        <label key={option} className={i === chosen ? "choice chosen" : "choice"}>
          <input type="radio" name={group} checked={i === chosen} onChange={() => onChoose(i)} />
          {option}
        </label>
      ))}
    </div>
  );
}

/** A chapter's presets, a radio group labelled with the chapter's name. */
export function Presets({
  name,
  presets,
  chosen,
  onChoose,
}: {
  name: string;
  presets: readonly Preset[];
  chosen: number;
  onChoose: (i: number) => void;
}) {
  return <Choices label={name} options={presets.map((p) => p.label)} chosen={chosen} onChoose={onChoose} className="presets" />;
}

/** "Adjust": a disclosure that holds a chapter's sliders, named for its chapter, so each chapter's
 * Adjust says which it opens. */
export function Adjust({ name, children }: { name: string; children: ReactNode }) {
  const [open, setOpen] = useState(false);
  const id = useId();
  return (
    <div className="adjust">
      <button type="button" className="adjust-toggle" aria-label={`Adjust: ${name}`} aria-expanded={open} aria-controls={id} onClick={() => setOpen((o) => !o)}>
        Adjust
      </button>
      <div id={id} className="controls" hidden={!open}>
        {children}
      </div>
    </div>
  );
}

/** A labelled slider for one of a demo's parameters. */
export function Control(props: {
  label: string;
  value: number;
  min: number;
  max: number;
  step?: number;
  onChange: (value: number) => void;
}) {
  const { label, value, min, max, step = 1, onChange } = props;
  return (
    <label className="control">
      <span>
        {label}: <b>{value}</b>
      </span>
      <input
        type="range"
        min={min}
        max={max}
        step={step}
        value={value}
        aria-label={label}
        onChange={(e) => onChange(Number(e.target.value))}
      />
    </label>
  );
}

/** What a demo is doing: running, announced politely, or failed, as an alert. */
export function Status({ loading, error }: { loading: boolean; error?: string }) {
  return (
    <>
      <p className="loading" role="status">
        {loading ? "Running…" : ""}
      </p>
      {error !== undefined && (
        <p className="error" role="alert">
          {error}
        </p>
      )}
    </>
  );
}
