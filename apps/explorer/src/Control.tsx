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

/** What a demo shows while it loads or when it fails. */
export function Status({ loading, error }: { loading: boolean; error?: string }) {
  if (error !== undefined) {
    return <p className="error">{error}</p>;
  }
  return loading ? <p className="loading">Running…</p> : null;
}
