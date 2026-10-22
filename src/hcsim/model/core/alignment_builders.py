"""Builders for alignment targets: from a plan table, or synthetic.

`from_frame` reads a plan table as a `FrameSpec` describes it:

- **layout**: `long` (a row per group and month) or `wide` (a column per month);
- **time keys**: a long table's month column, or a wide table's month headers, read as months
  through an optional format;
- **aggregation**: how rows that share a group and month combine (`sum`, `mean`, or `last`),
  as a plan fanned out by job family sums to its cost centers;
- **a population filter**: a population's rules, on whatever of their columns the table has;
- **exclusions**: values of named columns whose rows are dropped, such as hedge rows;
- **anchors**: targets set at the window's first month, such as each group's current
  headcount, so a plan's path starts where the workforce is.

Rows outside the window are dropped, and a negative target left after aggregation is clipped
to zero. Everything dropped, summed, or clipped is counted in the target's normalization.

`synthetic` builds targets without a table, for tests and studies: `constant`, `linear` (from a
base to a final value), `step` (a jump at a month), `dates` (given values at given months), and
`relative` (a base scaled by a factor at given months).
"""

from __future__ import annotations

from collections.abc import Collection, Mapping, Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Literal

import numpy as np
import pandas as pd

from hcsim.model.core.alignment import (
    AlignmentError,
    AlignmentTarget,
    Interpolation,
    Marginal,
)
from hcsim.model.data.population import PopulationSpec


class Aggregation(StrEnum):
    SUM = "sum"
    MEAN = "mean"
    LAST = "last"


@dataclass(frozen=True)
class FrameSpec:
    grouping: str
    layout: Literal["long", "wide"] = "long"
    #: A long table's value and month columns.
    value: str = "target_headcount"
    time: str = "month_end"
    #: How month keys are written, for `pd.to_datetime`; None reads ISO 8601.
    time_format: str | None = None
    #: A long table's band column, if it has one.
    band: str | None = None
    aggregate: Aggregation = Aggregation.SUM
    population: PopulationSpec | None = None
    exclusions: Mapping[str, Collection[object]] = field(default_factory=dict)
    anchors: Mapping[str, float] = field(default_factory=dict)


def from_frame(
    frame: pd.DataFrame,
    spec: FrameSpec,
    *,
    start: pd.Period | str,
    months: int,
    planned_depth: int,
    interpolation: Interpolation = Interpolation.HOLD_LAST,
) -> AlignmentTarget:
    counts: dict[str, int] = {}
    rows = frame.copy()
    if spec.grouping not in rows.columns:
        raise AlignmentError(f"the table has no grouping column {spec.grouping!r}")
    for column, values in spec.exclusions.items():
        if column not in rows.columns:
            raise AlignmentError(f"the table has no column {column!r} to exclude by")
        drop = rows[column].isin(list(values))
        counts[f"excluded_by_{column}"] = int(drop.sum())
        rows = rows[~drop]
    if spec.population is not None:
        keep = np.ones(len(rows), dtype=bool)
        for column, admitted in spec.population.rules.items():
            if column in rows.columns:
                keep &= rows[column].astype(str).isin(admitted).to_numpy()
        counts["outside_population"] = int((~keep).sum())
        rows = rows[keep]
    long = _long(rows, spec)
    first = pd.Period(start, freq="M")
    last = first + months - 1
    ordinal = np.array([m.ordinal for m in long["month"]], dtype=np.int64)
    inside = (ordinal >= first.ordinal) & (ordinal <= last.ordinal)
    counts["outside_window"] = int((~inside).sum())
    long = long[inside]
    grouped = long.groupby(["group", "month"], sort=True)
    counts["duplicates_aggregated"] = int(len(long) - grouped.ngroups)
    method = {"sum": "sum", "mean": "mean", "last": "last"}[Aggregation(spec.aggregate).value]
    cells = grouped["target"].agg(method).reset_index()
    if "band" in long.columns:
        cells["band"] = grouped["band"].agg(method).to_numpy()
    for group, value in spec.anchors.items():
        cells = cells[~((cells["group"] == group) & (cells["month"] == first))]
        cells = pd.concat(
            [cells, pd.DataFrame({"group": [group], "month": [first], "target": [float(value)]})],
            ignore_index=True,
        )
    counts["anchored"] = len(spec.anchors)
    negative = cells["target"] < 0
    counts["negatives_clipped"] = int(negative.sum())
    cells.loc[negative, "target"] = 0.0
    if cells.empty:
        raise AlignmentError("nothing of the table is left to target")
    return AlignmentTarget(
        (_marginal(spec.grouping, cells),),
        first,
        months,
        planned_depth,
        interpolation,
        {k: v for k, v in counts.items() if v},
    )


def _long(rows: pd.DataFrame, spec: FrameSpec) -> pd.DataFrame:
    if spec.layout == "long":
        missing = sorted({spec.value, spec.time} - set(rows.columns))
        if missing:
            raise AlignmentError(f"a long table needs columns {missing}")
        out = pd.DataFrame(
            {
                "group": rows[spec.grouping].astype(str).to_numpy(),
                "month": _months(rows[spec.time], spec.time_format),
                "target": rows[spec.value].to_numpy(dtype=np.float64),
            }
        )
        if spec.band is not None:
            out["band"] = rows[spec.band].to_numpy(dtype=np.float64)
        return out
    keys = [c for c in rows.columns if c != spec.grouping and c not in spec.exclusions]
    melted = rows.melt(
        id_vars=[spec.grouping], value_vars=keys, var_name="key", value_name="target"
    )
    melted = melted.dropna(subset=["target"])
    return pd.DataFrame(
        {
            "group": melted[spec.grouping].astype(str).to_numpy(),
            "month": _months(melted["key"], spec.time_format),
            "target": melted["target"].to_numpy(dtype=np.float64),
        }
    )


def _months(keys: pd.Series, time_format: str | None) -> pd.Series:
    try:
        stamps = pd.to_datetime(keys.astype(str), format=time_format or "ISO8601")
    except (ValueError, TypeError) as error:
        raise AlignmentError(f"time keys don't read as dates: {error}") from None
    return pd.Series(stamps.dt.to_period("M").to_numpy(), index=keys.index)


def _marginal(grouping: str, cells: pd.DataFrame) -> Marginal:
    groups = sorted(cells["group"].unique())
    dates = sorted(set(cells["month"]))
    values = np.full((len(groups), len(dates)), np.nan)
    row = {g: i for i, g in enumerate(groups)}
    col = {d: j for j, d in enumerate(dates)}
    i = cells["group"].map(row).to_numpy(dtype=np.int64)
    j = np.array([col[m] for m in cells["month"]], dtype=np.int64)
    values[i, j] = cells["target"].to_numpy(dtype=np.float64)
    bands = None
    if "band" in cells.columns and cells["band"].notna().any():
        bands = np.full_like(values, np.nan)
        bands[i, j] = cells["band"].to_numpy(dtype=np.float64)
    return Marginal(grouping, tuple(groups), tuple(dates), values, bands)


class Shape(StrEnum):
    CONSTANT = "constant"
    LINEAR = "linear"
    STEP = "step"
    DATES = "dates"
    RELATIVE = "relative"


def synthetic(
    shape: Shape | str,
    grouping: str,
    base: Mapping[str, float],
    *,
    start: pd.Period | str,
    months: int,
    planned_depth: int | None = None,
    final: Mapping[str, float] | float | None = None,
    at: pd.Period | str | None = None,
    jump: Mapping[str, float] | float = 0.0,
    values: Mapping[str, Sequence[float]] | None = None,
    factors: Sequence[float] | None = None,
    dates: Sequence[pd.Period | str] = (),
) -> AlignmentTarget:
    """A synthetic target for every group in `base`, the groups' starting headcount:

    - `constant`: `base` every month;
    - `linear`: a line from `base` to `final` over the window;
    - `step`: `base`, then `base + jump` from month `at`;
    - `dates`: `values[group]` at `dates`;
    - `relative`: `base` times `factors` at `dates`.
    """
    shape = Shape(shape)
    first = pd.Period(start, freq="M")
    groups = sorted(base)
    if not groups:
        raise AlignmentError("a synthetic target needs at least one group")
    window = [first + k for k in range(months)]
    b = np.array([float(base[g]) for g in groups])

    def per_group(x: Mapping[str, float] | float) -> np.ndarray:
        return (
            np.array([float(x[g]) for g in groups])
            if isinstance(x, Mapping)
            else np.full(len(groups), float(x))
        )

    if shape is Shape.CONSTANT:
        months_used, grid = window, np.repeat(b[:, None], months, axis=1)
    elif shape is Shape.LINEAR:
        if final is None:
            raise AlignmentError("a linear target needs a final value")
        end = per_group(final)
        steps = np.linspace(0.0, 1.0, months) if months > 1 else np.ones(1)
        months_used, grid = window, b[:, None] + (end - b)[:, None] * steps[None, :]
    elif shape is Shape.STEP:
        if at is None:
            raise AlignmentError("a step target needs the month it steps at")
        moment = pd.Period(at, freq="M")
        after = np.array([m >= moment for m in window])
        grid = np.where(after[None, :], (b + per_group(jump))[:, None], b[:, None])
        months_used = window
    else:
        chosen = [pd.Period(d, freq="M") for d in dates]
        if not chosen:
            raise AlignmentError(f"a {shape.value} target needs dates")
        months_used = chosen
        if shape is Shape.DATES:
            if values is None or sorted(values) != groups:
                raise AlignmentError("a dates target needs values for every group")
            grid = np.array([[float(v) for v in values[g]] for g in groups])
        else:
            if factors is None or len(factors) != len(chosen):
                raise AlignmentError("a relative target needs one factor per date")
            grid = b[:, None] * np.asarray(factors, dtype=np.float64)[None, :]
        if grid.shape != (len(groups), len(chosen)):
            raise AlignmentError("a dates target needs one value per date for every group")
    target = Marginal(grouping, tuple(groups), tuple(months_used), np.maximum(grid, 0.0))
    return AlignmentTarget((target,), first, months, planned_depth or months)
